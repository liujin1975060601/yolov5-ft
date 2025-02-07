# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import cv2

import numpy as np
import torch
from tqdm import tqdm
from val import process_batch_poly


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_bigimg_dataloader, img2pts_paths
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, rot_nms,  big_nms,
                           coco80_to_coco91_class, colorstr, increment_path, print_args,
                           scale_coords,scale_coords_poly, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, poly_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync,torch_distributed_zero_first
from tools.plotbox import plot_one_rot_box
from detect_big import detect_big



def load_files(path):
    files = sorted(os.listdir(path))
    image_files = [os.path.join(path, x) for x in files]
    label_files = img2pts_paths(image_files)
    return image_files, label_files


def read_image_label(im_file, pts_file):#参考dataset.py里面的verify_image_label
    image = cv2.imread(im_file)

    base_name, extension = os.path.splitext(pts_file)
    lb_file = base_name+'.txt'
    with open(lb_file) as f:
        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
        read_l = len(l)
        l = np.array(l, dtype=np.float32)#[nt,5]
        assert(l.shape[0]==read_l)
    
    with open(pts_file) as f:
        p = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if len(p):#参考dataset.py里面的verify_image_label
            for i in range(len(p)):
                if p[i]==['-']:
                    xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                    #p[i] = [str(round(xc-w/2,6)),'0','0','0','0','0','0','0']
                    p[i] = [str(round(xc-w/2,6)),str(round(yc-h/2,6)),
                            str(round(xc+w/2,6)),str(round(yc-h/2,6)),
                            str(round(xc+w/2,6)),str(round(yc+h/2,6)),
                            str(round(xc-w/2,6)),str(round(yc+h/2,6))]
            p = np.array(p, dtype=np.float32)#list转np矩阵[nt,4*2=8]

    nl = len(l)
    if nl:
        clss = l[:, 0].reshape(-1, 1)#[nt,1]，我感觉reshape(-1, 1)不必要，本来l[:, 0]就是在dim=1就只有长度1
        boxs = np.clip(l[:, 1:], 0, 1)#[nt,4] ,np.clip的意思是设定上下限,目标框的坐标和wh都限制在0,1范围内
        l = np.concatenate((clss, boxs), axis=1)#l从list转换成np矩阵[nt,1+4=5]
        #
        p = np.clip(p, 0, 1)
        l = np.concatenate((l, p), axis=1)#l[l(5)+p(4*2)]
        assert l.shape[1] == 13, f'labels require 5+8=13 columns, {l.shape[1]} columns detected'
        assert (l >= 0).all(), f'negative label values {l[l < 0]}'
        assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
        l = np.unique(l, axis=0)  # remove duplicate rows,  同一目标重复标注，合并
        if len(l) < nl:
            segments = np.unique(segments, axis=0)
            msg = f'WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
    else:
        ne = 1  # label empty
        l = np.zeros((0, 4+1 + 4*2), dtype=np.float32)
    return image, l#[nt, 4+1 + 4*2]

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        ab_thres=3.0,
        iou_thres=0.1,  # NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        fold=2,
        subsize=512,
        overlap=100,
        mask_dir=[]
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt = next(model.parameters()).device, True  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=False)
        stride, pt = model.stride, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(6, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    # colors
    colors = [
        (54, 67, 244),
        (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121),
        (180, 105, 255)]
    image_files, label_files = load_files(data['val_big'])
    for image_path, label_path in tqdm(zip(image_files, label_files), desc=s, total=len(image_files)):
        t1 = time_sync()
        image, label = read_image_label(image_path, label_path)#参考dataset.py里面的verify_image_label
        if(image.shape[0] < subsize):
            image = cv2.resize(image, (imgsz,imgsz))
        #image[H,W, C]   label[nt, 1(cls)+4(box) + 4*2]
        p = Path(image_path)
        save_path = str(save_dir / p.name)
        #
        keeps,dataloader = detect_big(model,half,device, image, imgsz, batch_size,subsize,overlap, conf_thres,iou_thres,ab_thres,fold,mask_dir)
        seen += 1
        # plot
        # for k in keeps:
        #     xyxy = k[:8].cpu().numpy()
        #     cls = k[-1].cpu()
        #     plot_one_rot_box(xyxy, image, color=colors[int(cls)],   dir_line=True)
        # cv2.imwrite(save_path, image)

        targets = torch.zeros((label.shape[0],1+4*2),device=device)
        targets[:, 0] = torch.from_numpy(label[:, 0])  # 拷贝cls部分
        targets[:, 1:] = torch.from_numpy(label[:, 1+4:])  # 拷贝除了4个框之外的部分
        #targets = torch.from_numpy(label)#targets[4+1 + 4*2] label[nt, 4+1 + 4*2]
        #targets = targets[..., 4:]#把前面的4(box)删掉了！targets[1(cls) + 4*2]
        #targets = targets.to(device)#targets[1(cls) + 4*2]
        if len(keeps):#keeps[nt][10 = 4(pts)*2+1(conf)+1(cls)]-->out[nt,10 = 4(pts)*2+1(conf)+1(cls)]
            out = torch.stack(keeps, dim=0)
        else:
            out = torch.zeros(0)

        height, width = image.shape[:2]

        targets[:, 1:] *= torch.Tensor([width, height, width, height, width, height, width, height]).to(device)  # to pixels
        labels = targets#[1(cls)+4(pts)*2]
        pred = out#[10 = 4(pts)*2+1(conf)+1(cls)]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
        if nl:
            correct = process_batch_poly(pred, labels, iouv)#pred[10 = 4(pts)*2+1(conf)+1(cls)] vs labels[9 = 1(cls)+4(pts)*2]
            if plots:
                confusion_matrix.process_batch_poly(pred, labels)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 8].cpu(), pred[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)
        
    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        print(threshs)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.4g' * 6  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference sub img at shape {shape}, %.1fms NMS per big image' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/Guge.yaml', help='dataset.yaml path')###
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp130/weights/best.pt', help='model.pt path(s)')###
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')###
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=768, help='inference size (pixels)')###
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')###
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')###
    parser.add_argument('--fold', type=int, default=2, help='倍角')###
    #opt.data = 'data/Guge.yaml'
    #opt.weights = 'runs/train/exp130/weights/best.pt'
    #opt.batch_size = 16
    #opt.imgsz = 768
    #opt.conf_thres = 0.001
    #opt.iou_thres = 0.1
    #opt.ab_thres = 3.0
    #opt.fold = 2
    
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--subsize', type=int, default=512, help='subsize')
    parser.add_argument('--overlap', type=int, default=100, help='overlap')
    opt = parser.parse_args()
    # opt.data = 'data/dota.yaml'
    # opt.weights = 'weights/best.pt'
    # opt.batch_size = 16
    # opt.imgsz = 768
    # opt.conf_thres = 0.001
    # opt.iou_thres = 0.1
    # opt.ab_thres = 3.0
    opt.subsize = opt.imgsz
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    opt.mask_dir = data_dict['mask_dir']

    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
