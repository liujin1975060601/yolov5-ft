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
from utils.metrics import ConfusionMatrix, ap_per_class, poly_iou, ap_per_class_dir
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from tools.plotbox import plot_one_rot_box


def vector_dot(labels1, labels2):
    """
    labels1: (N, 8)
    labels2 pred: (M, 8)
    """
    device = labels1.device
    n = labels1.shape[0]
    m = labels2.shape[0]
    dot = torch.zeros((n, m), dtype=torch.float32).to(device)
    if n:
        vec1 = rbox2vec(labels1)
        vec2 = rbox2vec(labels2)
        s1 = torch.sqrt((vec1 ** 2).sum(dim=1))
        s2 = torch.sqrt((vec2 ** 2).sum(dim=1))
        for i in range(n):
            for j in range(m):
                cos_t = (vec1[i] * vec2[j]).sum() / s1[i] * s2[j]
                dot[i][j] = cos_t
    return dot

        

def rbox2vec(rbox):
    cx = rbox[:, ::2].sum(dim=1) / 4
    cy = rbox[:, 1::2].sum(dim=1) / 4
    px = (rbox[:, 0] + rbox[:, 2]) / 2
    py = (rbox[:, 1] + rbox[:, 3]) / 2
    dx = px-cx
    dy = py-cy
    L = torch.sqrt(dx ** 2 + dy ** 2)
    return torch.stack((dx/L, dy/L), dim=1)


def process_batch_poly(detections, labels, iouv, dotv=0.71):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2, x3,y3, x4,y4) format.
    Arguments:
        detections (Array[N, 10]), x1, y1, x2, y2, x3,y3, x4,y4 conf, class
        labels (Array[M, 9]), class, x1, y1, x2, y2, x3,y3, x4,y4
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    dir_correct = torch.zeros(detections.shape[0], dtype=torch.bool, device=iouv.device)
    # iou
    iou = poly_iou(labels[:, 1:], detections[:, :8])
    # dot
    dot = vector_dot(labels[:, 1:], detections[:, :8])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 9]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou, dir]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        # dir匹配
        #dir index
        iou_match = matches[matches[:, 2] >= iouv[0]]
        dir_match = dot[iou_match[:, 0].long(), iou_match[:, 1].long()]
        dir_correct[correct[:, 0]] = dir_match.abs() >= dotv
    # print(correct[:, 0])
    # print(dir_correct)
    # print(matches)
    # print(dir_match)
    return correct, dir_correct

def load_files(path):
    files = sorted(os.listdir(path))
    image_files = [os.path.join(path, x) for x in files]
    label_files = img2pts_paths(image_files)
    return image_files, label_files


def read_image_label(p1, p2):
    image = cv2.imread(p1)
    with open(p2) as f:
        p = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if len(p):
            p = np.array(p, dtype=np.float32)
            clss = p[:, 0].reshape(-1, 1)
            boxs = np.clip(p[:, 1:], 0, 1)
            p = np.concatenate((clss, boxs), axis=1)
        else:
            p = np.zeros((0, 9), dtype=np.float32)
    return image, p
    


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
        fold=1,
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
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'dacc')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    aacc = 0.0
    loss = torch.zeros(6, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    dir_stats = []
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
    image_files, label_files = load_files(data['val'])
    for image_path, label_path in tqdm(zip(image_files, label_files), desc=s, total=len(image_files)):
        t1 = time_sync()
        image, label = read_image_label(image_path, label_path)
        p = Path(image_path)
        save_path = str(save_dir / p.name)
        dataloader = create_bigimg_dataloader(image, imgsz, batch_size, sub_size=subsize, over_lap=overlap)
        t2 = time_sync()
        dt[0] += t2 - t1
        det_results = {}
        seen += 1
        for xys, cimg, im in dataloader:
            im = im.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = model(im, augment=augment, val=True)[0]
            # NMS
            pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold,mask_dir=mask_dir)

            for i, det in enumerate(pred):
                if len(det):
                    # xy = torch.tensor([[xy[0], xy[1]]])
                    xy = xys[i]
                    xy = xy.repeat(1, 4)
                    xy = xy.to(device)
                    det[:, :8] = scale_coords_poly(im.shape[2:], det[:, :8], cimg.shape[1:])
                    det[:, :8] = (xy + det[:, :8]).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        item = {
                            'xyxy': list(xyxy),
                            'conf': conf
                        }
                        items = det_results.setdefault(int(cls), [])
                        items.append(item)
        dt[1] += time_sync() - t2
        # 大图nms
        t3 = time_sync()
        keeps = []
        for cls, dets in det_results.items():
            det_list = []
            scores = []
            for item in dets:
                det_list.append(item['xyxy'])
                scores.append(item['conf'])
            det_list = torch.tensor(det_list)
            det_list = det_list.to(device)
            scores = torch.tensor(scores)
            scores = scores.to(device)
            indexes = big_nms(det_list, scores, iou_thres)
            cls = torch.tensor([cls]).to(device)
            for idx in indexes:
                xyxy = det_list[idx]
                keeps.append(torch.cat((xyxy, scores[idx][None], cls), dim=0))
        dt[2] += time_sync() - t3
        # plot
        # for k in keeps:
        #     xyxy = k[:8].cpu().numpy()
        #     cls = k[-1].cpu()
        #     plot_one_rot_box(xyxy, image, color=colors[int(cls)],   dir_line=True)
        # cv2.imwrite(save_path, image)

        targets = torch.from_numpy(label)
        targets = targets.to(device)
        if len(keeps):
            out = torch.stack(keeps, dim=0)
        else:
            out = torch.zeros(0)

        height, width = image.shape[:2]

        targets[:, 1:] *= torch.Tensor([width, height, width, height,width, height, width, height]).to(device)  # to pixels
        labels = targets
        pred = out
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                dir_stats.append(torch.zeros(0, dtype=torch.bool))
                continue
        if nl:
            correct,dir_correct = process_batch_poly(pred, labels, iouv)
            if plots:
                confusion_matrix.process_batch_poly(pred, labels)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            dir_correct = torch.zeros(pred.shape[0], dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 8].cpu(), pred[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)
        dir_stats.append(dir_correct.cpu())
        
    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    dir_tp = []
    for bs in dir_stats:
        for x in bs:
            dir_tp.append(x)
    dir_tp = np.array(dir_tp)
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, cls_dir_acc = ap_per_class_dir(*stats, dir_tp, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        aacc = cls_dir_acc.mean()
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.4g' * 5  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, aacc))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], cls_dir_acc[i]))

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
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')
    parser.add_argument('--fold', type=int, default=1, help='fold angle')
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
    opt.data = 'data/ucas.yaml'
    opt.weights = 'paperdatas/train/ucas-640-fold1-aug-ms/weights/best.pt'
    # opt.weights = 'runs/train/ucas-640-dir-cbam-88.41/weights/best.pt'
    # opt.weights = 'runs/train/exp/weights/best.pt'
    opt.batch_size = 4
    opt.imgsz = 640
    opt.conf_thres = 0.001
    opt.iou_thres = 0.1
    opt.ab_thres = 3.0
    opt.fold = 1
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
