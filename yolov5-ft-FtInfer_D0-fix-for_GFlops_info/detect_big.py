# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import imp
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.datasets import create_bigimg_dataloader
from detect import detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadBigImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, big_nms,check_dataset,
                           increment_path,  print_args, scale_coords_poly,strip_optimizer, xyxy2xywh, rot_nms)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync, torch_distributed_zero_first
from tools.plotbox import plot_one_rot_box
from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast

def detect_big(model,half,device, image, imgsz, batch_size,subsize,overlap, conf_thres,iou_thres,ab_thres,fold,mask_dir,threshs=torch.zeros(0)):
    dataloader = create_bigimg_dataloader(image, imgsz, batch_size, sub_size=subsize, over_lap=overlap)
    det_results = {}
    for xys, cimg, im in dataloader:
        im = im.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        if 0:
            pred = model(im, augment=0, val=True)[0]
            #pred[2=Detect+Detect2][nl][b,a,H,W,Dectct(4(box)+1(conf)+nc) or Detect2(5=1(p)+2(dir)+2(ab))]
            pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold,mask_dir=mask_dir,threshs=threshs)
        else:
            pred = detect(model, False, im, False,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir,threshs=threshs)
        #pred[b][nt,10=4(pts)*2+1(conf)+1(cls)]

        for i, det in enumerate(pred):
            if len(det):
                # xy = torch.tensor([[xy[0], xy[1]]])
                xy = xys[i]
                xy = xy.repeat(1, 4)#4个点
                xy = xy.to(device)
                det[:, :8] = (xy + scale_coords_poly(im.shape[2:], det[:, :8], cimg.shape[1:])).round()
                #det[:, :8]除以im.shape[2:]再乘以cimg.shape[1:]加上左上角坐标加上左上角坐标xy得到大图绝对坐标
                
                for *xyxy, conf, cls in reversed(det):
                    item = {
                        'xyxy': list(xyxy),
                        'conf': conf
                    }
                    items = det_results.setdefault(int(cls), [])
                    items.append(item)
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
    return keeps,dataloader#keeps[nt][4(pts)*2+1(conf)+1(cls)]

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.1,  # confidence threshold
        ab_thres=3.0,
        iou_thres=0.1,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        half=True,  # use FP16 half-precision inference
        plot_label=True,
        dir_line=True,
        fold=2,
        subsize=512,
        overlap=100,
        data='',
        mask_dir=[]
        ):
    source = str(source)
    save_img = True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    # stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    stride, names = model.stride, model.names

    # Half
    pt = True
    if pt:
        model.model.half() if half else model.model.float()
    # Dataloader

    dataset = LoadBigImages(source, img_size=imgsz, stride=stride, auto=False, sub_size=subsize, over_lap=overlap)
    bs = 1  # batch_size

    train_path = Path(weights).parent.parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        conf_thres = min(threshs)

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
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

    # for path, im, im0s, vid_cap, s in dataset:
    for path, cut_imgs, convert_imgs, image,  s in dataset:#利用dataset切割管理器，取出每块信息，LoadBigImages::__next__
        print(path)
        p = path
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem)
        if not os.path.exists(str(save_dir / 'labels')):
            os.makedirs(str(save_dir / 'labels'))
        s += '%gx%g ' % image.shape[:2]
        det_results = {}
        t1 = time_sync()
        batch_size = 1
        t0 = time_sync()
        keeps,dataloader = detect_big(model,half,device, image, imgsz, batch_size,subsize,overlap, conf_thres,iou_thres,ab_thres,fold,mask_dir,threshs)
        #keeps[nt][4(pts)*2+1(conf)+1(cls)]
        t3 = time_sync()
        '''
        for cut_img, im in zip(cut_imgs, convert_imgs):
            xy, cimg = cut_img['xy'], cut_img['patch']
            #得到切图左上角坐标xy和切图cimg

            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize, val=True)[0] #pred[2=Detect+Detect2][nl][b,a,H,W,4(box)+1(conf)+nc]
            #pred[2=Detect+Detect2][nl][b,a,H,W,Dectct(4(box)+1(conf)+nc) or Detect2(5=1(p)+2(dir)+2(ab))]
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold,mask_dir=mask_dir)
            #pred[batchs][nt,10=pts[8]+conf[1]+clsid[1]]
            dt[2] += time_sync() - t3

            for i, det in enumerate(pred):#batch循环  得到det[nt,10=pts[8]+conf[1]+clsid[1]]
                seen += 1

                if len(det):#nt
                    xy = torch.tensor([[xy[0], xy[1]]])
                    xy = xy.repeat(1, 4)
                    xy = xy.to(device)
                    det[:, :8] = scale_coords_poly(im.shape[2:], det[:, :8], cimg.shape)
                    det[:, :8] = (xy + det[:, :8]).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    for *xyxy, conf, cls in reversed(det):#xyxy[nt,4*2]
                        item = {
                            'xyxy': list(xyxy),
                            'conf': conf
                        }
                        items = det_results.setdefault(int(cls), [])
                        items.append(item)#将目标item记录在归于每一类cls的字典items里


        # 大图nms
        objs = []
        for cls, dets in det_results.items():#按类cls循环
            det_list = []
            scores = []
            for item in dets:#每个目标
                det_list.append(item['xyxy'])
                scores.append(item['conf'])
            det_list = torch.tensor(det_list)#det_list[nt,8=4*2]
            det_list = det_list.to(device)
            scores = torch.tensor(scores)#scores[nt]
            scores = scores.to(device)
            print('NMS前', names[int(cls)], len(det_list))
            indexes = big_nms(det_list, scores, iou_thres)#indexes是nms去重过滤之后剩下的索引
            print('NMS后', names[int(cls)], len(indexes))
            for idx in indexes:#根据indexes这个遍历nms去重过滤之后剩下的索引
                xyxy = det_list[idx]#xyxy[8=4*2]
                conf = float('{:.2f}'.format(scores[idx]))#conf[1]
                line = (cls, *xyxy, conf) # label format
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                #xyxy = [x.cpu() for x in xyxy]#xyxy[8]:list
                obj = torch.cat((xyxy.cpu(),torch.tensor([conf]),torch.tensor([cls])),dim=0)#obj[4(pts)*2+1(conf)+1(cls)]
                objs.append(obj)
                #plot_one_rot_box(xyxy.cpu().numpy(), image, color=colors[int(cls)%len(colors)], label=label, dir_line=dir_line, line_thickness=line_thickness)
        for obj in objs:
            pts = obj[:8]
            conf = obj[8]
            cls = obj[9]
            label = '{} {:.2f}'.format(names[int(cls)], conf) if plot_label else None
            plot_one_rot_box(pts, image, color=colors[int(cls)%len(colors)], label=label, dir_line=dir_line, line_thickness=line_thickness)
        '''
        # plot
        for k in keeps:#keeps[nt][4(pts)*2+1(conf)+1(cls)]
            pts = k[:8].cpu().numpy()#k[4(pts)*2+1(conf)+1(cls)]
            conf = k[8].cpu()
            cls = k[-1].cpu()
            label = '{} {:.2f}'.format(names[int(cls)], conf) if plot_label else None
            plot_one_rot_box(pts, image, color=colors[int(cls)%len(colors)], label=label, dir_line=True)

        cv2.imwrite(save_path, image)

        # # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t0:.3f}s)')




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')
    parser.add_argument('--plot_label',action='store_true', default=True, help='plot labels')
    parser.add_argument('--fold', type=int, default=1, help='fold angle')
    parser.add_argument('--subsize', type=int, default=512, help='subsize')
    parser.add_argument('--overlap', type=int, default=100, help='overlap')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    opt.weights = 'runs/train/exp130/weights/best.pt'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS/val/images'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA512/val-big/images'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS_split/val/images'
    #opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS/val/images'
    opt.source = '/home/liu/data/home/liu/workspace/darknet/datas/GuGe/big/images'
    opt.conf_thres = 0.3
    opt.iou_thres = 0.1
    opt.imgsz = [768, 768]
    opt.fold = 2
    opt.ab_thres = 3.0
    opt.data = 'data/Guge.yaml'
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    opt.mask_dir = data_dict['mask_dir']
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
