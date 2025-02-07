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
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, rot_nms,
                           coco80_to_coco91_class, colorstr, increment_path, print_args,
                           scale_coords,scale_coords_poly, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, poly_iou, ap_per_class_dir
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync



def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')



# def process_batch(detections, labels, iouv):
#     """
#     Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
#     Arguments:
#         detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#         labels (Array[M, 5]), class, x1, y1, x2, y2
#     Returns:
#         correct (Array[N, 10]), for 10 IoU levels
#     """
#     correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
#     iou = box_iou(labels[:, 1:], detections[:, :4])
#     x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
#     if x[0].shape[0]:
#         matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
#         if x[0].shape[0] > 1:
#             matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#             # matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         matches = torch.Tensor(matches).to(iouv.device)
#         correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
#     return correct

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



@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        ab_thres=3.0,
        iou_thres=0.1,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        fold=2,
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
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
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
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=False,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'dacc')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(6, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    dir_stats = []
    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        targets2 = targets.clone()
        targets[...,2:10] = targets[..., 6:].clone()
        targets = targets[..., :10]
        if pt:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # inference
        if training:
            out = []
            train_out = []
            tmp = model(im, augment=augment)
            for t in tmp:
                out.append(t[0])
                train_out.extend(t[1])
        else:
            out, train_out = model(im, augment=augment, val=True)
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets2.to(device), fold=fold)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height,width, height, width, height]).to(device)  # to pixels
        t3 = time_sync()
        out = rot_nms(out, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold,mask_dir=mask_dir)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    # dir
                    dir_stats.append(torch.zeros(0, dtype=torch.bool))
                continue


            predn = pred.clone()
            scale_coords_poly(im.shape[2:], predn[:, :8], shape, shapes[si][1])

            # Evaluate
            if nl:
                tbox = labels[:, 1:]
                scale_coords_poly(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct, dir_correct = process_batch_poly(predn, labelsn, iouv, dotv=0.5)
                if plots:
                    confusion_matrix.process_batch_poly(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                dir_correct = torch.zeros(pred.shape[0], dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 8].cpu(), pred[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)
            dir_stats.append(dir_correct.cpu())

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
      
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

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
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

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
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')
    parser.add_argument('--fold', type=int, default=1, help='倍角')
    opt = parser.parse_args()
    opt.half = True
    opt.data = 'data/Guge.yaml'
    opt.weights = 'runs/train/exp12/weights/best.pt'
    opt.imgsz = 768
    opt.ab_thres = 3.0
    '''
    opt.data = 'data/hrsc2016.yaml'
    # opt.weights = 'weights/ucas-640-dir-88.89/weights/best.pt'
    opt.weights = 'paperdatas/train/4.5/hrsc-768-se-aug-ms-fold2/weights/best.pt'
    # opt.weights = 'runs/train/exp4/weights/best.pt'
    opt.imgsz = 1280
    opt.ab_thres = 7.0
    '''
    opt.fold = 2
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
