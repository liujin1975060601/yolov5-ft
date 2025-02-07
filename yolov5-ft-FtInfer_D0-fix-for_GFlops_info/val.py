# YOLOv5 ������ by Ultralytics, GPL-3.0 license
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
from numpy import seterr
seterr('raise')
import torch
from tqdm import tqdm
import csv

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
                           scale_coords,scale_coords_poly, xywh2xyxy, xyxy2xywh,
                           non_max_suppression, non_max_suppression_ft, get_ft_num)
from utils.metrics import ConfusionMatrix, ap_per_class, poly_iou, box_iou_ft
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.loss import loop_loss
from detect import detect, detect_with_cms
from general.MyString import replace_last_path

import pickle

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')



def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # 0.5~0.95 0.05
    iou = box_iou(labels[:, 1:], detections[:, :4]) # [nt, np] iou 0, 0
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches

def vector_dot(labels1, labels2):
    """
    labels1: (N, 8)
    labels2: (N, 8)
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


def process_batch_poly(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2, x3,y3, x4,y4) format.
    Arguments:
        detections (Array[N, 10]), x1, y1, x2, y2, x3,y3, x4,y4 conf, class
        labels (Array[M, 9]), class, x1, y1, x2, y2, x3,y3, x4,y4
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # iou
    iou = poly_iou(labels[:, 1:], detections[:, :8])
    #labels[nlable,1+8=9]  #detections[nDetection,8+1+1=10]-->iou[nlable,nDetection]
    # dot
    dot = vector_dot(labels[:, 1:], detections[:, :8])
    
    #在2维数据iou[nlable,nDetection]中选择True元素集合，生成匹配对集合[nid(labels)],[nid(detections)]
    #labels[:, 0:1]==detections[:, 9] labels中每一个元素与detections里面每个元素比较，相等的话就生成匹配对集合[nid(labels)],[nid(detections)]
    #以上两个集合通过 & 操作符求交集
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 9]))  # IoU above threshold and classes match
    #x[0]=[nid(labels)],x[1]=[nid(detections)]  x是两个长度一样的tensor集成的list
    assert(len(x)==2 and x[0].shape[0]==x[1].shape[0])
    if x[0].shape[0]:#存在匹配
        #torch.stack(x, 1)是两个list在维度为1上拼接得到[nid,2]
        #iou[x[0], x[1]]维度替换为[nid]，然后通过[:, None]扩展一维[1]-->[nid,1]
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [id_label, id_detection, iou]
        #[nid,2] cat [nid,1]得到matches[nid,3=(id_label, id_detection, iou)]
        if x[0].shape[0] > 1:#有一个以上匹配的时候需要剔除掉重复匹配，保证1对1
            matches = matches[matches[:, 2].argsort()[::-1]]#按iou从大到小排序
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]#对第1列id_detection去重，因为排序了，删除同一列排序后面iou小的
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]#对第0列id_label去重，因为排序了，删除同一列排序后面iou小的
            #至此，每一行每一列都只有唯一一个1对1匹配
        matches = torch.Tensor(matches).to(iouv.device)#numpy转gpu
        #初值correct里面全部置0，下面仅在有匹配>=iouv的地方给1
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        #得到预测框detection中每个编号对应的真值框编号，有可能
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches#[detections.shape[0], iouv.shape[0]]


def process_batch_ft(detections_ft, labels_ft, d_c, l_c, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections_ft (Array[N, 2+4*coef]), x1, y1, x2, y2, conf, class
        labels_ft (Array[M, 2+4*coef]), class, x1, y1, x2, y2
        d_c, l_c 
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections_ft.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # iou = box_iou(labels[:, 1:], detections[:, :4])
    iou = box_iou_ft(labels_ft, detections_ft)#-->iou[len(labels_ft),len(detections_ft)]
    x = torch.where((iou >= iouv[0]) & (l_c[:, 0:1] == d_c))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches



@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        ab_thres=3.0,
        iou_thres=0.45,  # NMS IoU threshold
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
        pts=None,
        cms_num=0,
        fxs_func=None,
        nms_ft = True,
        map_ft = True,
        workers=2,
        ):
    # Initialize/load model and set device
    training = model is not None

    if training:  # called by train.py
        device, pt = next(model.parameters()).device, True  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
        pts = model.get_module_byname('Detect2') is not None if pts is None else pts
        cms_s = model.get_module_byname('SubclassInfer') is not None
        ftinfer_m = model.get_module_byname('FTInfer')
        if ftinfer_m is not None:
            ft_length = ftinfer_m.ft_length
            ft_infer = True
        else:
            ft_length = 0
            ft_infer = False
        cms_start = 10 if pts else 6
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt = model.stride, model.pt
        pts = model.model.get_module_byname('Detect2') is not None
        cms_num = 0
        cms_s = False
        cms_start = 10 if pts else 6
        ft_infer = False
        # ft_start  = 14 if pts else 6
        ft_length = 0
        p = model.model.get_module_byname('FTInfer')
        if p:
            ft_infer = True
            cms_start += p.ft_length
            ft_length = p.ft_length
        for key, num in zip(['SubclassInfer', 'ValueInfer'], [1, 'nv']):
            p = model.model.get_module_byname(key)
            if p:
                if isinstance(num, int):
                    cms_s = True
                    cms_num += num 
                else: 
                    cms_num += getattr(p, num)
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
        data = check_dataset(data)  # check data, if fail, exit
        
        cms_config = data.get('cms_config', None)
        train_path = data['train']
        ft_coef = data.get('ft_coef', 0)
        if ft_coef == -1:
            ft_coef = get_ft_num(train_path)
        fxs_path = Path(weights).parent.resolve() / 'fxs_func.npy'
        if fxs_path.exists():
            fxs_func = np.load(fxs_path, allow_pickle=True)
            for i in range(fxs_func.shape[0]):
                if fxs_func[i] is not None:
                    LOGGER.info(f"ValueInfer[{i}]: {fxs_func[i]['y'].min()} ~ {fxs_func[i]['y'].max()}")
        else:
            fxs_func = None


    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz
            model(torch.zeros(1, 3, imgsz2[0], imgsz2[1]).to(device).type_as(next(model.model.parameters())))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # dataloader = create_dataloader(data[task], imgsz2, batch_size, stride, single_cls, pad=pad, rect=False,
        #                                prefix=colorstr(f'{task}: '))[0]
        hyp = model.model.hyp
        #val_count = hyp.get('val_count',300)
        val_count = data.get('val_count', 0)
        dataloader, dataset = create_dataloader(data[task], imgsz2, batch_size, stride, single_cls, rect=False, pad=pad,
                                      hyp=hyp,
                                      prefix=colorstr(f'{task}: '),
                                      pts=pts, cms=cms_config if cms_config is not None else False,
                                      fxs=fxs_func,
                                      ft_coef=ft_coef,
                                      workers=workers, 
                                      debug_samples=0,sample_count=val_count)
        if fxs_func is not None:
            fxs_func = fxs_func[dataset.reorder]
        
    mask_dir = data.get('mask_dir', 0)   # mask_dir can be not exist.
    assert isinstance(mask_dir, (int, list)), f'{mask_dir}, Mask Dir Error.'
    mask_dir = [mask_dir] * nc if isinstance(mask_dir, int) else mask_dir
    if len(mask_dir) != nc:
        mask_dir = mask_dir + [0] * (nc - len(mask_dir))

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%15s' + '%11s' * 12) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh', 'ft_xy', 'ft_coef', 'cms_s', 'cms_v')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(10, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    cms_s_val = [0, 0]    # P, GT(only match)
    cms_v_val = torch.zeros((cms_num - 1) if cms_s else max(1, cms_num), device=device)
    cms_v_val2 = np.zeros((cms_num - 1) if cms_s else max(1, cms_num))
    cms_v_num = torch.zeros((cms_num - 1) if cms_s else max(1, cms_num), device=device)
    ft_center_val = torch.zeros(1, device=device)
    ft_coef = max((ft_length - 2) // 4, 0)
    ft_coef_val = torch.zeros(ft_coef, device=device)
    ft_num = torch.zeros(1, device=device)
    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        '''
        这里是旋转框模型的处理方法
        targets14 = targets.clone()#targets14[nobj,14=1(batch)+1(cls)+4(box)+4*2(pts)]
        ##把targets14中间的4(box)删掉了！变成了targets[nobj,10]
        targets[...,2:10] = targets[..., 6:].clone()#targets[nobj,1(batch)+1(cls)+4*2(pts)]
        targets = targets[..., :10]#把中间的4(box)删掉了！targets[nobj,1(batch)+1(cls)+4*2(pts)]
        '''
        targets = targets.to(device)
        targets_all = targets.clone().to(device)#targets_all[nobj,14=1(batch)+1(cls)+4(box)+4*2(pts) {+cms+cms_mask}]
        ##把targets_all中间的4(box)删掉了！变成了targets[nobj,10+cms+cms_mask]
        targets6 = targets[..., :6].clone().to(device)
        if pt:
            im = im.to(device, non_blocking=True)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1#dt[0]预处理时间


        # Inference
        # out,train_out = detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=data['mask_dir'], dt=dt)
        if not pts:
            if training:
                out,train_out = [],[]
                tmp = model(im, augment=augment)
                for t in tmp:
                    out.append(t[0])
                    train_out.extend(t[1])
            else:
                out, train_out = model(im, augment=augment, val=True)
            dt[1] += time_sync() - t2
            

            # Loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets_all.to(device), fold=fold, mask_dir=mask_dir)[1]  # box, obj, cls
                n_loop = compute_loss.n_loop
            else:
                n_loop = 8
            # NMS
            targets6[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = []  # for autolabelling

            pbox_out = []
            for i in out[0]:
                pbox_out.append(i.view(i.shape[0], -1, i.shape[-1]))
            pCms = [None] * len(out[0])
            pFt = [None] * len(out[0])
            sv_id = 0
            if ft_infer:
                sv_id = 1
                pFt = out[sv_id]
                for i, j in enumerate(pFt):
                    pFt[i] = j.view(j.shape[0], -1, j.shape[-1])
                pFt = torch.cat(pFt, 1) # B, features, 2+4*coef
            pFt = None if isinstance(pFt, list) else pFt
            if cms_num > 0:
                sv_id += 1
                if cms_s:
                    pCms_s = out[sv_id]
                    sv_id += 1
                    for i in range(len(pCms_s)):
                        cms_conf, cms_cls = pCms_s[i].max(-1, keepdim=True)
                        pCms[i] = torch.cat([cms_conf, cms_cls], dim=-1)
                if len(out) > sv_id:    # cms_s+v pCms_s[nl][b,a,H,W,ns] pCms_v[nl][b,a,H,W,nv]
                    pCms_v = out[sv_id]
                    for i in range(len(pCms_v)):
                        pCms[i] = pCms_v[i] if not cms_s else torch.cat([pCms[i], pCms_v[i]], dim=-1)
                for i, j in enumerate(pCms):
                    pCms[i] = j.view(j.shape[0], -1, j.shape[-1])
                pCms = torch.cat(pCms, 1) 
            pCms = None if isinstance(pCms, list) else pCms
            pbox_out = torch.cat(pbox_out, 1) 
            
            t3 = time_sync()
            # 水平框nms
            if not (nms_ft and ft_infer):
                out, indices = non_max_suppression(pbox_out, conf_thres, iou_thres, labels=lb, multi_label=False, agnostic=single_cls, return_indices=True)
            # 傅里叶nms
            else:
                out, out_ft, indices = non_max_suppression_ft(pbox_out, pFt, conf_thres, iou_thres, labels=lb, multi_label=False, agnostic=single_cls, return_indices=True)

            dt[2] += time_sync() - t3
            
            if cms_s:
                pv, tv = 2, 1   # if subclass exist conf, class
            else:
                pv, tv = 0, 0
            # Metrics
            for si, pred in enumerate(out):#batch内循环   pred[nt,10=2*4+1(conf)+1(cls)]
                labels = targets6[targets6[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    
                    if ft_length > 0:
                        labels_ft = targets_all[targets6[:, 0] == si, 6:6+ft_length].clone()
                        labels_ft[:, :2] = labels_ft[:, :2] * torch.Tensor([width, height]).to(device)
                        labels_ft[:, 2:] = labels_ft[:, 2:] * torch.Tensor([width, width, height, height]).to(device).repeat((ft_length-2) // 4).view(1, -1)
                        
                    # 水平框的map
                    if not (map_ft and ft_infer):
                        correct, matches = process_batch(predn, labelsn, iouv)
                    # 傅里叶的map
                    else:
                        correct, matches = process_batch_ft(detections_ft=pFt[si][indices[si]], 
                                                            labels_ft=labels_ft, 
                                                            d_c=predn[:, 5], 
                                                            l_c=labels[:, 0:1],
                                                            iouv=iouv)   
                    if matches.shape[0] > 0:
                        if ft_infer or (cms_num > 0):
                            t_idx, p_idx, _ = torch.chunk(matches, 3, dim=-1)
                            t_idx, p_idx = t_idx.view(-1).long(), p_idx.view(-1).long()
                            if cms_num > 0:
                                pCms_i = pCms[si][indices[si]]
                                cmss, cmss_mask = torch.chunk(targets_all[targets6[:, 0] == si, cms_start:], 2, dim=-1)
                                if cms_s:
                                    cmss_s = cmss[:, 0]
                                    cms_s_val[0] += ((pCms_i[p_idx, 1].long() == cmss_s[t_idx].long())*cmss_mask[t_idx, 0]).sum().tolist()
                                    cms_s_val[1] += cmss_mask[t_idx, 0].sum().tolist()
                                if cms_num - int(cms_s) > 0:
                                    cmss_v = cmss[:, tv:]
                                    l1_v = torch.abs((pCms_i[p_idx][:, pv:] - cmss_v[t_idx])*cmss_mask[t_idx][:, tv:]).sum(0)
                                    tmp, tmp2, tmp3 = pCms_i[p_idx][:, pv:].cpu().numpy(), cmss_v[t_idx].cpu().numpy(), cmss_mask[t_idx][:, tv:].cpu().numpy()
                                    l1_v2 = np.zeros(l1_v.shape, np.float32)
                                    for fi, fxs in enumerate(fxs_func[tv:]):
                                        a1 = np.interp(tmp[:, fi], fxs['x'], fxs['y'])    # x -> y
                                        b1 = np.interp(tmp2[:, fi], fxs['x'], fxs['y'])    # x -> y
                                        l1_v2[fi] += np.abs((a1 - b1) * tmp3[:, fi] / np.maximum(np.abs(b1), 1e-8)).sum(0)
                                    cms_v_val += l1_v
                                    cms_v_val2 += l1_v2
                                    cms_v_num += cmss_mask[t_idx][:, tv:].sum(0)
                            if ft_infer:
                                pFt_i = pFt[si][indices[si]]
                                ft_labels_xy = labels_ft[:, :2].clone() # a0, c0
                                # ft_labels_xy = ft_labels_xy *  torch.Tensor([width, height]).to(device)
                                # ft_labels_coef = targets_all[targets6[:, 0] == si, 8:6+ft_length]
                                ft_labels_coef = labels_ft[:, 2:].clone()
                                assert t_idx.shape[0]<=min(len(pFt[si][indices[si]]),len(labels_ft))
                                #ft_labels_xy[t_idx] [nt, 2] 
                                # temp1 = torch.sqrt(((pFt_i[p_idx][:, :2] - ft_labels_xy[t_idx]) ** 2).sum(-1))  # ((a0-at0)^2 + (c0-at0)^2)^0.5
                                ft_center_val += torch.sqrt((((pFt_i[p_idx][:, :2] - ft_labels_xy[t_idx])) ** 2 ).sum(-1)).mean()
                                ft_coef_min = loop_loss(n_loop,ft_labels_coef[t_idx],pFt_i[p_idx][:, 2:])                               
                                temp1 = pFt_i[p_idx][:, 2:].clone().view(-1, ft_coef, 4).float()
                                temp2 = ft_coef_min.view(-1, ft_coef, 4).float()
                                ft_coef_val = ft_coef_val + torch.abs(torch.sqrt((temp1 * temp1).sum(-1)) - torch.sqrt((temp2 * temp2).sum(-1))).sum(0)
                                # ft_coef_val += torch.abs(pFt_i[p_idx][:, 2:] - ft_coef_min).mean()
                                ft_num += t_idx.shape[0]
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
                # Save/log
                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
        else:
            targets[...,2:10] = targets[..., 6:14].clone()#targets[nobj,1(batch)+1(cls)+4*2(pts)]
            targets = targets[..., :10]#把中间的4(box)删掉了！targets[nobj,1(batch)+1(cls)+4*2(pts)]
            # inference
            # out, train_out = detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir, dt=dt)
            out, train_out = detect_with_cms(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir, dt=dt)
            #train_out[2=Detect+Detect2][nl][b,a,H,W,Dectct(5+c) or Detect2(5=1+2(dir)+2(ab))]
            #out[b,nt,10=2*4+1(conf)+1(cls)]

            # Loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets_all.to(device), fold=fold, mask_dir=mask_dir)[1]  # box, obj, cls

            #targets[nobj,10 = 1(batch)+1(cls)+4*2(pts)]
            targets[:, 2:] *= torch.Tensor([width, height, width, height,width, height, width, height]).to(device)  # to pixels
            # Metrics
            if cms_s:
                pv, tv = cms_start + 2, 1
            else:
                pv, tv = cms_start, 0
            for si, pred in enumerate(out):#batch内循环   pred[nt,10=2*4+1(conf)+1(cls)]
                labels = targets[targets[:, 0] == si, 1:]#labels[nt,9=1(cls)+4*2(pts)]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue


                predn = pred.clone()
                scale_coords_poly(im.shape[2:], predn[:, :8], shape, shapes[si][1])

                # Evaluate
                if nl:
                    tpts = labels[:, 1:]#tpts[nt,8=4*2(pts)]
                    scale_coords_poly(im[si].shape[1:], tpts, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tpts), 1)  #labelsn[nt,9=1+4*2] native-space labels
                    correct, matches = process_batch_poly(predn[:, :10], labelsn, iouv)#predn[nt,10=2*4+1(conf)+1(cls)]
                    if matches.shape[0] > 0:
                        if cms_num > 0:
                            t_idx, p_idx, _ = torch.chunk(matches, 3, dim=-1)
                            t_idx, p_idx = t_idx.view(-1).long(), p_idx.view(-1).long()
                            cmss, cmss_mask = torch.chunk(targets_all[targets6[:, 0] == si, cms_start:], 2, dim=-1)
                            if cms_s:
                                cmss_s = cmss[:, 0]
                                cms_s_val[0] += ((predn[p_idx, 11].long() == cmss_s[t_idx].long())*cmss_mask[t_idx, 0]).sum().tolist()
                                cms_s_val[1] += cmss_mask[t_idx, 0].sum().tolist()
                            if cms_num - int(cms_s) > 0:
                                cmss_v = cmss[:, tv:]
                                l1_v = torch.abs((predn[p_idx][:, pv:] - cmss_v[t_idx])*cmss_mask[t_idx][:, tv:]).sum(0)
                                tmp, tmp2, tmp3 = predn[p_idx][:, pv:].cpu().numpy(), cmss_v[t_idx].cpu().numpy(), cmss_mask[t_idx][:, tv:].cpu().numpy()
                                l1_v2 = np.zeros(l1_v.shape, np.float32)
                                for fi, fxs in enumerate(fxs_func[tv:]):
                                    a1 = np.interp(tmp[:, fi], fxs['x'], fxs['y'])    # x -> y
                                    b1 = np.interp(tmp2[:, fi], fxs['x'], fxs['y'])    # x -> y
                                    l1_v2[fi] += np.abs((a1 - b1) * tmp3[:, fi] / np.maximum(np.abs(b1), 1e-8)).sum(0)
                                cms_v_val += l1_v
                                cms_v_val2 += l1_v2
                                cms_v_num += cmss_mask[t_idx][:, tv:].sum(0)

                    #得到预测框pred对应的真值匹配bool矩阵correct[预测目标数量,iou阈值数量]
                    if plots:
                        confusion_matrix.process_batch_poly(predn[:, :10], labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                #pred的8列是预测目标conf，9列是预测目标cls
                stats.append((correct.cpu(), pred[:, 8].cpu(), pred[:, 9].cpu(), tcls)) #(correct, conf, pcls, tcls)

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
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs, py = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names, cut=False)
        #p[nc] r[nc] ap[nc,10] f1[nc] ap_class[nc] threshs[nc]
        # 保存到文件
        with open(save_dir / 'status.pkl', 'wb') as f:
            pickle.dump([py,ap,names], f)

        #p[nc] r[nc] ap[nc,10] f1[nc] ap_class[nc] threshs[nc]
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        train_path = weights.parent.parent if not training and not isinstance(weights, str) else save_dir
        np.save(train_path / 'threshs.npy', threshs)
    else:
        # nt = torch.zeros(1)
        f1 = torch.zeros(1)
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        threshs = torch.zeros(1)

    cms_v_val = cms_v_val/(cms_v_num + 1e-8)
    cms_v_val2 = cms_v_val2/(cms_v_num.cpu().numpy() + 1e-8)
    ft_center_val = ft_center_val / (ft_num.sum() + 1e-8)
    ft_coef_val = ft_coef_val / (ft_num + 1e-8)
    
    if cms_v_val.shape[0] > 1:
        LOGGER.info(('%20s' + '%11.4g' * len(cms_v_val)) % ('ValueInfer', *cms_v_val.tolist()))
        LOGGER.info(('%20s' + '%11.4g' * len(cms_v_val2)) % ('ValueInfer[R]', *cms_v_val2.tolist()))

    # Print results
    pf = '%15s' + '%11i' * 2 + '%11.4g' * 6   # print format
    LOGGER.info((pf + '%11.4g' * 5) % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(), 
                threshs.mean(), 
                ft_center_val, ft_coef_val.mean(), 
                cms_s_val[0]/(cms_s_val[1] + 1e-8),
                cms_v_val.mean(), cms_v_val2.mean()))
    if (verbose or (not training)) and nc >= 1 and len(stats):
        if nc < 100:
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]))
        with open(save_dir / 'classes_map.csv', 'w', newline='') as file_map:
            writer = csv.writer(file_map)
            writer.writerow(['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh'])
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in ["all", seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()] ])
            # Print results per class
            for i, c in enumerate(ap_class):
                writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]]])
    
    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz
        shape = (batch_size, 3, imgsz2[0], imgsz2[1])
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

def run_rot(data,
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
        workers=2
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
                                       workers=workers, prefix=colorstr(f'{task}: '),sample_count=1000)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(6, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        targets14 = targets.clone()#targets14[nobj,14=1(batch)+1(cls)+4(box)+4*2(pts)]
        ##把targets14中间的4(box)删掉了！变成了targets[nobj,10]
        targets[...,2:10] = targets[..., 6:].clone()#targets[nobj,1(batch)+1(cls)+4*2(pts)]
        targets = targets[..., :10]#把中间的4(box)删掉了！targets[nobj,1(batch)+1(cls)+4*2(pts)]
        if pt:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1#dt[0]预处理时间

        # inference
        out,train_out = detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=data['mask_dir'], dt=dt)
        #train_out[2=Detect+Detect2][nl][b,a,H,W,Dectct(5+c) or Detect2(5=1+2(dir)+2(ab))]
        #out[b,nt,10=2*4+1(conf)+1(cls)]

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets14.to(device), fold=fold, mask_dir=data['mask_dir'])[1]  # box, obj, cls

        #targets[nobj,10 = 1(batch)+1(cls)+4*2(pts)]
        targets[:, 2:] *= torch.Tensor([width, height, width, height,width, height, width, height]).to(device)  # to pixels
        # Metrics
        for si, pred in enumerate(out):#batch内循环   pred[nt,10=2*4+1(conf)+1(cls)]
            labels = targets[targets[:, 0] == si, 1:]#labels[nt,9=1(cls)+4*2(pts)]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue


            predn = pred.clone()
            scale_coords_poly(im.shape[2:], predn[:, :8], shape, shapes[si][1])

            # Evaluate
            if nl:
                tpts = labels[:, 1:]#tpts[nt,8=4*2(pts)]
                scale_coords_poly(im[si].shape[1:], tpts, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tpts), 1)  #labelsn[nt,9=1+4*2] native-space labels
                correct = process_batch_poly(predn, labelsn, iouv)#predn[nt,10=2*4+1(conf)+1(cls)]
                #得到预测框pred对应的真值匹配bool矩阵correct[预测目标数量,iou阈值数量]
                if plots:
                    confusion_matrix.process_batch_poly(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            #pred的8列是预测目标conf，9列是预测目标cls
            stats.append((correct.cpu(), pred[:, 8].cpu(), pred[:, 9].cpu(), tcls)) #(correct, conf, pcls, tcls)

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
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        #p[nc] r[nc] ap[nc,10] f1[nc] ap_class[nc] threshs[nc]
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        train_path = weights.parent.parent if not training else save_dir
        np.save(train_path / 'threshs.npy', threshs)
    else:
        nt = torch.zeros(1)
        f1 = torch.zeros(1)
        threshs = torch.zeros(1)
    
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        with open(save_dir / 'map.csv', 'w', newline='') as file_map:
            writer = csv.writer(file_map)
            writer.writerow(['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh'])
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in ["all", seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()] ])
            # Print results per class
            for i, c in enumerate(ap_class):
                writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]]])

        pf = '%20s' + '%11i' * 2 + '%11.4g' * 6  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()))
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]))
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
    parser.add_argument('--data', type=str, default=ROOT / 'data/UCAS_ft.yaml', help='dataset.yaml path')###
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp_UCAS5/weights/best.pt', help='model.pt path(s)')###
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')###
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[640,640], help='inference size (pixels)')###
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')### 0.45
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')###
    parser.add_argument('--fold', type=int, default=2, help='倍角')###
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default=ROOT / 'runs/val/COCO_S', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--nms_ft', action='store_true', help='use ft coefs to calc nms')
    parser.add_argument('--map_ft', action='store_true', help='use ft coefs to calc map')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()
    opt.half = True

    opt.nms_ft = True
    opt.map_ft = True
    opt.iou_thres = 0.45 if opt.nms_ft==False else 0.2
    opt.conf_thres = 0.001
    opt.imgsz = [640,896]

    #dota1.5
    # opt.data = 'data/dota_ft.yaml'
    # opt.weights = 'runs/train/exp_dota_ft4/weights/best.pt'
    #voc_segment     opt.weights = 'runs/train/exp_voc_segment13/weights/best.pt'
    #opt.weights = 'runs/train/exp3_voc/weights/best.pt'
    #opt.data = 'data/voc_segment.yaml'
    #hrsc2016
    #opt.data = 'data/hrsc2016.yaml'
    #opt.weights = 'runs/train/exp_hrsc20162/weights/best.pt'
    #opt.source = r'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/GuGe/images'
    #Guge
    # opt.data = 'data/Guge_ft.yaml'
    # opt.weights = 'runs/train/exp_Guge_ft6/weights/last.pt'
    #opt.batch_size = 16
    # opt.imgsz = 768
    # opt.conf_thres = 0.001
    # opt.iou_thres = 0.1
    # opt.ab_thres = 3.0
    #coco2017
    opt.data = 'data/coco_ft.yaml'
    opt.weights = 'runs/train/exp_coco_ft13/weights/best_mAP50.pt'

    #hrsc2016_ft
    #opt.data = 'data/hrsc2016_ft.yaml'
    #opt.weights = 'runs/train/exp_hrsc2016_ft2/weights/best.pt'
    #dota1.5-ft liujin
    # opt.data = './data/dota_ft.yaml'
    # opt.weights = './runs/train/exp_dota_ft/weights/best.pt'
    opt.device = 0

    #coco
    # opt.data = 'data/coco_ft.yaml'
    # opt.weights = 'runs/train/coco_ft_yolov5m_55.045/weights/best_mAP50.pt'

    opt.name += f'_{Path(opt.data).stem}'
    opt.project = replace_last_path(opt.project,Path(opt.data).stem)
    
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        if 0:
            name = opt.name + '_' + Path(opt.data).stem
            name_s = ['hv', 'ft']
            iou_t = [0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
            for i in range(2):
                opt.nms_ft = name_s[i] == 'ft'
                for j in range(len(iou_t)):
                    opt.name = name + f'_{name_s[i]}{iou_t[j]*100:.0f}'
                    opt.iou_thres = iou_t[j]
                    run(**vars(opt))
        else:
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
