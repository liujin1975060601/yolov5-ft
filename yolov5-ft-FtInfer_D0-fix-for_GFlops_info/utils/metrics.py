# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

#from DOTA_devkit.polyiou_cpu import poly_iou_cpu64
from DOTA_devkit.polyiou import polyiou

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), cut=False):
    #tp[nt,10]
    #conf[nt]
    #pred_cls[nt]
    #target_cls[]gtnt]
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)#[nt]按排序重新整理顺序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]#得到排序后的[nt]
    #tp[nt,10]  conf[nt]  pred_cls[nt]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    f1 = np.zeros((nc, 1000))
    ic = np.zeros(nc)
    theshes = torch.ones(len(names))
    #ap[nc,10]  p[nc,1000]  r[nc,1000]
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            #theshes[int(c)] = 1.0
            py.append(np.ones(1000))
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            t_thresh = np.interp(-px, -conf[i], conf[i]) # t_thresh interp

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], cut=cut)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
            # Compute F1 (harmonic mean of precision and recall)
            f1[ci] = 2 * p[ci] * r[ci] / (p[ci] + r[ci] + 1e-16)#f1[nc,1000]
            ic[ci] = f1[ci].argmax()#ic[cls]
            #theshes[int(c)] = ic[ci] / px.shape[0]
            theshes[int(c)] = np.clip(t_thresh[int(ic[ci])],np.min(conf[i]),np.max(conf[i]))#ic[ci] / px.shape[0]


    # Compute F1 (harmonic mean of precision and recall)
    #f1 = 2 * p * r / (p + r + 1e-16)#f1[nc,1000]
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    #f1[nc,1000]
    ic = np.round(ic).astype(int)
    #ap[nc, 10=tp.shape[1]]
    return p[np.arange(nc), ic], r[np.arange(nc), ic], ap, f1[np.arange(nc), ic], unique_classes.astype('int32'), theshes, py
    '''
    idf1 = f1.mean(0).argmax()  # max F1 index
    return p[:, idf1], r[:, idf1], ap, f1[:, idf1], unique_classes.astype('int32'), theshes
    '''


def ap_per_class_dir(tp, conf, pred_cls, target_cls, dir_tp, plot=False, save_dir='.', names=()):

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    dir_tp = dir_tp[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    dtps = np.zeros(len(unique_classes), dtype=np.int)
    dfps = np.zeros(len(unique_classes), dtype=np.int)
    dtpfns = np.zeros(len(unique_classes), dtype=np.int)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            py.append(np.ones(1000)) 
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            # dir
            dfps[ci] = (1 - dir_tp[i]).sum()
            dtps[ci] = dir_tp[i].sum()
            dtpfns[ci] = n_l
            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    cls_dir_acc = dtps / (dfps + dtpfns)
    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32'), cls_dir_acc



def compute_ap(recall, precision, cut=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if cut:
        mrec = np.concatenate(([0.0], recall, [recall[-1]],[1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def process_batch_poly(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2, x3, y3, x4,y4) format.
        Arguments:
            detections (Array[N, 10]), x1, y1, x2, y2, x3, y3, x4,y4 conf, class
            labels (Array[M, 9]), class, x1, y1, x2, y2, x3, y3, x4,y4
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 8] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 9].int()
        iou = poly_iou(labels[:, 1:], detections[:, :8])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU



def ab_iou(ab1, ab2,eps=1e-7):
    
    inter = torch.min(ab1, ab2).prod(1)
    iou =  inter / ( ab1.prod(1) + ab2.prod(1) - inter + eps)
    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

import cv2

def ft2box(coefs, term=1, n=200):
    boxes = []
    theta_fine = torch.tensor(np.linspace(0, 2*np.pi, n, endpoint=True)).to(coefs.device)
    for i, coef1 in enumerate(coefs):
        an,bn,cn,dn = torch.split(coef1[2:].view(-1, 4), 1, dim=-1)
        term_ = an.shape[0] if term == -1 else term
        x_approx = sum([an[i]*torch.cos((i+1)*theta_fine) + bn[i]*torch.sin((i+1)*theta_fine) for i in range(term_)])
        y_approx = sum([cn[i]*torch.cos((i+1)*theta_fine) + dn[i]*torch.sin((i+1)*theta_fine) for i in range(term_)])
        xy = torch.vstack([x_approx, y_approx]).T
        xy[:, 0] += coef1[0]
        xy[:, 1] += coef1[1]
        xy = xy.cpu().long().numpy().astype(int)
        box = cv2.boxPoints(cv2.minAreaRect(xy)).reshape(-1).astype(np.float64)    # (4, 2) lu ru rb lb
        boxes.append(box)
    return np.array(boxes)

def ft2dir(coefs,cen):
    # a1, b1, c1, d1 = an[1], bn[1], cn[1], dn[1] # nt,1
    abcd = coefs[:, 2:6] #abcd[nt,4]
    a1, b1, c1, d1 = torch.split(abcd, 1, dim=-1)

    # if 1:
    if cen:
        tan2t = (2*(a1*b1 + c1*d1))/(a1**2 + c1**2 - b1**2 - d1**2) # x, y   2t~[-0.5pi, 0.5pi], t~[-0.25pi, 0.25pi]  
        # sin(A/2) = √{(1–cosA)/2}
        # cos(A/2) = √{(1+cosA)/2}
        cos2t = torch.sqrt(1 / (1 + tan2t**2)).view(-1,1)
        cos_sin = cos2t @ torch.tensor([[0.5, -0.5]], dtype=abcd.dtype, device=coefs.device) + 0.5    # nt,1 @ 1,2 -> [nt, 2]
        cos_sin = torch.sqrt(cos_sin)
        index = torch.where(tan2t < 0)[0]
        cos_sin[index, 1] *= -1 #cos_sin[nt,2]
        #
        '''
        cos_sin2t = torch.stack([a1**2 + c1**2 - b1**2 - d1**2, 2*(a1*b1 + c1*d1)],dim=1).squeeze(2)#[nt,2]
        sgn_sin = cos_sin2t[:,0] * cos_sin2t[:,1] < 0
        cos_sin2t = F.normalize(cos_sin2t,p=2,dim=-1)#[nt,2]
        cos_sin2t[:,0] = torch.abs(cos_sin2t[:,0])
        cos_t = torch.sqrt((1+cos_sin2t[:,0])/2)#[nt]
        sin_t = torch.sqrt((1-cos_sin2t[:,0])/2)#[nt]
        sin_t[sgn_sin] *= -1
        cos_sinx = torch.stack([cos_t,sin_t],dim=1)#[nt,2]
        '''
    else:
        a1,b1,c1,d1 = a1.float(),b1.float(),c1.float(),d1.float()
        cos_sin2t = torch.stack([a1**2 + c1**2 - b1**2 - d1**2, 2*(a1*b1 + c1*d1)],dim=1).squeeze(2)#[nt,2]
        cos_sin2t = F.normalize(cos_sin2t,p=2,dim=-1)#[nt,2]   

        cos_t = torch.sqrt((1+cos_sin2t[:,0])/2)#[nt]
        sin_t = torch.sqrt((1-cos_sin2t[:,0])/2)#[nt]
        sin_t[cos_sin2t[:,1]<0] *= -1

        cos_sin = torch.stack([cos_t,sin_t],dim=1)#[nt,2]

    return cos_sin

def ft2vector(coefs,cen=0):
    #coefs[nt, 2+4*coef]
    if not torch.is_tensor(coefs):
        coefs = torch.from_numpy(coefs)
    cos_sin = ft2dir(coefs,cen)#[nt,2]
    cos_t,sin_t = cos_sin[:,0],cos_sin[:,1]
    a1,b1,c1,d1 = torch.chunk(coefs[:,2:6],4,dim=1)#coefs[:, 2:6]-->a1[nt,term]
    #a1,b1,c1,d1 = coefs[:,2],coefs[:,3],coefs[:,4],coefs[:,5]
    ap= a1*cos_t+b1*sin_t#[nt,1]
    cp= c1*cos_t+d1*sin_t#[nt,1]
    bp=-a1*sin_t+b1*cos_t#[nt,1]
    dp=-c1*sin_t+d1*cos_t#[nt,1]
    return ap,bp,cp,dp

def ft2pts(coefs,cen=0):
    #coefs[nt, 2+4*coef]
    cos_sin = ft2dir(coefs,cen)#[nt,2]
    abcd = coefs[:, 2:6] #abcd[nt,4]
    
    # a1, b1, c1, d1 = an[1], bn[1], cn[1], dn[1] # nt,1
    if 0:
        s4c4 = torch.cat([cos_sin[:, 1:2].repeat([1,4]), cos_sin[:, 0:1].repeat([1,4])], dim=-1) #s4c4[nt,8]
        abcdsc = abcd.repeat([1, 2]) * s4c4 # as,bs,cs,ds, ac,bc,cc,dc  # abcdsc[nt, 8]

        mat = torch.tensor(
            [
                [0,1,0,0,1,0,0,0],[0,0,0,1,0,0,1,0], # x1 y1
                [-1,0,0,0,0,1,0,0],[0,0,-1,0,0,0,0,1]   # x2 y2
            ], dtype=abcd.dtype, device=coefs.device).T    # 4, 8
        
        mat2 = torch.tensor(
            [
                [1,0,1,0],  # x1
                [0,1,0,1],  # y1
                [-1,0,1,0],  # x2
                [0,-1,0,1]  # y2
            ], dtype=abcd.dtype, device=coefs.device).T 
        mat3 = torch.tensor([1,1,1,1,-1,-1,-1,-1], dtype=abcd.dtype, device=coefs.device).view(1, -1)

        xy_ = abcdsc @ mat   # nt, 4 [x1, y1, x2, y2]
        pts_ = xy_ @ mat2
        pts_ = pts_.repeat([1, 2]) * mat3
        pts_[:, 0::2] += coefs[:, 0:1]
        pts_[:, 1::2] += coefs[:, 1:2]    # nt, 8

        return pts_
    else:
        xc, yc = torch.split(coefs[:, :2].clone(), 1, dim=-1)
        an1, bn1, cn1, dn1 = torch.split(abcd, 1, dim=-1)
        m_sin = cos_sin[:, 1:2]
        m_cos = cos_sin[:, 0:1]
        scale=1.0
        a1 = scale*(an1 * m_cos + bn1 * m_sin)
        b1 = scale*(-an1 * m_sin + bn1 * m_cos)
        c1 = scale*(cn1 * m_cos + dn1 * m_sin)
        d1 = scale*(-cn1 * m_sin + dn1 * m_cos)   
        P0=torch.cat([xc-a1-b1,yc-c1-d1], dim=-1).view(-1, 2)
        P1=torch.cat([xc-a1+b1,yc-c1+d1], dim=-1).view(-1, 2)
        P2=torch.cat([xc+a1+b1,yc+c1+d1], dim=-1).view(-1, 2)
        P3=torch.cat([xc+a1-b1,yc+c1-d1], dim=-1).view(-1, 2)
        points = torch.cat([P0, P1, P2, P3], dim=1)
        return points


def box_iou_ft(coef1s, coef2s, n=15):
    '''
    coef1s [n1, 2 + ft_coef * 4]
    coef2s [n2, 2 + ft_coef * 4]
    '''
    ious = torch.zeros([coef1s.shape[0], coef2s.shape[0]]).float().to(coef1s.device)
    # boxes1 = ft2box(coef1s)
    # boxes2 = ft2box(coef2s)
    boxes1 = ft2pts(coef1s).cpu().numpy().astype(np.float64)
    boxes2 = ft2pts(coef2s).cpu().numpy().astype(np.float64)


    polys_1 = []
    polys_2 = []
    for i in range(len(boxes1)):
        tm_polygon = polyiou.VectorDouble([boxes1[i, 0], boxes1[i, 1],
                                            boxes1[i, 2], boxes1[i, 3],
                                            boxes1[i, 4], boxes1[i, 5],
                                            boxes1[i, 6], boxes1[i, 7]])
        polys_1.append(tm_polygon)

    for i in range(len(boxes2)):
        tm_polygon = polyiou.VectorDouble([boxes2[i, 0], boxes2[i, 1],
                                            boxes2[i, 2], boxes2[i, 3],
                                            boxes2[i, 4], boxes2[i, 5],
                                            boxes2[i, 6], boxes2[i, 7]])
        polys_2.append(tm_polygon)
    
    
    n = len(boxes1)
    m = len(boxes2)
    for i in range(n):
        for j in range(m):
            iou = polyiou.iou_poly(boxes1[i], boxes2[j])
            ious[i, j] = iou
    return ious


def poly_iou(poly1, poly2):
    device = poly1.device
    poly1 = poly1.cpu().numpy().astype(np.float64)
    poly2 = poly2.cpu().numpy().astype(np.float64)
    polys_1 = []
    polys_2 = []
    for i in range(len(poly1)):
        tm_polygon = polyiou.VectorDouble([poly1[i][0], poly1[i][1],
                                           poly1[i][2], poly1[i][3],
                                           poly1[i][4], poly1[i][5],
                                           poly1[i][6], poly1[i][7]])
        polys_1.append(tm_polygon)

    for i in range(len(poly2)):
        tm_polygon = polyiou.VectorDouble([poly2[i][0], poly2[i][1],
                                        poly2[i][2], poly2[i][3],
                                        poly2[i][4], poly2[i][5],
                                        poly2[i][6], poly2[i][7]])
        polys_2.append(tm_polygon)
    
    
    n = len(poly1)
    m = len(poly2)
    ious = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou = polyiou.iou_poly(poly1[i], poly2[j])
            ious[i][j] = iou
    return torch.from_numpy(ious).to(device)

    
    


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def bbox_ioas(region, boxes, eps=1E-7):
    """ Returns the intersection over boxes area given region, boxes. Boxes are x1y1x2y2
    region:       np.array of shape(4)
    boxes:       np.array of shape(n,4)
    returns:    np.array of shape(n)
    """
    boxesT = boxes.transpose()

    # Get the coordinates of bounding boxes
    x1 , y1 , x2 , y2  = region[0], region[1], region[2], region[3]
    x1s, y1s, x2s, y2s = boxesT[0], boxesT[1], boxesT[2], boxesT[3]

    # Intersection area
    inter_area = (np.minimum(x2, x2s) - np.maximum(x1, x1s)).clip(0) * \
                 (np.minimum(y2, y2s) - np.maximum(y1, y1s)).clip(0)
    #inter_area[n]

    # boxes area
    box2_area = (x2s - x1s) * (y2s - y1s) + eps #box2_area[n]

    # Intersection over boxes area
    return inter_area / box2_area  #-->iou[n]

def wh_iou(anchorsab, tab):
    # Returns the nxm IoU matrix. anchorsab is nx2, tab is mx2
    anchorsab = anchorsab[:, None]  # [na,1,2]
    tab = tab[None]  # [1,nt,2]
    inter_sec = torch.min(anchorsab, tab)  # [na,nt,2]
    inter = inter_sec.prod(2) # [na,nt]
    return inter / (anchorsab.prod(2) + tab.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=(), plot_f1=1,grid=1):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    #
    if grid:
        ax.grid(True)
    if plot_f1:
        # 定义P和R的取值范围
        P = np.linspace(0, 1, 400)
        R = np.linspace(0, 1, 400)
        P, R = np.meshgrid(P, R)
        # 计算F1值
        F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R)!=0)  # 处理除以零
        # 绘制指定等高线的等高线图
        levels = np.linspace(0.1, 0.9, 9)
        contour = plt.contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    #
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_pr_curves(pys, aps, save_path, names=(), methods=[], plot_f1=1, grid=1):
    assert len(pys) == len(aps)
    colors = ['red', 'green', 'blue', 'yellow', 'grey', 'black']
    px = np.linspace(0, 1, 1000)
    c = len(names)
    n = len(pys)
    assert methods==[] or len(methods)==n
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # 创建一个大图
    fig, axs = plt.subplots(1, c + 1, figsize=(5 * (c + 1), 5), tight_layout=True)
    
    for j in range(n):#methods j
        py = np.stack(pys[j], axis=1)
        ap = aps[j]

        # 绘制每个类的子图
        for i in range(c):#classes i
            if grid:
                axs[i].grid(True)
            if plot_f1:
                P = np.linspace(0, 1, 400)
                R = np.linspace(0, 1, 400)
                P, R = np.meshgrid(P, R)
                F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) != 0)
                levels = np.linspace(0.1, 0.9, 9)
                contour = axs[i].contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
                axs[i].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            axs[i].plot(px, py[:, i], linewidth=1, label=f'{methods[j]} {100*ap[i, 0]:.2f}%', color=colors[j % len(colors)])
            axs[i].set_title(names[i])
            axs[i].set_xlabel('Recall')
            axs[i].set_ylabel('Precision')
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            axs[i].legend()

        # 绘制所有类的平均 PR 曲线的子图
        if grid:
            axs[c].grid(True)
        if plot_f1:
            P = np.linspace(0, 1, 400)
            R = np.linspace(0, 1, 400)
            P, R = np.meshgrid(P, R)
            F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) != 0)
            levels = np.linspace(0.1, 0.9, 9)
            contour = axs[c].contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
            axs[c].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        axs[c].plot(px, py.mean(1), linewidth=1, label=f'{methods[j]} all classes {100 * ap[:, 0].mean():.2f}%', color=colors[j % len(colors)])
    
    axs[c].set_title('All Classes')
    axs[c].set_xlabel('Recall')
    axs[c].set_ylabel('Precision')
    axs[c].set_xlim(0, 1)
    axs[c].set_ylim(0, 1)
    axs[c].legend()
    
    # 保存图像
    fig.savefig(save_path / 'prs_comparison.png', dpi=250)
    plt.close()

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
