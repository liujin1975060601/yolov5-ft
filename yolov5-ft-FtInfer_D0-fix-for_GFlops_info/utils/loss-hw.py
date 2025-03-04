# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou, ab_iou, wh_iou
from utils.torch_utils import is_parallel
from utils.general import pts2dir


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # add
        # BCEdir = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        # BCEab = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['ab_pw']], device=device))
        BCEdir = nn.SmoothL1Loss(reduction='mean')
        BCEab = nn.SmoothL1Loss(reduction='mean')
        BCEpab = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['pab_pw']], device=device))


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-2] if is_parallel(model) else model.model[-2]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        # add
        self.BCEdir = BCEdir
        self.BCEab = BCEab
        self.BCEpab = BCEpab

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
        
        # add
        det2 = model.module.model[-1] if is_parallel(model) else model.model[-1]
        setattr(self, 'anchorsab', getattr(det2, 'anchors'))

    def __call__(self, p, targets, fold):  # predictions, targets, model
        device = targets.device
        pts_targets = torch.clone(targets[:, 6:])
        dir_targets = pts2dir(pts_targets, fold_angle=fold)
        targets = targets[:, :6]
        p1 = p[:3]
        p2 = p[3:]
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # add
        tcls, tbox, indices, anchors, tdir, tab, anchorsab = self.build_targets_dir(p1, p2, targets, dir_targets)  # targets
        ldir, lab, lpab = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # Losses
        for i, (pi, pj) in enumerate(zip(p1, p2)):  # layer index, layer predictions
            b, a, gj, gi, aab = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tpab = torch.zeros_like(pj[..., 0], device=device)
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2.6 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # add
                pjs = pj[b, aab, gj, gi]
                # theta
                cos_sin = pjs[:, 1:3].sigmoid() * 2 - 1
                ldir += self.BCEdir(cos_sin, tdir[i])
                # ab
                a_b = (pjs[:, 3:5].sigmoid() * 2) ** 2.6 * anchorsab[i]
                # a_b =  torch.exp(pjs[:, 3:5]) * anchorsab[i]
                a_b_iou = ab_iou(a_b, tab[i])
                lab += (1.0 - a_b_iou).mean()

                # pab
                tpab[b, aab, gj, gi] = a_b_iou.detach().clamp(0).type(tpab.dtype)
                

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            lpab += self.BCEpab(pj[..., 0], tpab)          
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ldir *= self.hyp['theta']
        lab *= self.hyp['ab']
        lpab *= self.hyp['pab']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + ldir + lab + lpab) * bs, torch.cat((lbox, lobj, lcls, ldir, lab, lpab)).detach()



    def build_targets_dir(self, p1, p2, targets, dir_targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tdir, tab, anchab = [], [], [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # add
        gain2 = torch.ones(5, device=targets.device)

        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        dir_targets = torch.cat((dir_targets.repeat(na, 1, 1), ai[:, :, None]), 2)


        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            anchorsab = self.anchorsab[i]
            gain[2:6] = torch.tensor(p1[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # add
            gain2[2:4] = torch.tensor(p2[i].shape)[[3, 2]]

            # Match targets to anchors
            t = targets * gain
            # add
            t2 = dir_targets * gain2
            if nt:
                # Matches
                # 筛选wh的anchor
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare

                t = t[j]  # filter
                # add
                t2 = t2[j]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

                # add
                t2 = t2.repeat((5, 1, 1))[j]

            else:
                t = targets[0]
                offsets = 0

                # add
                t2 = dir_targets[0]

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            if t2.shape[0] != 0:
                max_an_iou_idx = torch.max(wh_iou(anchorsab, t2[:, 2:4]), dim=0)[1]
            else:
                max_an_iou_idx = torch.zeros(0, dtype=torch.long, device=t2.device)

            indice = (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1), max_an_iou_idx)
            indices.append(indice)  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # add
            tdir.append(t2[:, :2])
            tab.append(t2[:, 2:4])
            # 从筛选出的网格中寻找与ab最匹配的anchor
            anchab.append(anchorsab[max_an_iou_idx])

        return tcls, tbox, indices, anch, tdir, tab, anchab