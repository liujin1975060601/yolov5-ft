# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from utils.metrics import bbox_iou, ab_iou, wh_iou
from utils.torch_utils import is_parallel
from utils.general import pts2dir
import math
from hungarian.hungarian_match import hungarian_match

criterion_p = nn.CrossEntropyLoss()

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
def bce_loss_without_sigmond(p, t):
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p))

def loop_loss(n_loop,ft_coefs_labels,ft_coef_pi):#ft_coefs_labels[nt,4*term]  ft_coef_min[nt, 4*term]
    if n_loop==0:
        return ft_coefs_labels
    #term = ft_coefs_labels.shape[1] // 4
    # a_ft -> [nt, term]
    nt = ft_coefs_labels.shape[0]
    #temp = torch.split(ft_coefs_labels.view(nt, -1, 4), 1, dim = -1)
    a_ft, b_ft, c_ft, d_ft = [coef.squeeze(-1) for coef in torch.split(ft_coefs_labels.view(nt, -1, 4), 1, dim = -1)] # 4| nt, term, 1
    
    ft_coef_loop = torch.zeros([n_loop, nt, ft_coefs_labels.shape[1]]).to(ft_coef_pi.device)#ft_coef_loop[n_loop, nt, term*4] ={8, nt, [k=4] * 4}
    # min_loss = 1e6s
    for t in range(n_loop):
        coef_num = a_ft.shape[-1]   # k  
        angle = torch.tensor(2*math.pi*t/n_loop)
        for k in range(coef_num):
            k_angle = (k+1) * angle
            t_cos = torch.cos(k_angle)
            t_sin = torch.sin(k_angle)
            
            # temp = torch.cos(k_angle) * a_ft[:, k] + torch.sin(k_angle) * b_ft[:, k]
        
        #   ak` = cos(k*2pi*t/n_loop) * ak + sin(k*2pi*t/n_loop) * bk 
            ft_coef_loop[t, :, 4*k + 0] = t_cos * a_ft[:, k] + t_sin * b_ft[:, k] #[nt, 1]
        #   bk` = sin(k*2pi*t/n_loop) * ak - cos(k*2pi*t/n_loop) * bk
            ft_coef_loop[t, :, 4*k + 1] =-t_sin * a_ft[:, k] + t_cos * b_ft[:, k]

        #   ck` = cos(k*2pi*t/n_loop) * ck + sin(k*2pi*t/n_loop) * dk 
            ft_coef_loop[t, :, 4*k + 2] = t_cos * c_ft[:, k] + t_sin * d_ft[:, k]
        #   dk` = sin(k*2pi*t/n_loop) * ck - cos(k*2pi*t/n_loop) * dk
            ft_coef_loop[t, :, 4*k + 3] =-t_sin * c_ft[:, k] + t_cos * d_ft[:, k]
        #   4 [nt, 1]

    # ft_coef_loop - pi_f[:, 2:].detach().view(1, nt, term*4)  ({n_loop, nt, term*4} - {1, nt, term*4}).mean ->  {n_loop , nt}
    # ft_coef_loop[n_loop, nt, term*4]
    #pi_ft_coef = pi_f[:, 2:].clone()[None, ...]
    # ft_coef_loop = ft_coef_loop[1::2]
    loss_nt = torch.abs(ft_coef_loop - ft_coef_pi) # [n_loop, nt, term*4]
    loss_nt = loss_nt.mean(-1)  # [n_loop, nt]
    nt_indices = torch.argmin(loss_nt.T, dim=-1) # loss_nt.T[nt,n_loop] -> [nt]
    ft_coef_min = ft_coef_loop[nt_indices, torch.arange(ft_coefs_labels.shape[0])] # [nt, term*4]
    # ft_coef_loop[n_loop, nt, term*4] -> [nt_indices, :] -> ft_coef_min[nt, term*4]
    return ft_coef_min

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False,n_loop=8, hungarian=False,sort_obj_iou=0,D0=None):
        self.sort_obj_iou = sort_obj_iou
        device = next(model.parameters()).device  # get model device
        self.h = model.hyp  # hyperparameters
        self.hungarian_flag = hungarian
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['obj_pw']], device=device))
        self.order = {
            'Detect':None, 
            'Detect2':None,
            'FTInfer':None,
            'SubclassInfer':None,   
            'ValueInfer':None
        }
        # check whether it has Detect2, SubclassInfer or ValueInfer
        start = 0
        for i, h in enumerate(model.yaml['head']):  # each module only support one now
            if h[2] in ['Detect', 'Detect2', 'FTInfer', 'SubclassInfer', 'ValueInfer']:
                self.order[h[2]] = (start, start+3)
                start += 3
        # add
        # BCEdir = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['theta_pw']], device=device))
        # BCEab = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['ab_pw']], device=device))
        if self.order['Detect2'] is not None:
            MSELoss = nn.MSELoss(reduction='mean')
            L1Loss = nn.L1Loss(reduction='mean')
            BCEdir = nn.SmoothL1Loss(reduction='mean')
            BCEab = nn.SmoothL1Loss(reduction='mean')
            BCEpab = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['pab_pw']], device=device))


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=self.h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = self.h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.get_module_byname('Detect') if is_parallel(model) else model.get_module_byname('Detect')  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, self.h, autobalance

        self.offset = self.h.get('offset', 1)
        if(self.offset==0):
            self.gr=0
        
        # add
        if self.order['Detect2'] is not None:
            self.MSELoss = MSELoss
            self.L1Loss = L1Loss
            self.BCEdir = BCEdir
            self.BCEab = BCEab
            self.BCEpab = BCEpab

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
        
        # add
        det2 = model.module.get_module_byname('Detect2') if is_parallel(model) else model.get_module_byname('Detect2')
        if det2 is not None:
            setattr(self, 'anchorsab', getattr(det2, 'anchors'))
        else:
            setattr(self, 'anchorsab', [None] * len(self.anchors))
        
        self.svinfer = 0
        det3 = model.module.get_module_byname('SubclassInfer') if is_parallel(model) else model.get_module_byname('SubclassInfer')
        if det3 is not None:
            setattr(self, 'ns', getattr(det3, 'ns'))
            self.BCEcls_sub = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h.get('cls_sub_pw',1)], device=device))
            self.svinfer += 1
        
        det4 = model.module.get_module_byname('ValueInfer') if is_parallel(model) else model.get_module_byname('ValueInfer')
        if det4 is not None:
            setattr(self, 'nv', getattr(det4, 'nv'))
            # self.MSEvinfer = nn.MSELoss(reduction='mean')
            self.MSEvinfer = nn.SmoothL1Loss()
            self.svinfer += self.nv
        
        det_ft = model.module.get_module_byname('FTInfer') if is_parallel(model) else model.get_module_byname('FTInfer')
        if det_ft is not None:
            setattr(self, 'ft_coef', getattr(det_ft, 'ft_coef'))
            self.ft_len = 2 + 4 * self.ft_coef
            self.ftloss_xy = nn.SmoothL1Loss()
            self.ftloss_coef = nn.SmoothL1Loss()
            self.lamda = det_ft.lamda
            self.d0 = det_ft.d0
        else:
            self.ft_len = 0
        
        self.n_loop=n_loop
        # self.lamda = lamda
        self.D0 = torch.from_numpy(D0).to(device=device)

    def __call__(self, p, targets, fold, mask_dir = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], paths = None):  # predictions, targets, model
        device = targets.device
        #targets[1(b)+1(cls)+4(box)  +  4*2]来自于dataset
        dir_scale = torch.zeros(len(mask_dir), device=device)#mask_dir
        dir_scale[np.array(mask_dir) > 0] = 1.0
         #p[nl ~ nl][b,a,GH,GW,4+1+class ~ 1+2+2]

        # pH = p[:3]#pH[nl][b,a,GH,GW,4+1+class]
        # pD = p[3:]#pD[nl][b,a,GH,GW,1+2+2]
        #pH[nl][b,a,GH,GW,4+1+class]
        #pD[nl][b,a,GH,GW,1+2+2]
        #pS[nl][b,a,GH,GW,ns]
        #pV[nl][b,a,GH,GW,nv]
        pH = p[self.order['Detect'][0]:self.order['Detect'][1]]
        pD = p[self.order['Detect2'][0]:self.order['Detect2'][1]] if self.order['Detect2'] is not None else [None] * len(pH)
        pF = p[self.order['FTInfer'][0]:self.order['FTInfer'][1]] if self.order['FTInfer'] is not None else [None] * len(pH)
        pS = p[self.order['SubclassInfer'][0]:self.order['SubclassInfer'][1]] if self.order['SubclassInfer'] is not None else [None] * len(pH)
        pV = p[self.order['ValueInfer'][0]:self.order['ValueInfer'][1]] if self.order['ValueInfer'] is not None else [None] * len(pH)

        bs = pH[0].shape[0]
        assert (pF is None) or pF[0].shape[0]==bs
        assert ((paths is None) or len(paths)==bs)

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # add

        # tcls, tbox, indices, anchors, tdir, tab, anchorsab, svinfer = self.build_targets_dir(pH, pD, targets, fold)  # targets

        tcls, tbox, indices, anchors, tdir, tab, anchorsab, svinfer, ftinfer, tiou = self.build_targets_whole(pH, pD, pF, pS, pV, targets, fold)
        #tcls[nl][nt]
        #tbox[nl][nt,4]
        #indices[nl][5(b, a, gj, gi, aab)][nt]
        #anchors[nl][nt,2]
        #tdir[nl][nt,2]
        #tab[nl][nt,2]
        #anchorsab[nl][nt,2]
        #tiou[nl][nt]

        ldir, lab, lpab = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # lcos = torch.zeros(1, device=device)
        # lsin = torch.zeros(1, device=device)
        # la = torch.zeros(1, device=device)
        # lb = torch.zeros(1, device=device)
        lf_xy = torch.zeros(1, device=device)
        lf_coef = torch.zeros(1, device=device)
        ls = torch.zeros(1, device=device)
        lv = torch.zeros(1, device=device)
        # Losses
        for i, (pi, pi_dir, pi_f, pi_s, pi_v) in enumerate(zip(pH, pD, pF, pS, pV)):  # layer index, layer predictions
            #pi[b,a,GH,GW,4+1+class]
            #pi_dir[b,a,GH,GW,1+2+2]
            b, a, gj, gi, aab = indices[i]  # image, anchor, gridy, gridx
            #[nt]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            #tobj[b,a,GH,GW]
            nt = b.shape[0]  # number of targets
            if nt:
                dir_scale_objs = dir_scale[tcls[i]]#dir_scale_objs[nt]
                assert(dir_scale_objs.shape[0]==nt)
                dir_set = dir_scale_objs > 0
                not_dir_set = torch.logical_not(dir_set)
            
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                #ps[nt,4+1+class]
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                #pxy[nt,2]   -0.5~1.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 0~4
                #pwh[nt,2]   0~4
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #pbox[nt,4]
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                assert len(tiou)==0 or  iou.shape[0]==tiou[i].shape[0]
                #iou[nt]
                #lbox += (1.0 - iou[not_dir_set]).mean()\
                #      + self.MSELoss(pbox[dir_set,:2],tbox[i][dir_set,:2])# poly center xy loss
                if torch.nonzero(not_dir_set).shape[0] > 0:
                    lbox += (1.0 - iou[not_dir_set]).mean()
                if torch.nonzero(dir_set).shape[0] > 0:
                    lbox += self.L1Loss(pbox[dir_set,:2],tbox[i][dir_set,:2])# poly center xy loss
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)#tiou[i]
                #score_iou[nt]
                #tobj[b,a,GH,GW]
                if self.sort_obj_iou < 2:
                    if self.sort_obj_iou==1:
                        sort_id = torch.argsort(score_iou)
                        b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                else:
                    assert self.sort_obj_iou==2
                    tobj[b, a, gj, gi] += (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                    tobj = torch.where(tobj > 1, torch.tensor(1), tobj)
                    assert torch.all((tobj >= 0) & (tobj <= 1)), "Not all elements are in the [0, 1] range."

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    #t[nt,class]==self.cn==0
                    t[range(nt), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # add
                if pi_dir is not None:
                    pi_dir = pi_dir[b, :, gj, gi]
                    #pi_dir[b,a,h,w,1+2+2]-->pi_dir[nt,na,1+2+2]
                    na_dir = pi_dir.shape[1]

                    pi_dirs = pi_dir[np.arange(nt), aab, :][dir_set]#aab[nt]
                    #pi_dirs[nt_dir,1(p)+2(q)+2(a,b)]
                    nt_dir = pi_dirs.shape[0]
                    tpab = torch.zeros((nt_dir, na_dir), device=device)#torch.zeros_like(pi_dir[:,:, 0], device=device)
                    #tpab[nt_dir,na]
                    # theta
                    q2 = pi_dirs[:, 1:3].sigmoid() * 2 - 1
                    assert(q2.shape[0]==nt_dir)
                    #q2[nt_dir,2]
                    if 0:
                        Lq2 = torch.norm(q2, dim=1, keepdim=True)
                        q2 = q2 / (Lq2 + 1e-8)  # 避免0除问题

                    tdir[i] = tdir[i][dir_set]
                    #q2[nt_dir,2]  #tdir[i][nt_dir,2] dir_scale_objs[nt_dir]
                    ldir += self.BCEdir(q2, tdir[i])#dir_scale_objs
                    # lcos += self.BCEdir(q2[:, 0], tdir[i][:, 0])
                    # lsin += self.BCEdir(q2[:, 1], tdir[i][:, 1])
                    # ldir += (lcos + lsin)
                    # ab  #anchorsab[i][nt_dir,2]
                    a_b = (pi_dirs[:, 3:5].sigmoid() * 2) ** 2 * anchorsab[i][dir_set]
                    #a_b[nt_dir,2]
                    # a_b =  torch.exp(pi_dirs[:, 3:5]) * anchorsab[i]
                    tab[i] = tab[i][dir_set]#tab[i][nt_dir,2]
                    #a_b_iou = ab_iou(a_b, tab[i])
                    #a_b_iou[nt_dir]
                    #lab += (1.0 - a_b_iou).mean()
                    lab += self.MSELoss(a_b, tab[i])
                    # pab
                    '''
                    for at in range(self.na):
                        aab_sel = aab==at
                        pi_ps = pi_dir[b[aab_sel], at, gj[aab_sel], gi[aab_sel]]
                        #lpab+=self.L1Loss(pi_ps[:,0],torch.ones_like(pi_ps[:,0]))
                        not_aab_sel = aab!=at
                        pi_ps_n = pi_dir[b[not_aab_sel], at, gj[not_aab_sel], gi[not_aab_sel]]
                        #lpab+=self.L1Loss(pi_ps_n[:,0],torch.zeros_like(pi_ps_n[:,0]))
                        assert(pi_ps[:,0].shape[0]+pi_ps_n.shape[0]==b.shape[0])
                    '''
                    #pi_dir[nt,na_dir,1(p)+2(q)+2(a,b)] 
                    ppdir = pi_dir[..., 0][dir_set] #ppdir[nt_dir,na_dir]
                    aab_dir_set = aab[dir_set]#aab_dir_set[nt_dir]
                    if self.hyp['softmax_dir']:#AP50 = 0.9305
                        lpab += criterion_p(ppdir, aab_dir_set)#tpab[nt_dir,na_dir]  aab_dir_set[nt_dir]
                    else:
                        #tpab[nt_dir,na_dir]
                        tpab[np.arange(nt_dir),aab_dir_set] = 1.0
                        lpab += self.BCEpab(ppdir, tpab)#tpab[nt_dir,na_dir]

                if pi_s is not None or pi_v is not None:
                    # ps_sv = pi[b, a, gj, gi]
                    # svinfer should be [filter, cms(s+v) * 2]
                    svinfer_i, mask_sv_i = torch.chunk(svinfer[i], 2, dim=-1)
                    v_start = 0
                    if pi_s is not None:
                        ps_s = pi_s[b, a, gj, gi]
                        sinfer_i = svinfer_i[:, 0].long()
                        mask_s_i = mask_sv_i[:, 0] > 0
                        v_start = 1
                        # calc subclass loss
                        t = torch.full_like(ps_s, 0, device=device)  # targets
                        #t[nt,class]==self.cn==0
                        t[range(nt), sinfer_i] = self.cp
                        t = t[mask_s_i]
                        ls += self.BCEcls_sub(ps_s[mask_s_i], t)  # BCE

                    if pi_v is not None:
                        ps_v = pi_v[b, a, gj, gi]
                        for v in range(0, ps_v.shape[-1]):
                            vinfer_i = svinfer_i[:, v + v_start]
                            mask_v_i = mask_sv_i[:, v + v_start] > 0
                            ps_vv = ps_v[..., v][mask_v_i]
                            vinfer_i = vinfer_i[mask_v_i]
                            lv += self.MSEvinfer(ps_vv.sigmoid(), vinfer_i)
                    
                if pi_f is not None:
                    pi_f = pi_f[b, a, gj, gi]#pi_f[nt, 2 + term*4]
                    assert nt==pi_f.shape[0]
                    # a0, c0
                    # pi_f[:, :2] = pi_f[:, :2].sigmoid() * 2 - 0.5    (-0.5~1.5)
                    ft_xy_pi = 0.5 + self.lamda * self.D0[None,:2] * (2 * pi_f[:, :2].sigmoid() - 1)  #pi_f[:, :2].sigmoid() * 2 - 0.5 #ft_xy_pi[nt, 2]

                    lf_xy += self.ftloss_xy(ft_xy_pi, ftinfer[i][:, :2]) #[nt, 2]
                    # term
                    term = (pi_f.shape[1]-2) // 4
                    
                    # pi_f[:, 2:] = pi_f[:, 2:].sigmoid() * 2 - 1   -1~1    0~1 
                    #anchors_coef = 2 * anchors[i].repeat(1, 2).reshape(-1, 4) #[nt,2]->#[nt,4]   
                    # 将b张量的形状从 [nt, 2] 重塑为 [nt, 4] ，使其为 [x, x, y, y]
                    anchors_coef = self.d0 * anchors[i].view(nt, 2, 1).repeat(1, 1, 2).view(nt, -1) #[nt,2]->#[nt,4]
                    # pi * anchors /            
                    ft_coef_pi = self.D0[None,2:] * (pi_f[:, 2:].sigmoid() * 2 - 1) * anchors_coef.repeat(1, term)#[nt,4term]*[nt,4term]-->[nt,4term]

                    ft_coefs_labels = ftinfer[i][:, 2:] #[nt, term*4] labels    * fea wh 
                    if self.n_loop==0:
                        lf_coef += self.ftloss_coef(ft_coef_pi, ft_coefs_labels)   # a1,b1,....an,bn,cn,dn 
                    else:
                        assert term==ft_coefs_labels.shape[1] // 4
                        ft_coef_min = loop_loss(self.n_loop,ft_coefs_labels,ft_coef_pi)
                        # ft_coef_loop[n_loop, nt, term*4] -> [nt_indices, :] -> ft_coef_min[nt, term*4]
                        # new -> ftinfer
                        lf_coef += self.ftloss_coef(ft_coef_pi, ft_coef_min)   # a1,b1,....an,bn,cn,dn 

            pobj = pi[..., 4]
            if self.h.get('pn_obj', 1) == 1:
                obji = self.BCEobj(pobj, tobj)
            else:#self.h['max_obj']默认256，dota1.5这种数据集里面最大目标数量可能超过256爆仓
                # 计算 tobj > 0 的损失
                # 创建一个等于0的掩码，只有一定比例的元素为True
                mask = torch.rand_like(tobj) < self.h.get('pn_obj', 1)  # mask_ratio 是你要保留的比例
                mask_obj,mask_noobj = tobj > 0, (tobj == 0) & mask
                pim = torch.cat([pobj[mask_obj], pobj[mask_noobj]], dim=0)
                tm = torch.cat([tobj[mask_obj], tobj[mask_noobj]], dim=0)
                pn_scale = 1 #((mask_obj.sum() + mask_noobj.sum()) / tobj.numel()).detach()
                obji = pn_scale * self.BCEobj(pim, tm)

            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        
        if pD[0] is not None:
            ldir *= self.hyp['theta']
            lab *= self.hyp['ab']
            lpab *= self.hyp['pab']
        
        if pF[0] is not None:
            lf_xy *= self.hyp.get('ft_infer_xy', 0.25)
            lf_coef *= self.hyp.get('ft_infer_coef', 1.0)

        if pS[0] is not None:
            ls *= self.hyp.get('subclass_infer', 0.02)
        
        if pV[0] is not None:
            lv *= self.hyp.get('value_infer', 0.02)
        
        return (lbox + lobj + lcls + ldir + lab + lpab + lf_xy + lf_coef + ls + lv) * bs, torch.cat((lbox, lobj, lcls, ldir, lab, lpab, lf_xy, lf_coef, ls, lv)).detach()

    def build_targets_whole(self, pH, pD, pF, pS, pV, target_all, fold):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt, svlength = self.na, target_all.shape[0], self.svinfer   # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        tdir, tab, anch2, tiou = [], [], [], []
        tft = []
        svinfer = []
        gain = torch.ones(7, device=target_all.device)  # normalized to gridspace gain
        gain_dir = torch.ones(4, device=target_all.device)
        gain_ft = torch.ones(self.ft_len, device=target_all.device)
        ai = torch.arange(na, device=target_all.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        has_D = pD[0] is not None
        has_F = pF[0] is not None
        has_S = pS[0] is not None
        has_V = pV[0] is not None
        
        targets = target_all[:, :6] #[nt,6]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=target_all.device).float() * g  # offsets
        if has_D:
            pts_targets = torch.clone(target_all[:, 6:14]) #[nt,4*2]
            dir_targets = pts2dir(pts_targets, fold_angle=fold) #[nt,2+2]
            dir_targets = dir_targets.repeat(na, 1, 1)
        if has_F:
            start_ft = 14 if has_D else 6
            ft_targets = torch.clone(target_all[:, start_ft:start_ft + self.ft_len])
            ft_targets = ft_targets.repeat(na, 1, 1)
        for i in range(self.nl):
            anchors, shape = self.anchors[i], pH[i].shape
            anchors2 = self.anchorsab[i]
            t_dir = torch.zeros([0, 4], device=target_all.device)
            cms_infer = torch.zeros([0, svlength], device=target_all.device)
            t_ft = torch.zeros([0, self.ft_len], device=target_all.device)
            # Match targets to anchors
            if nt:
                # Matches
                gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xywh gain
                t = targets * gain  # shape(3,n,7)  [img, cls, x, y, w, h, a]
                
                # Matches
                # 筛选wh的anchor,得到真值与各个anchors的相似性，越接近1越相似
                #[na,nt,2]/[na,?,?]=r[na,nt,2]
                #None是补充插入扩展一个长度=1的维度，后面的维度[2]可以省略，长度=1的维度可以在矩阵除法过程中做广播
                #r = t[..., 4:6] / anchors[:, None]  # wh ratio
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                #r[na,nt,2]
                #第1个max(r, 1 / r)得到一个大于1的数，越接近1越相似，找到与对应anchors的相似度
                  #两个维度r[na,nt,2]的数组之间找最大值维度依然是[na,nt,2]
                #第2个max(2)是在[na,nt,2]的第2个维度上找最大值(对应的值是[0],索引(0,1)是[1])
                # 得到torch.max(r, 1 / r).max(2)[0]的维度[na,nt]，因此j是一个[na,nt]的bool数组
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                #j[na,nt]

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                if has_D:
                    gain_dir[2:4] = torch.tensor(pD[i].shape)[[3, 2]]
                    # dir_targets = torch.cat((dir_targets.repeat(na, 1, 1), ai[:, :, None]), 2)
                    t_dir = dir_targets * gain_dir  # 
                    t_dir = t_dir[j]

                if has_S or has_V:
                    cms_infer = torch.clone(target_all[:, 14 if has_D else 6:]) #[nt, ns+nv]
                    cms_infer = cms_infer.repeat(na, 1, 1)
                    cms_infer = cms_infer[j]

                if has_F:
                    gain_ft[0:2] = torch.tensor(pF[i].shape)[[3, 2]]

                    gain_ft[2:] = torch.tensor(pF[i].shape)[[3,3,2,2]].repeat((self.ft_len - 2) // 4)   
                    t_ft = ft_targets * gain_ft
                    t_ft = t_ft[j]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                if self.offset:
                    gxi = gain[[2, 3]] - gxy  # inverse
                    j, k = ((gxy % 1 < g) & (gxy > 1)).T
                    l, m = ((gxi % 1 < g) & (gxi > 1)).T
                    j = torch.stack((torch.ones_like(j), j, k, l, m))
                    t = t.repeat((5, 1, 1))[j]
                    if has_D:
                        t_dir = t_dir.repeat((5, 1, 1))[j]
                    if has_S or has_V:
                        cms_infer = cms_infer.repeat((5, 1, 1))[j]
                    if has_F:
                        t_ft = t_ft.repeat((5, 1, 1))[j]
                    offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                else:
                    offsets = torch.zeros_like(gxy)
                if anchors2 is not None and t_dir.shape[0] != 0:
                    max_an_iou_idx = torch.max(wh_iou(anchors2, t_dir[:, 2:4]), dim=0)[1] 
                else:
                    max_an_iou_idx = torch.zeros(0, dtype=torch.long, device=t_dir.device)
            else:
                t = targets[0]
                offsets = 0
                max_an_iou_idx = torch.zeros(0, dtype=torch.long, device=t_dir.device)

            if self.hungarian_flag:
                # t [nt1, 7]   nt1 是包含根据比例过滤5邻域后的匹配gird数目
                # b, c, x, y, w, h, a
                # 0, 1, 2, 3, 4, 5, 6
                a = t[:, 6].long()
                grid_box = t[:, 2:6].clone()# x, y, w, h
                grid_box[:, 0:2] = t[:, 2:4].floor() + 0.5
                grid_box[:, 2:] = anchors[a]    # x, y, an_w, an_h
                gt_box = (target_all[:, :6].clone() * gain[:6])[:, 2:6]    # gt[nt, 4]
                ious, mathcied_ious, (gt_idx, grid_idx) = hungarian_match(grid_box, gt_box)
                tiou.append(mathcied_ious)

                t_dir = t_dir[grid_idx]
                max_an_iou_idx = max_an_iou_idx[grid_idx]
                offsets = offsets[grid_idx]
                t = t[grid_idx]

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1), max_an_iou_idx))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            if has_F:
                tft.append(torch.cat([t_ft[:, :2] - gij, t_ft[:, 2:]], 1))
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

            tdir.append(t_dir[:, :2])
            tab.append(t_dir[:, 2:4])
            anch2.append(anchors2[max_an_iou_idx] if anchors2 is not None else [])
            svinfer.append(cms_infer)   # cms + cms_mask

        return tcls, tbox, indices, anch, tdir, tab, anch2, svinfer, tft, tiou
    
