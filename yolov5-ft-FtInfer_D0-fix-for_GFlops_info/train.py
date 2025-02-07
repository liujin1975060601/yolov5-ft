"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# import val_big
import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader, cms_normalize_with_config, cms_check_latest
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, get_latest_run_exp, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer,
                           get_ft_num)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from tools.plotbox import plot_one_rot_box

from datetime import datetime, timedelta

import shutil

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, fold = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.fold

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    best_ap = best.with_name('best_mAP50.pt')

    # Hyperparameters
    hyp_file = save_dir / 'hyp.yaml'
    if(os.path.exists(hyp_file)):
        with open(hyp_file, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    else:
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
            LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        # Save run settings
        with open(hyp_file, 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
    
    #
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    assert(os.path.exists(data_dict['path']))

    train_path, val_path = data_dict['train'], data_dict['val']
    
    #names
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check

    #mask_dir
    mask_dir = data_dict.get('mask_dir', 0)   # mask_dir can be not exist.
    ft_coef = data_dict.get('ft_coef', 0)
    lamda = data_dict.get('lamda', 1.0)
    d0 = data_dict.get('d0', 2.0)
    if ft_coef == -1:
        ft_coef = get_ft_num(train_path)

    assert isinstance(mask_dir, (int, list)), f'{mask_dir}, Mask Dir Error.'
    mask_dir = [mask_dir] * nc if isinstance(mask_dir, int) else mask_dir
    if len(mask_dir) != nc:
        mask_dir = mask_dir + [0] * (nc - len(mask_dir))
    #
    cms_config = data_dict.get('cms_config', None)  # cms_config can be not exist.
    
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # add detect2 if any(mask_dir) else  

    # cms update model cfg
    if resume:
        cfg = None
    
    d2_flag = False
    if cfg is not None:
        subclass_infer, value_infer = 0, 0
        last_hindex = -1
        with open(cfg, encoding='ascii', errors='ignore') as f:
            cfg_dict = yaml.safe_load(f)  # model dict
        ft_flag = False
        for i, h in enumerate(cfg_dict['head']):
            if h[2] in ['Detect', 'Detect2', 'SubclassInfer', 'ValueInfer', 'FTInfer']:
                if last_hindex == -1:
                    last_hindex = i
                if h[2] == 'Detect2':
                    d2_flag = True
                if h[2] == 'FTInfer':
                    ft_flag = True
        # pts
        # if any(mask_dir) and not d2_flag:
        #     # for h in cfg_dict['head'][4:14][::-1]:
        #     #     cfg_dict['head'].insert(last_hindex, deepcopy(h))
        #     [cfg_dict['head'].insert(last_hindex, deepcopy(h)) for h in cfg_dict['head'][4:14][::-1]]
        #     cfg_dict['head'][last_hindex][0] = 13    # new branch behind module 13
        #     cfg_dict['head'].append(deepcopy(cfg_dict['head'][-1]))
        #     for i in range(len(cfg_dict['head'][-1][0])):
        #         cfg_dict['head'][-1][0][i] += 10    # new output layer
        #     cfg_dict['head'][-1][2] = 'Detect2'  # module name
        #     cfg_dict['head'][-1][3] = ['nc', 'anchorsab']  # module args
        #     cfg_dict['anchorsab'] = cfg_dict.get('anchorsab', cfg_dict['anchors'])
        #     d2_flag = True
        # ft
        # if ft_coef > 0 and not ft_flag:
        #     [cfg_dict['head'].insert(last_hindex, deepcopy(h)) for h in cfg_dict['head'][4:14][::-1]]
        #     cfg_dict['head'][last_hindex][0] = 13    # new branch behind module 13
        #     cfg_dict['head'].append(deepcopy(cfg_dict['head'][-1]))
        #     for i in range(len(cfg_dict['head'][-1][0])):
        #         cfg_dict['head'][-1][0][i] += 10    # new output layer
        #     cfg_dict['head'][-1][2] = 'FTInfer'  # module name
        #     cfg_dict['head'][-1][3] = ['ft_coef', 'lamda', 'd0','anchors']  # module args
        #     cfg_dict['ft_coef'] = ft_coef
        #     cfg_dict['lamda'] = lamda
        #     cfg_dict['d0'] = d0
        # cms
        # if cms_config is not None:
        #     srange = []
        #     for i in cms_config:
        #         if i['type'] == 'int':
        #             subclass_infer += 1
        #             srange.append(i['range'])
        #         elif i['type'] == 'float':
        #             value_infer += 1
        #         else:
        #             raise TypeError('CMS CONFIG ERROR')
        #     if subclass_infer > 0:
        #         assert subclass_infer == 1, 'Do not support subclass more than 1 yet.'
        #         [cfg_dict['head'].insert(last_hindex, deepcopy(h)) for h in cfg_dict['head'][4:14][::-1]]
        #         cfg_dict['head'][last_hindex][0] = 13    # new branch behind module 13
        #         cfg_dict['head'].append(deepcopy(cfg_dict['head'][-1]))
        #         for i in range(len(cfg_dict['head'][-1][0])):
        #             cfg_dict['head'][-1][0][i] += 10    # new output layer
        #         cfg_dict['head'][-1][2] = 'SubclassInfer'  # module name
        #         cfg_dict['head'][-1][3] = ['srange', 'anchors']  # module args
        #         cfg_dict['srange'] = srange[0]
        #         pass    # add later
        #     if value_infer > 0:
        #         [cfg_dict['head'].insert(last_hindex, deepcopy(h)) for h in cfg_dict['head'][4:14][::-1]]
        #         cfg_dict['head'][last_hindex][0] = 13    # new branch behind module 13
        #         cfg_dict['head'].append(deepcopy(cfg_dict['head'][-1]))
        #         for i in range(len(cfg_dict['head'][-1][0])):
        #             cfg_dict['head'][-1][0][i] += 10    # new output layer
        #         cfg_dict['head'][-1][2] = 'ValueInfer'  # module name
        #         cfg_dict['head'][-1][3] = ['num_v', 'anchors']  # module args
        #         cfg_dict['num_v'] = value_infer
        # cfg = cfg_dict

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if(os.path.basename(weights)=='last.pt'):#resume==True mode
            if(ckpt['epoch']==-1):
                ckpt['epoch']=epochs-1
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), ft_coef=ft_coef).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), ft_coef=ft_coef).to(device)  # create

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness, best_ap50 = 0, 0.0, 0.0
    if pretrained:
        if resume:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']
                best_ap50 = ckpt.get('best_ap50', 0.0)

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'

        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    if resume:
        d2_flag = model.get_module_byname('Detect2') is not None
        cms_num = 0
        for key, num in zip(['SubclassInfer', 'ValueInfer'], [1, 'nv']):
            p = model.get_module_byname(key)
            if p:
                if isinstance(num, int):
                    cms_num += num 
                else: 
                    cms_num += getattr(p, num)
        if cms_num > 0:
            cms_config = model.cms_config
        else:
            cms_config = None        
    else:
        cms_num = len(cms_config) if cms_config else 0
        
    pts = d2_flag
    fxs_flag = False
    fxs_func = None
    if (Path(train_path) / 'fxs_func.npy').exists():
        fxs_func = np.load(Path(train_path).parent.resolve() / 'fxs_func.npy', allow_pickle=True)
        if fxs_func[-1] != cms_check_latest(path=[train_path, val_path]):
            fxs_flag = True
    else:
        fxs_flag = cms_config is not None
    if fxs_flag:
        cms_normalize_with_config(cms_path=Path(train_path).parent.resolve() / 'cms_config.json',
                                    path=[train_path, val_path],
                                    save_path=Path(train_path).parent.resolve())
        fxs_func = np.load(Path(train_path).parent.resolve() / 'fxs_func.npy', allow_pickle=True)

    if fxs_func is not None:
        fxs_func = fxs_func[:-1]

    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=opt.augment, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True,
                                              pts=pts, cms=cms_config if cms_config is not None else False,
                                              fxs=fxs_func,
                                              ft_coef=ft_coef,
                                              save_dir=save_dir,debug_samples=10)
    
    if fxs_func is not None:
        fxs_func = fxs_func[dataset.reorder]
        np.save(w / 'fxs_func.npy', fxs_func, allow_pickle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches / epoch
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    # Process 0
    if RANK in [-1, 0]:
        val_count = data_dict.get('val_count', 0)
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=False, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '),
                                       pts=pts, cms=cms_config if cms_config is not None else False,
                                       fxs=fxs_func,ft_coef=ft_coef,
                                       save_dir=save_dir,debug_samples=0,sample_count=val_count)[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)#[ntotal,13]
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            # if plots:
            #     plot_labels(labels[:, :5], names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    m = de_parallel(model).get_module_byname('Detect') # number of detection layers (used for scaling hyp['obj'])
    nl = m.nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= ((imgsz if isinstance(imgsz,int) else max(imgsz)) / 640) ** 2 * 3. / nl  # scale to image size and layers

    #model.info(img_size=opt.imgsz) #fail
    mft = de_parallel(model).get_module_byname('FTInfer') 
    mft.D0 = torch.from_numpy(dataset.D0).to(device=device)
    #ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights','FTInfer'])
    mft = ema.ema.get_module_byname('FTInfer') 
    mft.D0 = torch.from_numpy(dataset.D0).to(device=device)

    imgszs = [opt.imgsz,[512,512],[640,640],[480,640],[768,768],[896,896],[1024,1024]]
    for size in imgszs:
        print(size)
        model.info(img_size=size) #ok

    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    #dataset.labels[nimage][nt,1+4+8] 根据数据集dataset.labels[]得到每类静态权重类权重model.class_weights[nc]与类数量成反比
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    #model.class_weights[nc]
    model.names = names
    model.mask_dir = mask_dir
    data_dict['mask_dir'] = deepcopy(model.mask_dir)

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = [0 for i in range(14)]  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls) + dir+ab+pab+ftxy+ftcoef+cms_s+cms_v
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model,n_loop=hyp.get('n_loop',8), hungarian=hyp.get('hm',0), sort_obj_iou=hyp.get('sort_obj_iou',0),D0=dataset.D0)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    epoch = start_epoch
    mloss = None
    last_save_time = datetime(2019, 1, 1, 1, 1, 1)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            #model.class_weights[nc]   maps[nc]初值=0
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            #cw[nc]是根据上轮多类map动态调节的类别权重，第一轮maps[:]==0所以和model.class_weights一样
            #maps[nc]
            #cw[n] 类权重与类数量成反比，与每类map误差成正比
            #dataset.labels[nimages][nt,1+4(box)+4(pts)*2]
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            #iw[nimages]
             # nn*wn / (iw.sum(0) + n*wn) = neg_alpha--> nw = neg_alpha * iw.sum(0) / (nn-neg_alpha*n)
            neg_set = iw == 0
            neg_count = iw[neg_set].shape[0]#负样本数量
            neg_alpha = hyp.get('neg_alpha', 0.03)#负样本所占权重比例
            #
            #if neg_count > 0 and neg_count/iw.shape[0] > neg_alpha:
            #   nw = neg_alpha * iw.sum(0) / (neg_count-neg_alpha*iw.shape[0])
            #   iw += nw
            #   neg_alpha_val = iw[neg_set].sum(0) / iw.sum(0)
            #   print(neg_alpha_val)
            if neg_count > 0:
                nw = neg_alpha * iw.sum(0) / ((1 - neg_alpha)*neg_count)
                iw[neg_set] += nw #原来=0,+nw其实就是=nw
                neg_alpha_val = iw[neg_set].sum(0) / iw.sum(0)
                if start_epoch==epoch:#仅第一个epoch显示负样本比例
                    print("neg_alpha_val=",neg_alpha_val)
            #根据iw[dataset.n]在dataset.n这么多图像中随机选择索引dataset.indices[dataset.n]
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # mloss = torch.zeros(3, device=device)  # mean losses
        mloss = torch.zeros(10, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 14) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'dir', 'ab', 'pab', 'ft_xy', 'ft_coef', 'cms_s', 'cms_v', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            #imgs[b,3,H,W]   targets[nt,14=1[b]+1[class]+4[box]+2*4]
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward imgs[b,3,H,W]  pred[6][b,a,GH,GW,4+1+class]  targets[nt,1(batch) + 1(cls)+4(box)+4*2]
                loss, loss_items = compute_loss(pred, targets.to(device), fold=fold, mask_dir=model.mask_dir, paths=paths)  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                #if ni - last_opt_step > 4:
                    #break#TODO!
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                #mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                #mloss = [(mloss[j] * i + loss) / (i + 1) for j, loss in enumerate(loss_items) if not torch.isnan(loss)]
                for j, loss in enumerate(loss_items):
                    if not torch.isnan(loss):
                        mloss[j] = (mloss[j] * i + loss) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 11 + '%10s') % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], f'{imgs.shape[-1]}x{imgs.shape[-2]}'))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------
            # break#TODO!
        # noval = False
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop  # epochs
            fi = None
            if epoch >= data_dict.get('val_epoch', 0) and (not noval or final_epoch):  # Calculate mAP
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           iou_thres=0.2 if ft_coef > 0 else 0.45,
                                           ab_thres=3.0,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss,
                                           fold=fold,
                                           pts=pts,
                                           cms_num=cms_num,
                                           fxs_func=fxs_func
                                           )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi and fi > 0:
                    torch.save(ckpt, best)
                    #
                    # if epoch >= data_dict.get('val_epoch', 0):
                    #     print(f"save {epoch}.pt to best.pt with best_fitness:{best_fitness}")
                    src,dst = str(save_dir / 'threshs.npy'),str(w / 'threshs.npy')
                    #os.system(f'cp {src} {dst}')
                    shutil.copy(src, dst)
                    
                if best_ap50 < results[2]:
                    best_ap50 = results[2]
                    torch.save(ckpt, best_ap) # Path --> str() ****/weights/best.pt --> ****/weights/best_mAP50.pt

                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and (fi!=None and stopper(epoch=epoch, fitness=fi)):
                break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best, best_ap:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best_ap:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            conf_thres=0.001,
                                            iou_thres=0.2 if ft_coef > 0 else 0.45, # best pycocotools results at 0.65
                                            ab_thres=3.0,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,
                                            fold=fold,
                                            # mask_dir=model.mask_dir,
                                            pts=pts,
                                            cms_num=cms_num,
                                            fxs_func=fxs_func
                                            )  # val best model with plots
                    if is_coco:
                        if mloss!=None:
                            callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/dota.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-ft.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[640,640], help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')   # r'E:\PyCharmProject\yolov5_rot_imsize_0627\runs\train\exp255\weights\last.pt'
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=300, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--fold', type=int, default=1, help='fold angle')
    parser.add_argument('--augment', default=1, action='store_true', help='augment')
    parser.add_argument('--hm', action='store_true', default=False, help='计算loss时使用匈牙利匹配')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')


    # parser.add_argument('--pts', type=bool, action='store_false', default=False, help='use pts or not')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
	
	#coco2017
    opt.data = 'data/coco_ft.yaml'#切记里面的数据集路径存在可用！
    opt.cfg = 'models/yolov5l-coco_ft.yaml'#
    opt.weights = 'weights/yolov5l.pt'#
    #opt.resume = 'runs/train/exp_coco_ft2/weights/last.pt'
    opt.imgsz = [640,896]
    #GuGe
    # opt.data = 'data/Guge_ft.yaml'#切记里面的数据集路径存在可用！
    # opt.cfg = 'models/yolov5s-guge_ft.yaml'#
    # opt.weights = 'weights/yolov5s.pt'#
    # opt.imgsz = [640,640]
    #Dota1.5
    # opt.data = 'data/dota_ft.yaml'#切记里面的数据集路径存在可用！
    # opt.cfg = 'models/yolov5l-dota_ft.yaml'#
    # opt.weights = 'weights/yolov5l.pt'#
    # opt.imgsz = [640,896]
    #
    # opt.data = './data/dota.yaml'#切记里面的数据集路径存在可用！
    # opt.resume = False #'runs/train/exp_UCAS_ft/weights/last.pt'
    # #opt.cfg = './models/yolov5l.yaml'#
    # opt.cfg = './models/yolov5l-dota.yaml'#
    # opt.weights = 'weights/yolov5l.pt'
    # opt.imgsz = 640

    #Dota1.5   自标注3种数据集训练
    # opt.data = './data/dota_ft.yaml'
    # opt.resume = False #'./runs/train/exp_dota_ft_label_train48/weights/last.pt'
    # opt.cfg = './models/yolov5s-dota_ft.yaml'
    # opt.weights = 'weights/yolov5s.pt'
    # opt.imgsz = 640
    # opt.epochs = 200
    # ckpt_path = get_latest_run_exp(search_dir='./runs/train/')  # specified or most recent path
    # if(os.path.exists(ckpt_path+'/weights/last.pt')):
    #     opt.resume = True
    #opt.imgsz = 640
    
    # opt.noval = False
    # opt.batch_size = 8
    # opt.epochs = 100
    # opt.noautoanchor = False
    # # opt.single_cls = True
    # opt.hyp = 'data/hyps/hyp.ucas.yaml'
    opt.fold = 2
    opt.device = 0
    # opt.image_weights = True
    
    opt.name += f'_{Path(opt.data).stem}'
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)

    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        if not os.path.exists(ckpt):
            print(f'\033[91m{ckpt} not exists.\033[0m')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        #opt.cfg = ''  # reinstate
        if(os.path.basename(ckpt)=='last.pt'):
            opt.weights = ckpt  # reinstate
        opt.resume = True if os.path.exists(opt.weights) else False
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        if os.path.exists(opt.data) and os.path.exists(opt.cfg):
            opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
                check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            if opt.evolve:
                opt.project = str(ROOT / 'runs/evolve')
                opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
            # ckpt_path = get_latest_run_exp(search_dir='./runs/train/')  # specified or most recent path
            # if(not os.path.exists(ckpt_path)):
            #     opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
            # else:
            #     opt.save_dir = ckpt_path
        else:
            print(f'\033[91mdata_path:{opt.data} or {opt.cfg} not exist.注意路径大小写.\033[0m')
    
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
