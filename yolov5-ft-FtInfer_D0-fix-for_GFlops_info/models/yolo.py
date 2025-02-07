# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys, os
from copy import deepcopy
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)
from torchsummary import summary

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)#这里又经过了一个卷积层确保了输出通道数是na(5+nc)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):            
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])  # conv x[nl][bs,a*(no=4+1+nc),H,W]
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            #x[i][bs,a*(no=4+1+nc),H,W]-->x[i][bs,na,no=4+1+nc,H,W]-->x[i][bs,na,H,W,no=4+1+nc]
            #self.no = nc + 5  # number of outputs per anchor维度放最后，便于推理时集中搜集信息

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:#初始时分配网格坐标
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                    #self.grid[i][1,na,H,W,2]是网格整数坐标    self.anchor_grid[i][1,na,H,W,2]
                    #注意：stride就是一个网格包含多少像素   anchors[]里面就是锚定框相对于网格的倍数*stride就是锚定框的像素数，详见_make_grid

                y = x[i].sigmoid()#y[bs,na,H,W,no=4+1+nc]
                if self.inplace:#true
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]#xy self.stride[i]意思是尺度i输出的一格代表输入多少像素，因此输出的box就是绝对像素坐标
                    # y[..., 2:4] = (y[..., 2:4] * 2) ** 2.6 * self.anchor_grid[i]  # wh
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z.append(y.view(bs, -1, self.no))
                z.append(y)#y[bs,na,H,W,no=4+1+nc]
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z

        # return x if self.training else (torch.cat(z, 1), x)
        return x if self.training else (z, x)#z[nl][bs,na,H,W,no=4+1+nc]

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        #yv[ny,nx]  xv[ny,nx]
        #torch.stack((xv, yv), 2)-->[ny,nx,2]-->[1,na,ny,nx,2]
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float() #返回grid网格左上角编号

        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float() #返回目标anchors的像素数量
        #self.anchors[3,3,2]-->anchor_grid[1,self.na=3,ny, nx,2]
        return grid, anchor_grid


class Detect2(Detect):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect2, self).__init__(nc, anchors, ch, inplace)
        self.nc = nc  # number of classes
        self.no = 5  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])  # conv  x[nl][bs,a*(no=1+2+2),H,W]
            bs, _, ny, nx = x[i].shape  # x(bs,15,20,20) to x(bs,3,20,20,5)
            assert(self.no==1+2+2)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            #x[i][bs,a*(no=1+2+2),H,W]-->x[i][bs,na,no=1+2+2,H,W]-->x[i][bs,na,H,W,no=1+2+2]
            #self.no = 5  # number of outputs per anchor维度放最后，便于推理时集中搜集信息

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:#初始时分配网格坐标
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()#y[bs,na,H,W,no=1(p)+2(dir)+2(ab)]
                if self.inplace:
                    y[..., 1:3] = y[..., 1:3] * 2 - 1   # theta
                    # y[..., 3:5] = (y[..., 3:5] * 2) ** 2.6 * self.anchor_grid[i]  # ab
                    y[..., 3:5] = (y[..., 3:5] * 2) ** 2 * self.anchor_grid[i]  # ab
                # if self.inplace:
                #     y = x[i]
                #     y[..., 0] = y[..., 0].sigmoid()
                #     y[..., 1:3] = y[..., 1:3].sigmoid() * 2 - 1   # theta
                #     y[..., 3:5] = torch.exp(y[..., 3:5]) * self.anchor_grid[i]  # ab
                else:
                    raise Exception("Detect2 forward error")
                # z.append(y.view(bs, -1, self.no))
                z.append(y)#y[bs,na,H,W,no=1(p)+2(dir)+2(ab)]
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        # return x if self.training else (torch.cat(z, 1), x)
        return x if self.training else (z, x)#z[nl][bs,na,H,W,no=1(p)+2(dir)+2(ab)]


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, ft_coef=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        if ft_coef and ft_coef != self.yaml['ft_coef']:
            LOGGER.info(f"Overriding model.yaml ft_coef={self.yaml['ft_coef']} with ft_coef={ft_coef}")
            self.yaml['ft_coef'] = ft_coef  # override yaml value


        # add for float range
        self.srange = self.yaml.get('srange', None)
        self.module_idx = {
        }
        self.cms_config = None
        self.mask_dir = [0] * nc
        for i, h in enumerate(self.yaml['head']):
            if h[2] in ['Detect', 'Detect2', 'SubclassInfer', 'FTInfer', 'ValueInfer']:
                self.module_idx[h[2]] = i + len(self.yaml['backbone'])

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        s = 256  # 2x min stride
        stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])[:3]
        m = self.get_module_byname('Detect')  # Detect()水平框输出层
        if isinstance(m, Detect):
            m.inplace = self.inplace
            m.stride = stride  # forward
            #[:3意思是前三个分支的输出][1,na*(no=4+1+nc),H,W]   x.shape[-2]意思是输出阵列H的长度
            #输出一格相当于原始图像中多少像素
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])[:3]  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1) #anchors[]的意思是目标长度的像素数转换成网格倍数
            self._initialize_biases(m) #only run once 针对水平框的1(obj)+nc输出做了特殊的bias初始化
        # add
        m = self.get_module_byname('Detect2')# Detect2()斜框输出层
        if isinstance(m, Detect2):
            m.inplace = self.inplace
            m.stride = stride  # forward
            #[:3意思是前三个分支的输出][1,na*(no=1+2+2),H,W]   x.shape[-2]意思是输出阵列H的长度
            #输出一格相当于原始图像中多少像素
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1) #anchors[]的意思是目标长度的像素数转换成网格倍数
            #self._initialize_biases(m) #这个只是针对水平框，斜框这么弄就错误了！针对水平框的1(obj)+nc输出做了特殊的bias初始化，造孽啊
        # Init weights, biases
        m = self.get_module_byname('FTInfer')
        if m:
            m.stride = stride
            m.anchors /= m.stride.view(-1, 1, 1) 
        self.stride = stride
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        # add
        out = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # y.append(x)  # save output
            # y.append(x)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            # add
            if type(m) in [Detect, Detect2, SubclassInfer, FTInfer, ValueInfer]:
                #x[nl][bs,na,H,W,no=4(box)+1(conf)+nc]
                #x[nl][bs,na,H,W,no=1+2+2]
                if self.training:
                    out.extend(x)# only x
                else:
                    out.append(x)#[b,a,H,W,4(box)+1(conf)+nc for Detect or 5=1+2+2 for Detect2]; list(z, x) 
                # out.append(x)
        # feat = y[18]
        # show_heatmap(feat.cpu().detach().numpy()[0])
        # 输出热图时候同时输出y
        # return out, y
        return out#out[2=Detect+Detect2][3=3scales ouputs][b,a,H,W,4(box)+1(conf)+nc or 5=1+2+2]

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.get_module_byname('Detect').nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, m, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-2]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.self.get_module_byname('Detect')  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.get_module_byname('Detect')  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def get_module_byname(self, name:str):
        if not hasattr(self, 'module_idx'):
            self.module_idx = {
            }
            for i, h in enumerate(self.yaml['head']):
                if h[2] in ['Detect', 'Detect2', 'SubclassInfer', 'ValueInfer', 'FTInfer']:
                    self.module_idx[h[2]] = i + len(self.yaml['backbone'])
        idx = self.module_idx.get(name, -1)
        return self.model[idx] if idx != -1 else None


def parse_model(d, ch):  # model_dict, input_channels(3) 解析yaml格式的模型文件结构
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    anchorsab = d.get('anchorsab', None)

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ## add
    save2 = set()
    ##
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):#（f, n, m, args)对应yaml文件里面的from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings,作为嵌套执行的python字符串命令，这里返回一个commom定义模型类
        for j, a in enumerate(args):
            try:
                args[j] = d.get(a, None) or eval(a)  if isinstance(a, str) else a  # eval strings
            except (NameError, KeyError):
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, CBAM, CSPC, CSPCAM, CSPSAM]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, CSPC,CSPCAM, CSPSAM]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in [Detect, Detect2, ValueInfer, SubclassInfer, FTInfer]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int) and m not in [ValueInfer, SubclassInfer, FTInfer]:  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        # elif m is CBAM:
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module里面包含若干个小层
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 判断是否需要保存当前层的输出
        # save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                save2.add(x % i)
        layers.append(m_)#m_压入模型module列表
        if i == 0:
            ch = []
        ch.append(c2)
    save = list(save2)
    return nn.Sequential(*layers), sorted(save)


class ValueInfer(nn.Module):
    def __init__(self, nv, anchors=(), ch=()):
        '''
        na: 模型anchor的个数
        nv: 模型本输出分支二级属性的个数
        '''
        super().__init__()
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.nv = nv
        self.m = nn.ModuleList(nn.Conv2d(x, self.nv * self.na , 1) for x in ch)
    
    def forward(self, x):
        z = []  # inference output
        for i in range(self.na):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])  # conv x[nl][bs,a*(nv),H,W]
            bs, _, ny, nx = x[i].shape  # x(bs,a*(nv),H,W) to x(bs,3,H,W,nv)
            x[i] = x[i].view(bs, self.na, self.nv, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                z.append(x[i].sigmoid())
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        return x if self.training else (z, x)


class SubclassInfer(nn.Module):
    def __init__(self, ns=1, anchors=(), ch=()):
        '''
        na: 模型anchor的个数
        ns: 模型本输出分支二级属性的个数
        '''
        super().__init__()
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.ns = ns
        self.m = nn.ModuleList(nn.Conv2d(x, self.ns * self.na , 1) for x in ch)
    
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])  # conv x[nl][bs,a*(ns),H,W]
            bs, _, ny, nx = x[i].shape  # x(bs,a*(ns),H,W) to x(bs,3,H,W,ns)
            x[i] = x[i].view(bs, self.na, self.ns, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                z.append(x[i].sigmoid())
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        return x if self.training else (z, x)


class FTInfer(nn.Module):
    stride=None
    def __init__(self, ft_coef, lamda, d0, anchors=(), ch=(), inplace=True):
        '''
        na: 模型anchor的个数
        ft_coef: 模型本输出ft coef的个数
        '''
        super().__init__()
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.ft_coef = ft_coef
        self.ft_length = 2 + 4 * ft_coef
        self.m = nn.ModuleList(nn.Conv2d(x, self.ft_length * self.na , 1) for x in ch)
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.inplace = inplace
        self.lamda = lamda
        self.d0 = d0
        self.D0 = None

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])  # conv x[nl][bs,a*(nv),H,W]
            bs, _, ny, nx = x[i].shape  # x(bs,a*(nv),H,W) to x(bs,3,H,W,nv)
            x[i] = x[i].view(bs, self.na, self.ft_length, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i],self.anchor_grid[i] = self._make_grid(nx, ny, i) #self.anchor_grid[i][1,na=3,H,W,4]

                y = x[i].sigmoid()
                if self.inplace: # y[..., 0:2] [b,na=3,H,W,4coef]
                    y[..., 0:2] = (0.5 + self.lamda * (self.D0[None,None,None,None,:2] if self.D0 is not None else 1) * (2 * y[..., 0:2] - 1) + self.grid[i]) * self.stride[i] #center a0,c0
                    #anchors_coef = 2 * np.repeat(self.anchor_grid[i], [2, 2], axis=-1) #[nt,2]->[nt,4]
                    anchors_coef = self.d0 * self.anchor_grid[i]#anchors_coef[1,na=3,H,W,4*ft_coef]   abcd->xy 
                    b = y[..., 2:].shape[0]
                    anchors_coef = anchors_coef.expand(b, -1, -1, -1, -1) #anchors_coef[b,na=3,H,W,4*ft_coef]
                    #ty = y[..., 2:] * 2 - 1
                    y[..., 2:] = (self.D0[None,None,None,None,2:] if self.D0 is not None else 1) * (2 * y[..., 2:] - 1)*anchors_coef #->[b,na=3,H,W,4*ft_coef]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    coef = y[..., 2:] * 2 - 1
                    y = torch.cat((xy, coef), -1)
                z.append(y)
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        return x if self.training else (z, x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = next(self.parameters()).device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        #yv[ny,nx]  xv[ny,nx]
        #torch.stack((xv, yv), 2)-->[ny,nx,2]-->[1,na,ny,nx,2]
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float() #返回grid网格左上角编号grid[1, self.na=3, H, W, 2]
        #tt = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2, 1)).expand((1, self.na, ny, nx, 2, 2))
        #tt = tt.reshape(1, self.na, ny, nx, -1)
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2, 1)).expand((1, self.na, ny, nx, 2, 2)).reshape(1, self.na, ny, nx, -1).repeat(1,1,1,1, self.ft_coef).float() #返回anchors的像素数量
        #anchor_grid[1, self.na, ny, nx, 4] img shape
        return grid, anchor_grid



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = 'yolov5m-cam-ucas.yaml'
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    # model = Model(opt.cfg)
    # model.train()
    summary(model, (3, 640, 640), batch_size=1, device="cuda")
    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
