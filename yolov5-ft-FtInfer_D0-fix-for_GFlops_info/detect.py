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
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_ft, print_args, scale_coords, scale_coords_ft, scale_coords_poly,strip_optimizer, xyxy2xywh, rot_nms,
                           rot_nms_with_cms, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from models.experimental import attempt_load
from tools.plotbox import plot_one_rot_box, plot_one_box, plot_one_box_with_cms

from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from tools.fft.fftinside import render_fourier_curve,render_fourier_curve2

#out,train_out = detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=data['mask_dir'])


def detect_with_cms(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle,mask_dir=[],threshs=torch.zeros(0),dt=[]):
    # inference
    cms_s = model.cms_s if isinstance(model, DetectMultiBackend) else (model.modules.get_module_byname('SubclassInfer') if hasattr(model, 'module') else model.get_module_byname('SubclassInfer')) is not None
    if dt!=[]:
        # inference
        t2 = time_sync()
        if training:
            out,train_out = [],[]
            tmp = model(im, augment=augment)
            for t in tmp:
                out.append(t[0])
                train_out.extend(t[1])
        else:
            out, train_out = model(im, augment=augment, val=True)
        t3 = time_sync()
        dt[1] += t3 - t2#dt[1]推理时间

        # NMS  pts[4*2]部分全部变成绝对坐标
        #pred[2=Detect+Detect2][nl][b,a,H,W,Dectct(5+c) or Detect2(5=1+2(dir)+2(ab))]
        out = rot_nms_with_cms(out, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir, cms_s=cms_s)
        #out[b][nt,10=2*4+1(conf)+1(cls)]
        dt[2] += time_sync() - t3#dt[2]后处理时间
        
        return out,train_out
    else:
        pred = model(im, augment=0, val=True)[0]
        #pred[2=Detect+Detect2][nl][b,a,H,W,Dectct(4(box)+1(conf)+nc) or Detect2(5=1(p)+2(dir)+2(ab))]
        pred = rot_nms_with_cms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir, cms_s=cms_s)
        #pred[b][nt,10=4(pts)*2+1(conf)+1(cls)]
        return pred

def detect_cms(model, training, im, augment, conf_thres, iou_thres, dt=[], agnostic=False, cms_s=False, cms_num=0, ft_infer=False):
    t2 = time_sync()
    if training:
        out,train_out = [],[]
        tmp = model(im, augment=augment)
        for t in tmp:
            out.append(t[0])
            train_out.extend(t[1])
    else:
        out, train_out = model(im, augment=augment, val=True)
    if dt!=[]:
        dt[1] += time_sync() - t2
            
    # NMS
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
        pFt = torch.cat(pFt, 1)
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
    # out, indices = non_max_suppression(pbox_out, conf_thres, iou_thres, agnostic=agnostic, return_indices=True)
    if pFt is None or False:
        out, indices = non_max_suppression(pbox_out, conf_thres, iou_thres, agnostic=agnostic, return_indices=True)
    else:
        out, out_ft, indices = non_max_suppression_ft(pbox_out, pFt, conf_thres, iou_thres, agnostic=agnostic, return_indices=True)

    
    if pFt is not None:
        pFt = [pFt[i][idx] if idx.shape[0] > 0 else torch.zeros(0, pFt.shape[-1], device=pFt.device) for i, idx in enumerate(indices)]
        for i, j in enumerate(out):
            out[i] = torch.cat([j, pFt[i]], -1)
    
    if pCms is not None:
        pCms = [pCms[i][idx] if idx.shape[0] > 0 else torch.zeros(0, pCms.shape[-1], device=pCms.device) for i, idx in enumerate(indices)]
        for i, j in enumerate(out):
            out[i] = torch.cat([j, pCms[i]], -1)
    dt[2] += time_sync() - t3
    return out


def detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle,mask_dir=[],threshs=torch.zeros(0),dt=[]):
    # inference
    if dt!=[]:
        # inference
        t2 = time_sync()
        if training:
            out,train_out = [],[]
            tmp = model(im, augment=augment)
            for t in tmp:
                out.append(t[0])
                train_out.extend(t[1])
        else:
            out, train_out = model(im, augment=augment, val=True)
        t3 = time_sync()
        dt[1] += t3 - t2#dt[1]推理时间

        # NMS  pts[4*2]部分全部变成绝对坐标
        #out[2=Detect+Detect2][nl][b,a,H,W,Dectct(5+c) or Detect2(5=1+2(dir)+2(ab))]
        out = rot_nms(out, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir, threshs=threshs)
        #out[b][nt,10=2*4+1(conf)+1(cls)]
        dt[2] += time_sync() - t3#dt[2]后处理时间
        
        return out,train_out
    else:
        pred = model(im, augment=0, val=True)[0]
        #pred[2=Detect+Detect2][nl][b,a,H,W,Dectct(4(box)+1(conf)+nc) or Detect2(5=1(p)+2(dir)+2(ab))]
        pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir,threshs=threshs)
        #pred[b][nt,10=4(pts)*2+1(conf)+1(cls)]
        return pred

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.1,  # confidence threshold
        ab_thres=3,
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        half=True,  # use FP16 half-precision inference
        plot_label=True,
        dir_line=False,
        fold=2,
        data='',
        nc=8,
        mask_dir=[],
        thresh_scale=0.2,
        plot_inside=0
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    else:
        if not os.path.exists(source):
            print(f'\033[91mimages_path:{source} not exist.\033[0m')

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    # model = attempt_load(weights, map_location=device, fuse=True)
    # stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    stride, names, mask_dir = model.stride, model.names, mask_dir or model.mask_dir

    # imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    pt = True
    half = True
    if pt:
        model.model.half() if half else model.model.float()
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
        bs = 1  # batch_size
    
    train_path = Path(weights).parent.parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        threshs = torch.from_numpy(threshs)
    else:
        threshs = torch.ones(len(names)) * conf_thres
    threshs = threshs * thresh_scale
    conf_thres = threshs.to(device)


    fxs_path = Path(weights).parent.resolve() / 'fxs_func.npy'
    if fxs_path.exists():
        fxs_func = np.load(fxs_path, allow_pickle=True)
        fxs_func = fxs_func
        for i in range(fxs_func.shape[0]):
            if fxs_func[i] is not None:
                LOGGER.info(f"ValueInfer[{i}]: {fxs_func[i]['y'].min()} ~ {fxs_func[i]['y'].max()}")
    else:
        fxs_func = None

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    pts = model.pts
    cms_num = model.cms_num
    ft_infer = model.ft_infer
    ft_range = [6, 6+model.ft_length] if ft_infer else None

    amp_stat = [[0 for i in range((model.ft_length - 2) // 4 + 1)] for j in range(nc)]  # [abcd1~n, num]
    for idimg,(path, im, im0s, vid_cap, s) in enumerate(dataset):
        # print(path)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        if pts:
            pred, _ = detect_with_cms(model, False, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir,threshs=threshs, dt=dt)
        else:
            pred = detect_cms(model, False, im,augment, conf_thres, iou_thres, dt=dt, 
                              cms_s=model.cms_s, cms_num=cms_num,
                              ft_infer=ft_infer)

        '''
        pred = model(im, augment=augment, visualize=visualize, val=True)[0]
        #pred[2=Detect+Detect2][nl][b,a,H,W,Detect(5+c) or Detect2(5=1+2(dir)+2(ab))]
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold,mask_dir=mask_dir, threshs=threshs)
        #pred[b][nt,10=2*4+1(conf)+1(cls)]
        dt[2] += time_sync() - t3
        '''

        colors = [
        (54, 67, 244),
        # (99, 30, 233),
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

        # Process predictions
        if pts:
            for i, det in enumerate(pred):  # per image batch循环
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem)
                if not os.path.exists(str(save_dir / 'labels')):
                    os.makedirs(str(save_dir / 'labels'))
                s += '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :8] = scale_coords_poly(im.shape[2:], det[:, :8], im0.shape).round()

                    # Print results
                    for c in det[:, 9].unique():
                        n = (det[:, 9] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :10])):
                        line = [cls, *xyxy, conf] # label format
                        if det.shape[-1] != 10:
                            line.extend(det[j, 10:])
                        line = tuple(line)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        xyxy = [x.cpu() for x in xyxy]
                        label = '{} {:.2f}'.format(names[int(cls)], conf)
                        if not plot_label:
                            label = None
                        plot_one_rot_box(np.array(xyxy), im0, color=colors[int(cls)%len(colors)], label=label, dir_line=dir_line, line_thickness=line_thickness)
                    cv2.imencode(p.suffix, im0)[1].tofile(save_path)
                else:
                    with open(txt_path+'.txt', 'w') as f:
                        pass
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1] * 1E3:.1f}ms")
        else:
            for i, det in enumerate(pred):
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem)
                if not os.path.exists(str(save_dir / 'labels')):
                    os.makedirs(str(save_dir / 'labels'))
                s += '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # xyxy ->
                    det[:, :4] = scale_coords(im[i].shape[1:], det[:, :4], im0.shape).round()  # native-space pred
                    if ft_infer:
                        det[:, ft_range[0]:ft_range[1]] = scale_coords_ft(im[i].shape[1:], det[:, ft_range[0]:ft_range[1]], im0.shape).round()  # native-space pred

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # save_flag = False
                    if plot_inside:
                        img_inside = im0.copy()
                    for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                        line = [cls, *xyxy, conf] # label format
                        # if int(cls) in [20, 22, 23]:
                            # save_flag = True
                        line = tuple(line)
                        cms_sub = None
                        cms_value = None
                        start_n = ft_range[1] if ft_infer else 6
                        if cms_num != 0:
                            fxs_n = 0
                            if model.cms_s:
                                cms_sub = '{} {:.3f}'.format(int(det[-j, 7]), det[-j, 6])
                                start_n += 2
                                fxs_n = 1
                            if det.shape[-1] > start_n:
                                cms_value = det[j, start_n:].tolist()
                                cms_value = [np.interp(cms, fxs_func[fxs_n + k]['x'], fxs_func[fxs_n + k]['y']) for k, cms in enumerate(cms_value)]
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        xyxy = [x.cpu() for x in xyxy]
                        label = '{} {:.2f}'.format(names[int(cls)], conf)
                        ft_label = det[j, ft_range[0]:ft_range[1]].cpu().numpy() if ft_infer else None
                        # if not plot_label:
                        #     label = None
                        #     continue
                        # plot_one_box(np.array(xyxy), im0, color=colors[int(cls)%len(colors)], label=label, line_thickness=line_thickness)
                        if not ft_infer:
                            plot_one_box_with_cms(np.array(xyxy), im0, 
                                                color=colors[int(cls)%len(colors)], 
                                                label=label, line_thickness=line_thickness, 
                                                cms_sub=cms_sub, cms_value=cms_value,
                                                ft_label=ft_label)
                        else:
                            plot_one_box(np.array(xyxy), im0, 
                                         color=colors[int(cls)%len(colors)], 
                                         label=label, line_thickness=line_thickness,
                                         ft_label=ft_label,
                                         amp_stat=amp_stat[int(cls)])
                    # if save_flag:
                    cv2.imencode(p.suffix, im0)[1].tofile(save_path)
                    # cv2.imwrite(save_path, im0)
                    #
                    if plot_inside:
                        ft_label = det[:, ft_range[0]:ft_range[1]]
                        height, width, _ = im0.shape
                        #img_inside = np.zeros((height, width, 3), dtype=np.uint8)
                        cls = det[:, 5]

                        img_inside = render_fourier_curve2(img_inside,ft_label,cls,200)
                        # 显示图像
                        # cv2.imshow('Rendered Image', img_inside)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        # 获取文件的目录名和基文件名
                        dirname, basename = os.path.split(save_path)
                        # 获取文件名和扩展名
                        name, ext = os.path.splitext(basename)
                        # 创建新的文件名
                        new_basename = f"{name}_inside{ext}"
                        # 创建新的保存路径
                        new_save_path = os.path.join(dirname, new_basename)
                        cv2.imencode(p.suffix, img_inside)[1].tofile(new_save_path)
                        plot_inside-=1
                else:
                    with open(txt_path+'.txt', 'w') as f:
                        pass
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1] * 1E3:.1f}ms")

    total_time = dt[1]
    n_img = len(dataset)
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    print(f"total inference time: {total_time}, \
        every inference time:{total_time / n_img}, fps:{1/(total_time / n_img)}\nSaveDir:\n{save_dir}")
    for i in range(nc):
        print(f'{i:02d}: ',' '.join([f"{amp/(amp_stat[i][-1]+1e-5):.2f}" for amp in amp_stat[i][:-1]]))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--thresh_scale', type=float, default=0.2, help='confidence scale')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')
    parser.add_argument('--plot_label',action='store_true', help='plot labels')
    parser.add_argument('--fold',type=int, default=1, help='fold angle')
    parser.add_argument('--plot_inside',type=int, default=0, help='plot_inside')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #opt.weights = 'weights/yolov5l.pt'
    #opt.source = '/home/liu/work/datasets/HRSC/val/images'

    '''#hrsc2016
    opt.data = 'data/hrsc2016.yaml'
    opt.weights = 'runs/train/exp1/weights/hrsc-best/best.pt'
    #opt.weights = 'runs/train/exp_hrsc20162/weights/best.pt'
    #opt.source = r'/home/liu/data/home/liu/workspace/darknet/datas/GuGe/val/images'
    #opt.source = r'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val/images'
    opt.source = r'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/HRSC2016/test/images'
    opt.source = r'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/HRSC2016/img'
    '''
    #dota1.5
    # opt.data = 'data/dota.yaml'
    # opt.weights = 'runs/DOTA1.5/train/exp3/weights/best.pt'
    # opt.source = r'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/dota1.5/val500/images_500'
    #Guge
    #opt.data = 'data/Guge.yaml'#切记里面的数据集路径存在可用！
    #opt.weights = 'runs/train/exp_Guge27/weights/best.pt'
    #opt.source = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/GuGe/val/images'
    #voc_segment
    #opt.data = 'data/voc_segment.yaml'
    #opt.weights = 'runs/train/exp_voc_segment2/weights/best.pt'
    #opt.source = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val/images'
    #coco2017
    opt.data = 'data/coco_ft.yaml'
    opt.weights = 'runs/train/l-66.34-30.91/weights/best_mAP50.pt'
    opt.plot_inside = 0
    #4090


    opt.source = r'E:/datas/coco2017/val/images'

    # opt.weights = './runs/train/exp_dota_ft/weights/best.pt'
    # opt.data = './data/dota_ft.yaml'
    # opt.source = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/DOTA1.5/val/patches/images'

    # # Guge
    # opt.weights = '/home/user/workspace/yolov5/yolov5-ft/yolov5-ft-iou_thresh-conf_thresh/runs/train/exp_Guge_ft9/weights/best_mAP50.pt'
    # opt.data = './data/Guge_ft.yaml'
    # opt.source = '/home/user/datas/Guge/patches/val/images'

    opt.name += f'_{Path(opt.data).stem}'

    opt.conf_thres = 0.1
    opt.iou_thres = 0.15
    # opt.imgsz = [768, 768]
    # opt.ab_thres = 7.0
    opt.fold = 2
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    #
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    if not os.path.exists(opt.source):
        source0 = opt.source
        opt.source = data_dict.get('val','')
        print(f'\033[91m{source0}not exists. change to {opt.source}.\033[0m')
    opt.nc = int(data_dict['nc'])  # number of classes
    opt.mask_dir = data_dict.get('mask_dir', None)
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
