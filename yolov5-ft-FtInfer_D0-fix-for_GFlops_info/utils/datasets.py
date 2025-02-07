# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import augment_hsv, copy_paste, letterbox, mixup, random_perspective, random_perspective_new
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn,pts2dir,
                           ft2ftnorm, ftnorm2ft)
from utils.torch_utils import torch_distributed_zero_first
from DOTA_devkit.image_split import CutImages

from general.MyString import replace_path

from torch.utils.data import Subset # 导入 Subset 类

import sys

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
# NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
NUM_THREADS = 1

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    paths = [p for p in paths if p is not None]
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=True,
                      pts=True, cms=False, fxs=None, ft_coef=0,save_dir='',debug_samples=0,sample_count=0):
    if(save_dir==''):
        debug_samples = 0
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      pts=pts, cms=cms,
                                      fxs=fxs, ft_coef=ft_coef,
                                      save_dir=save_dir,
                                      debug_samples=debug_samples)
    
    if(sample_count>0):
        if(sample_count < len(dataset)):
            dataset = Subset(dataset, torch.randperm(sample_count))
        else:
            print(f'\033[91m{sample_count} vs {len(dataset)}.\033[0m')

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle and sampler is None,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset



def create_bigimg_dataloader(image, imgsz, batch_size, sub_size=512, over_lap=640):

    dataset = BigImageDataset(image, imgsz, sub_size, over_lap)
    batch_size = min(batch_size, len(dataset))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            # img0 = cv2.imread(path)  # BGR
            img0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files



class LoadBigImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, sub_size=512, over_lap=100, load_label=False):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni  # number of files
        self.mode = 'image'
        self.auto = auto
        self.sub_size = sub_size
        self.over_lap = over_lap
        self.load_label = load_label
        assert self.nf > 0, f'No images or videos found in {p}. ' 


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        cut = CutImages(sub_size=self.sub_size, over_lap=self.over_lap)
        img0, cut_imgs = cut.cut_images(path) # BGR
        assert img0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '
        convert_imgs = []
        for sub_item in cut_imgs:
            # Padded resize
            sub_img = sub_item['patch']
            img = letterbox(sub_img, self.img_size, stride=self.stride, auto=self.auto)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            convert_imgs.append(img)


        if self.load_label:
            pts_path = self.pts_files[self.count - 1]
            with open(pts_path) as f:
                p = [x.split() for x in f.read().strip().splitlines() if len(x)]
                p = np.array(p, dtype=np.float32)
                p = np.clip(p, 0, 1)
            return path, cut_imgs, convert_imgs, img0, p

        return path, cut_imgs, convert_imgs, img0,  s


    def __len__(self):
        return self.nf  # number of files


class BigImageDataset(Dataset):

    def __init__(self, image, img_size=640, sub_size=512, over_lap=100, stride=32):
        cut = CutImages(sub_size=sub_size, over_lap=over_lap)
        _, cut_imgs = cut.cut_images(image) # BGR
        convert_imgs = []
        xy = []
        cuts = []
        for sub_item in cut_imgs:
            sub_img = sub_item['patch']
            cuts.append(sub_img)
            xy.append(np.array(sub_item['xy']))
            img = letterbox(sub_img, img_size, stride=stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            convert_imgs.append(img)
        self.cuts = cuts
        self.xy = xy
        self.convert_imgs = convert_imgs
      
 
    def __getitem__(self, index):   
        cut_img = self.cuts[index]
        conv_img = self.convert_imgs[index]
        xy = self.xy[index]

        return xy,cut_img, conv_img
 
    def __len__(self): 
        return len(self.cuts)





class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def img2pts_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.pts' for x in img_paths]

def img2suf_paths(img_paths, suffix):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + suffix for x in img_paths]

def img2cms_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.cms' for x in img_paths]

def img2ft_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.ft' for x in img_paths]

def filt_labels_H(labels, least_pixel_size, least_area=40):
    #labels = self.labels[index].copy()#[nobj,1(cls)+4(xyxy))]
    xyxy = labels[:,1:5]#xyxy[nobj,4]
    wh = xyxy[:,2:4]-xyxy[:,0:2]#wh[nobj,2]
    Lab = torch.norm(wh, p=2, dim=1)
    mask = (Lab > least_pixel_size) & (wh[:,0]>0) & (wh[:,1]>0) & (wh[:,0]*wh[:,1]>=least_area)
    labels = labels[mask]
    assert wh[mask].shape[0]==0 or torch.all(wh[mask] > 0), "所有元素必须大于0"
    return labels

class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', pts=False, cms=False, fxs=None,
                 ft_coef=0,save_dir='',debug_samples=0):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        # self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic = False  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2] if isinstance(img_size,int) else [-img_size[0] // 2, -img_size[1] // 2]
        self.stride = stride
        self.path = path
        self.ft_coef = ft_coef
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        #verify labels path by small files..
        err_count = 0
        for i,image_name in enumerate(self.img_files):
            if (not os.path.exists(image_name)):
                print(f'\033[91mimage_name:{image_name} not exist.注意路径大小写.\033[0m')
                if(err_count>3):
                    sys.exit()
                err_count+=1
            label_name = replace_path(image_name,'labels','.txt')
            label_path, name = os.path.split(label_name)
            if (not os.path.exists(label_path)):
                print(f'\033[91mlabel_path:{label_path} not exist{name}.注意路径大小写.\033[0m')
                if(err_count>3):
                    sys.exit()
                err_count+=1
            if(i>20):
                break
        # add
        self.pts_files = img2pts_paths(self.img_files) if pts else [None] * len(self.label_files)
        self.cms_files = img2cms_paths(self.img_files) if cms else [None] * len(self.label_files)
        self.ft_files = img2ft_paths(self.img_files) if ft_coef > 0 else [None] * len(self.label_files)
        self.cms_num = len(cms) if cms else 0
        self.cms_sub = [i for i, c_ in enumerate(cms) if c_['type'] == 'int'] if cms else []
        self.label_length = (5 + self.cms_num * 2) if not pts else (13 + self.cms_num * 2)
        self.pts = pts
        self.cms = cms
        if ft_coef > 0:
            self.label_length += (ft_coef * 4 + 2)
            self.ft_length = ft_coef * 4 + 2
        else:
            self.ft_length = 0
        self.end_i = 5 if not pts else 13
        self.reorder = list(range(self.cms_num)) if len(self.cms_sub) == 0 else self.cms_sub + [i for i in range(self.cms_num) if i not in self.cms_sub]
        #
        label_path = p if p.is_file() else Path(self.label_files[0]).parent
        cache_path = Path(str(label_path)+'_ft').with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files + self.pts_files + self.cms_files + self.ft_files)  # same hash
            assert cache['ft_coef'] == self.ft_coef
            assert all([cache['cms_sub'][i] == self.cms_sub[i] for i in range(len(self.cms_sub))]) # same cms
        except:
            cache, exists = self.cache_labels(cache_path, prefix, fxs), False  # cache
        #cache[n_images]形成的字典，里面每项3个内容：cache[n_images][0]是该图像标签labels[nt,5+8]  [1]是[640,640]该图像大小  [2]=[]
            
        self.D0 = self.get_D0(cache,ft_coef) #self.D0[4*ft_coef+2]

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs', 'cms_sub', 'ft_coef')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())#labels[:,13=1(cls)+4(xywh)+4*2(pts)]
        self.labels = list(labels)#labels是tuple转list
        self.shapes = np.array(shapes, dtype=np.float64)#shapes是tuple转array
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
        if(hyp!=None):
            self.least_pixel_size = hyp.get('least_pixel_size', 2) #at least 2 pixels for every object
            self.least_area =  hyp.get('least_area', 4) #at least 4 pixels^2 for every object
        else:
            self.least_pixel_size = 2
            self.least_area = 4

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            if isinstance(img_size,int):
                self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int64) * stride
            else:
                one_shape = np.ceil(np.array(img_size) / stride).astype(np.int64) * stride
                self.batch_shapes = np.repeat([one_shape],nb,axis=0) #([one_shape].repeat(nb,1)
            #self.batch_shapes[nb,2]

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()
        self.save_dir = save_dir
        self.debug_samples = debug_samples
    
    def get_D0(self,cache,k):
        # 假设 cache 是已定义的字典
        # 初始化最大值数组，长度为 4k+2
        max_values = np.zeros(4*k+2)

        # 遍历字典
        for key, val in cache.items():
            if isinstance(val,list) and len(val)==3:
                # val[0] 是我们需要的np数组
                arr = val[0]
                if isinstance(arr,np.ndarray):
                    if arr.shape[0]>0:
                        # 选择最后 4k+2 列
                        sub_array = arr[:, -4*k-2:]
                        # 计算每列的最大绝对值并更新最大值数组
                        max_values = np.maximum(max_values, np.max(np.abs(sub_array), axis=0))
        return max_values

    def cache_labels(self, path=Path('./labels_ft.cache'), prefix='', fxs=None):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        reorder = self.reorder
        ft_coef = self.ft_coef
        all_labels_masks = np.zeros([0, 2 * self.cms_num], dtype=np.float32)
        with Pool(NUM_THREADS) as pool:
            cms_error = []
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.img_files, self.label_files, self.pts_files, self.cms_files, self.ft_files, repeat(self.cms_num), repeat(reorder), repeat(ft_coef), repeat(prefix))),
                desc=desc, total=len(self.img_files))
            #verify_image_label是函数名，参数是->self.img_files, self.label_files, self.pts_files, repeat(prefix))
            #返回值是im_file, l, ....后面都不是重要的，重要的是前面2个->im_file[:]和l[:,1+4+2*4=13]
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg, cms_flag in pbar: 
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if cms_flag:
                    cms_error.append(im_file)
                if l is not None:
                    x[im_file] = [l, shape, segments]#x是一个字典，将文件名im_file作为key，检索出[l[:,1+4+2*4=13], shape, segments]
                    if self.cms_num > 0:
                        all_labels_masks = np.vstack([all_labels_masks, l[:, -2*self.cms_num:]])
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        if self.cms_num > 0:
            # 直方图归一
            assert len(fxs) == self.cms_num, 'ERROR: Did not load fxs_func.npy'
            for k, v in x.items():
                if v[0].shape[0] > 0:
                    for i in range(self.cms_num):
                        j = reorder[i]
                        if fxs[j] is not None:
                            v[0][:, -2*self.cms_num + i] = np.interp(v[0][:, -2*self.cms_num + i], fxs[j]['y'], fxs[j]['x'])  # 根据属性值映射关系，y插值到x
        x['cms_sub'] = self.cms_sub
        x['ft_coef'] = self.ft_coef
        #
        x['hash'] = get_hash(self.label_files + self.img_files + self.pts_files + self.cms_files + self.ft_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        #x字典末尾追加一些全局属性'hash','results',....
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
            with open(path.parent / 'cms_error.txt', 'w') as fp:
                fp.write('\n'.join(cms_error))
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x #x字典返回就是Cache

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):#这里往往被collate_fn(batch)调用batch次
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:#对单张图像进行数据增广
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            
            labels = self.labels[index].copy()#[nobj,13=1(cls)+4(box)+4*2]

            # show origin label
            # if labels.size:  # normalized xywh to pixel xyxy format转换为[左上角,右下角,绝对坐标x1,绝对坐标y1,...,绝对坐标x4,绝对坐标y4]
            #     labels_ori = labels.copy()
            #     temp1 = labels_ori[:, self.end_i: self.end_i+ 2 + 4 * self.ft_coef].copy()
            #     path_show = self.img_files[index]
            #     labels_ori[:, 1:self.end_i] = xywhn2xyxy(labels[:, 1:self.end_i],w,h,0,0)
            #     if self.ft_coef > 0:
            #        labels_ori[:, self.end_i: self.end_i+ 2 + 4 * self.ft_coef] = ftnorm2ft(labels[:, self.end_i: self.end_i+ 2 + 4 * self.ft_coef],w,h,0,0)
            #     show_ft(img, labels_ori[:, 1:self.end_i], labels_ori[:, self.end_i: self.end_i+2 + 4 * self.ft_coef], '_origin')

           
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)#img[H,W,C=3]
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            if labels.size:  # normalized xywh to pixel xyxy format转换为[左上角,右下角,绝对坐标x1,绝对坐标y1,...,绝对坐标x4,绝对坐标y4]
                labels[:, 1:self.end_i] = xywhn2xyxy(labels[:, 1:self.end_i], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                if self.ft_coef > 0:
                   labels[:, self.end_i: self.end_i + self.ft_length] = ftnorm2ft(labels[:, self.end_i: self.end_i + self.ft_length], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
           # [0,1,2,3]是box单独处理，[4,..]是pts单独处理

            #before aug show

            if self.augment and random.random() < hyp.get('augment', 0.6):#hyp['augment']是数据增广概率
                # path_show = self.img_files[index]
                #ft_image = show_ft(img, labels[:, 1:self.end_i], labels[:, self.end_i: self.end_i+2 + 4 * self.ft_coef], '0')
                img, labels = random_perspective_new(img, labels,
                                                degrees=hyp.get('degrees',0),
                                                translate=hyp.get('translate',0),
                                                scale=hyp.get('scale',1.0),
                                                shear=hyp.get('shear',0),
                                                perspective=hyp.get('perspective',0),
                                                flip = [hyp['fliplr'],hyp['flipud']],
                                                ft_length=self.ft_length,
                                                pts=self.pts)
                # HSV color-space
                augment_hsv(img, hgain=hyp.get('hsv_h',0.015), sgain=hyp.get('hsv_s',0.7), vgain=hyp.get('hsv_v',0.4))
                if 0:
                    cv2.imwrite('/home/liu/data/home/liu/workspace/darknet/datas/GuGe/image_warped'+str(random.random())+'.jpg',img)
        
        if self.labels[0] is not None:
            labels = filt_labels_H(torch.from_numpy(labels),self.least_pixel_size,self.least_area).numpy()
            
        # after aug show
        if self.debug_samples > 0:
            ft_image = show_ft(img, labels[:, 1:self.end_i], labels[:, self.end_i: self.end_i+self.ft_length] if self.ft_length > 0 else [])
            name = os.path.splitext(os.path.basename(self.img_files[index]))[0]
            debug_samples_path = str(self.save_dir) + '/debug_samples/'
            if(not os.path.exists(debug_samples_path)):
                os.makedirs(debug_samples_path, exist_ok=True)
            cv2.imencode('.jpg', ft_image)[1].tofile(f'{debug_samples_path}/aug[{index}]_{name}.jpg')
            self.debug_samples-=1

        nl = len(labels)  # number of labels
        if nl:
            # labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=False, eps=1E-3)
            labels[:, 1:self.end_i] = xyxy2xywhn(labels[:, 1:self.end_i], w=img.shape[1], h=img.shape[0], clip=False, eps=1E-3)
            if self.ft_coef > 0:
                labels[:, self.end_i: self.end_i + self.ft_length] = ft2ftnorm(labels[:, self.end_i: self.end_i + self.ft_length], w=img.shape[1], h=img.shape[0])

        labels_out = torch.zeros((nl, self.label_length+1))#labels_out[1(batch)+1(cls)+4(box)+4*2(pts)]
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)#labels_out[1(batch)+1(cls)+4(box)+4*2(pts)]
            #注意第0列在collate_fn里面填入batch
        #labels_out[nt,1(b)+13=(1(cls)+4(box)+4(pts)*2)]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes
        #与collate_fn里面的img, label, path, shapes = zip(*batch)对应

    @staticmethod
    def collate_fn(batch):#这里之前会调用batchsize次LoadImagesAndLabels::__getitem__(self, index)形成batch集合
        img, label, path, shapes = zip(*batch)
        #img[batch][C,H,W], label[batch][nt,1(b)+13=(1(cls)+4(box)+4(pts)*2)], path[batch], shapes(h0, w0), ((h / h0, w / w0), pad)
        #与__getitem__里面的return torch.from_numpy(img), labels_out, self.img_files[index], shapes对应
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
            #LoadImagesAndLabels::__getitem__里执行了labels_out[:, 1:] = torch.from_numpy(labels)
            #labels_out[1(batch)+1(cls)+4(box)+4*2(pts)]
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            # im = cv2.imread(path)  # BGR
            im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)    # window opencv didnot support chinese
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        if isinstance(self.img_size,int):#如果是单一长度，则按不变的长宽比缩放
            r = self.img_size / max(h0, w0)  # ratio
        else:
            # origin
            # im = cv2.resize(im, (self.img_size[1], self.img_size[0]),interpolation=cv2.INTER_AREA)
            # new
            r = min(self.img_size[0] / h0, self.img_size[1] / w0)
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = [self.img_size,self.img_size] if(isinstance(self.img_size,int)) else self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s[i] + x)) for i,x in enumerate(self.mosaic_border)]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s[0] * 2, s[1] * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[1] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[0] * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[1] * 2), min(s[0] * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:self.end_i] = xywhn2xyxy(labels[:, 1:self.end_i], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            if self.ft_coef > 0:
                labels[:, self.end_i: self.end_i + self.ft_length] = ftnorm2ft(labels[:, self.end_i: self.end_i + self.ft_length], w, h, padw, padh)

        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:5], *segments4):
        #np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        np.clip(x[0::2], 0, 2 * s[1], out=x[0::2])  # clip when using random_perspective()
        np.clip(x[1::2], 0, 2 * s[0], out=x[1::2])  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp.get('copy_paste',0))
    img4, labels4 = random_perspective_new(img4, labels4, segments4,
                                       degrees=self.hyp.get('degrees',0),
                                       translate=self.hyp.get('translate',0),
                                       scale=self.hyp.get('scale',1.0),
                                       shear=self.hyp.get('shear',0),
                                       perspective=self.hyp.get('perspective',0),
                                       flip = [self.hyp['fliplr'],self.hyp['flipud']],
                                       border=self.mosaic_border,
                                       ft_length=self.ft_length,
                                       pts=self.pts)  # border to remove
    # HSV color-space
    augment_hsv(img, hgain=self.hyp.get('hsv_h',0.015), sgain=self.hyp.get('hsv_s',0.7), vgain=self.hyp.get('hsv_v',0.4))

    return img4, labels4


def load_mosaic9(self, index):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:self.end_i] = xywhn2xyxy(labels[:, 1:self.end_i], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:self.end_i], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp.get('degrees',0),
                                       translate=self.hyp.get('translate',0),
                                       scale=self.hyp.get('scale',1.0),
                                       shear=self.hyp.get('shear',0),
                                       perspective=self.hyp.get('perspective',0),
                                       flip = [self.hyp['fliplr'],self.hyp['flipud']],
                                       border=self.mosaic_border)  # border to remove
    # HSV color-space
    augment_hsv(img, hgain=self.hyp.get('hsv_h',0.015), sgain=self.hyp.get('hsv_s',0.7), vgain=self.hyp.get('hsv_v',0.4))

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


# def verify_image_label(args):
#     # Verify one image-label pair
#     im_file, lb_file, prefix = args
#     nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
#     try:
#         # verify images
#         im = Image.open(im_file)
#         im.verify()  # PIL verify
#         shape = exif_size(im)  # image size
#         assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
#         assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
#         if im.format.lower() in ('jpg', 'jpeg'):
#             with open(im_file, 'rb') as f:
#                 f.seek(-2, 2)
#                 if f.read() != b'\xff\xd9':  # corrupt JPEG
#                     ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
#                     msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'
#
#         # verify labels
#         if os.path.isfile(lb_file):
#             nf = 1  # label found
#             with open(lb_file) as f:
#                 l = [x.split() for x in f.read().strip().splitlines() if len(x)]
#                 if any([len(x) > 8 for x in l]):  # is segment
#                     classes = np.array([x[0] for x in l], dtype=np.float32)
#                     segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
#                     l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
#                 l = np.array(l, dtype=np.float32)
#             nl = len(l)
#             if nl:
#                 assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
#                 assert (l >= 0).all(), f'negative label values {l[l < 0]}'
#                 assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
#                 l = np.unique(l, axis=0)  # remove duplicate rows
#                 if len(l) < nl:
#                     segments = np.unique(segments, axis=0)
#                     msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
#             else:
#                 ne = 1  # label empty
#                 l = np.zeros((0, 5), dtype=np.float32)
#         else:
#             nm = 1  # label missing
#             l = np.zeros((0, 5), dtype=np.float32)
#         return im_file, l, shape, segments, nm, nf, ne, nc, msg
#     except Exception as e:
#         nc = 1
#         msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
#         return [None, None, None, None, nm, nf, ne, nc, msg]

def verify_image_label(args):#cache_labels函数通过Pool对象回调调用verify_image_label函数
    # Verify one image-label pair
    im_file, lb_file, pts_file, cms_file, ft_file, cms_num, cms_reorder, ft_coef, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    cms_flag = False
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        num_ = 5
        if pts_file is not None:
            num_ += 8
        num_ += 2 * cms_num
        if ft_coef > 0:
            num_ += (2 + 4 * ft_coef)
        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            #读取txt标签文件存储到 l[nt,5=1+4]
            with open(lb_file) as f: # .txt
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]#读取yolo里一行一个目标的标签数据[nt][c x y w h]
                read_l = len(l)
                l = np.array(l, dtype=np.float32)#l[nt][5=1+4]->l[nt,5=1+4]
                assert(l.shape[0]==read_l)
                assert(read_l==0 or l.shape[1]==5)
                
            #add... 读取新增pts文件中的数据
            if pts_file is not None:
                if os.path.exists(pts_file):
                    with open(pts_file) as f: #.pts
                        p = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        #p = [['0','0','0','0','0','0','0','0'] for x in p if x==['-']]]
                else:
                    p = [['-']] * l.shape[0]
                for i in range(len(p)):
                    if p[i]==['-']:#无数据，针对非旋转目标的情况,填入水平框四个角点坐标
                        xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                        #p[i] = [str(round(xc-w/2,6)),'0','0','0','0','0','0','0']
                        p[i] = [str(round(xc-w/2,6)),str(round(yc-h/2,6)),
                                str(round(xc+w/2,6)),str(round(yc-h/2,6)),
                                str(round(xc+w/2,6)),str(round(yc+h/2,6)),
                                str(round(xc-w/2,6)),str(round(yc+h/2,6))]
                        #由于后面p = np.array(p, dtype=np.float32)要求字符串必须是数值的，'-'转数据矩阵会报错
                        #也就是水平框的四个顶点构建四边形
                p = np.array(p, dtype=np.float32)#list转np矩阵[nt,4*2=8]
                assert(p.shape[0]==l.shape[0])#pts的行数即目标数  和  lables里的行数目标数应该一致

            if cms_file is not None:
                if os.path.exists(cms_file):
                    with open(cms_file) as f:
                        c = [x.split(',') for x in f.read().strip().splitlines() if len(x)]
                    for i in range(len(c)):
                        if c[i]==['-']:
                            c[i] = [''] * cms_num
                    tl = [len(t) for t in c]
                    if max(tl) != cms_num or min(tl) != cms_num:
                        cms_flag = True
                        for i, j in enumerate(tl):
                            if j != cms_num:
                                c[i] = [''] * cms_num    # 将有错误的cms, 用'-'替代
                else:
                    c = [[''] * cms_num] * l.shape[0]  # 没找到文件则是空, 用'-'代替
                cs = list(map(lambda x: [True if x_.strip() != '' else False for x_ in x], c))
                c = list(map(lambda x: [x_.strip() if x_.strip() != '' else '0' for x_ in x], c))
                c = np.array(c, dtype=np.float32)# cms label
                cs = np.array(cs, dtype=np.float32)# cms mask label
                c = c[:, cms_reorder]
                cs = cs[:, cms_reorder]

            if ft_file is not None:
                ft_flag = False # re new
                if os.path.exists(ft_file): # ft file
                    with open(ft_file, 'r') as f:
                        ft = [x.split()[1:] for x in f.read().strip().splitlines() if len(x)]
                        if len(ft) > 0:
                            if len(ft[0]) < 2 + 4 * ft_coef:
                                ft_flag = True  
                            else:
                                ft_flag = False
                        else:
                            ft_flag = True
                else:
                    ft_flag = True
                if ft_flag: 
                    ft = []                    
                    ft_flag2 = False    # 不能从pts读取,则取box的四个顶点
                    if pts_file is not None:
                        ft_p = p
                    elif os.path.exists(img2pts_paths([lb_file])[0]):
                        with open(img2pts_paths([lb_file])[0]) as f:
                            ft_p = [x.split() for x in f.read().strip().splitlines() if len(x)]
                            #p = [['0','0','0','0','0','0','0','0'] for x in p if x==['-']]]
                        for i in range(len(ft_p)):
                            if ft_p[i]==['-']:
                                xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                                #p[i] = [str(round(xc-w/2,6)),'0','0','0','0','0','0','0']
                                ft_p[i] = [str(round(xc-w/2,6)),str(round(yc-h/2,6)),
                                        str(round(xc+w/2,6)),str(round(yc-h/2,6)),
                                        str(round(xc+w/2,6)),str(round(yc+h/2,6)),
                                        str(round(xc-w/2,6)),str(round(yc+h/2,6))]
                        ft_p = np.array(ft_p, dtype=np.float32)#list转np矩阵[nt,4*2=8]
                    else:
                        ft_flag2 = True
                    for i in range(l.shape[0]):
                        if not ft_flag2:
                            ft.append(compute_coefficients(ft_p[i],terms=ft_coef, interp=True))
                        else:
                            xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                            ft_p = [round(xc-w/2,6),round(yc-h/2,6),
                                    round(xc+w/2,6),round(yc-h/2,6),
                                    round(xc+w/2,6),round(yc+h/2,6),
                                    round(xc-w/2,6),round(yc+h/2,6)]
                            ft.append(compute_coefficients(ft_p,terms=ft_coef, interp=True))
                ft = np.array(ft, dtype=np.float32)#list转np矩阵[nt, 2 + 2 * ft_coef =8]
                if 2 + 4 * ft_coef < ft.shape[-1]:
                    ft = ft[:, :2 + 4 * ft_coef]

            nl = len(l)
            if nl:#l[1+4(xywh)]就是原始yolo标签txt数据的一行，一个目标的位置信息
                clss = l[:, 0].reshape(-1, 1)#[nt,1]，l[:, 0]的shape是l[:],reshape后变成l[nt,1]
                margin = 0
                boxs = np.clip(l[:, 1:], -margin, 1+margin)#[nt,4] ,np.clip的意思是设定上下限,目标框的坐标和wh都限制在0,1范围内
                l = np.concatenate((clss, boxs), axis=1)#l从list转换成np矩阵[nt,1+4=5]
                #add 把新增数据拼接到目标标签l[:,1+4]后面...
                length = 5
                if pts_file is not None:
                    # p = np.clip(p, 0, 1)#p[:,8]是新增的
                    l = np.concatenate((l, p), axis=1)#把pts数据p拼到原始yolo标签l后面:l[nt,5] cat p[nt,4*2=8]-->p[nt,5 + 4*2=8]
                    assert l.shape[1] == 13, f'labels require 5+8=13 columns, {l.shape[1]} columns detected'
                    length += 8
                if ft_file is not None:
                    l = np.concatenate((l, ft), axis=1)  # l[nt,box + [pts] + [ft]]
                if cms_file is not None:
                    l = np.concatenate((l, c, cs), axis=1)  # l[nt,box + [pts] + cms + cms mask]
                assert l.shape[1] == num_, f'labels require 5+8(pts)+2+4*ft_coef+ns+nv columns, {l.shape[1]} columns detected'
                assert (l[:, :5] >= 0).all(), f'negative label values {l[l < 0]}'
                # assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                nl2 = len(l)
                l = np.unique(l, axis=0)  # remove duplicate rows,  同一目标重复标注，合并
                if len(l) < nl2:
                    segments = np.unique(segments, axis=0)
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, num_), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, num_), dtype=np.float32)
        if cms_flag:
            if msg != '':
                msg = msg + ' and CMS labels Error'
            else:
                msg = f'{prefix}WARNING: {im_file}: CMS labels Error'
        return im_file, l, shape, segments, nm, nf, ne, nc, msg, cms_flag
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        msg = msg + ' and CMS labels Error' if cms_flag else msg
        return [im_file, None, None, None, nm, nf, ne, nc, msg, cms_flag]


def compute_coefficients(xy, terms=2, interp=False): # [0 ~ terms] 4  a0c0 a2~d2 6 2
    x = np.array(xy[0::2])
    y = np.array(xy[1::2])  # points 
    if interp:
        x = np.concatenate([x, [x[0]]], dtype=np.float32)
        y = np.concatenate([y, [y[0]]], dtype=np.float32)
        ori = np.linspace(0, 1, x.shape[0], endpoint=True)
        gap = np.linspace(0, 1, max(terms*2, x.shape[0] - 1), endpoint=False) #点的总数(x.shape[0]-1)应该>=2*terms(即阶数-1)
        x = np.interp(gap, ori, x)  # 0, x0; 0.25, x1; 0.5, x2; 0.75, x3; 1, x0;
        y = np.interp(gap, ori, y)
        N = x.shape[0]
    else:
        N = x.shape
    t = np.linspace(0, 2*np.pi, N, endpoint=False)  # t = t*2pi/n
    a0 = 1./N * sum(x)
    c0 = 1./N * sum(y)
    
    an, bn, cn, dn = [np.zeros(1 + terms) for i in range(4)]
    
    for k in range(1, (N // 2) + 1):    # 1,2,...,int(N/2)
        if k > terms:
            break
        an[k] = 2./N * sum(x * np.cos(k*t))
        bn[k] = 2./N * sum(x * np.sin(k*t))
        cn[k] = 2./N * sum(y * np.cos(k*t))
        dn[k] = 2./N * sum(y * np.sin(k*t))
    list_coef = [a0, c0]
    for k in range(1, an.shape[0]):
        list_coef.append(an[k])
        list_coef.append(bn[k])
        list_coef.append(cn[k])
        list_coef.append(dn[k])
    return list_coef    # a0, c0, a1, b1, c1, d1, ... ak, bn, ck, dk

def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, quality=75)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats


def cms_normalize_with_config(cms_path, path, save_path, scale_up=1., middle=True):
    cms_config = json.load(open(cms_path, 'r'))
    save_path = Path(save_path)
    cms_labels = {} # 记录CMS标签的字典, key为文件名
    mask_label = {} # 记录CMS标签中是否为空,和cms_labels字典一一对应
    cms_length = len(cms_config)  # 记录CMS标签应该有多少个属性值
    name = [c['name'] for c in cms_config]
    ex = [i for i, c in enumerate(cms_config) if c['type'] == 'int']
    # error_list = []
    cms_num = np.zeros([1, cms_length])
    fstr = []
    for i in range(cms_length):
        if i in ex:
            fstr.append('{:d}')
        else:
            fstr.append('{:.6f}')
    fstr = ','.join(fstr) + '\n'
    assert len(name) == cms_length, f'name you offerd did not match cms length. Name:{len(name)}, cms_length: {cms_length}'

    # 读取所有CMS标签,并且获取cms_length
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise Exception(f'{p} does not exist')
        all_path = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        all_path = img2cms_paths(all_path)
        assert all_path, f'No images found'
    except Exception as e:
        raise Exception(f'Error loading data from {path}')
    for f in all_path:
        fp = Path(f)
        if fp.exists():
            with open(fp, 'r') as fr:
                txt = [x.split(',') for x in fr.read().splitlines() if x.strip != '']
                tl = [len(t) for t in txt]
                if (max(tl) != cms_length) or (min(tl) != cms_length):
                    for i, j in enumerate(tl):
                        if txt[i][0] != '-':
                            txt[i] = txt[i] + [''] * (cms_length - j) if j < cms_length else txt[i][:cms_length]
                cms_labels[fp.stem] = txt

    # 用于记录CMS中各个值类属性值的最大值、最小值
    max_, min_ = np.zeros([1, cms_length], dtype=np.float64), np.zeros([1, cms_length], dtype=np.float64)

    # 根据读入的标签进行处理，转换成numpy arry同时记录对应的mask，并记录CMS中各个值类属性值的最大值、最小值
    for k, v in cms_labels.items():
        for i, v_ in enumerate(v):
            if v_ == ['-']:
                v[i] = [''] * cms_length
        mask = list(map(lambda x: [1 if x_.strip() != '' else 0 for x_ in x], v))   # 1 exist, 0 not exist
        v = list(map(lambda x: [x_.strip() if x_.strip() != '' else '0' for x_ in x], v))
        cms_labels[k] = np.array(v, dtype=np.float64)
        if cms_labels[k].shape[0] > 0:
            max_v = cms_labels[k].max(0).reshape(1, -1)
            min_v = cms_labels[k].min(0).reshape(1, -1)
            max_ = np.vstack([max_, max_v]).max(0).reshape(1, -1)
            min_ = np.vstack([min_, min_v]).min(0).reshape(1, -1)
        mask_label[k] = np.array(mask, dtype=bool)
        cms_num += mask_label[k].sum(0).reshape(1, -1)

    # 扩大范围， scale_up = 1时无效，可用于避免属性值中最大值完全对应1，最小值完全对应0
    max_n = max_ * scale_up 
    min_n = min_ * np.where(min_ >= 0, (2 - scale_up), scale_up)

    # ex中无效
    for i in ex:
        max_n[i] = max_[i]
        min_n[i] = min_[i]

    # 取出所有的属性值进行计算
    all_labels = np.zeros([0, cms_length], dtype=np.float64)
    all_mask = np.zeros([0, cms_length], dtype=bool)
    for k, v in cms_labels.items():
        all_labels = np.vstack([all_labels, v])
        all_mask = np.vstack([all_mask, mask_label[k]])
    
    fxs = [None] * (cms_length + 1)   # 用于记录各个属性值对应的 f(x)，应该包含x（0~1），y（属性值）
    for k in range(cms_length):
        if k in ex:
            continue            # 不参与归一的属性值跳过
        num = int(cms_num[0, k])    # 有效属性k的个数
        tmp = all_labels[:, k][all_mask[:, k]]  # 提取 有效的属性k
        assert num > 0, 'Error 0'
        assert tmp.shape[0] == num, 'Error 1'
        y = np.hstack([min_n[0, k], tmp, max_n[0, k]]) if scale_up != 1 else tmp
        y = y[np.argsort(y)]    # 将属性值从小到大排序
        tmp_x = np.where(np.hstack([1, np.diff(y)]) != 0)[0]   # 记录属性值不相同的首个index
        tmp = np.unique(y)
        num_x = np.r_[- tmp_x[:-1] + tmp_x[1:], num - tmp_x[-1]]
        y = y[tmp_x]            
        assert y.shape[0] == tmp.shape[0], 'Error 2'
        if middle:
            # tmp3 = tmp_x + (num_x - 1) / 2
            x = (tmp_x + (num_x - 1) / 2) / (num - 1)  # 有多个相同值取中间
            x = x.reshape(-1)
        else:
            x = np.linspace(start=0, stop=1, num=(num + 2) if scale_up != 1 else num, endpoint=True, dtype=np.float64)  # 不做处理，根据tmp_x直接提取
            x = x[tmp_x]
        # x = np.linspace(start=0, stop=1, num=(num + 2) if scale_up != 1 else num, endpoint=True, dtype=np.float64)
        fs = {'x': x, 'y': y}
        fxs[k] = fs
    fxs[-1] = get_hash(all_path)
    np.save(save_path / 'fxs_func.npy', fxs)


def cms_check_latest(path):
    all_path = []
    for pa in path:
        p = Path(pa.replace('images', 'labels'))
        fs = p.rglob('*.cms')
        for f in fs:
            all_path.append(str(f))
    return get_hash(all_path)

def calc_fxs(all_labels, all_mask, cms_length, ex, middle=True):
    fxs = [None] * cms_length   # 用于记录各个属性值对应的 f(x)，应该包含x（0~1），y（属性值）
    all_mask = all_mask.astype(bool)
    for k in range(cms_length):
        if k in ex:
            continue            # 不参与归一的属性值跳过
        tmp = all_labels[:, k][all_mask[:, k]]  # 提取 有效的属性k
        num = tmp.shape[0]    # 有效属性k的个数
        assert num > 0, 'Error 0'
        y = tmp
        y = y[np.argsort(y)]    # 将属性值从小到大排序
        tmp_x = np.where(np.hstack([1, np.diff(y)]) != 0)[0]   # 记录属性值不相同的首个index
        tmp = np.unique(y)
        num_x = np.r_[- tmp_x[:-1] + tmp_x[1:], num - tmp_x[-1]]
        y = y[tmp_x]            
        assert y.shape[0] == tmp.shape[0], 'Error 2'
        if middle:
            # tmp3 = tmp_x + (num_x - 1) / 2
            x = (tmp_x + (num_x - 1) / 2) / (num - 1)  # 有多个相同值取中间
            x = x.reshape(-1)
        else:
            x = np.linspace(start=0, stop=1, num=num, endpoint=True, dtype=np.float64)  # 不做处理，根据tmp_x直接提取
            x = x[tmp_x]
        # x = np.linspace(start=0, stop=1, num=(num + 2) if scale_up != 1 else num, endpoint=True, dtype=np.float64)
        fs = {'x': x, 'y': y}
        fxs[k] = fs
    return fxs

def show_ft(img, xy_rect=(),ft_labels=()):
    # h, w, _ = img.shape
    new_img = img.copy()
    theta_fine = np.linspace(0, 2*np.pi, 200)
    for i, xy_label in enumerate(xy_rect):
        if len(ft_labels) > i:#for ft
            label = ft_labels[i]
            an,bn,cn,dn = [abcd.reshape(-1) for abcd in np.split(label[2:].reshape(-1, 4), 4, axis=-1)]
            x_approx = sum([an[i]*np.cos((i+1)*theta_fine) + bn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            y_approx = sum([cn[i]*np.cos((i+1)*theta_fine) + dn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            xy = np.vstack([x_approx + label[0], y_approx + label[1]]).T# * (w, h)
            xy = xy.astype(np.int32)
            cv2.polylines(new_img, [xy], True, (0,0,255), 2)
        if xy_label.shape[-1] == 4:#for h rect
            cv2.rectangle(new_img, (int(xy_label[0]), int(xy_label[1])), (int(xy_label[2]), int(xy_label[3])), (0,255,0), 2)
        elif xy_label.shape[-1] == 12:# for pts
            pts_ = xy_label[4:].reshape([-1, 2]).astype(np.int32)
            cv2.polylines(new_img, [pts_], True, (0,0,255), 2)
    return new_img
