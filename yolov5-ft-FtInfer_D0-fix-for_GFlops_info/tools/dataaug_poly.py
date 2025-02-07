import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import cv2
import os
import numpy as np
import random
import sys
sys.path.append('./')
sys.path.append('../')

class ImageAugPoly:
    def __init__(self) -> None:
        self.augments = [seq, oneof_aug, fliplr, flipud, flipone, rotate, rotate2, shear, crop, blur, translate]

    def augment(self, image, labels):
        polys = []
        for label in labels:
            polys.append(Polygon(label[5:].reshape(4, 2), label=label[0]))
        boxes = PolygonsOnImage(polys, shape=image.shape)
        na = len(self.augments)
        idx = random.randint(0, na - 1)
        aug_img, boxes_aug = self.augments[idx](image, boxes)
        boxes_aug = boxes_aug.remove_out_of_image().clip_out_of_image()
        n = len(boxes_aug)
        if n:
            points = []
            bbox = []
            clss = []
            for i in range(len(boxes_aug.polygons)):
                after = boxes_aug.polygons[i]
                n = len(after.coords)
                if n < 4 or n > 4:
                    continue
                xmin, ymin = np.min(after.xx), np.min(after.yy)
                xmax, ymax = np.max(after.xx), np.max(after.yy)
                points.append(after.coords.reshape(-1))
                clss.append(after.label)
                bbox.append([xmin, ymin, xmax, ymax])
            points = np.array(points)
            clss = np.array(clss)
            bbox = np.array(bbox)
            if len(clss):
                clss = clss.reshape(-1, 1)
                aug_labels = np.concatenate((clss, bbox, points), axis=1)
            else:
                aug_labels = np.zeros((0, 13), dtype=np.float32)
        else:
            aug_labels = np.zeros((0, 13), dtype=np.float32)
        
        return aug_img, aug_labels


def augment_poly(image, labels, r=[-45, 45]):
    polys = []
    for label in labels:
        polys.append(Polygon(label[5:].reshape(4, 2), label=label[0]))
    boxes = PolygonsOnImage(polys, shape=image.shape)
    aug_img, boxes_aug = rotate(image, boxes)
    boxes_aug = boxes_aug.remove_out_of_image().clip_out_of_image()

    n = len(boxes_aug)
    if n:
        points = []
        bbox = []
        clss = []
        for i in range(len(boxes_aug.polygons)):
            after = boxes_aug.polygons[i]
            n = len(after.coords)
            if n < 4 or n > 4:
                continue
            xmin, ymin = np.min(after.xx), np.min(after.yy)
            xmax, ymax = np.max(after.xx), np.max(after.yy)
            points.append(after.coords.reshape(-1))
            clss.append(after.label)
            bbox.append([xmin, ymin, xmax, ymax])
        points = np.array(points)
        clss = np.array(clss)
        bbox = np.array(bbox)
        if len(clss):
            clss = clss.reshape(-1, 1)
            aug_labels = np.concatenate((clss, bbox, points), axis=1)
        else:
            aug_labels = np.zeros((0, 13), dtype=np.float32)
    else:
        aug_labels = np.zeros((0, 13), dtype=np.float32)
    return aug_img, aug_labels
    



def read_anno(path, file_name):
    bboxes = []
    with open(os.path.join(path, file_name), 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            segmentation = [int(float(x)) for x in line[:8]]
            pos = []
            for i in range(0, len(segmentation), 2):
                pos.append((segmentation[i], segmentation[i + 1]))
            catgeory = line[8]
            bboxes.append(Polygon(pos, label=catgeory))
    return bboxes


def seq(image, bbs):
    seqe = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 1.5)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    # Augment BBs and images.
    image_aug, bbs_aug = seqe(image=image, polygons=bbs)
    return image_aug, bbs_aug


def fliplr(image, bbs, rate=1):
    aug = iaa.Fliplr(rate)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def flipud(image, bbs, rate=1):
    aug = iaa.Flipud(rate)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def flipone(image, bbs, rate=0.5):
    aug = iaa.OneOf([
        iaa.Fliplr(rate),
        iaa.Flipud(rate)
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def rotate(image, bbs, rotate_angle=(-90, 90)):
    aug = iaa.Affine(rotate=rotate_angle)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def rotate2(image, bbs, rotate_angle=(-45, 45)):
    aug = iaa.Affine(rotate=rotate_angle)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def shear(image, bbs, shear_angle=(-16, 16)):
    aug = iaa.Affine(shear=shear_angle)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def translate(image, bbs):
    aug = iaa.OneOf([
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.ElasticTransformation(alpha=(0, 2.0), sigma=0.1),
        iaa.ScaleX((0.5, 1.5)),
        iaa.ScaleY((0.5, 1.5)),
        iaa.TranslateX(percent=(-0.1, 0.1)),
        iaa.TranslateY(percent=(-0.1, 0.1))
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def crop(image, bbs):
    aug = iaa.Sequential([
        iaa.CropAndPad(percent=(-0.2, 0.2), keep_size=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def oneof_aug(image, bbs):
    aug = iaa.OneOf([
        # iaa.Affine(rotate=(-45, 45)),
        iaa.AdditiveGaussianNoise(scale=0.2 * 255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def resize(image, bbs, size=(1280, 1280)):
    image_rescaled = ia.imresize_single_image(image, size)
    bbs_rescaled = bbs.on(image_rescaled)
    return image_rescaled, bbs_rescaled


def blur(image, bbs):
    aug = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 11)),
        iaa.AverageBlur(k=((5, 11), (1, 3))),
        iaa.BilateralBlur(
            d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
        iaa.MedianBlur(k=(3, 11))
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


# 加云或者雾
def clouds(image, bbs):
    aug = iaa.OneOf([
        iaa.Clouds(),
        iaa.Fog()
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def get_augs():
    augs = [seq, resize, oneof_aug, fliplr, flipud, flipone, rotate, rotate2, shear, crop, blur, translate]
    return augs


def xyxy2points(xyxy):
    return xyxy[0], xyxy[1], xyxy[2], xyxy[1], xyxy[2], xyxy[3], xyxy[0], xyxy[3]


def gen_img():
    img_path = r'/home/LIESMARS/2019286190105/datasets/final-master/HRSC/train/images'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/HRSC/train/labelTxt'
    aug_img_path = r'runs/detect/images'
    aug_label_path = r'runs/detect/labels'
    images = os.listdir(img_path)
    for img in images:
        src_image = cv2.imread(os.path.join(img_path, img))
        label = img.split('.')[0] + '.txt'
        polys = read_anno(label_path, label)
        bbs = PolygonsOnImage(polys, shape=src_image.shape)
        augs = get_augs()
        t = 1
        for aug in augs:
            image_aug, bbs_aug = aug(src_image, bbs)
            # 去除在图片外的，剪切部分在图片里的
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
            aug_name = img.split('.')[0] + '-' + str(t)
            aug_label_name = 'aug_' + aug_name + '.txt'
            with open(os.path.join(aug_label_path, aug_label_name), 'w') as f:
                for i in range(len(bbs_aug.polygons)):
                    after = bbs_aug.polygons[i]
                    n = len(after.coords)
                    if n < 4 or n > 4:
                        continue
                    coords = after.coords
                    line = ""
                    for p in coords:
                        line += str(p[0]) + ' ' + str(p[1]) + ' '
                    line += after.label
                    f.write(line + '\n')
            aug_img_name = 'aug_' + aug_name + '.jpg'
            cv2.imwrite(os.path.join(aug_img_path, aug_img_name), image_aug)
            t += 1

if __name__ == '__main__':
    gen_img()