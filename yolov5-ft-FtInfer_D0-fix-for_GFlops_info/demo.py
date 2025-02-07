import os

import cv2
import numpy as np
from numpy.lib import utils
from scipy.spatial import distance
import torch
from tools.plotbox import plot_one_rot_box
from utils.autoanchor import kmean_anchors, kmean_anchors_ab


def order_corners(boxes):
    """
        Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
        Sorted corners have the following restrictions:
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """

    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y)  # sort y
    x_sorted = torch.zeros_like(x, dtype=x.dtype)
    for i in range(x.shape[0]):
        x_sorted[i] = x[i, y_indices[i]]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(y.shape[0]):
        y_sorted[i, :2] = y_sorted[i, :2][x_bottom_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_top_indices[i]]
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()


def sortpts_clockwise(A):
    A = A.reshape(4, 2)
    # Sort A based on Y(col-2) coordinates
    sortedAc2 = A[np.argsort(A[:, 1]), :]

    # Get top two and bottom two points
    top2 = sortedAc2[0:2, :]
    bottom2 = sortedAc2[2:, :]

    # Sort top2 points to have the first row as the top-left one
    sortedtop2c1 = top2[np.argsort(top2[:, 0]), :]
    top_left = sortedtop2c1[0, :]

    # Use top left point as pivot & calculate sq-euclidean dist against
    # bottom2 points & thus get bottom-right, bottom-left sequentially
    sqdists = distance.cdist(top_left[None], bottom2, 'sqeuclidean')
    rest2 = bottom2[np.argsort(np.max(sqdists, 0))[::-1], :]

    # Concatenate all these points for the final output
    return np.concatenate((sortedtop2c1, rest2), axis=0).flatten()


def plot_clockwise():
    p1 = r'G:\dataset\UCAS_AOD\UCAS50\labels\train'
    p2 = r'G:\dataset\UCAS_AOD\UCAS50\labels\order'
    p3 = r'G:\dataset\UCAS_AOD\UCAS50\images\train'
    files = os.listdir(p1)
    files = list(filter(lambda x: x.endswith('.pts'), files))
    images = [x.split('.')[0] + '.png' for x in files]
    color1 = (205, 90, 106)  # BGR GT蓝色
    color2 = (180, 105, 255)  # BGR 纠正后的粉色
    for i, file in enumerate(files):
        print('cur image:', images[i])
        img = cv2.imread(os.path.join(os.path.join(p3, images[i])))
        h, w = img.shape[:2]
        with open(os.path.join(p1, file), 'r') as f:
            p = [x.split() for x in f.read().strip().splitlines() if len(x)]
            p = np.array(p, dtype=np.float32)
            p[:, 1::2] *= w
            p[:, 2::2] *= h
            for y in p:
                print(y[1:])
                plot_one_rot_box(y[1:], img, color1, leftop=True, radius=5)
                order = sortpts_clockwise(y[1:])
                order += 10
                print(order)
                plot_one_rot_box(order, img, color2, leftop=True, radius=5)
        cv2.imwrite(os.path.join(p2, images[i]), img)


if __name__ == '__main__':
    kmean_anchors_ab('data/dota.yaml', img_size=768, thr=4)
    # kmean_anchors('data/dota.yaml', img_size=768, thr=4)
