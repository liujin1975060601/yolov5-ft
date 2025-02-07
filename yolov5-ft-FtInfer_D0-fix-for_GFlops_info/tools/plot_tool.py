import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

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


def count_cls_num(path, cls):
    n = len(cls)
    count = np.zeros(n, dtype=np.int)
    files = os.listdir(path)
    files = filter(lambda x: x.endswith('.txt'), files)
    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                line = line.split(' ')
                count[int(line[0])] += 1
    return count



def plot_one_rot_box(x, img, color=None, label=None, line_thickness=None, leftop=False, radius=3, dir_line=False):
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    x = np.int32(x)
    leftop_x = (x[0], x[1])
    if dir_line:
        x1, y1 = (x[0] + x[2]) / 2, (x[1]+x[3]) / 2
        x2, y2 = (x[4] + x[6]) / 2, (x[5] + x[7]) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.arrowedLine(img, (int(cx), int(cy)), (int(x1), int(y1)), (0, 0, 255), thickness=tl)
    x = x.reshape((-1, 1, 2))
    cv2.polylines(img, [x], True, color, thickness=tl)
    if leftop:
        cv2.circle(img, leftop_x, radius, color, -1)
    if label:
        tf = max(tl - 1, 1)
        cv2.putText(img, label, leftop_x, 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)


def plot_ucas_aod():
    image_path = r'H:\BaiduNetdiskDownload\UCAS\train\images'
    label_path = r'H:\BaiduNetdiskDownload\UCAS\train\labels_pts_cls'
    save_path = r'H:\BaiduNetdiskDownload\UCAS\train\results'
    images = os.listdir(image_path)
    for image in images:
        print(image)
        label_name = image.split('.')[0] + '.pts'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width = src_img.shape[:2]
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                points = line.strip().split(' ')
                arr = []
                for i, x in enumerate(points[1:]):
                    if i % 2 == 0:
                        arr.append(float(x) * width)
                    else:
                        arr.append(float(x) * height)
                plot_one_rot_box(np.array(arr), src_img, dir_line=False, label=None, leftop=True, radius=6,
                                 color=colors[int(points[0])])
            cv2.imwrite(os.path.join(save_path, image), src_img)

def plot_hrsc2016():
    image_path = r'H:\BaiduNetdiskDownload\HRSC\train\images'
    label_path = r'H:\BaiduNetdiskDownload\HRSC\train\labels'
    save_path = r'H:\BaiduNetdiskDownload\HRSC\train\results'
    images = os.listdir(image_path)
    for image in images:
        print(image)
        label_name = image.split('.')[0] + '.pts'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width = src_img.shape[:2]
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                points = line.strip().split(' ')
                arr = []
                for i, x in enumerate(points[1:]):
                    if i % 2 == 0:
                        arr.append(float(x) * width)
                    else:
                        arr.append(float(x) * height)
                plot_one_rot_box(np.array(arr), src_img, dir_line=False, label=None, leftop=False, radius=6,
                                 color=colors[int(points[0])])
            cv2.imwrite(os.path.join(save_path, image), src_img)



def plot_dota():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768-1.0/train/images'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768-1.0/train/labels'
    save_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768-1.0/train/results'
    images = os.listdir(image_path)
    for image in images[:1000]:
        print(image)
        label_name = image.split('.')[0] + '.pts'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width = src_img.shape[:2]
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                points = line.strip().split(' ')
                arr = []
                for i, x in enumerate(points[1:]):
                    if i % 2 == 0:
                        arr.append(float(x) * width)
                    else:
                        arr.append(float(x) * height)
                plot_one_rot_box(np.array(arr), src_img, dir_line=False, label=None, leftop=True, radius=5, line_thickness=2,
                                 color=colors[int(points[0])])
            cv2.imwrite(os.path.join(save_path, image), src_img)

if __name__ == '__main__':
    plot_dota()
