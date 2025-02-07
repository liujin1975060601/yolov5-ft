import os
import os.path as osp
import random
import cv2
import shutil
import numpy as np


def generate():
    p1 = r'D:\BaiduNetdiskDownload\UCAS_AOD\UCAS50\images\train'
    p2 = r'D:\BaiduNetdiskDownload\UCAS_AOD\UCAS50\labels\train'
    p3 = r'D:\BaiduNetdiskDownload\UCAS_AOD\UCAS50\labels\train1'
    images = os.listdir(p1)
    for image in images:
        img = cv2.imread(osp.join(p1, image))
        h, w = img.shape[:2]
        label_name = image.split('.')[0] + '.txt'
        new_lines = []
        with open(osp.join(p2, label_name), 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                new_line = [line[0]]
                points = [float(x) for x in line[1:]]
                for i in range(len(points)):
                    if i % 2 == 0:
                        points[i] *= w
                    else:
                        points[i] *= h

                xmin, ymin = min(points[::2]), min(points[1::2])
                xmax, ymax = max(points[::2]), max(points[1::2])
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                bbox = bbox_2_yolo(bbox, w, h)
                bbox = [str(round(x, 6)) for x in bbox]
                new_line.extend(bbox)
                new_lines.append(new_line)
        with open(osp.join(p3, label_name), 'w') as f:
            for line in new_lines:
                f.write(' '.join(line) + '\n')
        pts_name = image.split('.')[0] + '.pts'
        shutil.copy(osp.join(p2, label_name), osp.join(p3, pts_name))


def bbox_2_yolo(bbox, img_w, img_h):
    # bbox矩形框, 左上角坐标 , 宽, 高
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    centerx = bbox[0] + w / 2
    centery = bbox[1] + h / 2
    dw = 1 / img_w
    dh = 1 / img_h
    centerx *= dw
    w *= dw
    centery *= dh
    h *= dh
    return centerx, centery, w, h


def generate_ucas():
    p1 = '/home/LIESMARS/2019286190105/datasets/final-master/UCAS_AOD/images'
    p2 = '/home/LIESMARS/2019286190105/datasets/final-master/UCAS_AOD/labelTxt'
    p3 = '/home/LIESMARS/2019286190105/datasets/final-master/UCAS_AOD/labels'

    images = os.listdir(p1)
    for image in images:
        img = cv2.imread(osp.join(p1, image))
        h, w = img.shape[:2]
        label_name = image.split('.')[0] + '.txt'
        with open(osp.join(p2, label_name), 'r') as f:
            p = [x.split() for x in f.read().strip().splitlines() if len(x)]
            p = np.array(p, dtype=np.float32)
        cls = p[:, 0].reshape(-1, 1)
        poly = p[:, 1:9]
        poly[:, ::2] /= w
        poly[:, 1::2] /= h

        bbox = p[:, 10:]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:] / 2
        bbox[:, ::2] /= w
        bbox[:, 1::2] /= h
        
        poly = np.concatenate((cls, poly), axis=1)
        bbox = np.concatenate((cls, bbox), axis=1)

        with open(osp.join(p3, label_name), 'w') as f:
            for box in bbox:
                line = [str(int(box[0]))]
                for x in box[1:]:
                    line.append(str(x))
                f.write(' '.join(line) + '\n')
        
        pts_name = image.split('.')[0] + '.pts'
        with open(osp.join(p3, pts_name), 'w') as f:
            for box in poly:
                line = [str(int(box[0]))]
                for x in box[1:]:
                    line.append(str(x))
                f.write(' '.join(line) + '\n')



def merge_data():
    folders = ['PLANE', 'CAR']
    path = '/home/LIESMARS/2019286190105/datasets/final-master/UCAS_AOD'
    out_img = '/home/LIESMARS/2019286190105/datasets/final-master/UCAS_AOD/images'
    out_label = '/home/LIESMARS/2019286190105/datasets/final-master/UCAS_AOD/labelTxt'
    for folder in folders:
        p = os.path.join(path, folder)
        images = list(filter(lambda x: x.endswith('.png'), os.listdir(p)))
        labels = [x.replace('.png', '.txt') for x in images]
        for image in images:
            print(image)
            name = folder + '_'+image
            shutil.copy(os.path.join(p, image), os.path.join(out_img, name))
        for label in labels:
            print(label)
            with open(os.path.join(p, label), 'r') as f:
                new_lines = []
                for line in f:
                    line = line.strip().split('\t')
                    line = [str(folders.index(folder))] + line
                    new_lines.append(line)
            name = folder + '_' + label
            with open(os.path.join(out_label, name), 'w') as f:
                for line in new_lines:
                    f.write(' '.join(line) + '\n')               


def select_val():
    p1 = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/images/train'
    p2 = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/images/val2'
    p3 = p2.replace('images', 'labels')
    if not osp.exists(p2):
        os.makedirs(p2)
    if not osp.exists(p3):
        os.makedirs(p3)
    images = os.listdir(p1)
    random.shuffle(images)
    for image in images[:200]:
        print(image)
        shutil.move(osp.join(p1, image), osp.join(p2, image))
        name = image.split('.')[0]
        shutil.move(osp.join(p1.replace('images', 'labels'), name+'.txt'), osp.join(p3, name+'.txt'))
        shutil.move(osp.join(p1.replace('images', 'labels'), name+'.pts'), osp.join(p3, name+'.pts'))


def generate_empty_txt(image_path, label_path):
    """
    对比图片和标签，为没有目标的图片补充空白标签
    :param image_path: 图片路径
    :param label_path: 标签路径
    :return:
    """

    def _create_txt(path, file_name):
        shutil.copy('a.txt', os.path.join(path, file_name))

    images = os.listdir(image_path)
    images = [x.split('.')[0] for x in images]
    labels = os.listdir(label_path)
    labels = set([x.split('.')[0] for x in labels])
    for image in images:
        if image not in labels:
            _create_txt(label_path, image + '.txt')




def ucas_split_topts():
    p1 = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA512/val-big/images'
    p2 = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA512/val-big/labelTxt'
    p3 = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA512/val-big/labels'
    # classes = ['ship']
    # classes = ['plane', 'car']
    classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
                  'ship', 'tennis-court',
                  'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
                  'helicopter', 'container-crane']
    images = os.listdir(p1)
    for image in images:
        img = cv2.imread(osp.join(p1, image))
        h, w = img.shape[:2]
        # h, w = 512, 512
        label_name = image.split('.')[0] + '.txt'
        new_lines = []
        with open(osp.join(p2, label_name), 'r') as f:
            p = [x.split() for x in f.read().strip().splitlines() if len(x)]
        for line in p:
            new_line = [str(classes.index(line[8]))]
            for i, x in enumerate(line[:8]):
                if i % 2 == 0:
                    new_line.append(str(float(x) / w))
                else:
                    new_line.append(str(float(x) / h))
            new_lines.append(new_line)

        pts_name = image.split('.')[0] + '.pts'
        print(pts_name)
        with open(osp.join(p3, pts_name), 'w') as f:
            for line in new_lines:
                f.write(' '.join(line) + '\n')



def generate_hsrc2016():
    file1 = r'G:\dataset\HRSC2016\ImageSets\test.txt'
    p1 = r'G:\dataset\HRSC2016\HRSC\images'
    p2 = r'G:\dataset\HRSC2016\HRSC\labelTxt'
    o1 = r'G:\dataset\HRSC2016\HRSC\val\images'
    o2 = r'G:\dataset\HRSC2016\HRSC\val\labelTxt'
    with open(file1, 'r') as f:
        names = f.read().strip().split()

    for name in names:
        print(name)
        img_name = name + '.png'
        txt_name = name + '.txt'
        shutil.move(osp.join(p1, img_name), osp.join(o1, img_name))
        shutil.move(osp.join(p2, txt_name), osp.join(o2, txt_name))


def read():
    p1 = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA512/train-512/labels'
    
    files = os.listdir(p1)
    for file in files[:100]:
        print(file)
        with open(osp.join(p1, file), 'r') as f:
            for line in f:
                print(line.strip())

if __name__ == '__main__':
    read()