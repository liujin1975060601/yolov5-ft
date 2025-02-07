import os
import sys

sys.path.append('./')
import os.path as osp
from pathlib import Path
import cv2
from DOTA_devkit.dota_utils import polygonToRotRectangle
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def filter_meta():
    """
    过滤dota标签中的meta信息
    """
    p1 = r'D:\dataset\DOTA\DOTA1.0-1.5\val\labelTxt-v1.0\labelTxt'
    p2 = r'D:\dataset\DOTA\DOTA1.0-1.5\val\labelTxt1.0'
    labels = os.listdir(p1)
    p1 = Path(p1)
    p2 = Path(p2)
    for label in labels:
        print(label)
        with open(p1 / label, 'r') as f:
            new_lines = []
            for line in f:
                cur = line.strip().split()
                if len(cur) > 8:
                    new_lines.append(line)

        with open(p2 / label, 'w') as f:
            for line in new_lines:
                f.write(line)


def makedirs(p):
    if not osp.exists(p):
        os.makedirs(p)


class DOTALabelTools:
    def __init__(self, img_path, labelTxt_path, output, classes):
        self.img_path = img_path
        self.labelTxt_path = labelTxt_path
        self.output = output
        self.classes = classes
        makedirs(self.output)

    def read_dota_labels(self, label):
        new_lines = []
        with open(label, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) > 0:
                    new_lines.append(line)
        return new_lines

    def _bbox2yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return [centerx, centery, w, h]

    def dota2yolo(self, points, w, h):
        xmin, ymin, xmax, ymax = min(points[::2]), min(points[1::2]), \
                                 max(points[::2]), max(points[1::2])
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        return self._bbox2yolo(bbox, w, h)

    def dota2pts(self, points, w, h):
        pts = []
        # rect = np.array(points, dtype=np.float32).reshape(4, 2)
        # rect = cv2.minAreaRect(rect)
        # box = cv2.boxPoints(rect).reshape(-1)

        for i, x in enumerate(points):
            if i % 2 == 0:
                pts.append(x / w)
            else:
                pts.append(x / h)
        return pts

    def generate_labels(self):
        images = os.listdir(self.img_path)
        for image in images:
            print(image)
            cur_name = image.split('.')[0]
            label_name = cur_name + '.txt'
            pts_name = cur_name + '.pts'
            img = cv2.imread(osp.join(self.img_path, image))
            h, w = img.shape[:2]
            annos = self.read_dota_labels(osp.join(self.labelTxt_path, label_name))
            yolo_lines = []
            pts_lines = []
            for ann in annos:
                cls = self.classes.index(ann[8])
                points = [float(x) for x in ann[:8]]
                bbox = self.dota2yolo(points, w, h)
                yolo_line = [cls] + bbox
                pts = self.dota2pts(points, w, h)
                pts_line = [cls] + pts

                yolo_lines.append(yolo_line)
                pts_lines.append(pts_line)

            assert len(yolo_lines) == len(pts_lines), "label convert error"

            with open(osp.join(self.output, label_name), 'w') as f1, \
                    open(osp.join(self.output, pts_name), 'w') as f2:
                for l1, l2 in zip(yolo_lines, pts_lines):
                    l1 = [str(x) for x in l1]
                    l2 = [str(x) for x in l2]
                    f1.write(' '.join(l1) + '\n')
                    f2.write(' '.join(l2) + '\n')


    def _generate(self, image):
        cur_name = image.split('.')[0]
        label_name = cur_name + '.txt'
        pts_name = cur_name + '.pts'
        img = cv2.imread(osp.join(self.img_path, image))
        h, w = img.shape[:2]
        annos = self.read_dota_labels(osp.join(self.labelTxt_path, label_name))
        yolo_lines = []
        pts_lines = []
        for ann in annos:
            cls = self.classes.index(ann[8])
            points = [float(x) for x in ann[:8]]
            bbox = self.dota2yolo(points, w, h)
            yolo_line = [cls] + bbox
            pts = self.dota2pts(points, w, h)
            pts_line = pts

            yolo_lines.append(yolo_line)
            pts_lines.append(pts_line)

        assert len(yolo_lines) == len(pts_lines), "label convert error"

        with open(osp.join(self.output, label_name), 'w') as f1, \
                open(osp.join(self.output, pts_name), 'w') as f2:
            for l1, l2 in zip(yolo_lines, pts_lines):
                l1 = [str(x) for x in l1]
                l2 = [str(x) for x in l2]
                f1.write(' '.join(l1) + '\n')
                f2.write(' '.join(l2) + '\n')
    
    def multi_generate_labels(self, process=8):
        images = os.listdir(self.img_path)
        images = os.listdir(self.img_path)
        with Pool(process) as p:
            list(tqdm(p.imap(self._generate, images),total=len(images),desc='generate labels'))
        p.close()
        p.join()


def generate():
    img_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768-1.0/val/images'
    labelTxt_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768-1.0/val/labelTxt1.0'
    output = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768-1.0/val/labels'
    dotav1_5_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
                        'ship', 'tennis-court',
                        'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
                        'swimming-pool',
                        'helicopter', 'container-crane']
    dotav1_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
                      'ship', 'tennis-court',
                      'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
                      'helicopter']
    hrsc_classes = ['ship']
    ucas_classes = ['plane', 'car']
    dtl = DOTALabelTools(img_path, labelTxt_path, output, dotav1_classes)
    dtl.multi_generate_labels(process=16)


if __name__ == '__main__':
    generate()
