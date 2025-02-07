import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
import xml.dom.minidom
import numpy as np
import os
import math
import sys
import glob


def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w / 2, h / 2
    x_ru, y_ru = w / 2, h / 2
    x_ld, y_ld = -w / 2, -h / 2
    x_rd, y_rd = w / 2, -h / 2

    x_lu_ = math.cos(theta) * x_lu + math.sin(theta) * y_lu + box[0]
    y_lu_ = -math.sin(theta) * x_lu + math.cos(theta) * y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:

        if child_of_root.tag == 'Img_SizeWidth':
            img_width = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeHeight':
            img_height = int(child_of_root.text)
        if child_of_root.tag == 'HRSC_Objects':
            box_list = []
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Object':
                    label = 1
                    # for child_object in child_item:
                    #     if child_object.tag == 'Class_ID':
                    #         label = NAME_LABEL_MAP[child_object.text]
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            tmp_box[4] = float(node.text)

                    tmp_box = coordinate_convert_r(tmp_box)
                    # assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    # if len(tmp_box) != 0:
                    box_list.append(tmp_box)
            # box_list = coordinate_convert(box_list)
            # print(box_list)
    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2 as cv
    from shutil import copyfile

    src_image_path = r'G:\dataset\HRSC2016\FullDataSet\AllImages'
    src_xml_path = r'G:\dataset\HRSC2016\FullDataSet\Annotations'
    txt_path = r'G:\dataset\HRSC2016\HRSC\labelTxt'
    out_img_path = 'G:\dataset\HRSC2016\HRSC\images'
    mkdir(txt_path)
    mkdir(out_img_path)

    src_imgs = glob.glob(f'{src_image_path}/*.bmp')

    for img_path in src_imgs:
        print(img_path)
        try:
            ori_image = cv.imread(img_path)
            x_path = img_path[:-3].replace('AllImages', 'Annotations') + 'xml'
            img_height, img_width, gtbox_labels = read_xml_gtbox_and_label(x_path)
            if len(gtbox_labels) == 0:
                print("0")

            # for gtbox_label in gtbox_labels:
            #     tl = np.asarray([gtbox_label[0], gtbox_label[1]], np.float32)
            #     tr = np.asarray([gtbox_label[2], gtbox_label[3]], np.float32)
            #     br = np.asarray([gtbox_label[4], gtbox_label[5]], np.float32)
            #     bl = np.asarray([gtbox_label[6], gtbox_label[7]], np.float32)
            #     box = np.asarray([bl, tl, tr, br], np.float32)
            #     box = np.int0(box)
            #     cv.drawContours(ori_image, [box], 0, (255, 255, 255), 1)
            # plt.imshow(ori_image)
            # plt.show()
            # print(gtbox_labels)
            name = os.path.split(img_path)[-1][:-4]
            with open(f"{txt_path}/{name}.txt", 'w') as f:
                for ann in gtbox_labels:
                    f.write(f"{ann[0]} {ann[1]} {ann[2]} {ann[3]} {ann[4]} {ann[5]} {ann[6]} {ann[7]} ship 0\n")
            cv.imwrite(f'{out_img_path}/{name}.png', ori_image)
        except:
            print(img_path)
