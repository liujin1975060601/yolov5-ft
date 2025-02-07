import cv2
import random
import os
import numpy as np
# from utils.metrics import ft2vector, ft2pts
import torch

import json
from PIL import ImageDraw, ImageFont, Image
font_path = os.path.dirname(os.path.dirname(__file__)) + '/font/simhei.ttf'

def ft2xy(an,bn,cn,dn,theta_fine,term):
    term = an.shape[0] if term<=0 else term
    x_approx = sum([an[i]*np.cos(i*theta_fine) + bn[i]*np.sin(i*theta_fine) for i in range(term)])
    y_approx = sum([cn[i]*np.cos(i*theta_fine) + dn[i]*np.sin(i*theta_fine) for i in range(term)])
    return x_approx,y_approx

def plot_one_box(x, img, color=None, label=None, line_thickness=None, ft_label=None, amp_stat=None, show_amp=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    tf = max(tl - 1, 1)  # font thickness
    if ft_label is None:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    else:
        theta_fine = np.linspace(0, 2*np.pi, 200)
        an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
        an,bn,cn,dn = np.insert(an,0,ft_label[0]),np.insert(bn,0,0),np.insert(cn,0,ft_label[1]),np.insert(dn,0,0)



        x_approx,y_approx = ft2xy(an,bn,cn,dn,theta_fine,0)#an.shape[0]
        contour_approx = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [contour_approx], isClosed=True, color=color, thickness=line_thickness)
        # term =1 
        #an,bn,cn,dn = an[:1],bn[:1],cn[:1],dn[:1] #ellipse only
        # x_approx1,y_approx1 = ft2xy(an,bn,cn,dn,theta_fine,2)
        # contour_approx1 = np.array(list(zip(x_approx1, y_approx1)), dtype=np.int32).reshape((-1, 1, 2))
        # cv2.polylines(img, [contour_approx1], isClosed=True, color=[255,255,255], thickness=2)

        #
        # ap,bp,cp,dp = ft2vector(ft_label.reshape(1,len(ft_label)),0)#ap[1,term]
        # assert len(ap)==1 and len(bp)==1 and len(cp)==1 and len(dp)==1
        # a0,c0 = an[0],cn[0]
        # cv2.arrowedLine(img, (int(np.round(a0)), int(np.round(c0))), (int(np.round(a0+ap[0])), int(np.round(c0+cp[0]))), (0, 0, 255), 2)
        # cv2.arrowedLine(img, (int(np.round(a0)), int(np.round(c0))), (int(np.round(a0+bp[0])), int(np.round(c0+dp[0]))), (0, 255, 0), 2)
        #
        # if label:
        #     cv2.putText(img, label, (round(ft_label[0]), round(ft_label[1] - 2)), 0, tl / 6, [225, 255, 255], thickness=max(tf//2,1), lineType=cv2.LINE_AA)

        # points = ft2pts(torch.Tensor(ft_label.reshape([1, -1])))  # 1, 8
        # contour_approx = np.array(list(zip(points[0, 0::2], points[0, 1::2])), dtype=np.int32).reshape((-1, 1, 2))
        # cv2.polylines(img, [contour_approx], isClosed=True, color=[0, 255, 0], thickness=2)

        c1 = (round(ft_label[0]), round(ft_label[1] - 15))
        amp = [0 for i in range(1, an.shape[0])]
        if amp_stat is None:
            amp_stat = [0 for i in range(0, an.shape[0])]
        for i_ in range(1, an.shape[0]):
            amp[i_ - 1] = np.sqrt(an[i_] ** 2 + bn[i_] ** 2 + cn[i_] ** 2 + dn[i_] ** 2) 
            amp_stat[i_ - 1] += amp[i_ - 1]
            amp[i_ - 1] = amp[i_ - 1] / max(img.shape) * 30
            amp_stat[-1] += 1


        if label:
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=max(tf//2,1))[0]
            cv2.rectangle(img, (round(ft_label[0]), round(ft_label[1])), (round(ft_label[0]) + t_size[0], round(ft_label[1] - 2) - t_size[1] - 3), (255,255,255), -1)
            cv2.putText(img, label, (round(ft_label[0]), round(ft_label[1] - 2)), 0, tl / 6, [50, 50, 50], thickness=max(tf//2,1), lineType=cv2.LINE_AA)

        if show_amp:
            cv2.rectangle(img, (c1[0], c1[1] + 3), (c1[0] + len(amp) * 6 + 3, int(c1[1] - max(amp) * 5 - 3)), (255,255,255), -1)
            for i_ in range(1, an.shape[0]):
                offset_amp = i_ * 6 - 3
                cv2.rectangle(img, (c1[0] + offset_amp, c1[1]), (c1[0] + offset_amp + 3, int(c1[1] - amp[i_ - 1] * 5)), (255,255,0), -1)





def plot_one_box_with_cms(x, img, color=None, label=None, line_thickness=None, cms_sub=None, cms_value=None, ft_label=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        cms_name = ['长度(cm): ', '宽度(cm): ', '最大缝宽(mm): ', '面积(cm^2): ']
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font1 = ImageFont.truetype(font_path, 30)#was ('simhei.ttf', 30) /usr/share/fonts/truetype/Arial.ttf
        t_size = font1.getsize(label)
        # c2 = (c1[0] + t_size[0] + 6, c1[1] + t_size[1] + 12)
        up_label = True
        if (int(x[1]) - t_size[1] - 1) > 0: # label x1 > 0  y>0
            c1 = (int(x[0]), int(x[1]) - t_size[1] - 1) # lu
            c2 = (int(x[0]) + t_size[0] + 4, int(x[1]) - 1)     # rb
            lb1 = (c1[0] + 3, c1[1]) # lu
        else:
            c1 = (int(x[0]), int(x[1]) + tl + 1) # lu
            c2 = (int(x[0]) + t_size[0] + 3, int(x[1]) + tl + 1 + t_size[1])     # rb
            lb1 = (c1[0] + 3, c1[1]) # lu
            up_label = False

        if cms_sub or cms_value:
            length = 0
            if cms_sub:
                length += 1
            if cms_value:
                length += len(cms_value)
            t1_size = 0
            for i, j in enumerate(cms_value):
                t1_size = max(font1.getsize(f'{cms_name[i]}{j:.2f}')[0], t1_size)
            if (int(x[3]) + (t_size[1] + 10) * (length + 1)) < img.shape[0]:  # box bottom + cms label  cms往下画
                c3 = (int(x[0]), int(x[3])) # box lb
                c4 = (int(x[0]) + t1_size + 4, int(x[3]) + (t_size[1] + 10) * length) # box lb + text
                lb2 = [int(x[0]) + 3, int(x[3]) - tl]    # cms txt origin
            else:
                c3 = (int(x[0]), int(x[1]) + tl) # box lu
                c4 = (int(x[0]) + t1_size + 4, int(x[1]) + 10 + (t_size[1] + 10) * len(cms_value))
                if not up_label:
                    c3 = (c3[0], c3[1] + t_size[1])
                    c4 = (c4[0], c4[1] + t_size[1])
                lb2 = [c3[0] + 4, c3[1] + 10]
            
        offset = t_size[1] + 10
        draw.rectangle((c1, c2), fill=tuple(color), outline=tuple(color))
        if cms_sub or cms_value:
            draw.rectangle((c3, c4), fill=tuple(color), outline=tuple(color))
        draw.text(lb1, label, font=font1,fill=(225, 255, 255))#图片上加入中文，使用与cv2.putText类似, 提供左上角的位置
        if cms_sub is not None:
            draw.text((lb2[0], lb2[1]), f'cms_s: {cms_sub}', font=font1,fill=(225, 255, 255))
            lb2[1] += offset
        if cms_value is not None:
            for i, j in enumerate(cms_value):
                if i <= 4:
                    draw.text((lb2[0], lb2[1]), f'{cms_name[i]}{j:.2f}', font=font1,fill=(225, 255, 255))
                    lb2[1] += offset
                else:
                    draw.text((lb2[0], lb2[1]), '......', font=font1,fill=(225, 255, 255))
        if ft_label is not None:
            '''
            ft_label[0:2] *= np.array([img.shape[1],img.shape[0]])
            for i in range((ft_label.shape[0]-2)//4):
                start=2+i*4
                ft_label[start:start+4] *= np.array([img.shape[1],img.shape[1],img.shape[0],img.shape[0]])
            '''
            theta_fine = np.linspace(0, 2*np.pi, 200)
            an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
            #an,bn,cn,dn = an[:1],bn[:1],cn[:1],dn[:1] #ellipse only
            x_approx = sum([an[i]*np.cos((i+1)*theta_fine) + bn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            y_approx = sum([cn[i]*np.cos((i+1)*theta_fine) + dn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            xy = np.vstack([x_approx, y_approx]).T #* (img.shape[1], img.shape[0])
            xy[:, 0] += ft_label[0]
            xy[:, 1] += ft_label[1]
            xy = xy.astype(np.int32).reshape(-1).tolist()
            draw.polygon(xy, outline=tuple(color), width=tl)
        img[:, :, :] = np.array(pil_img)#转换为PIL库可以处理的图片形式
  
def drawRect(img, pos, **kwargs):
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.rectangle(pos, **kwargs)
    img.paste(Image.alpha_composite(img, transp))

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
        cv2.circle(img, leftop_x, radius, color, tl)
    if label:
        tf = max(tl - 1, 1) 
        cv2.putText(img, label, leftop_x, 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)






def plot_images_from_8points():
    image_path = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/val/images'
    label_path = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/val/labelTxt1.5'
    images = os.listdir(image_path)
    save_path = './runs/detect/results'

    for image in images:
        # 读取label
        label_name = image.split('.')[0] + '.txt'
        src_img = cv2.imread(os.path.join(image_path, image))
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                segmentation = [int(float(x)) for x in line[:8]]
                category = line[8]
                # xmin, ymin, xmax, ymax = min(segmentation[::2]), min(segmentation[1::2]), \
                #                          max(segmentation[::2]), max(segmentation[1::2])
                # xyxy = np.array([xmin, ymin, xmax, ymax])
                # plot_one_box(xyxy, src_img, label=category)
                plot_one_rot_box(segmentation, src_img, label=category, dir_line=True, leftop=True)
            cv2.imwrite(os.path.join(save_path, image), src_img)


def plot_images_from_xywh():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/UCAS50/images/train'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/UCAS50/labels/train'
    images = os.listdir(image_path)
    save_path = '../runs/detect/exp2'
    # clses = ['triangle_horizontal', 'triangle_vertical', 'triangle_oblique', 'dangerous_goods', 'dangerous']
    for image in images:
        label_name = image.split('.')[0] + '.txt'
        src_img = cv2.imread(os.path.join(image_path, image))
        # width, height = src_img.shape[:2]
        height, width = src_img.shape[:2]
        label = os.path.join(label_path, label_name)
        if not os.path.exists(label):
            continue
        with open(label, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                # cls = int(line[0])
                # category = clses[cls]
                category = line[0]
                xywh = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
                xyxy = np.array(xywh2xyxy(xywh, width, height))
                plot_one_box(xyxy, src_img, label=category)
            cv2.imwrite(os.path.join(save_path, image), src_img)


def plot_images_from_rot():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/images'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/labels1.5'
    save_path = r'/home/LIESMARS/2019286190105/finalwork/yolov5/runs/detect/images'
    images = os.listdir(image_path)
    for image in images:
        print(image)
        label_name = image.split('.')[0] + '.pts'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width  = src_img.shape[:2]
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                points = line.strip().split(' ')
                arr = []
                for i, x in enumerate(points[1:]):
                    if i % 2 == 0:
                        arr.append(float(x) * width)
                    else:
                        arr.append(float(x) * height)
                plot_one_rot_box(np.array(arr), src_img, dir_line=True, label=points[0], leftop=True)
                # plot_one_rot_box(np.array(arr), src_img, dir_line=False, label=None)
            cv2.imwrite(os.path.join(save_path, image), src_img)


def xyrot2xy(x, width, height):
    x = np.array(x, dtype=np.float)
    x[:, 0] *= width
    x[:, 1] *= height
    return x


def xywh2xyxy(x, width, height):
    x, y, w, h = x[0], x[1], x[2], x[3]
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    x1 *= width
    x2 *= width
    y1 *= height
    y2 *= height
    return [int(x1), int(y1), int(x2), int(y2)]


if __name__ == '__main__':
    # plot_images_from_xywh()
    plot_images_from_8points()
    # plot_images_from_rot()
