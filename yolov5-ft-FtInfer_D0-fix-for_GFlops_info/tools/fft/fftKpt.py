import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from fftcurve import compute_coefficients,gen_curve,find_vectors
from tqdm import tqdm

def fourier_to_contour(fourier_desc):
    contour_reconstructed = []
    for t in np.linspace(0, 1, len(fourier_desc[0]), endpoint=False):
        x = fourier_desc[0][0] + sum([fourier_desc[0][i] * np.cos(2 * np.pi * i * t) - fourier_desc[1][i] * np.sin(2 * np.pi * i * t) for i in range(1, len(fourier_desc[0]))])
        y = fourier_desc[2][0] + sum([fourier_desc[2][i] * np.cos(2 * np.pi * i * t) + fourier_desc[3][i] * np.sin(2 * np.pi * i * t) for i in range(1, len(fourier_desc[2]))])
        contour_reconstructed.append([x, y])

    return np.array(contour_reconstructed, np.int32).reshape((-1, 1, 2))

def plot_fourier(plt,objs):
    # 初始化图表
    plt.figure()
    #plt.title(file_name)  # 添加标题
    for class_id,xy,coefs in objs:
        if(xy!=[]):
            [x, y] = xy
            # 绘制原始的多边形
            plt.plot(x, y, '.-', color="gray", linewidth=0.5)    
        # 使用傅里叶系数绘制曲线
        x_approx, y_approx = gen_curve(coefs, 8, 80)
        plt.plot(x_approx, y_approx, color=color1[class_id], linewidth=2)
    plt.gca().invert_yaxis()
    # 设置x和y轴的范围
    plt.xlim(0, 1)
    plt.ylim(1, 0)  # 注意，由于我们反转了y轴，所以范围也需要调换

    plt.axis('equal')
    plt.show()

def contour_fix_pts(contour,n):
    step = 0.08 / n
    epsilon = step * cv2.arcLength(contour, True)
    while True:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == n:
            break
        epsilon += step  # Adjust epsilon value
    return approx


def gen_one_contour(src_name,labels_path, terms, terms_export, show,show_hbox=False):
    name = os.path.splitext(os.path.basename(src_name))[0]
    seg_name = data_path + '/SegmentationClass/'+name+'.png'
    # 读取图像
    if os.path.exists(seg_name):
        seg_image = cv2.imread(seg_name)
        height, width, channels = seg_image.shape
    else:
        image_name = data_path + '/images/'+name+'.jpg'
        image_name = cv2.imread(image_name)
        height, width, channels = image_name.shape

    label_file = labels_path + '/'+name+'.txt'
    label_exist = os.path.exists(label_file)
    if not label_exist:
        txt = open(label_file, 'w')

        # 遍历每种颜色
        objs = []
        for class_id, color in enumerate(colors):
            # 创建一个二值图像
            mask = cv2.inRange(seg_image, color, color)
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.imshow(f"Contours{class_id}", mask)
            #cv2.waitKey(0)
                
            # 遍历轮廓
            for contour in contours:
                if cv2.contourArea(contour) > 0:
                    # 计算轮廓的外接矩形
                    left, top, w, h = cv2.boundingRect(contour)
                    cx,cy = left + w/2, top + h/2

                    x,y = np.squeeze(np.transpose(contour, (1, 2, 0))) #[2,npts]

                    # 计算轮廓的傅立叶系数
                    coefs = compute_coefficients(x, y, terms)

                    x_approx, y_approx = gen_curve(coefs, terms, terms_export)

                    objs.append((class_id,[cx,cy,w,h],[x_approx, y_approx]))
                    #fourier_desc = cv2.ximgproc.fourierDescriptor(contour)
                    #an, bn, cn, dn = fourier_desc[0], fourier_desc[1], fourier_desc[2], fourier_desc[3]

                    # 使用傅立叶描述符恢复轮廓
                    #contour_reconstructed = fourier_to_contour(fourier_desc)

                    # 创建输出文件
                    if label_exist:
                        with open(label_file, 'r') as file:
                            lines_txt = file.readlines()
                    else:
                        txt.write(f"{class_id} {cx/width} {cy/height} {w/width} {h/height}")
                        for i in range(terms_export):
                            txt.write(f" {x_approx[i]/width} {y_approx[i]/height} 2")
                        txt.write("\n")
                       
                    # 可视化
                    if 0:
                        image = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
                        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                        cv2.drawContours(image, [contour_reconstructed], -1, (0, 0, 255), 2)
        txt.close()
    else:
        with open(label_file, 'r') as file:
            lines_txt = file.readlines()
        objs = []
        for i,line in enumerate(lines_txt):
            # 计算傅里叶系数
            data_obj = list(map(float, line.split()))
            class_id = round(data_obj[0])
            [cx,cy,w,h] = data_obj[1:5]
            cx*=width
            cy*=height
            w*=width
            h*=height
            x_approx = [round(t * width) for t in data_obj[5::3]]#round(data_obj[5::2] * width)
            y_approx = [round(t * height) for t in data_obj[6::3]]#round(data_obj[6::2] * height)
            objs.append((class_id,[cx,cy,w,h],[x_approx, y_approx]))

    if 0:
        # 初始化图表
        plt.figure()
        #plt.title(file_name)  # 添加标题
        plot_fourier(plt,objs)
    else:
        if label_exist or show:
            image = cv2.imread(src_name)
            h, w, channels = image.shape
            kw,kh = w/width,h/height
            if show_hbox:
                images = []
                for term in range(2,terms):  # 循环从2阶到8阶
                    image2 = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
                    images.append(image2)
            for class_id,rect,pts in objs:
                [cx,cy,w,h] = rect
                if pts!=[]:
                    [x, y] = pts
                    x = [t * kw for t in x]
                    y = [t * kh for t in y]
                    # 合并x和y坐标到一个[n, 1, 2]形状的数组
                    contour = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [contour], isClosed=True, color=colors[class_id], thickness=2)
                    #cv2.circle(image, (round(x[0]), round(y[0])), 4, (0, 255, 0), 2)
                # 在原始图像上绘制外接矩形框
                cv2.rectangle(image, (round(cx - w/2), round(cy - h/2)), (round(cx + w/2), round(cy + h/2)), (0, 255, 0), 1)
            cv2.imshow("Contours", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    #windows
    #data_path = 'I:/datas/voc_segment_benchmark/val'
    #615
    data_path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val'
    #
    images_path = data_path + '/images/'
    labels_path = data_path + '/kpts/'
    if not os.path.exists(labels_path):
        os.mkdir(labels_path)

    # 读取颜色列表
    # 获取当前文件所在的目录
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    colors = []
    with open(os.path.join(current_dir_path, 'classes_colors.txt'), 'r') as f:
        for line in f:
            colors.append(tuple(map(int, line.split())))
        color1 = [tuple(val/255. for val in color) for color in colors]
        colors = [(b, g, r) for r, g, b in colors]

    images_files = [file for file in os.listdir(images_path) if file.endswith('.jpg')]

    terms=9
    terms_export = 32

    for idx, file_name in enumerate(tqdm(images_files)):
        # 图像文件名
        src_name = os.path.join(images_path, file_name)
        #src_name = data_path + '/images/2008_000041.jpg'
        gen_one_contour(src_name,labels_path, terms, terms_export, idx<8, 1)


