import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from fftcurve import compute_coefficients,gen_curve,find_vectors
from tqdm import tqdm
from shapely.geometry import Polygon
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

def rect_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 计算两个矩形的相交区域
    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    area_intersection = x_intersection * y_intersection

    area_rect1 = w1 * h1
    area_rect2 = w2 * h2

    # 计算交并比
    iou = area_intersection / (area_rect1 + area_rect2 - area_intersection)

    return iou

def poly_iou(contour1, contour2, hull):
    # 计算轮廓的凸包
    if(hull):
        contour1 = cv2.convexHull(contour1)
        contour2 = cv2.convexHull(contour2)

    # 转换为Shapely多边形
    polygon1 = Polygon(contour1[:, 0, :])
    polygon1 = polygon1.buffer(0)
    assert polygon1.is_valid
    polygon2 = Polygon(contour2[:, 0, :])
    polygon2 = polygon2.buffer(0)
    assert polygon2.is_valid

    # 计算交集多边形
    intersection = polygon1.intersection(polygon2)

    # 计算交集多边形的面积
    area_intersection = intersection.area

    # 计算两个轮廓的面积
    area_contour1 = polygon1.area
    area_contour2 = polygon2.area

    # 计算IoU_poly
    iou_poly = area_intersection / (area_contour1 + area_contour2 - area_intersection)
    return iou_poly

def gen_one_iou(src_name,labels_path, terms, iou_pairs,scale=1,rep_time=8):
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
    
    ft_file = labels_path + '/'+name+'.ft'
    ft_exist = os.path.exists(ft_file)
    if ft_exist:
        with open(label_file, 'r') as file:
            lines_txt = file.readlines()
        with open(ft_file, 'r') as file:
            lines_ft = file.readlines()
        objs = []
        for i,line in enumerate(lines_ft):
            # 计算傅里叶系数
            data_ft = list(map(float, line.split()))
            class_id = data_ft[0]
            an = [data_ft[1]]+data_ft[3::4]
            an = [x * width for x in an]
            bn = [0]+data_ft[4::4]
            bn = [x * width for x in bn]
            cn = [data_ft[2]]+data_ft[5::4]
            cn = [x * height for x in cn]
            dn = [0]+data_ft[6::4]
            dn = [x * height for x in dn]
            coefs = [an,bn,cn,dn]
            objs.append((class_id,[],coefs))
        for class_id,xy,coefs in objs:
            if xy!=[]:
                [x, y] = xy
                x = [t for t in x]
                y = [t for t in y]
                # 合并x和y坐标到一个[n, 1, 2]形状的数组
                contour = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))
                #cv2.polylines(image, [contour], isClosed=True, color=colors[class_id], thickness=1)
                #cv2.circle(image, (round(x[0]), round(y[0])), 4, (0, 255, 0), 2)
            # 使用傅里叶系数绘制曲线
            x_approx, y_approx = gen_curve(coefs, terms, 80)
            x_approx = [t for t in x_approx]
            y_approx = [t for t in y_approx]
            contour = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
            rect = cv2.boundingRect(contour)
            [x, y, w, h] = rect
            # 生成随机水平和垂直平移量
            for i in range(rep_time):
                tx = np.random.uniform(-scale*w, scale*w)
                ty = np.random.uniform(-scale*h, scale*h)
                # 对contour中的每个点进行平移
                contour_translated = contour + [int(tx), int(ty)]
                rect_translated = cv2.boundingRect(contour_translated)
                iou_rect = rect_iou(rect, rect_translated)
                iou_poly = poly_iou(contour,contour_translated,0)
                iou_pairs.append((iou_rect,iou_poly))
        
# 定义拟合的曲线模型，这里使用一个二次多项式
def quadratic_curve(x, a, b, c):
    return a * x**2 + b * x + c

# 绘制垂线
def plot_vhline(plt,x_coords,y_coords,x_value,y_value,color):
    plt.axvline(x=x_value, color=color, linestyle='--', linewidth=1)
    plt.axhline(y=y_value, color=color, linestyle='--', linewidth=1)
    # 在图中添加文本显示特殊点的坐标
    plt.text(x_value, min(y_coords), f'x={x_value:.2f}', ha='center')
    plt.text(min(x_coords), y_value, f'y={y_value:.2f}', va='bottom')

if __name__ == "__main__":
    #windows
    #data_path = 'I:/datas/voc_segment_benchmark/val'
    #615
    data_path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val'
    #277
    #data_path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val'

    images_path = data_path + '/images/'
    labels_path = data_path + '/labels/'
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
    terms_export = 8

    iou_pairs=[]
    for idx, file_name in enumerate(tqdm(images_files)):
        # 图像文件名
        src_name = os.path.join(images_path, file_name)
        #src_name = data_path + '/images/2008_000041.jpg'
        gen_one_iou(src_name,labels_path, terms, iou_pairs, 1, 8)
    #print(iou_pairs)
    # 提取 x 和 y 坐标
    x_coords, y_coords = zip(*iou_pairs)

    # 创建散点图
    plt.scatter(x_coords, y_coords, label='Points', color='b', marker='o',s=1)

    # 使用curve_fit进行拟合
    params, covariance = curve_fit(quadratic_curve, x_coords, y_coords)
    # 拟合的参数
    a, b, c = params
    # 生成拟合的曲线数据点
    x_fit = np.linspace(min(x_coords), max(x_coords), 100)
    y_fit = quadratic_curve(x_fit, a, b, c)
    plt.plot(x_fit, y_fit, label='Quadratic Fit', color='red')

    x_value = 0.5  # 你要查找的 x 值
    y_value = quadratic_curve(x_value, a, b, c)  # 使用拟合的参数计算对应的 y 值
    print(f"When x={x_value}, y={y_value}")
    # 绘制垂线
    plot_vhline(plt,x_coords,y_coords,x_value,y_value,'grey')

    # 定义一个用于求解的函数
    def find_x(x, a, b, c, y_value):
        return quadratic_curve(x, a, b, c) - y_value
    y_value = 0.5  # 你要查找的 y 值
    x_solution = fsolve(find_x, 0.5, args=(a, b, c, y_value))
    x_value = x_solution[0]
    print(f"When x={x_value}, y={y_value}")
    # 绘制垂线
    plot_vhline(plt,x_coords,y_coords,x_value,y_value,'red')


    # 可选：添加标签、标题等
    plt.xlabel('iou')
    plt.ylabel('poly_iou')
    plt.title('poly_iou vs iou')
    plt.legend()

    # 显示散点图
    plt.show()