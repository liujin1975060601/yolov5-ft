import numpy as np
import scipy.integrate as integrate
import torch
from tools.plotbox import plot_one_box

def is_point_inside_fourier_curve(fourier_coeffs, x, y):
    """
    判断点 (x, y) 是否在由傅里叶级数描述的闭合曲线内部
    :param fourier_coeffs: np数组, shape为 [4n-2]，包含傅里叶级数的系数
                           数组前2个元素是a0, b0, 后面4(n-1)个元素是从1到n-1阶的傅里叶系数
    :param x: 待判定点的 x 坐标
    :param y: 待判定点的 y 坐标
    :return: 如果点在闭合曲线内部，返回 True；否则返回 False
    """
    # 解析傅里叶系数
    n = (len(fourier_coeffs) + 2) // 4
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    a[0] = fourier_coeffs[0]
    c[0] = fourier_coeffs[1]
    for i in range(1, n):
        a[i] = fourier_coeffs[2 + 4 * (i - 1)]
        b[i] = fourier_coeffs[3 + 4 * (i - 1)]
        c[i] = fourier_coeffs[4 + 4 * (i - 1)]
        d[i] = fourier_coeffs[5 + 4 * (i - 1)]
    
    # 待判定的点
    z0 = x + 1j * y

    def fourier_series_curve(a, b, c, d, n, t):
        X = np.sum([a[i] * np.cos(i * t) + b[i] * np.sin(i * t) for i in range(n)], axis=0)
        Y = np.sum([c[i] * np.cos(i * t) + d[i] * np.sin(i * t) for i in range(n)], axis=0)
        return X, Y

    def fourier_series_derivative(a, b, c, d, n, t):
        X_prime = np.sum([i * (-a[i] * np.sin(i * t) + b[i] * np.cos(i * t)) for i in range(n)], axis=0)
        Y_prime = np.sum([i * (-c[i] * np.sin(i * t) + d[i] * np.cos(i * t)) for i in range(n)], axis=0)
        return X_prime, Y_prime

    def integrand(t):
        X, Y = fourier_series_curve(a, b, c, d, n, t)
        X_prime, Y_prime = fourier_series_derivative(a, b, c, d, n, t)
        z = X + 1j * Y
        dz_dt = X_prime + 1j * Y_prime
        return dz_dt / (z - z0)

    # 数值积分计算包围数
    W, _ = integrate.quad(lambda t: integrand(t).real, 0, 2 * np.pi, limit=100)
    W_imag, _ = integrate.quad(lambda t: integrand(t).imag, 0, 2 * np.pi, limit=100)
    W = (W + 1j * W_imag) / (2 * np.pi * 1j)

    # 判定点是否在闭合曲线内部
    WA = abs(W)
    inside = WA > 0.0001
    if inside:
        print(f'W={W}, {WA}')
        
    return inside

def render_fourier_curve(im0, objs, cls):
    """
    在图像上渲染多个目标的傅里叶闭合曲线围成的区域，并根据目标类别使用不同颜色。
    :param im0: 原始图像
    :param objs: np数组，shape为 [nt, 4n-2]，包含目标的傅里叶系数
    :param cls: np数组，shape为 [nt]，包含每个目标的整数类别标号
    :return: 渲染后的图像
    """
    # 定义颜色列表，根据类别选择颜色
    colors = [
        (255, 0, 0),     # 类别 0 -> 蓝色
        (0, 255, 0),     # 类别 1 -> 绿色
        (0, 0, 255),     # 类别 2 -> 红色
        (255, 255, 0),   # 类别 3 -> 青色
        (255, 0, 255),   # 类别 4 -> 品红色
        (0, 255, 255),   # 类别 5 -> 黄色
        (255, 255, 255), # 类别 6 -> 白色
    ]

    height, width, _ = im0.shape

    cls_int = cls.to(torch.int)

    # 对图像的每个像素进行检查
    for y in range(height):
        for x in range(width):
            for i, obj in enumerate(objs):
                if is_point_inside_fourier_curve(obj, x, y):
                    category = cls_int[i]
                    color = colors[category % len(colors)]
                    im0[y, x] = color
                    break  # 一旦找到包含该点的目标，就不再检查其他目标

    return im0



import numpy as np
import torch
import cv2
import random
def draw_x(image, center, color, size=5, thickness=1):
    """
    在图像上绘制叉叉
    :param image: 待绘制的图像
    :param center: 叉叉中心点 (x, y)
    :param color: 叉叉颜色 (B, G, R)
    :param size: 叉叉的大小
    :param thickness: 叉叉线条的厚度
    """
    x, y = center
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)

def render_fourier_curve2(img_inside, objs, cls, n):
    """
    在图像上渲染多个目标的傅里叶闭合曲线围成的区域，并根据目标类别使用不同颜色。
    :param img_inside: 原始图像
    :param objs: np数组，shape为 [nt, 4n-2]，包含目标的傅里叶系数
    :param cls: np数组，shape为 [nt]，包含每个目标的整数类别标号
    :param n: 随机生成点的数量
    :return: 渲染后的图像
    """
    # 定义颜色列表，根据类别选择颜色
    colors = [
        (255, 0, 0),     # 类别 0 -> 蓝色
        (0, 255, 0),     # 类别 1 -> 绿色
        (0, 0, 255),     # 类别 2 -> 红色
        (255, 255, 0),   # 类别 3 -> 青色
        (255, 0, 255),   # 类别 4 -> 品红色
        (0, 255, 255),   # 类别 5 -> 黄色
        (255, 255, 255), # 类别 6 -> 白色
        (120, 120, 0), # 非目标 -> 红色
    ]

    height, width, _ = img_inside.shape

    cls_int = cls.to(torch.int)

    # 生成傅里叶闭合曲线的图像
    bboxs=[]
    for i, obj in enumerate(objs):
        category = cls_int[i]
        color = colors[category % (len(colors)-1)]
        fourier_curve,bbox = compute_fourier_curve(obj)
        bboxs.append(bbox)
        plot_one_box(None, img_inside, 
                    color=colors[category%len(colors)],
                    ft_label=obj.cpu().numpy(),line_thickness=1,show_amp=0)

    if 1:
        for _ in range(n):
            # 目标模式或非目标模式
            if random.random() < 0.8:  # 目标模式
                idx = random.randint(0, len(objs) - 1)
                obj = objs[idx]
                category = cls_int[idx]
                color = colors[category % len(colors)]
                x_min, y_min, x_max, y_max = bboxs[idx]
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
            else:  # 非目标模式
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                color = colors[-1]  # 灰色

            # 检查点是否在目标区域内部
            #inside = cv2.pointPolygonTest(fourier_curve, (x, y), False) >= 0
            category = -1
            for i, obj in enumerate(objs):
                if is_point_inside_fourier_curve(obj, x, y):
                    category = cls_int[i]
                    break  # 一旦找到包含该点的目标，就不再检查其他目标
            if category>=0:
                cv2.circle(img_inside, (x, y), radius=4, color=colors[category % len(colors)], thickness=-1)
            else:
                draw_x(img_inside, (x, y), color=colors[-1], size=4, thickness=2)

    return img_inside

def compute_fourier_curve(obj):
    """
    根据傅里叶系数计算闭合曲线
    :param obj: 傅里叶系数数组
    :return: 闭合曲线的点集
    """
    t = np.linspace(0, 2 * np.pi, num=100)
    curve = np.zeros((len(t), 2), dtype=np.int32)
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for i in range(len(t)):
        x, y = 0, 0
        for k in range(len(obj) // 2):
            x += obj[2 * k] * np.cos(k * t[i]) - obj[2 * k + 1] * np.sin(k * t[i])
            y += obj[2 * k] * np.sin(k * t[i]) + obj[2 * k + 1] * np.cos(k * t[i])
        curve[i, 0] = int(x)
        curve[i, 1] = int(y)
        #
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return curve,bbox






if __name__ == "__main__":
    # 示例傅里叶系数
    n = 10
    fourier_coeffs = np.concatenate([np.random.randn(2), np.random.randn(4 * (n - 1))])

    # 待判定点
    x0, y0 = 0.5, 0.5

    # 调用函数判断点是否在闭合曲线内部
    is_inside = is_point_inside_fourier_curve(fourier_coeffs, x0, y0)
    print(f"The point ({x0}, {y0}) is {'inside' if is_inside else 'outside'} the closed curve.")
