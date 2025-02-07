import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from fftcurve import compute_coefficients,gen_curve,find_vectors
from tqdm import tqdm

# Function to interpolate the array if its length is less than term
def interpolate_to_length(contour, term=32):#contour[npts,2]
    # Extract x and y
    x, y = contour[..., 0].squeeze(-1), contour[..., 1].squeeze(-1)
    # Check if the length is less than term
    if len(x) < term:
        # Perform interpolation for x and y to have a length of at least term
        x_new = np.interp(np.linspace(0, len(x) - 1, term), np.arange(len(x)), x)
        y_new = np.interp(np.linspace(0, len(y) - 1, term), np.arange(len(y)), y)
        new_ct = np.column_stack((x_new, y_new))
        if len(contour.shape)==3:
            return np.expand_dims(new_ct, axis=1)
        else:
            return new_ct
    else:
        return contour

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

def gen_one_fourier(src_name,labels_path, hull, terms, terms_export, show,show_hbox=False):
    name = os.path.splitext(os.path.basename(src_name))[0]
    seg_name = data_path + '/SegmentationClass/'+name+'.png'
    # 读取图像
    if os.path.exists(seg_name):
        seg_image = cv2.imread(seg_name)
        height, width, channels = seg_image.shape
    else:
        image_name = data_path + '/images/'+name+'.jpg'
        image = cv2.imread(image_name)
        height, width, channels = image.shape

    label_file = labels_path + '/'+name+'.txt'
    label_exist = os.path.exists(label_file)
    if not label_exist:
        txt = open(label_file, 'w')
    
    ft_file = labels_path + '/'+name+'.ft'
    ft_exist = os.path.exists(ft_file)
    if not ft_exist:
        ft = open(ft_file, 'w')

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
                    if hull:
                        contour = cv2.convexHull(contour)
                    contour = interpolate_to_length(contour).astype(int)
                    # 计算轮廓的外接矩形
                    left, top, w, h = cv2.boundingRect(contour)

                    # 创建输出文件
                    if label_exist:
                        with open(label_file, 'r') as file:
                            lines_txt = file.readlines()
                    else:
                        txt.write(f"{class_id} {(left + w/2)/width} {(top + h/2)/height} {w/width} {h/height}\n")

                    x,y = np.squeeze(np.transpose(contour, (1, 2, 0))) #[2,npts]

                    # 计算轮廓的傅立叶系数
                    coefs = compute_coefficients(x, y, terms)
                    objs.append((class_id,[x, y],coefs))
                    #fourier_desc = cv2.ximgproc.fourierDescriptor(contour)
                    #an, bn, cn, dn = fourier_desc[0], fourier_desc[1], fourier_desc[2], fourier_desc[3]

                    # 使用傅立叶描述符恢复轮廓
                    #contour_reconstructed = fourier_to_contour(fourier_desc)

                    [an, bn, cn, dn] = coefs
                    ft.write(f"{class_id} {an[0]/width} {cn[0]/height} ")
                    for i in range(1, terms_export):
                        ft.write(f"{an[i]/width} {bn[i]/width} {cn[i]/height} {dn[i]/height} ")
                    ft.write("\n")
                    
                    # 可视化
                    if 0:
                        image = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
                        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                        cv2.drawContours(image, [contour_reconstructed], -1, (0, 0, 255), 2)
        txt.close()
        ft.close()
    else:
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

    if 0:
        # 初始化图表
        plt.figure()
        #plt.title(file_name)  # 添加标题
        plot_fourier(plt,objs)
    else:
        if ft_exist or show:
            image = cv2.imread(src_name)
            h, w, channels = image.shape
            kw,kh = w/width,h/height
            if show_hbox:
                image0 = image.copy()
                images = []
                for term in range(2,terms):  # 循环从2阶到8阶
                    image2 = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
                    images.append(image2)
            for class_id,xy,coefs in objs:
                if xy!=[]:
                    [x, y] = xy
                    x = [t * kw for t in x]
                    y = [t * kh for t in y]
                    # 合并x和y坐标到一个[n, 1, 2]形状的数组
                    contour = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [contour], isClosed=True, color=colors[class_id], thickness=1)
                    #cv2.circle(image, (round(x[0]), round(y[0])), 4, (0, 255, 0), 2)
                # 使用傅里叶系数绘制曲线
                x_approx, y_approx = gen_curve(coefs, terms, 80)
                x_approx = [t * kw for t in x_approx]
                y_approx = [t * kh for t in y_approx]
                contour_approx = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [contour_approx], isClosed=True, color=colors[int(class_id)], thickness=3)

                # 使用arrowedLine绘制目标方向矢量
                [a1, b1, c1, d1],points = find_vectors(coefs,0,kh/kw)
                a0,c0 = coefs[0][0],coefs[2][0]
                cv2.arrowedLine(image, (round(a0), round(c0)), (round(a0+a1), round(c0+c1)), (0, 0, 255), 2)
                cv2.arrowedLine(image, (round(a0), round(c0)), (round(a0+b1), round(c0+d1)), (0, 255, 0), 2)
                cv2.polylines(image, [points.astype(int).reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
                if show_hbox:
                    for term in range(2, terms):  # 循环从2阶到8阶
                        # 生成当前阶数的曲线
                        x_approx, y_approx = gen_curve(coefs, term, 80)
                        x_approx2 = [t * kw / 2 for t in x_approx]
                        y_approx2 = [t * kh / 2 for t in y_approx]
                        contour_approx2 = np.array(list(zip(x_approx2, y_approx2)), dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(images[term-2], [contour_approx2], isClosed=True, color=colors[int(class_id)], thickness=3)
                    x, y, w, h = cv2.boundingRect(contour_approx)
                    # 在原始图像上绘制外接矩形框
                    cv2.rectangle(image0, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            if show_hbox:
                # 现在我们有了所有不同阶数的曲线图像，可以将它们拼接在一起
                for term in range(2, terms):  # 循环从2阶到8阶
                    # 添加term值到每张图像的左上角
                    cv2.putText(images[term-2], f'Term {term}', (0, 0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                imageftn = cv2.hconcat(images)
                cv2.imshow("HBox", image0)
                cv2.imshow("imageftn", imageftn)
            cv2.imshow("Contours", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    #windows
    #data_path = 'I:/datas/voc_segment_benchmark/'
    #615
    data_path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val'
    #
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

    for idx, file_name in enumerate(tqdm(images_files)):
        # 图像文件名
        src_name = os.path.join(images_path, file_name)
        #src_name = data_path + '/images/2008_000041.jpg'
        gen_one_fourier(src_name,labels_path, 1, terms, terms_export, idx<8, 1)


