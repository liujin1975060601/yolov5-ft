import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from fftcurve import compute_coefficients,gen_curve,find_vectors
from tqdm import tqdm

def read_pol_file(file_path):
    objs = []
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每一行的数据
            values = line.strip().split()

            clsid = int(values[0])
            
            # 将字符串转换为浮点数并添加到数据列表
            pol = [float(value) for value in values[1:]] #pol[2n]
            objs.append((clsid,[(pol[i], pol[i + 1]) for i in range(0, len(pol), 2)]))

    return objs
# Function to interpolate the array if its length is less than term
def interpolate_to_length(contour, term=32):
    # Extract x and y
    x, y = contour[:, 0], contour[:, 1]
    # Check if the length is less than term
    if len(x) < term:
        # Perform interpolation for x and y to have a length of at least term
        x_new = np.interp(np.linspace(0, len(x) - 1, term), np.arange(len(x)), x)
        y_new = np.interp(np.linspace(0, len(y) - 1, term), np.arange(len(y)), y)
        return np.column_stack((x_new, y_new))
    else:
        return contour

def gen_one_poly(src_name,labels_path, terms, terms_export, show,show_hbox=False):
    name = os.path.splitext(os.path.basename(src_name))[0]
    # 读取图像
    image_name = data_path + '/images/'+name+'.jpg'
    image = cv2.imread(image_name)
    height, width, channels = image.shape

    pol_file = labels_path + '/'+name+'.pol'
    pol_exist = os.path.exists(pol_file)
    
    ft_file = labels_path + '/'+name+'.ft'
    ft_exist = os.path.exists(ft_file)
    if not ft_exist:
        objs = []
        if os.path.exists(pol_file):
            objs_file = read_pol_file(pol_file)
            ft = open(ft_file, 'w')
            for obj in objs_file:
                (class_id,points) = obj
                contour = np.array(points)#contour[npts,2]
                contour[:,0]*=width
                contour[:,1]*=height
                contour = interpolate_to_length(contour)
                contour=contour.astype(int)
                area = cv2.contourArea(contour)
                if True:
                    # 计算轮廓的外接矩形
                    left, top, w, h = cv2.boundingRect(contour)

                    x,y = contour[:,0],contour[:,1] #[2,npts]

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
                    
            ft.close()
    else:
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

    if len(objs)>0 and (ft_exist or show):
        image = cv2.imread(src_name)
        h, w, channels = image.shape
        kw,kh = w/width,h/height
        scale = 0.3
        if show_hbox:
            image0 = image.copy()
            images = []
            for term in range(2,terms):  # 循环从2阶到8阶
                image2 = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
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
                    x_approx2 = [round(t * kw *scale) for t in x_approx]
                    y_approx2 = [round(t * kh *scale) for t in y_approx]
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
    #data_path = 'E:/datas/new_road_damage/seg_0-1712-640/'
    #615
    data_path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/voc_segment/val'
    #
    images_path = data_path + '/images/'
    labels_path = data_path + '/labels/'

    images_files = [file for file in os.listdir(images_path) if file.endswith('.jpg')]

    # 读取颜色列表
    # 获取当前文件所在的目录
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    colors = []
    with open(os.path.join(current_dir_path, 'classes_colors.txt'), 'r') as f:
        for line in f:
            colors.append(tuple(map(int, line.split())))
        color1 = [tuple(val/255. for val in color) for color in colors]
        colors = [(b, g, r) for r, g, b in colors]

    terms=9
    terms_export = 8

    for idx, file_name in enumerate(tqdm(images_files)):
        # 图像文件名
        src_name = os.path.join(images_path, file_name)
        #src_name = data_path + '/images/2008_000041.jpg'
        gen_one_poly(src_name,labels_path, terms, terms_export, idx<8, 1)


