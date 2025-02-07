import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math
import cv2

def compute_coefficients(x, y, terms=8):
    N = len(x)
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    a0 = 1./N * sum(x)
    c0 = 1./N * sum(y)
    
    an, bn, cn, dn = [a0], [0], [c0], [0]
    
    for k in range(1, terms):
        an.append(2./N * sum(x * np.cos(k*t)))
        bn.append(2./N * sum(x * np.sin(k*t)))
        cn.append(2./N * sum(y * np.cos(k*t)))
        dn.append(2./N * sum(y * np.sin(k*t)))
        
    return [an, bn, cn, dn]

def gen_curve(coefs, term, nout):
    [an, bn, cn, dn] = coefs
    n_coefs = len(an)
    assert(len(bn)==n_coefs)
    assert(len(cn)==n_coefs)
    assert(len(dn)==n_coefs)
    term = min(term,n_coefs)
    assert(term <=n_coefs)
    theta_fine = np.linspace(0, 2*np.pi, nout)
    x_approx = sum([an[i]*np.cos(i*theta_fine) + bn[i]*np.sin(i*theta_fine) for i in range(term)])
    y_approx = sum([cn[i]*np.cos(i*theta_fine) + dn[i]*np.sin(i*theta_fine) for i in range(term)])
    return x_approx,y_approx
def shift_curve(coefs, term, shift, n_loop):
    [an, bn, cn, dn] = coefs
    n_coefs = len(an)
    assert(len(bn)==n_coefs)
    assert(len(cn)==n_coefs)
    assert(len(dn)==n_coefs)
    term = min(term,n_coefs)
    assert(term <=n_coefs)
    theta_fine = 2*np.pi*shift/n_loop#np.linspace(0, 2*np.pi, n_loop)
    ant,bnt,cnt,dnt = [0]*term,[0]*term,[0]*term,[0]*term
    ant[0] =  an[0]
    cnt[0] =  cn[0]
    for k in range(1,term):
        t_cos = np.cos(k*theta_fine)
        t_sin = np.cos(k*theta_fine)
        ant[k] =  an[k]*t_cos + bn[k]*t_sin
        bnt[k] = -an[k]*t_sin + bn[k]*t_cos
        cnt[k] =  cn[k]*t_cos + dn[k]*t_sin
        dnt[k] = -cn[k]*t_sin + dn[k]*t_cos
    return [ant, bnt, cnt, dnt]

def find_main_aixs(coefs, n_loop, ratio=1):
    [an, bn, cn, dn] = coefs
    n_coefs = len(an)
    assert(len(bn)==n_coefs)
    assert(len(cn)==n_coefs)
    assert(len(dn)==n_coefs)
    theta_fine = np.linspace(0, 2*np.pi, n_loop)#theta_fine[n_loop]
    t_cos = np.cos(theta_fine)#[n_loop]
    t_sin = np.sin(theta_fine)#[n_loop]
    at =  an[1]*t_cos + bn[1]*t_sin#at[n_loop]
    ct =  ratio*(cn[1]*t_cos + dn[1]*t_sin)#ct[n_loop]
    lengths = np.sqrt(at**2 + ct**2)
    max_t = np.argmax(lengths)
    #
    m_cos = np.cos(2*np.pi*max_t/n_loop)#[n_loop]
    m_sin = np.sin(2*np.pi*max_t/n_loop)#[n_loop]
    an,bn,cn,dn = np.array(an),np.array(bn),np.array(cn),np.array(dn)
    a1 =  an*m_cos + bn*m_sin#a1[n_loop]
    b1 = -an*m_sin + bn*m_cos#b1[n_loop]
    c1 =  cn*m_cos + dn*m_sin#c1[n_loop]
    d1 = -cn*m_sin + dn*m_cos#d1[n_loop]
    return max_t,[a1, b1, c1, d1]
def find_main_aixs_grad(coefs, ratio=1):
    [an, bn, cn, dn] = coefs
    n_coefs = len(an)
    assert(len(bn)==n_coefs)
    assert(len(cn)==n_coefs)
    assert(len(dn)==n_coefs)
    ratio2=ratio*ratio
    cos2t = 0.5 * (an[1]*an[1] + ratio2*cn[1]*cn[1] - bn[1]*bn[1] - ratio2*dn[1]*dn[1])
    sin2t = (an[1]*bn[1]+ratio2*cn[1]*dn[1])
    an,bn,cn,dn = np.array(an),np.array(bn),np.array(cn),np.array(dn)
    vec2t = np.array([cos2t, sin2t], dtype=np.float32)
    L = np.linalg.norm(vec2t)
    if L>0:
        vec2t /= L#vec2t[2]
        #
        m_cos = np.sqrt((1+vec2t[0])/2)
        m_sin = np.sqrt((1-vec2t[0])/2)
        if vec2t[1]<0:
            m_sin=-m_sin
        a1 =  an*m_cos + bn*m_sin#a1[n_loop]
        b1 = -an*m_sin + bn*m_cos#b1[n_loop]
        c1 =  cn*m_cos + dn*m_sin#c1[n_loop]
        d1 = -cn*m_sin + dn*m_cos#d1[n_loop]
    else:
        vec2t = np.array([an[1], cn[1]], dtype=np.float32)
        L = np.linalg.norm(vec2t)
        vec2t /= L
        m_cos = vec2t[0]
        m_sin = vec2t[1]
        a1 = an#a1[n_loop]
        b1 = bn#b1[n_loop]
        c1 = cn#c1[n_loop]
        d1 = dn#d1[n_loop]
    return [m_cos,m_sin],[a1, b1, c1, d1]
def find_vectors(coefs, n_loop, ratio=1):
    if n_loop>0:
        main_id,[ap, bp, cp, dp] = find_main_aixs(coefs, n_loop, ratio)#搜索最大值
    else:
        [m_cos,m_sin],[ap, bp, cp, dp] = find_main_aixs_grad(coefs,ratio)#求导
    xc,yc = coefs[0][0],coefs[2][0]
    a1,b1,c1,d1 = ap[1],bp[1],cp[1],dp[1]
    P0=[xc-a1-b1,yc-c1-d1]
    P1=[xc-a1+b1,yc-c1+d1]
    P2=[xc+a1+b1,yc+c1+d1]
    P3=[xc+a1-b1,yc+c1-d1]
    points = np.array([P0, P1, P2, P3], dtype=np.float32)
    return [a1, b1, c1, d1],points

def find_vectors_shift(coefs, term, shift, nloop):
    #main_id,[a1, b1, c1, d1] = find_main_aixs(coefs, n_loop)
    coefs2 = shift_curve(coefs,term,shift, nloop)
    xc,yc = coefs2[0][0],coefs2[2][0]
    [a1, b1, c1, d1]=[coefs2[0][1],coefs2[1][1],coefs2[2][1],coefs2[3][1]]
    P0=[xc-a1-b1,yc-c1-d1]
    P1=[xc-a1+b1,yc-c1+d1]
    P2=[xc+a1+b1,yc+c1+d1]
    P3=[xc+a1-b1,yc+c1-d1]
    points = np.array([P0, P1, P2, P3], dtype=np.float32)
    return [a1, b1, c1, d1],points
    

def coefs2area(coefs,term):
    [an, bn, cn, dn] = coefs
    n_coefs = len(an)
    assert(len(bn)==n_coefs)
    assert(len(cn)==n_coefs)
    assert(len(dn)==n_coefs)
    term = min(term,n_coefs)
    assert(term <=n_coefs)
    area = np.pi * sum([i * (an[i]*dn[i] - bn[i]*cn[i]) for i in range(term)])
    return area

def format_coefs(coefs, term=2):
    [an, bn, cn, dn] = coefs
    # 合并列表
    n = len(an)
    term = min(term,n)
    result = [an[0], cn[0]]  # 开始先写入a0和c0
    for i in range(1, term):
        result.extend([an[i], bn[i], cn[i], dn[i]])  # 按照a, b, c, d的顺序加入结果列表
    return result
'''
# 创建围绕椭圆的点
theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
x_center, y_center = 0.5, 0.5  # 椭圆的中心
a, b = 0.4, 0.3  # 椭圆的主、次半轴长度

# 生成椭圆的点
x = a * np.cos(theta)
y = b * np.sin(theta)

# 旋转椭圆
alpha = np.pi / 4  # 旋转45度
rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
x_rotated, y_rotated = np.dot(rot_matrix, np.array([x, y]))

# 平移到中心
x_rotated += x_center
y_rotated += y_center
'''
if 0:
    x = [1, 1, 0, 0]
    y = [0 ,1, 1, 0]

    coefs = compute_coefficients(x, y)
    # 使用傅里叶系数绘制曲线
    x_approx, y_approx = gen_curve(coefs, 2, 80)
    # 绘制原始的多边形
    plt.plot(x + [x[0]], y + [y[0]], 'o-', color="gray")    
    plt.plot(x_approx, y_approx)
    plt.gca().invert_yaxis()
    # 设置x和y轴的范围
    plt.xlim(0, 1)
    plt.ylim(1, 0)  # 注意，由于我们反转了y轴，所以范围也需要调换

    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # 定义颜色表，例如：
    color1 = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
    colors = []
    # 获取当前文件所在的目录
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir_path, 'classes_colors.txt'), 'r') as f:
        for line in f:
            colors.append(tuple(map(int, line.split())))
        color1 = [tuple(val/255. for val in color) for color in colors]
        colors = [(b, g, r) for r, g, b in colors]
        
    #windows
    #data_path = 'E:/datas/GuGE/patches/'
    #data_path = 'E:/datas/DOTA1.5/patches'
    #data_path = 'E:/datas\HRSC2016/test'
    #615
    data_path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/dota1.5/patches'

    labels_path = data_path + "/labels"
    images_path = data_path + "/images"

    # 初始化一个空列表
    names = []
    fnames = data_path+'/names.txt'
    # 打开文件并读取内容
    if os.path.exists(fnames):
        with open(fnames, 'r') as file:
            names = [line.strip() for line in file.readlines()]

    pts_files = [file for file in os.listdir(labels_path) if file.endswith('.pts')]

    term=3
    term_export = 4

    for idx, file_name in enumerate(tqdm(pts_files)):#1702 nan a1,c1
        with open(os.path.join(labels_path, file_name), 'r') as file:
            lines_pts = file.readlines()
        txt_filename = os.path.join(labels_path, file_name.replace('.pts', '.txt'))
        if(os.path.exists(txt_filename)):
            with open(txt_filename, 'r') as file:
                lines_txt = file.readlines()
                
        lines_ft = None
        ft_filename = os.path.join(labels_path, file_name.replace('.pts', '.ft'))
        if(os.path.exists(ft_filename)):
            with open(ft_filename, 'r') as file:
                lines_ft = file.readlines()

        objs = []
        for i,line_pts in enumerate(lines_pts):
            data_pts = list(map(float, line_pts.split()))
            x = data_pts[::2]
            y = data_pts[1::2]
            
            # 计算傅里叶系数
            if lines_ft:
                data_ft = list(map(float, lines_ft[i].split()))
                id = int(data_ft[0])
                an = [data_ft[1]]+data_ft[3::4]
                bn = [0]+data_ft[4::4]
                cn = [data_ft[2]]+data_ft[5::4]
                dn = [0]+data_ft[6::4]
                coefs = [an,bn,cn,dn]
            else:
                line_txt = lines_txt[i]
                data_txt = list(map(float, line_txt.split()))
                id = int(data_txt[0])
                coefs = compute_coefficients(x, y, term)
            objs.append((id,[x, y],coefs))
        
        if lines_ft or idx<3:
            image_filename = os.path.join(images_path, file_name.replace('.pts', '.jpg'))
            if not os.path.exists(image_filename):
                image_filename = os.path.join(images_path, file_name.replace('.pts', '.bmp'))
            image_filename = os.path.join(image_filename.replace('\\', '/').replace('//', '/'))
            if os.path.exists(image_filename):
                image = cv2.imread(str(image_filename))
                show_hbox=True
                if show_hbox:
                    image0 = image.copy()
                height, width, channels = image.shape
                for i,(class_id,xy,coefs) in enumerate(objs):
                    if xy!=[]:
                        [x, y] = xy
                        x = [t * width for t in x]
                        y = [t * height for t in y]
                        # 合并x和y坐标到一个[n, 1, 2]形状的数组
                        contour = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image, [contour], isClosed=True, color=colors[int(class_id)%len(colors)], thickness=1)
                        #cv2.circle(image, (round(x[0]), round(y[0])), 4, (0, 255, 0), 2)
                    # 使用傅里叶系数绘制曲线
                    x_approx1, y_approx1 = gen_curve(coefs, term, 80)
                    x_approx = [t * width for t in x_approx1]
                    y_approx = [t * height for t in y_approx1]
                    contour_approx = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [contour_approx], isClosed=True, color=colors[int(class_id)%len(colors)], thickness=3)           
                    # 使用arrowedLine绘制目标方向矢量
                    [a1, b1, c1, d1],points = find_vectors(coefs,0,height/width)#find_vectors_shift(coefs, term, 1, 8) #
                    a0,c0 = coefs[0][0],coefs[2][0]
                    cv2.arrowedLine(image, (round(a0 * width), round(c0 * height)), (round((a0+a1)*width), round((c0+c1)*height)), (0, 0, 255), 2)
                    cv2.arrowedLine(image, (round(a0 * width), round(c0 * height)), (round((a0+b1)*width), round((c0+d1)*height)), (0, 255, 0), 2)
                    # 使用cv2.polylines绘制四边形
                    # 调整points数组
                    points[:, 0] *= width   # 对所有行的第一个元素（x分量）乘以width
                    points[:, 1] *= height  # 对所有行的第二个元素（y分量）乘以height
                    cv2.polylines(image, [points.astype(int).reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)

                    # 计算面积
                    if 1:
                        area_coef = coefs2area(coefs,term)
                        contour_approx1 = np.array(list(zip(x_approx1, y_approx1)), dtype=np.float32).reshape((-1, 1, 2))
                        area_pts = cv2.contourArea(contour_approx1)
                    print(f'area_coef={area_coef}/{area_pts}')
                    if show_hbox:
                        x, y, w, h = cv2.boundingRect(contour_approx)
                        # 在原始图像上绘制外接矩形框
                        cv2.rectangle(image0, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if show_hbox:
                    cv2.imshow("HBox", image0)
                    print(f'[{idx}]:{file_name}')
                cv2.imshow("Contours", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                # 初始化图表
                plt.figure()
                plt.title(file_name)  # 添加标题
                for id,xy,coefs in objs:
                    if xy!='':
                        [x, y] = xy
                        # 绘制原始的多边形
                        plt.plot(x + [x[0]], y + [y[0]], 'o-', color="gray")    
                    # 使用傅里叶系数绘制曲线
                    x_approx, y_approx = gen_curve(coefs, term, 80)
                    plt.plot(x_approx, y_approx,color=color1[id % len(color1)])
                    # 显示id在重心坐标处
                    a0,c0 = coefs[0][0],coefs[2][0]
                    plt.text(a0, c0, names[id] if names!=[] else str(id))
                    # 使用arrow绘制目标方向矢量
                    dx,dy = coefs[0][1], coefs[2][1]
                    L = math.sqrt(dx*dx+dy*dy)
                    plt.arrow(a0, c0, dx,dy, head_width=L*0.08, head_length=L*0.13, fc='red', ec='red')
                plt.gca().invert_yaxis()
                # 设置x和y轴的范围
                plt.xlim(0, 1)
                plt.ylim(1, 0)  # 注意，由于我们反转了y轴，所以范围也需要调换

                plt.axis('equal')
                plt.show()
        
        # 保存到 .ft 文件
        if lines_ft==None:
            with open(ft_filename, 'w') as file:
                for id,xy,coefs in objs:
                    formatted_coefs = format_coefs(coefs,term_export)  # 解包coefs并获取按正确格式排序的列表
                    formatted_strings = ["{:.7f}".format(val) for val in formatted_coefs]
                    file.write(str(id) + ' ' + ' '.join(formatted_strings) + '\n')
                    #file.write(' '.join(map(str, formatted_coefs)) + '\n')

'''
# 从文件中读取数据
with open("E:/datas/GuGE/patches/labels/韩国_镇海军港-unknown_9_8.pts", "r") as file:
    lines = file.readlines()

# 初始化图表
plt.figure()

# 对每一行的数据进行处理
for line in lines:
    data = list(map(float, line.split()))
    x = data[::2]
    y = data[1::2]
   
    # 计算傅里叶系数
    an, bn, cn, dn = compute_coefficients(x, y, 4)
    
    # 使用傅里叶系数绘制曲线
    x_approx, y_approx = gen_curve(an, bn, cn, dn, 2, 8)
    #theta_fine = np.linspace(0, 2*np.pi, 1000)
    #x_approx = sum([an[i]*np.cos(i*theta_fine) + bn[i]*np.sin(i*theta_fine) for i in range(len(an))])
    #y_approx = sum([cn[i]*np.cos(i*theta_fine) + dn[i]*np.sin(i*theta_fine) for i in range(len(cn))])

    # 绘制原始的多边形
    plt.plot(x + [x[0]], y + [y[0]], 'o-', color="gray")    
    plt.plot(x_approx, y_approx)

plt.gca().invert_yaxis()
# 设置x和y轴的范围
plt.xlim(0, 1)
plt.ylim(1, 0)  # 注意，由于我们反转了y轴，所以范围也需要调换

plt.axis('equal')
plt.show()
'''

'''
an, bn, cn, dn = compute_coefficients(x_rotated, y_rotated, terms=2)

x_approx, y_approx = gen_curve(an, bn, cn, dn, 2, 6)

plt.plot(x_approx, y_approx, 'b-')
plt.scatter(x_rotated, y_rotated, color='red')
for i, (xi, yi) in enumerate(zip(x_rotated, y_rotated)):
    plt.text(xi, yi, str(i+1))
    
plt.axis('equal')
plt.show()
'''
