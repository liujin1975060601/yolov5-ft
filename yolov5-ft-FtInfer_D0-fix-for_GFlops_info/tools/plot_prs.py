import os
import sys
import importlib
from pathlib import Path
def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]
    
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError: # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__) # won't be needed after that

if __name__ == '__main__' and not  __package__:
    import_parents(1)

from pathlib import Path
from utils.metrics import plot_pr_curves
import os
import pickle

import pandas as pd

if __name__ == "__main__":
    brow_path = r'runs\dota'
    #r'D:\Articles\9DPose\experiments\carview'
    if brow_path!='' and os.path.exists(brow_path):
        # 根目录路径
        base_path = Path(brow_path)
        # 遍历base_path下的所有子文件夹
        subfolders = [f for f in base_path.glob('*/') if f.is_dir()]
        # 将子文件夹路径转换为字符串列表
        #file_paths = [str(Path('.') / folder) for folder in subfolders if not '#' in str(folder)]
        #file_paths = [str(Path('.') / folder) for folder in subfolders if not '#' in str(folder) and os.path.exists(str(Path('.') / folder / 'val/status.pkl'))]
        file_paths=[]
        for folder in subfolders:
            if not '#' in str(folder):
                model_status = os.path.join('.',folder,'status.pkl')
                val_status = os.path.join('.',folder,'val/status.pkl')
                if os.path.exists(val_status) or os.path.exists(model_status):
                    file_paths.append(str(Path('.') / folder))
    else:
        file_paths = ['runs/train/kitti-2/bs=8_56.49/val',\
                      'runs/train/kitti-2/bs=16_61.78/val']
    pys,aps=[],[]
    names=[]
    methods=[]
    for i,path in enumerate(file_paths):
        # 从文件加载
        model_path = file_paths[i]
        if os.path.exists(model_path):
            val_path = model_path + '/val'
            pkf = val_path + '/status.pkl'
            if not os.path.exists(pkf):
                val_path = model_path
                pkf = val_path + '/status.pkl'
            bias_name = val_path+'/bias.txt'
            
            if os.path.exists(bias_name):
                with open(bias_name, 'r', encoding='utf-8') as file:
                    # 读取第一行
                    method_name = file.readline().strip()
            else:
                method_name = os.path.basename(file_paths[i])

            assert Path(pkf).is_file()
            with open(pkf, 'rb') as f:
                [py,ap,_] = pickle.load(f)
            pys.append(py)
            aps.append(ap)
            methods.append(method_name)
            #
            #excel read names..
            csv_name = val_path + '/classes_map.csv'
            if len(names)==0 and os.path.exists(csv_name):
                # 读取Excel文件
                df = pd.read_csv(csv_name)
                # 提取第一列的字符串形成names列表
                names = df['Class'].tolist()
                assert names[0]=='all'
                names = names[1:]
                # 打印names列表
                print(names)
        else:
            print(f'\033[91m{model_path} not found.\033[0m')
    assert(len(aps)==len(pys))
    #
    if(len(aps)):
        prs_path = brow_path if os.path.exists(brow_path) else './runs/val'
        if not os.path.exists(prs_path):
            os.mkdir(prs_path)
        plot_pr_curves(pys,aps, prs_path, names=names, methods=methods)
    else:
        print(f'\033[91mlen(aps)={len(aps)} aps not found.\033[0m')

