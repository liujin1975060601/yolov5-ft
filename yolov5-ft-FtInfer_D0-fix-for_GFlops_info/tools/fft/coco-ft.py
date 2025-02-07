# import json
import ujson as json
import copy
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from itertools import repeat


def compute_coefficients_interp(xy, terms=2, interp=True, return_list=False):
    x = np.array(xy[0::2])
    y = np.array(xy[1::2])
    if interp:
        x = np.concatenate([x, [x[0]]], dtype=np.float32)
        y = np.concatenate([y, [y[0]]], dtype=np.float32)
        ori = np.linspace(0, 1, x.shape[0], endpoint=True)
        gap = np.linspace(0, 1, terms*2, endpoint=False)
        x = np.interp(gap, ori, x)
        y = np.interp(gap, ori, y)
    N = len(x)
    t = np.linspace(0, 2*np.pi, N, endpoint=False)  # t = t*2pi/n
    a0 = 1./N * sum(x)
    c0 = 1./N * sum(y)
    
    an, bn, cn, dn = [np.zeros(1 + terms) for i in range(4)]
    
    for k in range(1, (N // 2) + 1):    # 1,2,...,int(N/2)
        if k > terms:
            break
        an[k] = 2./N * sum(x * np.cos(k*t))
        bn[k] = 2./N * sum(x * np.sin(k*t))
        cn[k] = 2./N * sum(y * np.cos(k*t))
        dn[k] = 2./N * sum(y * np.sin(k*t))
    if return_list:
        list_coef = [a0, c0]
        for k in range(1, an.shape[0]):
            list_coef.append(an[k])
            list_coef.append(bn[k])
            list_coef.append(cn[k])
            list_coef.append(dn[k])
        return list_coef    # a0, c0, a1, b1, c1, d1, ... ak, bn, ck, dk
    an[0] = a0
    cn[0] = c0
    return (an,bn,cn,dn), (x,y)


# dict image_id {image name, w,h,cls, xywh, segments}
def read_json(coco_json):
    with open(coco_json, 'r') as fp:
        coco_dict = json.load(fp)
    r_dict = {}
    images = coco_dict['images']
    for image in images:
        r_dict[image['id']] = {
            'file_name': image['file_name'],
            'height': image['height'],
            'width': image['width'],
            'bbox': [], # xywh, xy 为中心点
            'segmentation': [],
            'categories': []    # 0~...
        }
    annotations = coco_dict['annotations']
    categories = coco_dict['categories']
    cls_ = {}
    names = []
    for i, category in enumerate(categories):
        cls_[category['id']] = i
        names.append(category['name'])
    for label in tqdm(annotations):
        # coco json bbox xywh, xy 为左上角
        r_dict[label['image_id']]['bbox'].append(np.array(label['bbox'], dtype=np.float32))
        r_dict[label['image_id']]['bbox'][-1][0] += r_dict[label['image_id']]['bbox'][-1][2] / 2
        r_dict[label['image_id']]['bbox'][-1][1] += r_dict[label['image_id']]['bbox'][-1][3] / 2
        r_dict[label['image_id']]['segmentation'].append(copy.deepcopy(label['segmentation']))
        r_dict[label['image_id']]['categories'].append(copy.deepcopy(cls_[label['category_id']]))
    return r_dict, names

def show_segment_label(coco_path, r_dict, name=None):
    coco_path = Path(coco_path)        
    colors = [
        (54, 67, 244),
        # (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121),
        (180, 105, 255)]
    cou = 0
    for k, v in tqdm(r_dict.items()):
    # for _ in range(1):
    #     k = '414673'
    #     v = r_dict[k]
        file = coco_path / v['file_name']
        # if file.stem[-6:] == '414673':
        #     print()
        flag = False
        i = 0
        seg0 = None
        mask = None
        for i, seg in enumerate(v['segmentation']):
            # if isinstance(seg, list):
            #     if len(seg) >= 2:
            #         seg0 = seg
            #         flag = True
            #         break
            if isinstance(seg, dict):
                size = seg['size']
                counts = seg['counts']
                mask = np.zeros(size[0]*size[1]).astype(np.uint8)
                start = 0
                for j, c in enumerate(counts):
                    mask[start:start + c] = 255 * (j % 2)
                    start += c
                mask = mask.reshape([size[1], size[0]]).T
                # _, mask0 = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                flag = True
                seg0 = contours
                break

        if not file.exists():
            continue
        if flag:
            img = cv2.imread(str(file))
            for i, seg in enumerate(v['segmentation']):
                flag_rle = False
                if isinstance(seg, list):
                    if len(seg) >= 2:
                        seg0 = seg
                    else:
                        continue
                if isinstance(seg, dict):
                    size = seg['size']
                    counts = seg['counts']
                    mask = np.zeros(size[0]*size[1]).astype(np.uint8)
                    start = 0
                    for j, c in enumerate(counts):
                        mask[start:start + c] = 255 * (j % 2)
                        start += c
                    mask = mask.reshape([size[1], size[0]]).T
                    # _, mask0 = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    flag_rle = True
                    seg0 = contours
                    # seg0 = np.concatenate(contours, axis=0).reshape(-1)

                bbox = v['bbox'][i]
                cls_ = v['categories'][i]
                c1 = [int(bbox[0] - bbox[2] / 2),
                    int(bbox[1] - bbox[3] / 2)
                ]
                c2 = [int(bbox[0] + bbox[2] / 2),
                    int(bbox[1] + bbox[3] / 2),
                ]
                if flag_rle:
                    empty = np.ones_like(img) * (98, 9, 11) * (mask[..., None] > 0)
                    img = empty*0.9 + img
                else:
                    for j, seg in enumerate(seg0):
                        seg = np.array(seg).reshape(-1, 2).astype(int)
                        cv2.polylines(img, [seg], isClosed=True, color=colors[cls_ % len(colors)], thickness=2)
                segm = np.concatenate(seg0, axis=0).astype(int).reshape(-1, 2)
                hull = cv2.convexHull(segm)
                cv2.polylines(img, [hull], isClosed=True, color=colors[(cls_+1) % len(colors)], thickness=2)
                if name is not None:
                    cv2.putText(img, name[cls_], c1, 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
                cv2.rectangle(img, c1, c2, [0,0,255], thickness=3)
            cv2.imwrite(f'{file.stem}-test.jpg', img)
            cou += 1
            if cou >= 5:
                return
            # cv2.imwrite(f'{file.stem}-empty.jpg', empty)
            # cv2.imwrite(f'{file.stem}-mask.jpg', mask)
            # return

def save_labels_single(args):
    v, save_path, normalize, terms = args
    p = save_path / v['file_name']
    txt_labels = []
    chft_labels = []
    w, h = v['width'], v['height']
    for bbox, cls_, seg in zip(v['bbox'], v['categories'], v['segmentation']):
        if isinstance(seg, list):
            pass
        elif isinstance(seg, dict):
            size = seg['size']
            counts = seg['counts']
            mask = np.zeros(size[0]*size[1]).astype(np.uint8)
            start = 0
            for j, c in enumerate(counts):
                mask[start:start + c] = 255 * (j % 2)
                start += c
            mask = mask.reshape([size[1], size[0]]).T
            seg, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segm = np.concatenate(seg, axis=0).astype(int).reshape(-1, 2)            
        hull = cv2.convexHull(segm).reshape(-1, 2)
        if normalize:
            bbox = (np.array(bbox).reshape(-1, 2) / np.array([w, h])).astype(np.float32).reshape(-1)
            hull = (hull / np.array([w, h])).astype(np.float32).reshape(-1)
        else:
            bbox = np.array(bbox).reshape(-1)
            hull = hull.reshape(-1)
            
        if terms > 0:
            hull = np.array(compute_coefficients_interp(hull, terms=terms, return_list=True)).astype(np.float32)
            
        bbox = [f'{x:.8f}' for x in bbox]
        hull = [f'{x:.8f}' for x in hull]
        txt_labels.append(f'{int(cls_)} {" ".join(bbox)}')
        chft_labels.append(f'{int(cls_)} {" ".join(hull)}')
    with open(p.with_suffix('.txt'), 'w') as fp:
        fp.write('\n'.join(txt_labels))
    if terms > 0:
        with open(p.with_suffix('.ft'), 'w') as fp:
            fp.write('\n'.join(chft_labels))
    else:
        with open(p.with_suffix('.ch'), 'w') as fp:
            fp.write('\n'.join(chft_labels))

def save_labels(r_dict, save_path, normalize=True, terms=32):
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    with Pool(5) as pool:
        pbar = pool.imap(save_labels_single, zip(r_dict.values(), repeat(save_path), repeat(normalize), repeat(terms)))
        for i in tqdm(pbar):
            pass


def save_ft_from_ch_single(args):
    file, terms = args
    with open(file, 'r') as fp:
        chs = [l.split() for l in fp.read().splitlines()]
    ft_labels = []
    for ch in chs:
        cls_ = ch[0]
        xy = np.array(ch[1:]).astype(np.float32)
        coef = [str(c) for c in compute_coefficients_interp(xy, terms=terms, return_list=True)]
        ft_labels.append(f'{int(cls_)} {" ".join(coef)}')
    with open(file.with_suffix('.ft'), 'w') as fp:
        fp.write('\n'.join(ft_labels))

def save_ft_from_ch(path, terms=4):
    path = Path(path)
    files = [i for i in path.rglob('*.ch')]
    with Pool(5) as pool:
        pbar = pool.imap(save_ft_from_ch_single, zip(files, repeat(terms)))
        for i in tqdm(pbar):
            pass

def show_ft(img_path, label_path, terms=-1):
    colors = [
        (54, 67, 244),
        # (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121),
        (180, 105, 255)
    ]
    img_path = Path(img_path)
    label_path = Path(label_path)
    for file in img_path.rglob('*'):
        if file.suffix.lower() in ['.jpg', '.png', '.bmp']:
            ft = label_path / file.with_suffix('.ft').name
            if ft.exists():
                image = cv2.imread(str(file))
                h, w, _ = image.shape
                with open(ft, 'r') as fp:
                    fts = [l.split() for l in fp.read().splitlines()]
                ch = ft.with_suffix('.ch')
                if ch.exists():
                    with open(ch, 'r') as fp:
                        hulls = [l.split()[1:] for l in fp.read().splitlines()]
                    hulls = [(np.array(hull).astype(np.float32).reshape([-1, 2]) * np.array([w, h]).reshape(-1, 2)).astype(np.int32) for hull in hulls]
                else:
                    hulls = None
                fts = np.array(fts).astype(np.float32)
                cls_ = fts[:, 0]
                coefs = fts[:, 1:]
                c_num = (coefs.shape[-1] - 2) // 4 + 1  # c_num = k + 1, a0~ak
                if terms != -1:
                    c_num = min(terms + 1, c_num)
                theta_fine = np.linspace(0, 2*np.pi, 100)
                for j in range(fts.shape[0]):
                    xy = coefs[j, :2]
                    coef = coefs[j, 2:]
                    an,bn,cn,dn = np.split(coef.reshape([-1, 4]), 4, axis=-1)
                    x_approx = sum([an[i-1]*np.cos(i*theta_fine) + bn[i-1]*np.sin(i*theta_fine) for i in range(1, c_num)])
                    y_approx = sum([cn[i-1]*np.cos(i*theta_fine) + dn[i-1]*np.sin(i*theta_fine) for i in range(1, c_num)])
                    x_approx += xy[0]
                    y_approx += xy[1]
                    xy2 = np.vstack([x_approx*w, y_approx*h]).T.astype(np.int32)
                    cv2.polylines(image, [xy2], isClosed=True, color=colors[int(cls_[j])%len(colors)], thickness=3)                 
                    cv2.imshow("Contours", image)
                    k = False
                    kk = True
                    while not k:
                        key = (cv2.waitKey(100) & 0xFF)
                        if key == 27:
                            k = True
                        if key == ord('w') or key == ord('W'):
                            if hulls is not None and kk:
                                cv2.polylines(image, [hulls[j]], isClosed=True, color=colors[int(cls_[j]+1)%len(colors)], thickness=3)             
                                cv2.imshow("Contours", image)
                                kk = False
                            else:
                                k = True
                        if key == ord('q') or key == ord('Q'):
                            cv2.destroyAllWindows()
                            return
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 4090
    # coco_root = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/coco2017labels'
    # coco_path = coco_root + '/coco/images'
    # labels_path = coco_root + '/coco/labels'
    # coco_json = coco_root + '/coco/annotations'
    # 4090-2
    # coco_root = '/home/liu/datas/coco2017'
    # coco_path = coco_root + '/train/images'
    # labels_path = coco_root + '/train/labels'
    # coco_json = coco_root + '/annotations'
    # CatBug
    coco_root = 'E:/datas/coco2017'
    coco_path = coco_root + '/val/images'
    labels_path = coco_root + '/val/labels'
    coco_json = coco_root + '/annotations'
    
    coco_path = Path(coco_path)
    labels_path = Path(labels_path)
    coco_json = Path(coco_json)
    for js, path in zip(['instances_train2017.json', 'instances_val2017.json'],['train2017', 'val2017']):
        if 1:
            r_dict, name = read_json(coco_json / js)
            save_labels(r_dict=r_dict,
                        save_path=labels_path / path,
                        terms=32)
        else:
        # show_segment_label(coco_path, r_dict, name)
        # save_ft_from_ch(path=r'E:\PyCharmProject\haitun\liu\labels',
        #                 terms=32)
            show_ft(img_path=coco_path / path,
                    label_path=labels_path / path,
                    terms=10)

    print('Done')
