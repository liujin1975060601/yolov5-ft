from pathlib import Path
import glob
import os

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
path = '/home/user/datas/dota1.5/patches/images'
prefix = 'Test'
try:
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # f = list(p.rglob('*.*'))  # pathlib
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
    img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    # img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
    assert img_files, f'{prefix}No images found'
except Exception as e:
    raise Exception(f'{prefix}Error loading data from {path}')