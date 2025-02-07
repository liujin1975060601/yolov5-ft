import os
import numpy as np
from pathlib import Path
import shutil

# path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/dota1.5/val500'

# img_list = np.array([f'{path}/images/{x}' for x in os.listdir(path + '/images') if os.path.splitext(x)[-1] in ['.bmp', '.jpg', '.png']])
# indices = np.random.choice(np.arange(len(img_list)), 500, replace=False)
# img_list = img_list[indices].tolist()
# with open(os.path.join(path, 'val_dota.txt'), 'w') as fp:
#     fp.write('\n'.join(img_list))


path = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/dota1.5/val500/val_dota.txt'
save = '/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b3612381/datas/dota1.5/val500/images_500'
with open(path, 'r') as fp:
    files = fp.read().splitlines()
save = Path(save)
for file in files:
    p = Path(file)
    shutil.copy(p, save / p.name)
print('Done')