# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import enum
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.spatial import distance
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
import sys

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness, ft2box, ft2pts
from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast,box2poly

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


# add
def pts2dir(pts, fold_angle=1):  # n * 8
    n = pts.shape[0]
    if n:
        xf, yf = (pts[:, 0] + pts[:, 2]) / 2, (pts[:, 1] + pts[:, 3]) / 2
        xb, yb = (pts[:, 4] + pts[:, 6]) / 2, (pts[:, 5] + pts[:, 7]) / 2
        dx = xf - xb
        dy = yf - yb
        L2 = dx ** 2 + dy ** 2
        L = torch.sqrt(L2)
        # 方向
        cos_t = dx / L
        sin_t = dy / L

        # 目标自身的宽高
        cx = pts[:, ::2].sum(dim=1) / 4
        cy = pts[:, 1::2].sum(dim=1) / 4
        # cx = (pts[:, 0] + pts[:, 2] + pts[:, 4] + pts[:, 6]) / 4
        # cy = (pts[:, 1] + pts[:, 3] + pts[:, 5] + pts[:, 7]) / 4
        txy = torch.zeros_like(pts, dtype=torch.float)
        px = pts[:, ::2] - cx[:, None]
        py = pts[:, 1::2] - cy[:, None]
        txy[:, ::2] = cos_t[:, None] * px + sin_t[:, None] * py
        txy[:, 1::2] = -sin_t[:, None] * px + cos_t[:, None] * py
        lx = (txy[:, 4] + txy[:, 6]) / 2
        rx = (txy[:, 0] + txy[:, 2]) / 2
        a = (rx - lx) / 2
        ty = (txy[:, 1] + txy[:, 7]) / 2
        by = (txy[:, 3] + txy[:, 5]) / 2
        b = (by - ty) / 2
        # assert any(a) > 0
        # assert any(b) > 0
        for i in range(n):
            if a[i] < 0:
                a[i] *= -1
            if b[i] < 0:
                b[i] *= -1
        
        if fold_angle == 2:
            cos_2t = 2 * (cos_t ** 2) - 1
            sin_2t = 2 * sin_t * cos_t
            cos_t = cos_2t
            sin_t = sin_2t
        
        out = torch.stack((cos_t, sin_t, a, b), dim=1)
    else:
        l = np.zeros((0, 4), dtype=np.float32)
        out = torch.from_numpy(l).to(pts.device)
    return out
def dirab2WH(dirab):
    assert(dirab.shape[1]==4)
    cos_t, sin_t, a,b = dirab[:,0],dirab[:,1],dirab[:,2],dirab[:,3]
    acos,bsin = a*cos_t, b*sin_t
    acos2,bsin2=acos*acos,bsin*bsin
    asin,bcos = a*sin_t, b*cos_t
    asin2,bcos2=asin*asin,bcos*bcos
    return 2*torch.sqrt(acos2+bsin2), 2*torch.sqrt(asin2+bcos2)


def sortpts_clockwise(A):
    # Sort A based on Y(col-2) coordinates
    sortedAc2 = A[np.argsort(A[:,1]),:]

    # Get top two and bottom two points
    top2 = sortedAc2[0:2,:]
    bottom2 = sortedAc2[2:,:]

    # Sort top2 points to have the first row as the top-left one
    sortedtop2c1 = top2[np.argsort(top2[:,0]),:]
    top_left = sortedtop2c1[0,:]

    # Use top left point as pivot & calculate sq-euclidean dist against
    # bottom2 points & thus get bottom-right, bottom-left sequentially
    sqdists = distance.cdist(top_left[None], bottom2, 'sqeuclidean')
    rest2 = bottom2[np.argsort(np.max(sqdists,0))[::-1],:]

    # Concatenate all these points for the final output
    return np.concatenate((sortedtop2c1,rest2),axis =0)



def show_heatmap(feature):
    heatmap = feature.sum(0)/ feature.shape[0]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # heatmap = 1.0 - heatmap # 也可以不写，就是蓝色红色互换的作用
    heatmap = cv2.resize(heatmap, (224,224)) # (224,224)指的是图像的size，需要resize到原图大小
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite('a.png', heatmap)


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Profile results: {time.time() - self.start:.5f}s')


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(name, opt):
    # Print argparser arguments
    LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''
def get_latest_run_exp(search_dir='.'):
    if(os.path.exists(search_dir)):
        folders = [f for f in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, f)) and f.startswith('exp')]
        #last_list = sorted(folders, key=lambda x: os.path.getctime(os.path.join(search_dir, x)), reverse=True)
        sorted_subfolders = sorted(folders, key=lambda x: os.path.getmtime(os.path.join(search_dir, x)), reverse=True)
        return os.path.join(search_dir, sorted_subfolders[0]) if folders else ''
    else:
        return ''

def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except OSError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def is_docker():
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_pip():
    # Is file in a pip package?
    return 'site-packages' in Path(__file__).resolve().parts


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
@WorkingDirectory(ROOT)
def check_git_status():
    # Recommend 'git pull' if code is out of date
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    print(colorstr('github: '), end='')
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg
    assert not is_docker(), 'skipping check (Docker image)' + msg
    assert check_online(), 'skipping check (offline)' + msg

    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f'up to date with {url} ✅'
    print(emojis(s))  # emoji-safe


def check_python(minimum='3.6.2'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result


@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            print(f'Found {url} locally at {file}')  # file already exists
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    try:
        if isinstance(data, (str, Path)):
            with open(data, errors='ignore') as f:
                data = yaml.safe_load(f)  # dictionary
    except Exception as e:
        print(f'\033[91mText error in {data}.\033[0m')

    # Parse yaml
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path
    if(not os.path.exists(path)):
        print(f'\033[91m{path} not exists.\033[0m')
        sys.exit()
    for k in 'train', 'val', 'test','val_big':
        if data.get(k):  # prepend path
            #data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]
            if isinstance(data[k], str):
                data[k] = str(path / data[k])
                # 将路径拆分成目录和文件名部分
                data_path, basename = os.path.split(data[k])
                # 将"images"替换为"labels"
                lables_path = os.path.join(data_path, "labels")
                if not os.path.exists(lables_path):
                    print(f'\033[91m{k} path:{lables_path} not exists.\033[0m')
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
                for x in data[k]:
                    if not os.path.exists(x):
                        print(f'\033[91m{k} path:{x} not exists.\033[0m')

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # download script
                root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    ZipFile(f).extractall(path=root)  # unzip
                    Path(f).unlink()  # remove zip
                    r = None  # success
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('Dataset not found.')

    # add for check cms_config.json, search in train dir
    if Path.exists(Path(train).parent.resolve() / 'cms_config.json'):
         with open(Path(train).parent.resolve() / 'cms_config.json', errors='ignore') as f:
            import json
            data['cms_config'] = json.load(f)
    else:
        print('No cms_config or train path is not a directory')
        data['cms_config'] = None

    return data  # dictionary


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
    return file


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    #labels[nimage][nt,1(cls)+4(box)+4(pts)*2]
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    #labels[total_objects,1(cls)+4(box)+4(pts)*2] 只统计目标类别，与图片无关
    classes = labels[:, 0].astype(np.int32)  # labels = [class xywh]
    #classes[total_objects]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class
    #weights[nc]

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1  无样本的类置为1个样本，避免下面/0错误
    weights = 1 / weights  # number of targets per class  类权重与类数量成反比
    weights /= weights.sum()  # normalize 确保总和=1
    return torch.from_numpy(weights) #array转tensor

def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80),master_mask=None,slave_rate=1.0):
    # Produces image weights based on class_weights and image contents
    #labels[nimages][nt,1(cls)+4(box)+4(pts)*2]
    #class_counts = np.array([np.bincount(x[:, 0].astype(np.int32), minlength=nc) for x in labels])
    # 计算每个图像的类别计数
    class_counts = np.array([
        np.bincount(
            np.where((x[:, 0] >= 0) & (x[:, 0] < nc), x[:, 0].astype(np.int32), 0),  # 超出范围的值设为 -1
            minlength=nc
        ) for x in labels
    ])
    #class_counts[nimages,nc]
    # 初始化权重矩阵
    if master_mask is not None:
        weights = np.zeros_like(class_counts, dtype=np.float32)
        assert len(master_mask) == len(weights), "master_mask 和 weights 的长度不匹配"
        weights[master_mask] = (class_weights.reshape(1, nc) * class_counts[master_mask])
        #
        image_weights = np.zeros(len(labels), dtype=np.float32)
        image_weights[master_mask] = weights[master_mask].sum(1)
        #
        master_sum = image_weights[master_mask].sum()
        slave_sum = slave_rate * master_sum
        master_count = master_mask.sum()
        slave_count = len(master_mask) - master_count
        # slave_rate = slave_count/master_count
        #weights[~master_mask] = slave_sum / slave_count
        #average_class_weight = class_weights.mean() #np.median(class_weights)#
        #weights[~master_mask] = slave_rate * average_class_weight * class_counts[~master_mask]
        #class_weights[nc]和class_counts[nimages,nc]的每一行做乘法，然后按行求和.sum(1)-->image_weights[nimages]
        objn = class_counts[~master_mask].sum(1) #objn[nimg]
        assert objn.shape[0]==slave_count
        obj_total = objn.sum()
        image_weights[~master_mask] = slave_sum * objn / obj_total
        #image_weights = weights.sum(1)
        #
        if 0:
            pos_w = image_weights[master_mask].sum()
            neg_w = image_weights[~master_mask].sum()
    else:
        image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    #image_weights[nimages]
    # index = random.choices(range(nimages), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y

    # add pts
    y[:, 4::2] = w * x[:, 4::2] + padw
    y[:, 5::2] = h * x[:, 5::2] + padh
    return y

def ftnorm2ft(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * y[:, 0] + padw    # a0
    y[:, 1] = h * y[:, 1] + padh  # c0
    # y[:, 2::4] *= w
    # y[:, 3::4] *= w
    # y[:, 4::4] *= h
    # y[:, 5::4] *= h
    y[:, 2:] *= torch.Tensor([w, w, h, h]).repeat((y.shape[-1] - 2) // 4) if isinstance(x, torch.Tensor) else np.array([w,w,h,h]).reshape(1, -1).repeat((y.shape[-1] - 2) // 4, axis=0).reshape(-1)
    return y

def ft2ftnorm(x, w=640, h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] /= w  # a0
    y[:, 1] /= h  # c0
    y[:, 2:] /= torch.Tensor([w, w, h, h]).repeat((y.shape[-1] - 2) // 4) if isinstance(x, torch.Tensor) else np.array([w,w,h,h]).reshape(1, -1).repeat((y.shape[-1] - 2) // 4, axis=0).reshape(-1)
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    # add
    y[:, 4::2] = x[:, 4::2] / w
    y[:, 5::2] = x[:, 5::2] / h

    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def scale_coords_ft(img1_shape, coords_ft, img0_shape, ratio_pad=None):
    # Rescale coords_ft (xyxy) from img1_shape to img0_shape
    # ft_   img1: 640 * 320, img0: 1280 * 640
    # coords_ft -> [a0,c0,a,b,c,d...]   a0,c0->center   img1
    if ratio_pad is None:  # calculate from img0_shape
        # gain min(0.5, 0.5)
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # pad 0,0
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords_ft[:, 0] -= pad[0]  # x padding
    coords_ft[:, 1] -= pad[1]  # y padding
    coords_ft /= gain#
    #coords_ft[2:4]*=1      x
    #coords_ft[4:6]*=gain   y
    return coords_ft

def scale_coords_poly(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, ::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords[:, :8] /= gain
    return coords




def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False):
    """Runs Non-Maximum Suppression (NMS) on inference results
    prediction: [batch, all_grid, 5+nc]
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """      
    # output indices  
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()

    nc = prediction.shape[2] - 5  # number of classes
    if isinstance(conf_thres, float):
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4] > conf_thres.min()  # candidates

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        grid = indices_grid[xc[xi]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # x[nobj,4(xywh)+1(obj)+cls]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # x[:,:4(xywh)] -> box[nobj,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:#
            i, j = (x[:, 5:] > conf_thres[None]).nonzero(as_tuple=False).T #i[nobj]目标行号, j[nobj]类列号
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            grid = grid[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            filter_conf = (conf > conf_thres[j]).view(-1)
            x = torch.cat((box, conf, j.float()), 1)[filter_conf]
            grid = grid[filter_conf]

        # Filter by class
        if classes is not None:
            grid = grid[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            grid = grid[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output, output_indices if return_indices else output


def non_max_suppression_ft(prediction, prediction_ft, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False):
    """Runs Non-Maximum Suppression (NMS) on inference results
    prediction: [batch, all_grid, 5+nc]ft
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """      
    # output indices  
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long() #indices_grid[nt0]


    nc = prediction.shape[2] - 5  # number of classes
    if isinstance(conf_thres, float):
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4] > conf_thres.min()  # candidates

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_ft = [torch.zeros((0, prediction_ft.shape[-1]), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    #prediction[batch,nt,5+c]  prediction_ft[batch,nt,2+4*term]
    for xi, (x,x_ft) in enumerate(zip(prediction, prediction_ft)):  # image index, image inference
        #x[nt0,5+c]  x_ft[nt0,2+4*term]  (xc[xi])[nt0]
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]# x[nt0,5+c]-->x[nt,5+c]
        x_ft = x_ft[xc[xi]]# x_ft[nt0,2+4*term]-->x_ft[nt,2+4*term]
        grid = indices_grid[xc[xi]] #indices_grid[nt]->grid[nt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf  x[nt,5+c]
        # x[nobj,4(xywh)+1(obj)+cls]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # x[:, :4(xywh)] -> box[nobj,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres[None]).nonzero(as_tuple=False).T #i[nobj]目标行号, j[nobj]类列号
            #依据前面的x[i, j + 5, None]，x[i, 5 + j, None]表示乘到第j列的class可信度
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float()), 1)#x[nobj,4(xyxy)+1(obj)+cls]->x[nobj,6=4(xyxy)+1(conf)+1(cls)
            x_ft = x_ft[i]
            grid = grid[i]
        else:  # best class only
            # x[nobj,4(xywh)+1(obj)+1(cls)]
            conf, j = x[:, 5:].max(1, keepdim=True)#选择class输出最大的conf作为最终可信度，j作为识别类标签
            # x[:, 5:]shape = [nobj,cls]
            # conf[nobj,1]
            # j[nobj,1]
            obj_filt = (conf > conf_thres[j]).view(-1) #class的可信度再过滤一轮
            x = torch.cat((box, conf, j.float()), 1)[obj_filt]  #后面x[:, 5:6]表示类id
            #x[nobj_filt_cls,6==4(box)+1(conf)+1(j.float()==cls)]  在1号维度拼起来[box+conf+cls]，再经过cls的conf过滤
            x_ft = x_ft[obj_filt]#x_ft[nt,2+4*term]->x_ft[nobj_filt_cls,2+4*term]  x_ft use the same filter as x
            grid = grid[obj_filt]#grid[nt]->grid[nobj_filt_cls]

        # Filter by class
        if classes is not None:#classes非空的话，就只检测classes所包含的类
            clsid = x[:, 5:6]
            grid = grid[(clsid == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(clsid == torch.tensor(classes, device=x.device)).any(1)]
            x_ft = x_ft[(clsid == torch.tensor(classes, device=x.device)).any(1)]
        # x[n,6=4(xywh)+1(obj)+1(cls)]
        # x_ft[n,2+4*term]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # delete small score boxes
            grid = grid[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x_ft = x_ft[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # x[n,6=4(xywh)+1(obj)+1(cls)]
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh) #c[n,1]  classes  c[nt] as class id

        # use ft_predict iou
        # boxes, scores = ft2box(x_ft) + c.cpu().numpy().astype(np.float64), x[:, 4]  
        polys, scores = (ft2pts(x_ft) + c).cpu().numpy().astype(np.float64), x[:, 4] #x_ft[n2,2+4*term]->polys[n2,8] x[n2,4]->scores[n2]

        polys2 = np.concatenate([polys, scores.cpu().numpy().astype(np.float64).reshape(-1, 1)], axis=-1)#polys2[n2,9=8(polys)+1(conf)]
        i = np.array(py_cpu_nms_poly_fast(polys2, iou_thres)) #i[n3] reserved objs

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output_indices[xi] = grid[i]
        output[xi] = x[i] #x[n,6]->x[i][n2,6]
        output_ft[xi] = x_ft[i] #x_ft[n,2+4*term]->x_ft[i][n2,2+4*term]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    # return (output, output_ft, output_indices) if return_indices else (output, output_ft)
    return (output, output_ft, output_indices) if return_indices else (output, output_ft)


def normalize_dir(q2):
    """
    q2: n * 2
    """
    L = torch.sqrt(q2[:, 0] ** 2 + q2[:, 1] ** 2)
    q2 = q2 / L[:, None]
    return q2

def ab2pts(ab, cxy, q2):
    n = ab.shape[0]
    pts = torch.zeros((n, 8), dtype=torch.float, device=ab.device)
    pts[:, 0] = cxy[:, 0] + q2[:, 0] * ab[:, 0] - q2[:, 1] * (-(ab[:, 1]))
    pts[:, 1] = cxy[:, 1] + q2[:, 1] * ab[:, 0] + q2[:, 0] * (-(ab[:, 1]))

    pts[:, 2] = cxy[:, 0] + q2[:, 0] * ab[:, 0] - q2[:, 1] * ab[:, 1]
    pts[:, 3] = cxy[:, 1] + q2[:, 1] * ab[:, 0] + q2[:, 0] * ab[:, 1]

    pts[:, 4] = cxy[:, 0] + q2[:, 0] * (-ab[:, 0]) - q2[:, 1] * ab[:, 1]
    pts[:, 5] = cxy[:, 1] + q2[:, 1] * (-ab[:, 0]) + q2[:, 0] * ab[:, 1]

    pts[:, 6] = cxy[:, 0] + q2[:, 0] * (-ab[:, 0]) - q2[:, 1] * (-(ab[:, 1]))
    pts[:, 7] = cxy[:, 1] + q2[:, 1] * (-ab[:, 0]) + q2[:, 0] * (-(ab[:, 1]))
    return pts


def two_fold_angle(q2):
    # t1 = np.arcsin(q2[:, 1])
    # t2 = np.arccos(q2[:, 0])
    factor = (q2[:, 1] > 0).float()
    factor[factor == 0. ] = -1.
    cos_t = torch.sqrt((1 + q2[:, 0]) / 2)
    sin_t = torch.sqrt((1 - q2[:, 0]) / 2 )
    sin_t *= factor
    q2 = torch.stack((cos_t, sin_t), dim=1)
    return q2




# def rot_nms(prediction, conf_thres=0.25, iou_thres=0.3, ab_thres=3.0, fold_angle=2):
#     """Runs Non-Maximum Suppression (NMS) on inference results
#     """
#     assert len(prediction) == 2
#     p1 = prediction[0]  # Detect
#     p2 = prediction[1]  # Detect2
    
#     nc = p1.shape[2] - 5  # number of classes
#     # 根据conf过滤p1
#     xc = p1[..., 4] > conf_thres  # candidates
#     # yc = p2[..., 0] > pab_thres

#     # Checks
#     assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#     assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

#     # Settings
#     time_limit = 10.0  # seconds to quit after

#     t = time.time()
#     output = [torch.zeros((0, 10), device=p1.device)] * p1.shape[0]
#     for xi, (x, y) in enumerate(zip(p1, p2)):  # image index, image inference

#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         # 根据水平框的conf筛选
#         x = x[xc[xi]]  # confidence
#         y = y[xc[xi]]

#         # 根据ab的conf筛选
#         # x = x[yc[xi]]  
#         # y = y[yc[xi]]

#         # 根据ab过滤
#         yab = (y[..., 3] > ab_thres) & (y[..., 4] > ab_thres)
#         y = y[yab]
#         x = x[yab]

#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

#         # 对y的输出进行处理
#         q2 = y[:, 1:3]
#         ab = y[:, 3:5]
#         q2 = normalize_dir(q2)

#         if fold_angle == 2:
#             q2 = two_fold_angle(q2)
#         # ab2pts
#         center_xy = x[:, :2]
#         pts = ab2pts(ab, center_xy, q2)

#         # cls
#         conf, j = x[:, 5:].max(1, keepdim=True)
#         pconf = y[:, 0][:,None]
#         conf = (conf + pconf) / 2
#         x = torch.cat((pts, conf, j.float()), 1)[conf.view(-1) > conf_thres]
#         # x = torch.cat((pts, pconf, j.float()), 1)[pconf.view(-1) > pab_thres]

#         clsses = j.unique()
#         keep = []
#         for clss in clsses:
#             y = x[x[..., 9].long() == clss][..., :9]

#             # Check shape
#             n = y.shape[0]  # number of boxes
#             if not n:  # no boxes
#                 continue
#             y = y.cpu().numpy().astype(np.float64)
#             i = py_cpu_nms_poly_fast(y, iou_thres)  # polygon-NMS
#             keep.extend(i)
#         output[xi] = x[keep]
       
#         # # Filter by class
#         # if classes is not None:
#         #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#         # if (time.time() - t) > time_limit:
#         #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
#             # break  # time limit exceeded

#     return output



def rot_nms(prediction, conf_thres=0.25, iou_thres=0.3, ab_thres=3.0, fold_angle=2, mask_dir=[], threshs=torch.zeros(0)):
    """Runs Non-Maximum Suppression (NMS) on inference results
    """
    # assert len(prediction) == 2
    pYolo = prediction[0]  # Detect  pYolo[nl][b,a,H,W,5+cls]
    pRot = prediction[1]  # Detect2  pRot[nl][b,a,H,W,5]
    nl = len(pYolo)
    batchs = pYolo[0].shape[0]
    nc = pYolo[0].shape[-1] - 5

    # 根据conf过滤
    targets = []
    tinfo = [] # featuremap, batch, anchor x, y
    for i in range(nl):
        # 第 batch_idx个批次
        yolo_map = pYolo[i]#yolo_map[b,na,H,W,5+c]
        for batch_idx in range(batchs):
            batch_data = yolo_map[batch_idx]#[na,H,W,5+c]
            #batch_data[na,H,W,5+c]
            na = batch_data.shape[0]
            for anchor_idx in range(na):
                anchor_data = batch_data[anchor_idx]#[H,W,5+c]
                #anchor_data[H,W,5+c]
                sim_vec = torch.nonzero(anchor_data[..., 4] > conf_thres)
                #sim_vec[nt,2] #取得所有非零元素True的坐标集合
                if len(sim_vec) > 0:
                    for x, y in sim_vec:
                        targets.append(anchor_data[x][y])#[nt][5+c]
                        tinfo.append([i, batch_idx, anchor_idx, x, y])#[nt][i, batch_idx, anchor_idx, x, y]

    output = [torch.zeros((0, 10), device=pYolo[0].device)] * batchs
    batch_targets = [None]*batchs
    for idx, target in enumerate(targets):
        info = tinfo[idx] # featuremap, batch, anchor x, y
        batch = info[1]
        feat = pRot[info[0]][batch].permute(1, 2, 0, 3).contiguous()#[H,W,na,1+2+2=5(dir)]
        anchor_data = feat[info[3]][info[4]]#anchor_data[na,1+2+2=5(dir)]
        #挑选P最大
        idex = anchor_data[:, 0].argmax()
        dir_target = anchor_data[idex]#dir_target[1+2+2=5(dir)]
        obj_merge = torch.cat((target, dir_target)).view(1, -1)#[1,5+c+5(dir)]
        if batch_targets[batch] is None:
            batch_targets[batch] = []
        batch_targets[batch].append(obj_merge)

    #batch_targets[b=0][nt][1,5+c+5(dir)]
    for xi in range(batchs):
        preds = batch_targets[xi]#[nt][1,5+c+5(dir)]
        if not preds:
            continue
        preds = torch.cat(preds, 0)#把preds这个list拼接成高一维度的tensor-->preds[nt,5+c+5(dir)]
        pYolo = preds[:, :5+nc]#[nt,5+nc]
        pRot = preds[:, 5+nc:]#[nt,5=1+2+2]

        # Compute conf
        pYolo[:, 5:] *= pYolo[:, 4:5]  # conf = obj_conf * cls_conf
        # pYolo=pYolo[pYolo[:,4]>conf_thres]#[nt_filt,5+nc]   这里没有意义, 前面已经判断过 obj_conf > conf_thres

        # 根据ab过滤
        if 1:
            mask_ab = (pRot[..., 3] > ab_thres) & (pRot[..., 4] > ab_thres)
            pRot = pRot[mask_ab]
            pYolo = pYolo[mask_ab]
            # If none remain process next image
            if not pYolo.shape[0]:
                continue

        # 对pRot的输出进行处理
        q2 = pRot[:, 1:3]
        #q2[nt_dir,2]
        if 0:
            q2 = normalize_dir(q2)#q2[nt][2]
        else:
            Lq2 = torch.norm(q2, dim=1, keepdim=True)
            q2 = q2 / (Lq2 + 1e-8)  # 避免0除问题
        if fold_angle == 2:
            q2 = two_fold_angle(q2)
        
        # ab2pts
        ab = pRot[:, 3:5]
        center_xy = pYolo[:, :2]#center_xy[nt][2]
        pts = ab2pts(ab, center_xy, q2)#pts[nt][8]

        # cls
        #pYolo[nt_filt,5+nc]
        conf, clsid = pYolo[:, 5:].max(1, keepdim=True)#conf[nt][1]  clsid[nt][1]
        #keepdim=True方便下面dim=1维度进行cat
        pRot = torch.cat((pts, conf, clsid.float()), 1)#,pYolo[:,:4]
        #pRot[nt,10=pts[8]+conf[1]+clsid[1]]
        assert(pRot.shape[0]==pYolo.shape[0])

        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        clsid = clsid.squeeze(1)#clsid[nt][1]-->clsid[nt]
        keep = []
        for clss in clsid.unique():#剔除重复类
            #分类子集进行nms，各类互不干扰
            cls_mask = clsid.long() == clss
            cls_idx = torch.nonzero(cls_mask).squeeze(1)
            if(cls_idx.shape[0]>0):
                if(mask_dir!=[] and mask_dir[clss]>0):
                    pcls = pRot[cls_idx, :9]#pcls[nt_cls,9]
                    #pcls = pRot[pRot[..., 9].long() == clss][..., :9]
                    # Check shape
                    assert(pcls.shape[0] > 0)#boxes number
                    pcls = pcls.cpu().numpy().astype(np.float64)#pcls[nt_cls,9]
                    id_nms = py_cpu_nms_poly_fast(pcls, iou_thres)  # polygon-NMS
                else:
                    # Batched NMS
                    pcls = pYolo[cls_idx]#pcls[nt_cls,5+nc]
                    #sort_idx = pcls[:, 4].argsort(descending=True)[:max_nms]
                    #pcls = pcls[sort_idx]  # sort by confidence
                    boxes, scores = pcls[:, :4], pcls[:, 4]  # boxes (offset by class), scores
                    box2poly(boxes,pRot,cls_idx)
                    #print(pRot[cls_idx])
                    id_nms = torchvision.ops.nms(xywh2xyxy(boxes), scores, iou_thres)  # NMS
                id_nms = cls_idx[id_nms] #注意上面id_nms是针对pcls的编号，需要还原到pYolo的编号，用了cls_idx=torch.nonzero进行原始编号的查询
                keep.extend(id_nms)#列表keep末尾加上列表id_nms
        output[xi] = pRot[torch.tensor(keep)]#用keep这个list对pRot的[nt]维度进行索引过滤

        #output[xi][nt,10=pts[8]+conf[1]+clsid[1]]
        if(threshs.shape[0]>0):
            rot_obj = torch.tensor([]).to('cuda')
            for t in output[xi]:#t[10=pts[8]+conf[1]+clsid[1]]
                if(t[8] > threshs[int(t[9])]):#根据t[8]conf针对不同类进行最终过滤,其中int(t[9])是类编号
                    rot_obj = torch.cat((rot_obj, t.unsqueeze(0)), dim=0)#t[1,10]搜集到集合rot_obj[:,10]
            output[xi] = rot_obj
       
    return output#output[batchs][nt,10=pts[8]+conf[1]+clsid[1]]


def rot_nms_with_cms(prediction, conf_thres=0.25, iou_thres=0.3, ab_thres=3.0, fold_angle=2, mask_dir=[], cms_s=False, ft_infer=False):
    """Runs Non-Maximum Suppression (NMS) on inference results
    """
    # assert len(prediction) == 2
    pYolo = prediction[0]  # Detect  pYolo[nl][b,a,H,W,5+cls]
    pRot = prediction[1]  # Detect2  pRot[nl][b,a,H,W,5]
    pCms = [None] * len(pYolo)
    pFt = [None] * len(prediction[0])
    sv_id = 2
    ft_length = 0
    pcms_flag = False

    nc = pYolo.shape[-1] - 5  # number of classes
    if isinstance(conf_thres, float):
        threshs = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        threshs = conf_thres.to(pYolo.device)
        conf_thres = float(conf_thres.min())

    if ft_infer:
        pFt = prediction[sv_id]
        for i, j in enumerate(pFt):
            pFt[i] = j.view(j.shape[0], -1, j.shape[-1])
        pFt = torch.cat(pFt, 1)
        ft_length = pFt.shape[-1]
        sv_id += 1
    pFt = None if isinstance(pFt, list) else pFt
    if cms_s:
        pCms_s = prediction[sv_id]
        sv_id += 1
        for i in range(len(pCms_s)):
            cms_conf, cms_cls = pCms_s[i].max(-1, keepdim=True)
            pCms[i] = torch.cat([cms_conf, cms_cls], dim=-1)
        pcms_flag = True
    if len(prediction) == sv_id+1:    # cms_s+v pCms_s[nl][b,a,H,W,ns] pCms_v[nl][b,a,H,W,nv]
        pCms_v = prediction[sv_id]
        for i in range(len(pCms_v)):
            pCms[i] = pCms_v[i] if not cms_s else torch.cat([pCms[i], pCms_v[i]], dim=-1)
        pcms_flag = True
    pCms = None if pCms[0] is None else pCms
    nl = len(pYolo)
    batchs = pYolo[0].shape[0]
    nc = pYolo[0].shape[-1] - 5

    # 根据conf过滤
    targets = []
    tinfo = [] # featuremap, batch, anchor x, y
    for i in range(nl):
        # 第 batch_idx个批次
        yolo_map = pYolo[i]#yolo_map[b,na,H,W,5+c]
        for batch_idx in range(batchs):
            batch_data = yolo_map[batch_idx]#[na,H,W,5+c]
            #batch_data[na,H,W,5+c]
            na = batch_data.shape[0]
            for anchor_idx in range(na):
                anchor_data = batch_data[anchor_idx]#[H,W,5+c]
                #anchor_data[H,W,5+c]
                sim_vec = torch.nonzero(anchor_data[..., 4] > conf_thres)
                #sim_vec[nt,2] #取得所有非零元素True的坐标集合
                if len(sim_vec) > 0:
                    for x, y in sim_vec:
                        targets.append(anchor_data[x][y])#[nt][5+c]
                        tinfo.append([i, batch_idx, anchor_idx, x, y])#[nt][i, batch_idx, anchor_idx, x, y]
    num_ = 10 if pCms is None else 10 +  + pCms[0].shape[-1]
    output = [torch.zeros((0, num_), device=pYolo[0].device)] * batchs
    batch_targets = [[] for i in range(batchs)]
    for idx, target in enumerate(targets):
        info = tinfo[idx] # featuremap, batch, anchor x, y
        batch = info[1]
        feat = pRot[info[0]][batch].permute(1, 2, 0, 3).contiguous()#[H,W,na,1+2+2=5(dir)]
        anchor_data = feat[info[3]][info[4]]#anchor_data[na,1+2+2=5(dir)]
        #挑选P最大
        idex = anchor_data[:, 0].argmax()
        dir_target = anchor_data[idex]#dir_target[1+2+2=5(dir)]
        obj_merge = torch.cat((target, dir_target), dim=-1) 
        if pFt is not None:
            feat_ft = pFt[info[0]][batch].permute(1, 2, 0, 3).contiguous()
            ft_data = feat_ft[info[3]][info[4]]
            ft_target = ft_data[idex]
            obj_merge = torch.cat((obj_merge, ft_target), dim=-1)
        if pCms is not None:
            feat_cms = pCms[info[0]][batch].permute(1, 2, 0, 3).contiguous()#[H,W,na,1+2+2=5(dir)]
            cms_data = feat_cms[info[3]][info[4]]
            cms_target = cms_data[idex]
            obj_merge = torch.cat((obj_merge, cms_target), dim=-1) #[1,5+c+5(dir)]
        obj_merge = obj_merge.view(1, -1)
        batch_targets[batch].append(obj_merge)

    #batch_targets[b=0][nt][1,5+c+5(dir)]
    for xi in range(batchs):
        preds = batch_targets[xi]#[nt][1,5+c+5(dir)]
        if not preds:
            continue
        preds = torch.cat(preds, 0)#把preds这个list拼接成高一维度的tensor-->preds[nt,5+c+5(dir)]
        pYolo = preds[:, :5+nc]#[nt,5+nc]
        pRot = preds[:, 5+nc:10+nc]#[nt,5=1+2+2]
        pFt = preds[:, 10+nc:10+nc+ft_length] if ft_infer else None
        pCms = preds[:, 10+nc+ft_length:] if pcms_flag else None

        # Compute conf
        pYolo[:, 5:] *= pYolo[:, 4:5]  # conf = obj_conf * cls_conf
        # pYolo=pYolo[pYolo[:,4]>conf_thres]#[nt_filt,5+nc]   

        # 根据ab过滤
        if 1:
            mask_ab = (pRot[..., 3] > ab_thres) & (pRot[..., 4] > ab_thres)
            pRot = pRot[mask_ab]
            pYolo = pYolo[mask_ab]
            pCms = pCms[mask_ab] if pCms is not None else None
            pFt = pFt[mask_ab] if pFt is not None else None
            # If none remain process next image
            if not pYolo.shape[0]:
                continue

        # 对pRot的输出进行处理
        q2 = pRot[:, 1:3]
        #q2[nt_dir,2]
        if 0:
            q2 = normalize_dir(q2)#q2[nt][2]
        else:
            Lq2 = torch.norm(q2, dim=1, keepdim=True)
            q2 = q2 / (Lq2 + 1e-8)  # 避免0除问题
        if fold_angle == 2:
            q2 = two_fold_angle(q2)
        
        # ab2pts
        ab = pRot[:, 3:5]
        center_xy = pYolo[:, :2]#center_xy[nt][2]
        pts = ab2pts(ab, center_xy, q2)#pts[nt][8]

        # cls
        #pYolo[nt_filt,5+nc]
        conf, clsid = pYolo[:, 5:].max(1, keepdim=True)#conf[nt][1]  clsid[nt][1]
        #keepdim=True方便下面dim=1维度进行cat
        pRot = torch.cat((pts, conf, clsid.float()), 1) 
        if pCms is not None:
            pRot = torch.cat((pRot, pCms), 1)#,pYolo[:,:4]
        if pFt is not None:
            pRot = torch.cat((pRot, pFt), 1)
        #pRot[nt,10=pts[8]+conf[1]+clsid[1]]
        assert(pRot.shape[0]==pYolo.shape[0])

        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        clsid = clsid.squeeze(1)#clsid[nt][1]-->clsid[nt]
        keep = []
        for clss in clsid.unique():#剔除重复类
            #分类子集进行nms，各类互不干扰
            cls_mask = clsid.long() == clss
            cls_idx = torch.nonzero(cls_mask).squeeze(1)
            if(cls_idx.shape[0]>0):
                if(mask_dir!=[] and mask_dir[clss]>0):
                    pcls = pRot[cls_idx, :9]#pcls[nt_filt,9]
                    #pcls = pRot[pRot[..., 9].long() == clss][..., :9]
                    # Check shape
                    assert(pcls.shape[0] > 0)#boxes number
                    pcls = pcls.cpu().numpy().astype(np.float64)#pcls[nt_filt,9]
                    id_nms = py_cpu_nms_poly_fast(pcls, iou_thres)  # polygon-NMS
                else:
                    # Batched NMS
                    pcls = pYolo[cls_idx]#pcls[nt_filt,5+nc]
                    #sort_idx = pcls[:, 4].argsort(descending=True)[:max_nms]
                    #pcls = pcls[sort_idx]  # sort by confidence
                    boxes, scores = pcls[:, :4], pcls[:, 4]  # boxes (offset by class), scores
                    box2poly(boxes,pRot,cls_idx)
                    #print(pRot[cls_idx])
                    id_nms = torchvision.ops.nms(xywh2xyxy(boxes), scores, iou_thres)  # NMS
                id_nms = cls_idx[id_nms] #注意上面id_nms是针对pcls的编号，需要还原到pYolo的编号，用了cls_idx=torch.nonzero进行原始编号的查询
                keep.extend(id_nms)#列表keep末尾加上列表id_nms
        output[xi] = pRot[torch.tensor(keep)]#用keep这个list对pRot的[nt]维度进行索引过滤

        #output[xi][nt,10=pts[8]+conf[1]+clsid[1]]
        if(threshs.shape[0]>0):
            rot_obj = torch.tensor([]).to('cuda')
            for t in output[xi]:#t[10=pts[8]+conf[1]+clsid[1]]
                if(t[8] > threshs[int(t[9])]):
                    rot_obj = torch.cat((rot_obj, t.unsqueeze(0)), dim=0)
            output[xi] = rot_obj
       
    return output#output[batchs][nt,10=pts[8]+conf[1]+clsid[1]]

def non_max_suppression_cms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """      
    # output indices  
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 2), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        grid = indices_grid[xc[xi]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            grid = grid[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            grid = grid[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            grid = grid[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            grid = grid[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output, output_indices if return_indices else output

def big_nms(det_list, scores, iou_thres):
    y = torch.cat((det_list, scores[:,None]), dim=1)
    y = y.cpu().numpy().astype(np.float64)
    i = py_cpu_nms_poly_fast(y, iou_thres)
    return i 



def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if(x['epoch']>=0):
        if x.get('ema'):
            x['model'] = x['ema']  # replace model with ema
        for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
            x[k] = None
        x['epoch'] = -1
        x['model'].half()  # to FP16
        for p in x['model'].parameters():
            p.requires_grad = False
        torch.save(x, s or f)
        mb = os.path.getsize(s or f) / 1E6  # filesize
        print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Print to screen
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')

    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :7]))  #
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data)}\n' +
                '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to YOLO outputs
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('example%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

# check .ft file
def get_ft_num(path):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    path = sb.join(path.rsplit(sa, 1))
    path = Path(path)
    for file in path.rglob('*.ft'):
        with open(file, 'r') as f:
            ft = [x.split()[1:] for x in f.read().strip().splitlines() if len(x)]
        if len(ft) == 0:
            continue
        else:
            return (len(ft[0])-2)//4
    return 0
