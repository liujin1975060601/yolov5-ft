import cv2
import numpy as np


class CutImages:
    def __init__(self,  sub_size=1024, over_lap=200):
        self.sub_size = sub_size
        self.over_lap = over_lap

    def cut_images(self, image):
        if isinstance(image, str):
            img0 = cv2.imread(image)  # BGR
        else:
            img0 = image
        # img0 = self.gdal_start(image_path)
        cut_imgs = self._cut_imgs(img0)
        return img0, cut_imgs
    # def gdal_start(self, img_path):
    #     dataset = gdal.Open(img_path)
    #     im_width = dataset.RasterXSize  # 栅格矩阵的列数
    #     im_height = dataset.RasterYSize  # 栅格矩阵的行数
    #     # im_bands = dataset.RasterCount  # 波段数
    #     im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    #     # im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    #     # im_proj = dataset.GetProjection()  # 获取投影信息
    #     # 16位转8位
    #     # 高， 宽
    #     # image = self.stretch_16to8(im_data)
    #     image = self.stretch_16to8_2(im_data)
    #     dataset = None
    #     # 拼接3通道
    #     image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    #     return image

    def stretch_16to8_2(self, bands, lower_percent=2, higher_percent=98):
        h, w = bands.shape
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands, lower_percent)
        d = np.percentile(bands, higher_percent)
        rate = (b - a) / (d - c)
        bands = np.ravel(bands)
        size = bands.size // 2
        t1 = a + (bands[:size] - c) * rate
        t1[t1 < a] = a
        t1 = t1.astype(np.uint8)
        t2 = a + (bands[size:] - c) * rate
        t2[t2 > b] = b
        t2 = t2.astype(np.uint8)
        # out = np.concatenate((t1, t2)).reshape((h, w))
        bands[:size] = t1[:]
        bands[size:] = t2[:]
        return bands.reshape((h, w))

    def _cut_imgs(self, img):
        height, width = img.shape[:2]
        sub_size = self.sub_size
        over_lap = self.over_lap
        cut_results = []
        # 从左到右，从上到下
        x, y = 0, 0
        h_last = False
        while y < height:
            if y + sub_size > height:
                y = height - sub_size
                h_last = True
            w_last = False
            x = 0
            while x < width:
                if x + sub_size > width:
                    x = width - sub_size
                    w_last = True
                patch = img[y:y + sub_size, x:x + sub_size].copy()
                cut = {
                    'xy': [x, y],
                    'patch': patch
                }
                cut_results.append(cut)
                if w_last:
                    break
                x += sub_size - over_lap
            if h_last:
                break
            y += sub_size - over_lap
        return cut_results