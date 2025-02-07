import numpy as np
import cv2
import matplotlib.pyplot as plt




def show_heatmap(feature):
    heatmap = feature.sum(0)/ feature.shape[0]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # heatmap = 1.0 - heatmap # 也可以不写，就是蓝色红色互换的作用
    heatmap = cv2.resize(heatmap, (224,224)) # (224,224)指的是图像的size，需要resize到原图大小
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite('a.png', heatmap)


if __name__ == "__main__":
    
    show_heatmap()