import numpy as np
from .poly_cpu import poly_cpu_nms32, poly_cpu_iou32, poly_cpu_nms32_slow, poly_cpu_nms64, poly_cpu_iou64, poly_cpu_nms64_slow

def poly_nms_cpu32(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float32)
    return poly_cpu_nms32(dets, thresh)

def poly_nms_cpu32_slow(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float32)
    return poly_cpu_nms32_slow(dets, thresh)

def poly_iou_cpu32(dets1, dets2):
    if dets1.shape[0] == 0 or dets2.shape[0] == 0:
        return np.zeros([dets1.shape[0], dets2.shape[0]]).astype(np.float32)
    return poly_cpu_iou32(dets1, dets2)

def poly_nms_cpu64(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float64)
    return poly_cpu_nms64(dets, thresh)

def poly_nms_cpu64_slow(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float64)
    return poly_cpu_nms64_slow(dets, thresh)

def poly_iou_cpu64(dets1, dets2):
    if dets1.shape[0] == 0 or dets2.shape[0] == 0:
        return np.zeros([dets1.shape[0], dets2.shape[0]]).astype(np.float64)
    return poly_cpu_iou64(dets1, dets2)

__all__ = ['poly_nms_cpu32', 'poly_iou_cpu32', 'poly_nms_cpu32_slow', 'poly_nms_cpu64', 'poly_iou_cpu64', 'poly_nms_cpu64_slow']