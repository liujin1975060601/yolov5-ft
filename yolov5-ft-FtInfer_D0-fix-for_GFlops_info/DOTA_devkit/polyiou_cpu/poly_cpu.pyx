import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "poly_cpu.hpp":
    void _poly_nms_float(int*, int*, np.float32_t*, int, float)
    void _poly_nms_float_slow(int*, int*, np.float32_t*, int, float)
    void _poly_iou_float(np.float32_t*, np.float32_t*, np.float32_t*, int, int)
    void _poly_nms_double(int*, int*, np.float64_t*, int, double)
    void _poly_nms_double_slow(int*, int*, np.float64_t*, int, double)
    void _poly_iou_double(np.float64_t*, np.float64_t*, np.float64_t*, int, int)

def poly_cpu_nms32(np.ndarray[np.float32_t, ndim=2] dets, float thresh):
    cdef int boxes_num = dets.shape[0]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 8]
    cdef np.ndarray[np.int32_t, ndim=1] order = scores.argsort()[::-1].astype(np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] sorted_dets = np.ascontiguousarray(dets[order, :8].copy(), dtype=np.float32)
    cdef int *keep_ptr = <int *>&keep[0]
    _poly_nms_float(keep_ptr, &num_out, &sorted_dets[0, 0], boxes_num, thresh)
    keep = keep[:num_out]
    return order[keep]

def poly_cpu_nms32_slow(np.ndarray[np.float32_t, ndim=2] dets, float thresh):
    cdef int boxes_num = dets.shape[0]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 8]
    cdef np.ndarray[np.int32_t, ndim=1] order = scores.argsort()[::-1].astype(np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] sorted_dets = np.ascontiguousarray(dets[order, :8].copy(), dtype=np.float32)
    cdef int *keep_ptr = <int *>&keep[0]
    _poly_nms_float_slow(keep_ptr, &num_out, &sorted_dets[0, 0], boxes_num, thresh)
    keep = keep[:num_out]
    return order[keep]

def poly_cpu_iou32(np.ndarray[np.float32_t, ndim=2] polys_n, np.ndarray[np.float32_t, ndim=2] polys_k):
    cdef int N = polys_n.shape[0]
    cdef int K = polys_k.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype = np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] polys_n1 = np.ascontiguousarray(polys_n[:, :8].copy(), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] polys_k1 = np.ascontiguousarray(polys_k[:, :8].copy(), dtype=np.float32)
    _poly_iou_float(&overlaps[0, 0], &polys_n1[0, 0], &polys_k1[0, 0], N, K)
    return overlaps

def poly_cpu_nms64(np.ndarray[np.float64_t, ndim=2] dets, float thresh):
    cdef int boxes_num = dets.shape[0]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] scores = dets[:, 8]
    cdef np.ndarray[np.int32_t, ndim=1] order = scores.argsort()[::-1].astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] sorted_dets = np.ascontiguousarray(dets[order, :8].copy(), dtype=np.float64)
    cdef int *keep_ptr = <int *>&keep[0]
    _poly_nms_double(keep_ptr, &num_out, &sorted_dets[0, 0], boxes_num, thresh)
    keep = keep[:num_out]
    return order[keep]

def poly_cpu_nms64_slow(np.ndarray[np.float64_t, ndim=2] dets, float thresh):
    cdef int boxes_num = dets.shape[0]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] scores = dets[:, 8]
    cdef np.ndarray[np.int32_t, ndim=1] order = scores.argsort()[::-1].astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] sorted_dets = np.ascontiguousarray(dets[order, :8].copy(), dtype=np.float64)
    cdef int *keep_ptr = <int *>&keep[0]
    _poly_nms_double_slow(keep_ptr, &num_out, &sorted_dets[0, 0], boxes_num, thresh)
    keep = keep[:num_out]
    return order[keep]

def poly_cpu_iou64(np.ndarray[np.float64_t, ndim=2] polys_n, np.ndarray[np.float64_t, ndim=2] polys_k):
    cdef int N = polys_n.shape[0]
    cdef int K = polys_k.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] overlaps = np.zeros((N, K), dtype = np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] polys_n1 = np.ascontiguousarray(polys_n[:, :8].copy(), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] polys_k1 = np.ascontiguousarray(polys_k[:, :8].copy(), dtype=np.float64)
    _poly_iou_double(&overlaps[0, 0], &polys_n1[0, 0], &polys_k1[0, 0], N, K)
    return overlaps
