//
// Created by dingjian on 18-5-24.
//

#ifndef DOTA_DEVKIT_POLY_CPU_HPP
#define DOTA_DEVKIT_POLY_CPU_HPP


// void _poly_nms(int* keep_out, int* num_out, const float* polys_host, int polys_num,
//             int polys_dim, float nms_overlap_thresh, int device_id);
void _poly_nms_float(int* keep_out, int* num_out, const float* polys_host, int polys_num, float nms_overlap_thresh);
void _poly_iou_float(float* overlaps, const float* Npolys, const float* Kpolys, int n, int k);
void _poly_nms_float_slow(int* keep_out, int* num_out, const float* polys_host, int polys_num, float nms_overlap_thresh);

void _poly_nms_double(int* keep_out, int* num_out, const double* polys_host, int polys_num, double nms_overlap_thresh);
void _poly_iou_double(double* overlaps, const double* Npolys, const double* Kpolys, int n, int k);
void _poly_nms_double_slow(int* keep_out, int* num_out, const double* polys_host, int polys_num, double nms_overlap_thresh);

#endif //DOTA_DEVKIT_POLY_NMS_HPP
