#include <omp.h>
#include "poly_cpu.hpp"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
using namespace std;
#define maxn 32
// #define TYPE double
#define EPS 1E-8

template <typename TYPE>
inline int sig(TYPE d){
    return(d>EPS)-(d<-EPS);
}

template <typename TYPE>
struct Point{
    TYPE x,y; Point(){}
    Point(TYPE x,TYPE y):x(x),y(y){}
    bool operator==(const Point&p)const{
        return sig(x-p.x)==0&&sig(y-p.y)==0;
    }
};

template <typename TYPE>
inline TYPE cross(Point<TYPE> o,Point<TYPE> a,Point<TYPE> b){ 
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}
template <typename TYPE>
inline TYPE area(Point<TYPE>* ps,int n){
    ps[n]=ps[0];
    TYPE res=0;
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        #pragma omp atomic
        res+= ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res * 0.5;
}
template <typename TYPE>
inline void lineCross(TYPE s1,TYPE s2,Point<TYPE> c,Point<TYPE> d,Point<TYPE>&p){
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
}
template <typename TYPE>
inline void polygon_cut(Point<TYPE>*p,int&n,Point<TYPE> a,Point<TYPE> b, Point<TYPE>* pp){
//    static Point<TYPE> pp[maxn];
    if(n<3){
        n=0;
        return;
    }
    int m=0;p[n]=p[0];
    TYPE s1, s2;
    s1=cross(a,b,p[0]);
    int sig_s1 = sig(s1);
    for(int i=0;i<n;i++){
        s2=cross(a,b,p[i+1]);
        // int sig_s1 = sig(s1);
        int sig_s2 = sig(s2);
        if(sig_s1>0) pp[m++]=p[i];
        // if(sig_s1!=sig(s2))
        if(sig_s1!=sig_s2)
            lineCross(s1,s2,p[i],p[i+1],pp[m++]);
        s1 = s2;
        sig_s1 = sig_s2;
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!(pp[i]==pp[i-1]))
            p[n++]=pp[i];
    while(n>1&&p[n-1]==p[0])n--;
}
template <typename TYPE>
inline TYPE intersectArea(Point<TYPE> a,Point<TYPE> b,Point<TYPE> c,Point<TYPE> d, Point<TYPE> o){
    Point<TYPE> p[16]={o,a,b};
    int s2=sig(cross(o,c,d));
    if(s2==0)return 0.0;
    if(s2==-1) swap(c,d);
    int n=3;
    Point<TYPE> pp[maxn];
    polygon_cut(p,n,o,c, pp);
    polygon_cut(p,n,c,d, pp);
    polygon_cut(p,n,d,o, pp);
    TYPE res=fabs(area(p,n));
    return s2 * res;
}

template <typename TYPE>
inline TYPE intersectArea(Point<TYPE>*ps1,int n1,Point<TYPE>*ps2,int n2){
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    TYPE res=0;
    Point<TYPE> o(0,0);
    #pragma omp parallel for
    for(int i=0;i<n1;i++){
        Point<TYPE> a=ps1[i];
        Point<TYPE> b=ps1[i+1];
        int s1=sig(cross(o,a,b));
        if(s1==0) continue;
        if(s1==-1) swap(a,b);
        #pragma omp parallel for
        for(int j=0;j<n2;j++){
            #pragma omp atomic
            res+= s1 * intersectArea(a, b,ps2[j],ps2[j+1], o);
        }
    }
    return res;//assumeresispositive!
}



template <typename TYPE>
TYPE devPolyIoU(TYPE const * const p, TYPE const * const q, int const * const pv, int const * const qv) {
    Point<TYPE> ps1[5], ps2[5];
    int n1 = 4;
    int n2 = 4;
    int x1, x2, y1, y2;
    x1 = (pv[0] > qv[0]) ? pv[0] : qv[0];
    x2 = (pv[1] < qv[1]) ? pv[1] : qv[1];
    y1 = (pv[2] > qv[2]) ? pv[2] : qv[2];
    y2 = (pv[3] < qv[3]) ? pv[3] : qv[3];
    if (x1 >= x2) return 0;
    if (y1 >= y2) return 0;

    Point<TYPE> o; 
    o.x = (pv[0] < qv[0]) ? pv[0] : qv[0];
    o.y = (pv[2] < qv[2]) ? pv[2] : qv[2];
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2] - o.x;
        ps1[i].y = p[i * 2 + 1] - o.y;

        ps2[i].x = q[i * 2] - o.x;
        ps2[i].y = q[i * 2 + 1] - o.y;
    }
    TYPE a1 = area(ps1,n1);
    TYPE a2 = area(ps2,n2);
    if(a1<0) reverse(ps1,ps1+n1);
    if(a2<0) reverse(ps2,ps2+n2);
    TYPE inter_area = intersectArea(ps1, n1, ps2, n2);
    TYPE union_area = fabs(a1) + fabs(a2) - inter_area;
    TYPE iou = inter_area / union_area;
    return iou;
}
//
//int main(){
//    double p[8] = {0, 0, 1, 0, 1, 1, 0, 1};
//    double q[8] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
//    vector<double> P(p, p + 8);
//    vector<double> Q(q, q + 8);
//    iou_poly(P, Q);
//    return 0;
//}

//int main(){
//    double p[8] = {0, 0, 1, 0, 1, 1, 0, 1};
//    double q[8] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
//    iou_poly(p, q);
//    return 0;
//}

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define MASK_TYPE unsigned long long
#define MASK_TYPE unsigned int
int const threadsPerBlock = sizeof(MASK_TYPE) * 8;

template <typename TYPE>
inline int findMin(const TYPE* v) {  
    int min = static_cast<int>(v[0]);  
    if (static_cast<int>(v[2]) < min) min = static_cast<int>(v[2]);  
    if (static_cast<int>(v[4]) < min) min = static_cast<int>(v[4]);  
    if (static_cast<int>(v[6]) < min) min = static_cast<int>(v[6]);  
    return static_cast<int>(min);  
}


template <typename TYPE>
inline int findMax(const TYPE* v) {  
    int max = static_cast<int>(v[0]);  
    if (static_cast<int>(v[2]) > max) max = static_cast<int>(v[2]);  
    if (static_cast<int>(v[4]) > max) max = static_cast<int>(v[4]);  
    if (static_cast<int>(v[6]) > max) max = static_cast<int>(v[6]);  
    return static_cast<int>(max);  
}

template <typename TYPE>
void findXY12(TYPE const * const p, int* v, const size_t n){
    #pragma omp parallel for
    for (int threadId=0;threadId<n;threadId++){
        if (threadId < n){
            v[threadId * 4 + 0]=findMin(p + threadId * 8);
            v[threadId * 4 + 1]=findMax(p + threadId * 8);
            v[threadId * 4 + 2]=findMin(p + 1 + threadId * 8);
            v[threadId * 4 + 3]=findMax(p + 1 + threadId * 8);
        }
    }
}


template <typename TYPE>
void poly_nms_kernel(const int n_polys, const TYPE nms_overlap_thresh,
                    const TYPE *dev_polys, MASK_TYPE *dev_mask,
                    const int* xy12) {
    const int col_blocks = DIVUP(n_polys, threadsPerBlock);
    #pragma omp parallel for
    for (int i=0;i<n_polys;i++){
        #pragma omp parallel for
        for (int j=i / threadsPerBlock;j<col_blocks;j++){
            MASK_TYPE t = 0;
            const int k_start = (j == 0) ? i % threadsPerBlock + 1 : 0; // 根据 j 的值设置 k 的起始值  
            #pragma omp parallel for
            for (int k=k_start;k<threadsPerBlock;k++){
                const int cur_idx = j * threadsPerBlock + k;
                if ((i < n_polys)&&(cur_idx < n_polys)){
                    if(devPolyIoU(dev_polys + i * 8, dev_polys + cur_idx * 8, xy12 + i * 4, xy12 + cur_idx * 4) > nms_overlap_thresh){
                        #pragma omp atomic
                        t |= 1ULL << k;
                    }
                }
            }
            dev_mask[i * col_blocks + j] = t;
        }
    }
}

template <typename TYPE>
void overlaps_kernel_nk(const int N, const int K, 
                        const TYPE* dev_polys_n, const int* xy12_n,
                        const TYPE* dev_polys_k, const int* xy12_k, 
                        TYPE* dev_overlaps) {

    #pragma omp parallel for
    for (int x=0;x<N;x++)
        #pragma omp parallel for
        for (int y=0;y<K;y++){
            dev_overlaps[x * K + y] = devPolyIoU(dev_polys_n + x * 8, dev_polys_k + y * 8,
                                                xy12_n + x * 4, xy12_k + y * 4);
        }
}



template <typename TYPE>
void _poly_nms_slow(int* keep_out, int* num_out, const TYPE* polys_host, int polys_num, TYPE nms_overlap_thresh) {

    std::vector<int> xy12(polys_num * 4);
    const int col_blocks = DIVUP(polys_num, threadsPerBlock);
    std::vector<MASK_TYPE> mask_host(polys_num * col_blocks);
    memset(&mask_host[0], 0, sizeof(MASK_TYPE) * polys_num * col_blocks);
    std::vector<MASK_TYPE> remv(col_blocks);
    memset(&remv[0], 0, sizeof(MASK_TYPE) * col_blocks);

    findXY12(polys_host, &xy12[0], polys_num);
    poly_nms_kernel(polys_num,
                    nms_overlap_thresh,
                    polys_host,
                    &mask_host[0],
                    &xy12[0]);
                

    // TODO: figure out it
    int num_to_keep = 0;
    for (int i = 0; i < polys_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;
    // 由于score按降序排列,必然是从前往后进行过滤,当前的框只需要考虑自身是否被过滤以及自己可以过滤哪些框
        if (!(remv[nblock] & (1ULL << inblock))) {
            // remv[nblock] 取 inblock 位   如果为0则保留(未被过滤)
            keep_out[num_to_keep++] = i;
            // remv[j] 合并当前保留框可以过滤的框
            MASK_TYPE *p = &mask_host[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
        }
    }
    *num_out = num_to_keep;
}


template <typename TYPE>
void _poly_iou(TYPE* overlaps, const TYPE* Npolys, const TYPE* Kpolys, int n, int k) {
    std::vector<int> xy12_n(n * 4);
    std::vector<int> xy12_k(k * 4);
    const size_t iou_size = n * k * sizeof(TYPE);

    findXY12(Npolys, &xy12_n[0], n);
    findXY12(Kpolys, &xy12_k[0], k);

    overlaps_kernel_nk(n, k,
                        Npolys, &xy12_n[0],
                        Kpolys, &xy12_k[0],
                        overlaps);
}


template <typename TYPE>
void _poly_nms(int* keep_out, int* num_out, const TYPE* polys_host, int polys_num, TYPE nms_overlap_thresh) {

    std::vector<int> xy12(polys_num * 4);
    const int col_blocks = DIVUP(polys_num, threadsPerBlock);
    std::vector<MASK_TYPE> remv(col_blocks);
    memset(&remv[0], 0, sizeof(MASK_TYPE) * col_blocks);

    findXY12(polys_host, &xy12[0], polys_num);

    // TODO: figure out it
    int num_to_keep = 0;
    for (int i = 0; i < polys_num; i++) {    
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;
        // 由于score按降序排列,必然是从前往后进行过滤,当前的框只需要考虑自身是否被过滤以及自己可以过滤哪些框
        if (!(remv[nblock] & (1ULL << inblock))) {
            // remv[nblock] 取 inblock 位   如果为0则保留(未被过滤)
            keep_out[num_to_keep++] = i;
            #pragma omp parallel for
            for (int j=nblock;j<col_blocks;j++){
                MASK_TYPE t = remv[j];
                const int k_start = (j == 0) ? inblock + 1 : 0; // 根据 j 的值设置 k 的起始值  
                #pragma omp parallel for
                for (int k=k_start;k<threadsPerBlock;k++){
                    const int cur_idx = j * threadsPerBlock + k;
                    if (!(t & (1ULL << k))){
                        if (cur_idx < polys_num){
                            if(devPolyIoU(polys_host + i * 8, polys_host + cur_idx * 8, &xy12[0] + i * 4, &xy12[0] + cur_idx * 4) > nms_overlap_thresh){
                                #pragma omp atomic
                                t |= 1ULL << k;
                            }
                        }
                    }
                }
                remv[j] = t;
            }
        }

    }
    *num_out = num_to_keep;
}

void _poly_nms_float(int* keep_out, int* num_out, const float* polys_host, int polys_num, float nms_overlap_thresh){
    _poly_nms(keep_out, num_out, polys_host, polys_num, nms_overlap_thresh);
}
void _poly_nms_double(int* keep_out, int* num_out, const double* polys_host, int polys_num, double nms_overlap_thresh){
    _poly_nms(keep_out, num_out, polys_host, polys_num, nms_overlap_thresh);
}

void _poly_nms_float_slow(int* keep_out, int* num_out, const float* polys_host, int polys_num, float nms_overlap_thresh){
    _poly_nms_slow(keep_out, num_out, polys_host, polys_num, nms_overlap_thresh);
}
void _poly_nms_double_slow(int* keep_out, int* num_out, const double* polys_host, int polys_num, double nms_overlap_thresh){
    _poly_nms_slow(keep_out, num_out, polys_host, polys_num, nms_overlap_thresh);
}

void _poly_iou_float(float* overlaps, const float* Npolys, const float* Kpolys, int n, int k){
    _poly_iou(overlaps, Npolys, Kpolys, n, k);
}
void _poly_iou_double(double* overlaps, const double* Npolys, const double* Kpolys, int n, int k){
    _poly_iou(overlaps, Npolys, Kpolys, n, k);
}