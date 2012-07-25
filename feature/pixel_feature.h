
#ifndef PIXEL_FEATURE_H
#define PIXEL_FEATURE_H
#include "image.h"
#include "coding.h"

// ***************************** //
// for pixel-wise feature coding
struct PixelFeatureOpt;
// pixel coding function (the function should be inline)
// c & p0~7: pixel and 8-neighbour
// dst: feature of the first bin
// dst_stride: stride to next bin
typedef void (*PixelFunc) (float *c, float *p0,float *p1, float *p2,float *p3,
        float *p4, float *p5,float *p6,float *p7, int img_stride,
        PixelFeatureOpt * opt);

// pixel coding options:
//      name: name of the pixel code
//      f: feature extraction function
//      image_depth: depth of image used
//      length: origin featue length
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
//      coded_length: coded featue length
struct PixelFeatureOpt
{
    char * name;
    PixelFunc f;
    int image_depth;
    double * param;
    int nparam;
    
    int length_sparse;
    int length;
        
    FloatSparseMatrix buffer;  
        
    CodeBook * codebook;    
};

void PixelFeatureInitBuffer(PixelFeatureOpt * opt)
{
//     printf("%d, %d\n", opt->length, opt->length_sparse);
    AllocateSparseMatrix(&opt->buffer, opt->length, 1, 1);
    AllocateColumnSparseMatrix(&opt->buffer, 0, opt->length_sparse);
}

void PixelFeatureReleaseBuffer(PixelFeatureOpt * opt)
{
    FreeSparseMatrix(&opt->buffer);
}

// pixel feature implementation

// raw gray pixel
inline void _PixelRAWGray (float *c, float *p0,float *p1, float *p2,float *p3,
        float *p4, float *p5,float *p6,float *p7, int img_stride,
        PixelFeatureOpt * opt)
{
    float * val = opt->buffer.p[0];
    int * bin = opt->buffer.i[0];
    val[0] = *c;
    bin[0] = 1;
}
        
// raw color pixel
inline void _PixelRAWColor (float *c, float *p0,float *p1, float *p2,float *p3,
        float *p4, float *p5,float *p6,float *p7, int img_stride,
        PixelFeatureOpt * opt)
{
    float * val = opt->buffer.p[0];
    int * bin = opt->buffer.i[0];
    val[0] = *c;
    bin[0] = 0;
    val[1] = *(c+img_stride);
    bin[1] = 1;
    val[2] = *(c+2*img_stride);
    bin[2] = 2;
}

// hog
inline void _PixelHOG (float *c, float *p0,float *p1, float *p2,float *p3,
        float *p4, float *p5,float *p6,float *p7, int img_stride,
        PixelFeatureOpt * opt)
{
    
    float gx, gy ;
    float angle, mod, nt, rbint ;
    int bint ;
            
    int num_ori = int(opt->param[0]);
    
    gy = 0.5f * (*p5 - *p1);
    gx = 0.5f * (*p3 - *p7);
    
    /* angle and modulus */
    angle = vl_fast_atan2_f (gy,gx) ;
    mod = vl_fast_sqrt_f (gx*gx + gy*gy) ;
    
    /* quantize angle */
    nt = vl_mod_2pi_f (angle) * (num_ori / (2*VL_PI)) ;
    bint = vl_floor_f (nt) ;
    rbint = nt - bint ;        
    
    float * val = opt->buffer.p[0];
    int * bin = opt->buffer.i[0];
    val[0] = (1 - rbint) * mod;    
    bin[0] = bint%num_ori;
    val[1] = (rbint) * mod;    
    bin[1] = (bint+1)%num_ori;
}

// hog of UoC

static double uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
static double vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};
        
inline void _PixelHOG9Bin (float *c, float *p0,float *p1, float *p2,float *p3,
        float *p4, float *p5,float *p6,float *p7, int img_stride,
        PixelFeatureOpt * opt)
{    
    float gx, gy ;
    float angle, mod, nt, rbint ;
    int bint ;    
        
    int num_ori = int(opt->param[0]);
    
    gy = 0.5f * (*p5 - *p1);
    gx = 0.5f * (*p3 - *p7);
    mod = vl_fast_sqrt_f (gx*gx + gy*gy) ;
    
    // snap to one of 18 orientations within 2*pi, degree=best_o*2*pi/18
    double best_dot = 0;
    int best_o = 0;
    for (int o = 0; o < 9; o++) {
        double dot = uu[o]*gx + vv[o]*gy;
        if (dot > best_dot) {
            best_dot = dot;
            best_o = o;
        } else if (-dot > best_dot) {
            best_dot = -dot;
            best_o = o+9;
        }
    }
    
    float * val = opt->buffer.p[0];
    int * bin = opt->buffer.i[0];
    val[0] = mod;    
    bin[0] = best_o;
}

// lbp 59
static unsigned int LBP59_Map[256]=
{0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12,
 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58,
 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24,
 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58,
 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
 58, 58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58,
 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58,
 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57};
 
inline void _PixelLBP59 (float *c, float *p0,float *p1, float *p2,float *p3,
        float *p4, float *p5,float *p6,float *p7, int img_stride,
        PixelFeatureOpt * opt)
{
    int unsigned bitString = 0 ;
    if(*p0 > *c) bitString |= 0x1 << 0; /*  E */
    if(*p1 > *c) bitString |= 0x1 << 1; /* SE */
    if(*p2 > *c) bitString |= 0x1 << 2; /* S  */
    if(*p3 > *c) bitString |= 0x1 << 3; /* SW */
    if(*p4 > *c) bitString |= 0x1 << 4; /*  W */
    if(*p5 > *c) bitString |= 0x1 << 5; /* NW */
    if(*p6 > *c) bitString |= 0x1 << 6; /* N  */
    if(*p7 > *c) bitString |= 0x1 << 7; /* NE */
    
    float * val = opt->buffer.p[0];
    int * bin = opt->buffer.i[0];
    val[0] = 1;
    bin[0] = LBP59_Map[bitString];
}


// color histogram

#endif
