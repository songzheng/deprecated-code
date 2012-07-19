// vlfeat headers
#include <vl/mathop.h>
#include <vl/imopv.h>
#include <math.h>
#include <string.h>

#ifndef PATCH_FEATURE_H
#define PATCH_FEATURE_H
#include "image.h"
#include "coding.h"

// ***************************** //
// for pixel-wise feature coding
struct PixelFeatureOpt;
// pixel coding function (the function should be inline)
// c & p0~7: pixel and 8-neighbour
// dst: feature of the first bin
// dst_stride: stride to next bin
typedef void (*PixelFunc) (float *c, float *p0,float *p1, float *p2,float *p3,float *p4, float *p5,float *p6,float *p7, int img_stride,
        float *dst, int dst_stride, PixelFeatureOpt * opt);

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
    int length;
    CodeBook * codebook;    
    int coded_length;
};

// pixel feature implementation

// hog
inline void _PixelHOG (float *c, float *p0,float *p1, float *p2,float *p3,float *p4, float *p5,float *p6,float *p7, int img_stride,
        float *dst, int dst_stride, PixelFeatureOpt * opt)
{
    
    float gx, gy ;
    float angle, mod, nt, rbint ;
    int bint ;
    
    
    // clear dst
    memset(dst, 0, sizeof(float)*opt->length);
    
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
        
    dst[(bint%num_ori)*dst_stride] = (1 - rbint) * mod;
    dst[((bint+1)%num_ori)*dst_stride] = (rbint) * mod;
    return;
}


// lbp
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
 
inline void _PixelLBP59 (float *c, float *p0,float *p1, float *p2,float *p3,float *p4, float *p5,float *p6,float *p7, int img_stride,
        float *dst, int dst_stride, PixelFeatureOpt * opt)
{
    memset(dst, 0, sizeof(float)*opt->length);
    int unsigned bitString = 0 ;
    if(*p0 > *c) bitString |= 0x1 << 0; /*  E */
    if(*p1 > *c) bitString |= 0x1 << 1; /* SE */
    if(*p2 > *c) bitString |= 0x1 << 2; /* S  */
    if(*p3 > *c) bitString |= 0x1 << 3; /* SW */
    if(*p4 > *c) bitString |= 0x1 << 4; /*  W */
    if(*p5 > *c) bitString |= 0x1 << 5; /* NW */
    if(*p6 > *c) bitString |= 0x1 << 6; /* N  */
    if(*p7 > *c) bitString |= 0x1 << 7; /* NE */
    
    dst[LBP59_Map[bitString]*dst_stride] = 1;
}


// color histogram


// ***************************** //
// for rectangle dense grid patch feature
struct RectPatchFeatureOpt;
// patch code function (the function should be inline)
typedef void (*RectPatchFunc) (FloatImage * img, FloatRect * patch_rect,
        float * dst, RectPatchFeatureOpt * opt);

// extract feature from rectangle patch option:
//      name: the patch feature name
//      f: the patch feature function
//      param, nparam: the patch feature parameter
//      pixel_opt: use the pixel feature if f == null
//      numbin_{x,y}: bin number in each patch
//      step_{x,y}: sampling step of each bin
//      size_{x,y}: patch size of each bin
//      bin_length: feature length for each bin
//      length: patch featue length
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
struct RectPatchFeatureOpt
{
    char * name;
    RectPatchFunc f;
    int image_depth;
    double * param;
    int nparam;
    
    PixelFeatureOpt pixel_opt; 
    
    int numbin_x, numbin_y;
    int step_x, step_y;
    int size_x, size_y;
    int bin_length;
    int length;
    CodeBook * codebook;
    int coded_length;
};


// ***************************** //
// for image pyramid
struct PyramidOpt
{
    FloatRect * pyramid;
    int npyramid;    
};

// ***************************** //
// main function for dense rectangle patch features
void RectPatchFeature(FloatImage * img, FloatImage * desp, FloatImage * patch_coordinate, RectPatchFeatureOpt * opt, PyramidOpt * pyra_opt)
{    
    int image_stride = img->height*img->width;
    
    // raw patch feature without grid grouping
    int n_raw_patch[2];
    
    // rectangle raw patch layout
    n_raw_patch[0] = int(1.0*(img->height-opt->size_y)/opt->step_y + 0.5) + 1;
    n_raw_patch[1] = int(1.0*(img->width-opt->size_x)/opt->step_x + 0.5) + 1;
       
    // allocate raw patch feature  
    FloatImage raw_patch_feat, raw_patch_norm;      
    AllocateImage(&raw_patch_feat, n_raw_patch[0], n_raw_patch[1], opt->bin_length);
    AllocateImage(&raw_patch_norm, n_raw_patch[0], n_raw_patch[1], 1);
    int patch_feat_stride = n_raw_patch[0] * n_raw_patch[1];
    
    if(opt->f == NULL)
    {
        PixelFeatureOpt *pixel_opt = &opt->pixel_opt;        
        assert(pixel_opt->image_depth == img->depth);
                
        // cache pixel-level feature
        FloatImage pixel_feat;
        bool pixel_coding = (pixel_opt->codebook != NULL);
        
        int feat_length;
        if(pixel_coding)
            feat_length = pixel_opt->coded_length;
        else
            feat_length = pixel_opt->length;
        
        AllocateImage(&pixel_feat, img->height-2, img->width-2, feat_length);
        
        
        // Set up a circularly indexed neighborhood using nine pointers.
        // |--------------|
        // | p0 | p1 | p2 |
        // |--------------|
        // | p7 | c  | p3 |
        // |--------------|
        // | p6 | p5 | p4 |
        // |--------------|
        
    	float *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *center;
        float * dst;
        
        FloatImage feat_raw_tmp;
        if(pixel_coding)
            AllocateImage(&feat_raw_tmp, pixel_opt->length, 1, 1);
                
        int pixel_feat_stride = pixel_feat.width * pixel_feat.height;
        
        for (int x = 1 ; x < img->width-1 ; ++ x) {
            p0 = img->p + (x-1)*img->stride;
            p1 = p0 + 1,
            p2 = p1 + 1,
            p3 = p2 + img->stride,
            p4 = p3 + img->stride,
            p5 = p4 - 1,
            p6 = p5 - 1,
            p7 = p6 - img->stride,
            center = p7 + 1;
            
            dst = pixel_feat.p + (x-1)*pixel_feat.stride;
            
            for (int y = 1 ; y < img->height-1 ; ++ y) {       
                
                if(pixel_coding)
                {
                    pixel_opt->f(center, p0, p1, p2, p3, p4, p5, p6, p7, image_stride,
                            feat_raw_tmp.p, 1, pixel_opt);
                    pixel_opt->codebook->f(feat_raw_tmp.p, dst, pixel_feat_stride, pixel_opt->codebook);
                }
                else
                {
                    pixel_opt->f(center, p0, p1, p2, p3, p4, p5, p6, p7, image_stride,
                            dst, pixel_feat_stride, pixel_opt);
                }
                                
                p0 ++;
                p1 ++;
                p2 ++;
                p3 ++;
                p4 ++;
                p5 ++;
                p6 ++;
                p7 ++;
                center ++;
                dst ++;
            }
        }         
        
        
        if(pixel_coding)
            FreeImage(&feat_raw_tmp);
        
        // allocate transposed convolution cache 1
        FloatImage tmp1;
        AllocateImage(&tmp1, n_raw_patch[1], img->height, 1);
        
        for(int n=0; n<feat_length; n++)
        {
            // subsampling to patch level
            // the coloumn and row is reverse for vl functions, so col --> row here
            vl_imconvcoltri_f (tmp1.p, tmp1.stride,
                    pixel_feat.p + n*pixel_feat_stride, pixel_feat.height, pixel_feat.width, pixel_feat.stride,
                    opt->size_x*2, /* filt size */
                    opt->step_x, /* subsampling step */
                    VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;
            
            vl_imconvcoltri_f (raw_patch_feat.p + patch_feat_stride*n, raw_patch_feat.stride,
                    tmp1.p, tmp1.height, tmp1.width, tmp1.stride,
                    opt->size_y*2,
                    opt->step_y,
                    VL_PAD_BY_CONTINUITY|VL_TRANSPOSE);   
        }
        
        FreeImage(&tmp1);
        FreeImage(&pixel_feat);
    }
    else
    {
        // patch level
        float *feat_tmp = new float[opt->length];
        FloatRect r;
        float *dst = raw_patch_feat.p;
        for(int x=0; x<n_raw_patch[1]*opt->step_x; x+=opt->step_x)
        {
            r.x1 = x;
            r.x2 = x + opt->size_x;
            for(int y=0; y<n_raw_patch[0]*opt->step_y; y+=opt->step_y)
            {
                
                r.y1 = y;
                r.y2 = y + opt->size_y;
                opt->f(img, &r, feat_tmp, opt);
                
                for(int i=0; i<opt->length; i++)
                    dst[i*patch_feat_stride] = feat_tmp[i];
                
                dst ++;
            }
        }
        delete[] feat_tmp;
    }
    
    /*********debug interrupt*********/
    MoveImage(&raw_patch_feat, desp);
    return;
    
    /*********debug interrupt*********/
    
    // normalize each raw patch
    NormL2Image(&raw_patch_feat, &raw_patch_norm);
            
    // group patch grids
    FloatImage patch_feat;      
    AllocateImage(&patch_feat, opt->bin_length*opt->numbin_x*opt->numbin_y, (n_raw_patch[0]-opt->numbin_y+1) * (n_raw_patch[1]-opt->numbin_x+1), 1);
    
    AllocateImage(patch_coordinate, 2, patch_feat.width, 1);
    float * dst = patch_coordinate->p;
    
    for(int ix=0; ix<n_raw_patch[1]-opt->numbin_x+1; ix++)
    {
        for(int iy=0; iy<n_raw_patch[0]-opt->numbin_y+1; iy++)
        {
            *(dst++) = float(iy * opt->step_y);
            *(dst++) = float(ix * opt->step_x);            
        }
    }
    
    for(int bin_x=0; bin_x<opt->numbin_x; bin_x++)
    {
        for(int bin_y=0; bin_y<opt->numbin_y; bin_y++)
        {
            int bin_start = bin_x * opt->numbin_y + bin_y;
            // add different normalize for each bin
            float bin_norm = 1.0f;
            for(int i=0; i<opt->bin_length; i++)
            {
                float * dst = patch_feat.p + bin_start * opt->bin_length + i;
                float * src = raw_patch_feat.p + patch_feat_stride*i;
                for(int ix=bin_x; ix<bin_x+n_raw_patch[1]-opt->numbin_x+1; ix++)
                {
                    for(int iy=bin_y; iy<bin_y+n_raw_patch[0]-opt->numbin_y+1; iy++)
                    {
                        *dst = bin_norm * src[ix*raw_patch_feat.stride + iy];
                        dst += patch_feat.height;
                    }
                }
            }
        }
    }
    
    FreeImage(&raw_patch_feat);
    FreeImage(&raw_patch_norm);
    
    // group and encode in pyramids
    if(opt->codebook == NULL)
    {
        MoveImage(&patch_feat, desp);
    }
    else
    {
        if(pyra_opt != NULL)
        {
            // encode in rectangle pyramid areas
            int nsplit = pyra_opt->npyramid;
            AllocateImage(desp, opt->coded_length, nsplit, 1);
            for(int i=0; i<nsplit; i++)
                EncodeRect(&patch_feat, pyra_opt->pyramid+i, patch_coordinate, desp->p+i*desp->height, 1, opt->codebook);
        }
        else
        {
            // encode each patch
            AllocateImage(desp, opt->coded_length, patch_feat.width, 1);        
            for(int i=0; i<patch_feat.width; i++)
                opt->codebook->f(patch_feat.p+i*patch_feat.height, desp->p+i*desp->height, 1, opt->codebook);
        }
    
        FreeImage(&patch_feat);
    }
}

#endif