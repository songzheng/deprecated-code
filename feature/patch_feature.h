// vlfeat headers
#include <mexutils.h>
#include <vl/mathop.h>
#include <vl/imopv.h>
#include <math.h>
#include <string.h>

#ifndef RECT_PATCH_FEATURE_H
#define RECT_PATCH_FEATURE_H
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
        float *dst, int dst_stride,
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
    float * param;
    int nparam;
    int length;
    CodeBook * codebook;    
    int coded_length;
};

// ***************************** //
// for rectangle dense grid patch feature
struct RectPatchFeatureOpt;
// patch code function (the function should be inline)
typedef void (*RectPatchFunc) (FloatImage * img, FloatRect * patch_rect,
        float * dst, int dst_stride, 
        RectPatchFeatureOpt * opt);

// extract feature from rectangle patch option:
//      f: the patch feature function
//      param, nparam: the patch feature parameter
//      pixel_opt: use the pixel feature if f == null
//      numbin_{x,y}: bin number in each patch
//      step_{x,y}: sampling step of each bin
//      size_{x,y}: patch size of each bin
//      norm_numbin_{x,y}: normalize over how many bins
//      bin_length: feature length for each bin
//      length: patch featue length
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
struct RectPatchFeatureOpt
{
    RectPatchFunc f; 
    float * param;
    int nparam;
    
    PixelFeatureOpt * pixel_opt; 
    
    int numbin_x, numbin_y;
    int step_x, step_y;
    int size_x, size_y;
    int norm_numbin_x, norm_numbin_y; 
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
void RectPatchFeature(FloatImage * img, FloatImage * desp, RectPatchFeatureOpt * opt, PyramidOpt * pyra_opt)
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
        PixelFeatureOpt *pixel_opt = opt->pixel_opt;        
        assert(pixel_opt->image_depth == img->depth);
                
        // cache pixel-level feature
        FloatImage pixel_feat;
        AllocateImage(&pixel_feat, img->height-2, img->width-2, pixel_opt->length);
        
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
        int pixel_feat_stride = pixel_feat.width * pixel_feat.height;
        
        for (int x = 1 ; x < img->width-1 ; ++ x) {
            p0 = img->p + (x-1)*img->width;
            p1 = p0 + 1,
            p2 = p1 + 1,
            p3 = p2 + img->stride,
            p4 = p3 + img->stride,
            p5 = p4 - 1,
            p6 = p5 - 1,
            p7 = p6 - img->stride,
            center = p7 + 1;
            
            dst = pixel_feat.p + (x-1)*img->width;
            
            for (int y = 1 ; y < img->height-1 ; ++ y) {
        
                pixel_opt->f(center, p0, p1, p2, p3, p4, p5, p6, p7, image_stride,
                        dst, pixel_feat_stride, 
                        opt->pixel_opt);
                
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
        
        // allocate transposed convolution cache 1
        FloatImage tmp1;
        AllocateImage(&tmp1, img->width, img->height, 1);
        
        for(int n=0; n<pixel_opt->length; n++)
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
        FloatRect r;
        float *dst = raw_patch_feat.p;
        for(int x=0; x<n_raw_patch[1]*opt->step_x; x+=opt->step_x)
        {
            r.x1 = x;
            r.x2 = x + opt->size_x;
            for(int y=0; y<n_raw_patch[1]*opt->step_y; y+=opt->step_y)
            {
                
                r.y1 = y;
                r.y2 = y + opt->size_y;
                opt->f(img, &r, dst++, patch_feat_stride, opt);
            }
        }
    }
    
    // compute l2 norm of each raw patch
    NormL2Image(&raw_patch_feat, &raw_patch_norm);
            
    // group patch grids and normalize
    FloatImage patch_feat;      
    AllocateImage(&patch_feat, opt->bin_length, (n_raw_patch[0]-opt->numbin_y+1) * (n_raw_patch[1]-opt->numbin_x+1), 1);
    /* ..................*/
    
    int * patch_coordinate = new int[2*patch_feat.width];
    FreeImage(&raw_patch_feat);
    FreeImage(&raw_patch_norm);
    
    // group and encode in pyramids
    if(pyra_opt != NULL)
    {
        // encode in rectangle pyramid areas
        int nsplit = pyra_opt->npyramid;
        AllocateImage(desp, opt->coded_length, nsplit, 1);
        for(int i=0; i<nsplit; i++)
            EncodeRect(&patch_feat, pyra_opt->pyramid+i, patch_coordinate, desp->p+i*desp->height, opt->codebook);
    }
    else
    {
        // encode each patch
        AllocateImage(desp, opt->coded_length, patch_feat.width, 1);        
        for(int i=0; i<patch_feat.width; i++)
            EncodeSingle(patch_feat.p+i*patch_feat.height, desp->p+i*desp->height, opt->codebook);
    }
    
    FreeImage(&patch_feat);
}

#endif