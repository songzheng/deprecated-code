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
// for rectangle dense grid patch feature
struct RectPatchFeatureOpt;
// patch code function (the function should be inline)
typedef void (*RectPatchFunc) (FloatImage * img, FloatRect * patch_rect, RectPatchFeatureOpt * opt);

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
    
    int length;
    
    FloatImage buffer_feat;
    FloatImage buffer_weight;
    
    int numbin_x, numbin_y;
    int sizebin_x, sizebin_y;
    
    CodeBook * codebook;
};

void RectPatchInitBuffer(int width, int height, RectPatchFeatureOpt * opt)
{
    // rectangle raw patch layout
    patch_height = int(1.0 * height / opt->sizebin_y + 0.5);
    patch_width = int(1.0 * width / opt->sizebin_x + 0.5);
    
    AllocateImage(&opt->buffer_feat, patch_height, patch_width, opt->length);
    AllocateImage(&opt->buffer_weight, patch_height, patch_width, 1);
}

void RectPatchReleaseBuffer(RectPatchFeatureOpt * opt)
{
    FreeImage(&opt->buffer_feat);
    FreeImage(&opt->buffer_weight);
}
// ***************************** //
// pixel to patch pooling
// average pooling
inline void RectPatchAveragePooling(FloatSparseMatrix * pixel_feat, int x, int y, 
        RectPatchFeatureOpt * opt)
{
    
    double xp = ((double)x+0.5)/(double)opt->sizebin_x - 0.5;
    double yp = ((double)y+0.5)/(double)opt->sizebin_y - 0.5;
    int ixp = (int)floor(xp+0.5);
    int iyp = (int)floor(yp+0.5);
        
    float * dst = patch_feat->p;
    int blocks[2] = {patch_feat->height, patch_feat->width};
    
    if (ixp >= 0 && iyp >= 0 && ixp < blocks[1] && iyp < blocks[0]) {
        for(int n=0; n<pixel_feat->block_num; n++)
        {
            float v = pixel_feat->p[0][n];
            int bin = pixel_feat->i[0][n];

            *(dst + ixp*blocks[0] + iyp + bin*blocks[0]*blocks[1]) += v;
        }
    }
}

// 2nd order moment pooling
// ??

// triangle filter pooling
inline void RectPatchTrianglePooling(FloatSparseMatrix * pixel_feat, int x, int y, 
        RectPatchFeatureOpt * opt)
{
    
    double xp = ((double)x+0.5)/(double)opt->sizebin_x - 0.5;
    double yp = ((double)y+0.5)/(double)opt->sizebin_y - 0.5;
    
    int ixp = (int)floor(xp);
    int iyp = (int)floor(yp);
    double vx0 = xp-ixp;
    double vy0 = yp-iyp;
    double vx1 = 1.0-vx0;
    double vy1 = 1.0-vy0;
    
    float * dst = patch_feat->p;
    int blocks[2] = {patch_feat->height, patch_feat->width};
    
    for(int n=0; n<pixel_feat->block_num; n++)
    {
        float v = pixel_feat->p[0][n];
        int bin = pixel_feat->i[0][n];
        
        if (ixp >= 0 && iyp >= 0) {
            *(dst + ixp*blocks[0] + iyp + bin*blocks[0]*blocks[1]) +=
                    vx1*vy1*v;
        }
        
        if (ixp+1 < blocks[1] && iyp >= 0) {
            *(dst + (ixp+1)*blocks[0] + iyp + bin*blocks[0]*blocks[1]) +=
                    vx0*vy1*v;
        }
        
        if (ixp >= 0 && iyp+1 < blocks[0]) {
            *(dst + ixp*blocks[0] + (iyp+1) + bin*blocks[0]*blocks[1]) +=
                    vx1*vy0*v;
        }
        
        if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
            *(dst + (ixp+1)*blocks[0] + (iyp+1) + bin*blocks[0]*blocks[1]) +=
                    vx0*vy0*v;
        }
    }
}
// max pooling
// ??

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
    // allocate buffer for raw patch feature  
    RectPatchInitBuffer(opt);        
    int n_raw_patch[2] = {opt->buffer_feat.height, opt->buffer_feat.width};           
    int patch_feat_stride = n_raw_patch[0] * n_raw_patch[1];
    
    if(opt->f == NULL)
    {
        // pixel-level buffer
        PixelFeatureOpt *pixel_opt = &opt->pixel_opt;        
        assert(pixel_opt->image_depth == img->depth);
        PixelFeatureInitBuffer(pixel_opt);
        
        int image_stride = img->height*img->width;
                       
        
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
                
                pixel_opt->f(center, p0, p1, p2, p3, p4, p5, p6, p7, image_stride,
                        pixel_opt);
                                
                opt->f_pooling(&opt->buffer, x, y, opt);
                
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
            for(int y=0; y<n_raw_patch[0]*opt->step_y; y+=opt->step_y)
            {
                
                r.y1 = y;
                r.y2 = y + opt->size_y;
                opt->f(img, &r, opt);
                
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