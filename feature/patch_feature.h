// vlfeat headers
#include <vl/mathop.h>
#include <vl/imopv.h>
#include <math.h>
#include <string.h>

#ifndef PATCH_FEATURE_H
#define PATCH_FEATURE_H
#include "image.h"
#include "coding.h"
#include "pixel_feature.h"


// ***************************** //
// for rectangle dense grid patch feature
struct RectPatchFeatureOpt;
// patch code function (the function should be inline)
typedef void (*RectPatchFunc) (FloatImage * img, FloatRect * patch_rect, RectPatchFeatureOpt * opt);
// pixel to patch pooling
typedef void (*RectPatchPoolingFunc)(FloatSparseMatrix * pixel_feat, int x, int y, RectPatchFeatureOpt * opt);

// extract feature from rectangle patch option:
//      name: the patch feature name
//      f: the patch feature function
//      param, nparam: the patch feature parameter
//      pixel_opt: use the pixel feature if f == null
//      numbin_{x,y}: bin number in each patch
//      sizebin_{x,y}: patch step of each bin
//      length: patch featue length of each bin
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
struct RectPatchFeatureOpt
{
    char * name;
    
    RectPatchFunc f;
    RectPatchPoolingFunc fp;
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

void RectPatchInitBuffer(int height, int width, RectPatchFeatureOpt * opt)
{
    // rectangle raw patch layout
//     printf("%d, %d, %d\n", opt->sizebin_y, opt->sizebin_x, opt->length);
    int patch_height = int(1.0 * height / opt->sizebin_y + 0.5);
    int patch_width = int(1.0 * width / opt->sizebin_x + 0.5);
    
    AllocateImage(&opt->buffer_feat, patch_height, patch_width, opt->length);
    AllocateImage(&opt->buffer_weight, patch_height, patch_width, 1);
}

void RectPatchNormalizeBuffer(RectPatchFeatureOpt * opt)
{    
    float * src = opt->buffer_feat.p;
    float * weight = opt->buffer_weight.p;
    
    int feat_stride = opt->buffer_feat.width*opt->buffer_feat.height;
//     mexPrintf("%dx%d\n", opt->buffer_feat.height, opt->buffer_feat.width);
    for(int x=0; x<opt->buffer_feat.width; x++)
    {
        for(int y=0; y<opt->buffer_feat.height; y++)
        {
            if(*weight == 0)
            {
                src ++;
                weight ++;
                continue;
            }
            
//             for(int n=0; n<opt->buffer_feat.depth; n++)
//                 src[n*feat_stride] /= *weight;
            src ++;
            weight ++;
        }
    }
}

void RectPatchReleaseBuffer(RectPatchFeatureOpt * opt)
{
    FreeImage(&opt->buffer_feat);
    FreeImage(&opt->buffer_weight);
}

// ***************************** //
// average pooling
inline void RectPatchAveragePooling(FloatSparseMatrix * pixel_feat, int x, int y, 
        RectPatchFeatureOpt * opt)
{
    
    double xp = ((double)x+0.5)/(double)opt->sizebin_x - 0.5;
    double yp = ((double)y+0.5)/(double)opt->sizebin_y - 0.5;
    int ixp = (int)floor(xp+0.5);
    int iyp = (int)floor(yp+0.5);
        
    float * dst = opt->buffer_feat.p;  
    int blocks[2] = {opt->buffer_feat.height, opt->buffer_feat.width};
    
    if (ixp >= 0 && iyp >= 0 && ixp < blocks[1] && iyp < blocks[0]) {
        for(int n=0; n<pixel_feat->block_num[0]; n++)
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
    
    int blocks[2] = {opt->buffer_feat.height, opt->buffer_feat.width};
    double xp = ((double)x+0.5)/(double)opt->sizebin_x - 0.5;
    double yp = ((double)y+0.5)/(double)opt->sizebin_y - 0.5;
    
    int ixp = (int)floor(xp);
    int iyp = (int)floor(yp);
    double vx0 = xp-ixp;
    double vy0 = yp-iyp;
    double vx1 = 1.0-vx0;
    double vy1 = 1.0-vy0;
    
    double v00 = vx0*vy0,
            v10 = vx1*vy0,
            v01 = vx0*vy1,
            v11 = vx1*vy1;
    
    float * dst = opt->buffer_feat.p;    
    
    for(int n=0; n<pixel_feat->block_num[0]; n++)
    {
        float v = pixel_feat->p[0][n];
        int bin = pixel_feat->i[0][n];
        
        if (ixp >= 0 && iyp >= 0) {
            *(dst + ixp*blocks[0] + iyp + bin*blocks[0]*blocks[1]) +=
                    v11*v;
            
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
// main function for dense rectangle patch features
void RectPatchFeature(FloatImage * img, FloatImage * patch_feat, FloatImage * patch_coordinate, RectPatchFeatureOpt * opt)
{    
    // allocate buffer for raw patch feature  
    RectPatchInitBuffer(img->height, img->width, opt);        
    int n_raw_patch[2] = {opt->buffer_feat.height, opt->buffer_feat.width};           
    int patch_feat_stride = n_raw_patch[0] * n_raw_patch[1];   
    
    if(opt->f == NULL)
    {
        opt->fp = RectPatchTrianglePooling;
        // pixel-level buffer
        PixelFeatureOpt *pixel_opt = &opt->pixel_opt;    
        assert(pixel_opt->image_depth == img->depth);
        PixelFeatureInitBuffer(pixel_opt);
        
//             mexPrintf("%d, %d\n", pixel_opt->buffer.width, pixel_opt->buffer.block_num[0]);
//             return;    
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
                        
            for (int y = 1 ; y < img->height-1 ; ++ y) {       
                
                pixel_opt->f(center, p0, p1, p2, p3, p4, p5, p6, p7, image_stride,
                       pixel_opt);
                                
                opt->fp(&pixel_opt->buffer, x, y, opt);
                
                p0 ++;
                p1 ++;
                p2 ++;
                p3 ++;
                p4 ++;
                p5 ++;
                p6 ++;
                p7 ++;
                center ++;
            }
        }       
        PixelFeatureReleaseBuffer(pixel_opt);
    }
    else
    {
        // patch level
        FloatRect r;
        for(int x=0; x<n_raw_patch[1]*opt->sizebin_x; x+=opt->sizebin_x)
        {
            r.x1 = x;
            r.x2 = x + opt->sizebin_x*2 - 1;
            for(int y=0; y<n_raw_patch[0]*opt->sizebin_y; y+=opt->sizebin_y)
            {
                
                r.y1 = y;
                r.y2 = y + opt->sizebin_y*2 - 1;
                opt->f(img, &r, opt);
            }
        }
    }
    
    // normalize raw patch
    RectPatchNormalizeBuffer(opt);
            
    // group patch grids    
    AllocateImage(&patch_feat, opt->length*opt->numbin_x*opt->numbin_y, (n_raw_patch[0]-opt->numbin_y+1) * (n_raw_patch[1]-opt->numbin_x+1), 1);
    
    // patch top-left corner
    AllocateImage(patch_coordinate, 2, patch_feat.width, 1);
    float * dst = patch_coordinate->p;
    
    for(int ix=0; ix<n_raw_patch[1]-opt->numbin_x+1; ix++)
    {
        for(int iy=0; iy<n_raw_patch[0]-opt->numbin_y+1; iy++)
        {
            *(dst++) = float(iy * opt->sizebin_y);
            *(dst++) = float(ix * opt->sizebin_x);            
        }
    }
    
    for(int bin_x=0; bin_x<opt->numbin_x; bin_x++)
    {
        for(int bin_y=0; bin_y<opt->numbin_y; bin_y++)
        {
            int bin_start = bin_x * opt->numbin_y + bin_y;
            // add different weight for each bin
            float bin_norm = 1.0f;
            for(int i=0; i<opt->length; i++)
            {
                float * dst = patch_feat.p + bin_start * opt->length + i;
                float * src = opt->buffer_feat.p + patch_feat_stride*i;
                for(int ix=bin_x; ix<bin_x+n_raw_patch[1]-opt->numbin_x+1; ix++)
                {
                    for(int iy=bin_y; iy<bin_y+n_raw_patch[0]-opt->numbin_y+1; iy++)
                    {
                        *dst = bin_norm * src[ix*n_raw_patch[0] + iy];
                        dst += patch_feat.height;
                    }
                }
            }
        }
    }
    RectPatchReleaseBuffer(opt);   
    
    // normalize each raw patch
    NormalizeColumn(&patch_feat);        
}

#endif