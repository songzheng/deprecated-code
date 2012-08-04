
#ifndef PIXEL_FEATURE_H
#define PIXEL_FEATURE_H

#include "image.h"

// ***************************** //
// for pixel-wise feature coding
struct PixelFeatureOpt;

typedef void (*FuncPixelFeatureInit)(PixelFeatureOpt * opt);
typedef void (*FuncPixelFeatureProc)(FloatImage * img, int x, int y, float * dst, PixelFeatureOpt * opt);

// pixel feature options:
//      image_depth: depth of image used
//      length: origin featue length
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
//      coded_length: coded featue length
struct PixelFeatureOpt
{
    char* name; //name of the pixel feature    
    FuncPixelFeatureInit func_init;
    FuncPixelFeatureProc func_proc;
    
    // feature options 
    int image_depth;
    double * param;
    int nparam;    
    int length;       
    
    // intermediate values
    int margin;
    int x1;
    int x2;
    int y1;
    int y2;
    int height;
    int width;
};

// ********************************* //
// pixel data implementation

// raw gray pixel 8-N & 4-N
void InitPixelGray8N(PixelFeatureOpt * opt)
{
    opt->image_depth = 1;
    opt->length = 9;
    opt->margin = 1;
}

inline void FuncPixelGray8N(FloatImage *img, int x, int y, float * dst,
        PixelFeatureOpt * opt)
{
    // Set up a circularly indexed neighborhood using nine pointers.
    // |--------------|
    // | p0 | p1 | p2 |
    // |--------------|
    // | p7 | c  | p3 |
    // |--------------|
    // | p6 | p5 | p4 |
    // |--------------|
    
    float * p = img->p + x*img->height + y;
    dst[0] = *(p-1-img->height);
    dst[1] = *(p-1);
    dst[2] = *(p-1+img->height);
    dst[3] = *(p+img->height);
    dst[4] = *(p+1+img->height);
    dst[5] = *(p+1);
    dst[6] = *(p+1-img->height);
    dst[7] = *(p-img->height);
    dst[8] = *p;
}
        
void InitPixelGray4N(PixelFeatureOpt * opt)
{
    opt->image_depth = 1;
    opt->length = 5;
    opt->margin = 1;
}

inline void FuncPixelGray4N(FloatImage *img, int x, int y, float * dst,
        PixelFeatureOpt * opt)
{
    // Set up a circularly indexed neighborhood using nine pointers.
    // |--------------|
    // |    | p0 |    |
    // |--------------|
    // | p3 | c  | p1 |
    // |--------------|
    // |    | p2 |    |
    // |--------------|
    
    float * p = img->p + x*img->height + y;
    
    dst[0] = *(p-1);
    dst[1] = *(p+img->height);
    dst[2] = *(p+1);
    dst[3] = *(p-img->height);
    dst[4] = *(p);
}

// raw color pixel
void InitPixelColor(PixelFeatureOpt * opt)
{
    opt->image_depth = 3;
    opt->length = 3;
    opt->margin = 0;
}

inline void FuncPixelColor(FloatImage *img, int x, int y, float * dst,
        PixelFeatureOpt * opt)
{
    float * p = img->p + x*img->height + y;
    dst[0] = *(p);
    dst[1] = *(p + img->width * img->height);
    dst[2] = *(p + 2*img->width * img->height);
}



// *********************************//
// entry

void InitPixelFeature(FloatMatrix * img, PixelFeatureOpt * opt)
{
    opt->func_init(opt);

    ASSERT(img->depth == opt->image_depth);    

    opt->x1 = opt->margin;
    opt->x2 = img->width-1-opt->margin;
    opt->y1 = opt->margin;
    opt->y2 = img->height-1-opt->margin;
    opt->width = img->width-2*opt->margin;
    opt->height = img->height-2*opt->margin;
//     mexPrintf("%d, %d, %d, %d, %d\n", img->depth, opt->image_depth, opt->height, opt->width, opt->margin);
}

void PixelFeature(FloatMatrix * img, FloatMatrix * feat, FloatMatrix * coord, PixelFeatureOpt * opt)
{    
    float * dst = feat->p;
    float * coord_y = coord->p;    
    float * coord_x = coord->p + coord->height * coord->width;    
    
    for (int x = opt->x1 ; x <= opt->x2 ; ++ x) {
        
        for (int y = opt->y1 ; y <= opt->y2 ; ++ y) {
            
            opt->func_proc(img, x, y, dst, opt);
            
            *(coord_y++) = (float)y;
            *(coord_x++) = (float)x;
            
            dst += opt->length;
        }
    }
    
}

#ifdef MATLAB_COMPILE

void MatReadPixelFeatureOpt(const mxArray * mat_opt, PixelFeatureOpt * opt)
{
    // get feature name
    mxArray * mx_name = mxGetField(mat_opt, 0, "name");
    ASSERT(mx_name != null);    
    opt->name = mxArrayToString(mx_name);
             
    // get parameters    
    mxArray * mx_param = mxGetField(mat_opt, 0, "param");
    if(mx_param == NULL || mxGetNumberOfElements(mx_param) == 0)
    {
        opt->param = NULL;
        opt->nparam = 0;
    }
    else
    {
        opt->param = (double *)mxGetPr(mx_param);
        opt->nparam = mxGetNumberOfElements(mxGetField(mat_opt, 0, "param"));
    }    
    
    opt->func_init = FUNC_INIT(PIXEL_FEATURE_NAME);
    opt->func_proc = FUNC_PROC(PIXEL_FEATURE_NAME);
}
#endif

#endif
