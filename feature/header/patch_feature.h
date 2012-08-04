#ifndef PATCH_FEATURE_H
#define PATCH_FEATURE_H
        
#include "image.h"
#include "pixel_feature.h"

// coding with pixel coding method
#include "coding.h"

// ***************************** //
// for patch feature extraction
                
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

struct PatchFeatureOpt;
typedef void (*FuncPatchFeatureInit)(FloatImage * img, PatchFeatureOpt * opt);
typedef void (*FuncPatchFeatureProc)(FloatImage * img, int x, int y, float * dst, PatchFeatureOpt * opt);

struct PatchFeatureOpt
{
    char * name;
    FuncPatchFeatureInit func_init;
    FuncPatchFeatureProc func_proc;
    
    int image_depth;
    double * param;
    int nparam;
    bool use_pixel_feature;
    PixelFeatureOpt pixel_opt; 
    CodingOpt pixel_coding_opt;
    
    int size_x, size_y;    
    int length;
    
    int height, width;
    bool use_default_patch;
};

// ***************************** //

// entry function for patch feature
void InitPatchFeature(FloatImage * img, PatchFeatureOpt * opt, FloatImage * coord = NULL)
{    
    if(opt->use_pixel_feature)
    {
        InitPixelFeature(img, &opt->pixel_opt);
        InitCoding(&opt->pixel_coding_opt);
    
        ASSERT(opt->pixel_opt.length == opt->pixel_coding_opt.length_input);
        opt->length = opt->pixel_coding_opt.length;
    }
    else
    {
    	opt->func_init(img, opt);
    }

    // default dense rectangle patch 
    if(coord == NULL){
        int step_x, step_y;
        step_x = opt->size_x/2;
        step_y = opt->size_y/2;

        opt->use_default_patch = true;
        opt->height = int(1.0 * img->height / step_y + 0.5);
        opt->width = int(1.0 * img->width / step_x + 0.5);
    }    
    else{
        opt->use_default_patch = false;
        opt->height = coord->height;
        opt->width = coord->width;
    }
}

void PatchFeature(FloatImage * img, FloatImage * feat, FloatImage * coord, PatchFeatureOpt * opt)
{        
    int size_x = opt->size_x, 
            size_y = opt->size_y;
        
    if(opt->use_default_patch)
    {        
        float * coord_y = coord->p;
        float * coord_x = coord->p + opt->height*opt->width;
        int step_x, step_y;
        step_x = size_x/2;
        step_y = size_y/2;
        
        for(int ix=0; ix<opt->width; ix++)
        {
            for(int iy=0; iy<opt->height; iy++)
            {
                *(coord_y++) = float(iy * step_y);
                *(coord_x++) = float(ix * step_x);
            }
        }
    }
    
    if(opt->use_pixel_feature)
    {
        FloatMatrix pixel_feat, pixel_coord;
        FloatSparseMatrix pixel_coding;
        PixelFeatureOpt * pixel_opt = &opt->pixel_opt;
        CodingOpt * pixel_coding_opt = &opt->pixel_coding_opt;
        
        // cache pixel level feature
        AllocateImage(&pixel_feat,
                pixel_opt->length,
                pixel_opt->height*pixel_opt->width,
                1);
        AllocateImage(&pixel_coord,
                pixel_opt->height,
                pixel_opt->width,
                2);
        
        PixelFeature(img, &pixel_feat, &pixel_coord, &opt->pixel_opt);
        
        // cache pixel level coded feature
        AllocateSparseMatrix(&pixel_coding,
                pixel_coding_opt->length,
                pixel_opt->height*pixel_opt->width,
                pixel_coding_opt->block_num,
                pixel_coding_opt->block_size);
        
        Coding(&pixel_feat, &pixel_coding, pixel_coding_opt);
        
        FreeImage(&pixel_feat);
        FreeImage(&pixel_coord);
        
        // pixel weight in patch
        FloatMatrix pixel_weight;
        AllocateImage(&pixel_weight, size_y, size_x, 1);
        for(int px=0; px<size_x; px++){
            for(int py=0; py<size_y; py++){
                float vx = 1 - abs(px+0.5f - 1.0f*size_x/2) / (1.0f*size_x/2),
                        vy = 1 - abs(py+0.5f - 1.0f*size_y/2) / (1.0f*size_y/2);
                
                pixel_weight.p[px*size_y + py] = vx * vy;
            }
        }
        
        // pool encoded feature to patches
        int npatch = coord->width * coord->height;
        float * coord_y = coord->p;
        float * coord_x = coord->p + npatch;
        
        for(int n=0; n<npatch; n++){
            int y = (int)(*(coord_y++));
            int x = (int)(*(coord_x++));
            
//         mexPrintf("%d, %d\n", y, x);
            float * dst = feat->p + n*opt->length;
            for(int px=0; px<size_x; px++){
                for(int py=0; py<size_y; py++){
                    float v = pixel_weight.p[px*size_y + py];
                    
                    int iy = MIN(MAX(y+py-pixel_opt->margin, 0), pixel_opt->height-1);
                    int ix = MIN(MAX(x+px-pixel_opt->margin, 0), pixel_opt->width-1);
                    
                    int idx = iy + ix*pixel_opt->height;
                    AddSparseMatrix(&pixel_coding, idx, v, dst);
                }
            }
        }
        FreeImage(&pixel_weight);
        
        FreeSparseMatrix(&pixel_coding);
    }
    else
    {    
        // patch level
        int npatch = coord->width * coord->height;
        float * coord_y = coord->p;
        float * coord_x = coord->p + npatch;
        for(int n=0; n<npatch; n++){
            int y = (int)(*(coord_y++));
            int x = (int)(*(coord_x++));
            float * dst = feat->p + n*opt->length;
            opt->func_proc(img, x, y, dst, opt);
        }
    }
    
}

#ifdef MATLAB_COMPILE
// matlab helper function
void MatReadPatchFeatureOpt(const mxArray * mat_opt, PatchFeatureOpt * opt)
{    
    // get feature name
    mxArray * mx_name = mxGetField(mat_opt, 0, "name");
    ASSERT(mx_name != null);
    opt->name = mxArrayToString(mx_name);
    
    mxArray * mx_pixel_opt = mxGetField(mat_opt, 0, "pixel_opt");
    
    if(mx_pixel_opt == NULL || mxIsEmpty(mx_pixel_opt))
    {
        MatReadPixelFeatureOpt(mx_pixel_opt, &opt->pixel_opt);
        opt->use_pixel_feature = true;
        opt->param = NULL;
        opt->nparam = 0;
        
        mxArray * mx_pixel_coding_opt = mxGetField(mat_opt, 0, "pixel_coding_opt");
        ASSERT(mx_pixel_coding_opt != NULL);
        MatReadCodingOpt(mx_pixel_coding_opt, &opt->pixel_coding_opt);
    }
    else
    {
        // use patch feature
        opt->use_pixel_feature = false;
        opt->param = (double *)mxGetPr(mxGetField(mat_opt, 0, "param"));
        opt->nparam = mxGetNumberOfElements(mxGetField(mat_opt, 0, "param"));
        opt->func_init = FUNC_INIT(PATCH_FEATURE_NAME);
        opt->func_proc = FUNC_PROC(PATCH_FEATURE_NAME);
    }
            
    // get patch size   
    COPY_INT_FIELD(size_x);
    COPY_INT_FIELD(size_y);
}
#endif

#endif