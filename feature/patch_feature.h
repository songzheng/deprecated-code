
#include "pgm.h"
#include "mathop.h"
#include "imopv.h"
#include <math.h>
#include <string.h>

struct FloatImage
{
    float * p;
    int width;
    int height;
    int stride;
};

// learning based encoding
struct CodeBook;
typedef void (*CodingFunc) (float * input, float * output, CodeBook * codebook);
struct CodeBook{
    char * name;
    CodingFunc f;
    float * mean;
    float * projection;
    float * centers;
    float * variances;
    
    int input_length;
    int length;
    int codebook_size;
};

// ***************************** //
// for pixel-wise feature coding
struct PixelFeatureOpt;
// pixel coding function
// img: the original image in Depth x Width x Height
// rect: area to extract pixel feature
// output: feature vector in Length x Width x Height
typedef void (*PixelFunc) (FloatImage * img, float * output, PixelFeatureOpt * opt);

// pixel coding options:
//      name: name of the pixel code
//      f: this function
//      length: coded featue length
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
struct PixelFeatureOpt
{
    char * name;
    PixelFunc f;
    float * param;
    int nparam;
    int length;
    CodeBook * codebook;    
};

// ***************************** //
// for rectangle dense grid patch feature
struct RectPatchFeatureOpt;
// patch code function 
typedef void (*PatchFunc) (FloatImage * img, float * output, int * patch_coor, int npatch, RectPatchFeatureOpt * opt);

// extract feature from rectangle patch option:
//      f: the patch feature function
//      param, nparam: the patch feature parameter
//      pixel_opt: use the pixel feature if f == null
//      numbin_{x,y}: bin number in each patch
//      numpixel_{x,y}: pixel in each bin
//      norm_numbin_{x,y}: normalize over how many bins
//      bin_length: feature length for each bin
//      length: patch featue length
//      param, nparam: code parameter
//      codebook: learning based encoding after extracting feature
struct RectPatchFeatureOpt
{
    PatchFunc f; 
    float * param;
    int nparam;
    
    PixelFeatureOpt * pixel_opt; 
    
    int numbin_x, numbin_y;
    int numpixel_x, numpixel_y;
    int norm_numbin_x, norm_numbin_y; 
    int bin_length;
    int length;
    CodeBook * codebook;
};


// ***************************** //
// for image pyramid
struct PyramidOpt
{
    FloatRect * pyramid;
    int npyramid;    
}

// ***************************** //
// main function
void Feature(FloatImage * img, FloatImage * desp, RectPatchFeatureOpt * opt, PyramidOpt * pyra_opt)
{
    
    // extract patch feature
    FloatImage patch_feat;
    int * patch_coordinate, npatch;
    AllocateFeature(&patch_feat, ...);
    
    if(opt->f == null)
    {
        FloatImage pixel_feat;
        AllocateFeature(&pixel_feat, ...);
        opt->pixel_opt->f(img, pixel_feat, opt->pixel_opt);

       
        for(int n=0; n<opt->pixel_opt->length; n++)
        {
            dst
            vl_imconvcoltri_f (dst, self->imHeight,
                    self->grads [bint], self->imWidth, self->imHeight,
                    self->imWidth,
                    self->geom.binSizeY, /* filt size */
                    1, /* subsampling step */
                    VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;
            
            vl_imconvcoltri_f (self->convTmp2, self->imWidth,
                    self->convTmp1, self->imHeight, self->imWidth,
                    self->imHeight,
                    self->geom.binSizeX,
                    1,
                    VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;
        }
    }else
    {
        opt->f(img, patch_feat, patch_coordinate, &npatch, opt);
    }
            
    // group and encode in pyramids
    if(pyra_opt != null)
    {
        int nsplit = pyra_opt->npyramid;
        AllocateFeature(desp, ...);
        for(int i=0; i<nsplit; i++)
            EncodePyramid(patch_feat, desp+i*dim, patch_coordinate, pyra_opt->pyramid+i)
    }
    else
    {
        EncodeFeature(patch_feat, desp);
    }
}