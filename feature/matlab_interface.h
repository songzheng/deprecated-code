
#include "mex.h"
#include "patch_feature.h"

#define _COPY_INT_FIELD(name, field) opt->name = (mxGetField(mat_opt, 0, (field))==NULL)?0:(int)mxGetScalar(mxGetField(mat_opt, 0, (field)))

void ReadPixelFeatureOpt(const mxArray * mat_opt, PixelFeatureOpt * opt)
{
    // get feature name
    mxArray * mx_name = mxGetField(mat_opt, 0, "name");
    if( mx_name == NULL)
    {
        opt->name = NULL;
        return;
    }
    
    opt->name = mxArrayToString(mx_name);
    
    // find pixel feature function
    opt->f = NULL;
    if(!strcmp(opt->name, "HOG"))
        opt->f = _PixelHOG9Bin;
    
    if(!strcmp(opt->name, "LBP"))
        opt->f = _PixelLBP59;
    
    if(opt->f == NULL)
        return;
         
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
    _COPY_INT_FIELD(image_depth, "image_depth");
    _COPY_INT_FIELD(length, "length");
    
    // get code book
    opt->codebook = NULL;
    // ........... to be implemented ..........//
}


void ReadRectPatchFeatureOpt(const mxArray * mat_opt, RectPatchFeatureOpt * opt)
{
    // get feature name
    mxArray * mx_name = mxGetField(mat_opt, 0, "name");
    opt->f = NULL;
    opt->name = NULL;
    if( mx_name == NULL || mxGetNumberOfElements(mx_name) == 0)
    {
        // use pixel feature
        mxArray * mx_pixel_opt = mxGetField(mat_opt, 0, "pixel_opt");
        if(mx_pixel_opt == NULL)
            return;
        
        ReadPixelFeatureOpt(mx_pixel_opt, &opt->pixel_opt);        
    }
    else
    {
        // use patch feature
        opt->name = mxArrayToString(mx_name);
        
        // find patch feature function
        if(opt->f == NULL)
            return;
                
        // get parameters
        opt->param = (double *)mxGetPr(mxGetField(mat_opt, 0, "param"));
        opt->nparam = mxGetNumberOfElements(mxGetField(mat_opt, 0, "param"));
        _COPY_INT_FIELD(image_depth, "image_depth");
        _COPY_INT_FIELD(length, "length");
    }
    
    // get grid parameters    
    _COPY_INT_FIELD(numbin_x, "numbin_x");
    _COPY_INT_FIELD(numbin_y, "numbin_y");
    _COPY_INT_FIELD(step_x, "step_x");
    _COPY_INT_FIELD(step_y, "step_y");
    _COPY_INT_FIELD(size_x, "size_x");
    _COPY_INT_FIELD(size_y, "size_y");
    _COPY_INT_FIELD(bin_length, "bin_length");
    _COPY_INT_FIELD(length, "length");
    
    // get code book
    opt->codebook = NULL;
    // ........... to be implemented ..........//
}

void ReadImage(const mxArray * mx_image, FloatImage * image)
{
    const mwSize ndim = mxGetNumberOfDimensions(mx_image);
    int dims[3];
    const mwSize * src_dims = mxGetDimensions(mx_image);
    dims[0] = src_dims[0];
    dims[1] = src_dims[1];
    if(ndim == 3)
        dims[2] = src_dims[2];
    else if(ndim == 2)
        dims[2] = 1;
    else
        return;
//     mexPrintf("%d x %d x %d\n", dims[0], dims[1], dims[2]);
//     return;
    AllocateImage(image, dims[0], dims[1], dims[2]);
    
    if(mxGetClassID(mx_image) == mxDOUBLE_CLASS)
    {
        double * src = (double *)mxGetPr(mx_image);

        for(int i=0; i<mxGetNumberOfElements(mx_image); i++)
            image->p[i] = (float)src[i];
    }
    else if(mxGetClassID(mx_image) == mxUINT8_CLASS)
    {        
        unsigned char * src = (unsigned char * )mxGetPr(mx_image);

        for(int i=0; i<mxGetNumberOfElements(mx_image); i++)
            image->p[i] = (float)src[i];
    }
        
}

mxArray * WriteImage(FloatImage * image)
{
    mwSize dims[3] = {image->height, image->width, image->depth};   
    
    mxArray * mx_image = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    double * dst = mxGetPr(mx_image);
    
    for(int i=0; i<mxGetNumberOfElements(mx_image); i++)
        dst[i] = (double)image->p[i];
    
    return mx_image;
}