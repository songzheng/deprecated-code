
#pragma once

#include <mexutils.h>
#define MATLAB_COMPILE

#include "patch_feature.h"
#include "matlab_interface.h"

// Raw feature code extraction
// •	Raw Pixel (Color / Gray)
// •	Local Normalized Pixel (center-surround)
// •	Gradient
// •	High Order Moment
// Input: Image
// Output: Dense Image Feature


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
        
    // copy input to matrix
    FloatImage img;
    MatCopyToFloatMatrix(prhs[0], &img);
        
    PixelFeatureOpt opt;
    MatReadPixelFeatureOpt(prhs[1], &opt);
    
    // initialize 
    InitPixelFeature(&img, &opt);
    
    // allocate return memory    
    FloatImage pixel_feat, pixel_coordinate;
    plhs[0] = MatAllocateFloatMatrix(&pixel_feat, opt.length, opt.height * opt.width, 1);
    plhs[1] = MatAllocateFloatMatrix(&pixel_coordinate, opt.height, opt.width, 2);
    
    // process
    PixelFeature(&img, &pixel_feat, &pixel_coordinate, &opt);    
    FreeImage(&img);
}