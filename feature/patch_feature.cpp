#pragma once

#include <mexutils.h>
#define MATLAB_MEMORY

#include "patch_feature.h"
#include "matlab_interface.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    FloatImage im;
    const mwSize * dims = mxGetDimensions(prhs[0]);
    ReadImage(prhs[0], &im);
    mexPrintf("%d x %d x %d\n", im.height, im.width, im.depth);
    
    RectPatchFeatureOpt opt;
    ReadRectPatchFeatureOpt(prhs[1], &opt);
    
    FloatImage desp, patch_coordinate;
    RectPatchFeature(&im, &desp, &patch_coordinate, &opt, NULL);
    
    plhs[0] = WriteImage(&desp);
    FreeImage(&im);    
    FreeImage(&desp);    
}