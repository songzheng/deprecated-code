
#include <mexutils.h>
#include "patch_feature.h"
#include "matlab_interface.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    FloatImage im;
    MatCopyToFloatMatrix(prhs[0], &im);
    
    bool use_default_patch = mxIsEmpty(prhs[1]);
    FloatImage patch_coord;
    PatchFeatureOpt opt;
    MatReadPatchFeatureOpt(prhs[2], &opt);
    
    if(!use_default_patch)
    {
        nlhs = 1;
        MatCopyToFloatMatrix(prhs[1], &patch_coord);               
        InitPatchFeature(&im, &opt, &patch_coord);
    }
    else
    {
        nlhs = 2;
        InitPatchFeature(&im, &opt);
        plhs[1] = MatAllocateFloatMatrix(&patch_coord, opt.height, opt.width, 2);
    }
    
    FloatImage patch_feat;
    plhs[0] = MatAllocateFloatMatrix(&patch_feat, opt.length, opt.height * opt.width, 1);    
    
    PatchFeature(&im, &patch_feat, &patch_coord, &opt);
    
    FreeImage(&im);    
    if(!use_default_patch) 
        FreeImage(&patch_coord);
}