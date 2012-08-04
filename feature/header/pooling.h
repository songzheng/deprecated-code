 

// ***************************** //
// for image pooling

struct PoolingOpt
{
    CodeBook * codebook;
};

void RegularGridTrianglePooling(FloatMatrix * image, FloatMatrix * feat_input, FloatMatrix * coord_input, 
        FloatMatrix * feat_output, FloatMatrix * coord_output, PoolingOpt * opt)
{    
}


//     // group and encode in pyramids
//     if(opt->codebook == NULL)
//     {
//         MoveImage(&patch_feat, desp);
//     }
//     else
//     {
//         if(pyra_opt != NULL)
//         {
//             // encode in rectangle pyramid areas
//             int nsplit = pyra_opt->npyramid;
//             AllocateImage(desp, opt->codebook->length, nsplit, 1);
//             for(int i=0; i<nsplit; i++)
//                 EncodeRect(&patch_feat, pyra_opt->pyramid+i, patch_coordinate, desp->p+i*desp->height, 1, opt->codebook);
//         }
//         else
//         {
//             // encode each patch
//             AllocateImage(desp, opt->coded_length, patch_feat.width, 1);        
//             for(int i=0; i<patch_feat.width; i++)
//                 opt->codebook->f(patch_feat.p+i*patch_feat.height, desp->p+i*desp->height, 1, opt->codebook);
//         }
//     
//         FreeImage(&patch_feat);
//     }