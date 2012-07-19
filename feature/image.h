#ifndef FLOAT_IMAGE_H
#define FLOAT_IMAGE_H

#include <assert.h>

struct FloatImage
{
    float * p;
    int width;
    int height;
    int depth;
    int stride;
};

struct FloatRect
{
    float x1;// left
    float y1;// top
    float x2;// right
    float y2;// bottom
};

// matlab style height x width x depth
void AllocateImage(FloatImage * img, int height, int width, int depth)
{
#ifdef MATLAB_MEMORY
    img->p = (float *)mxCalloc(width * height * depth, sizeof(float));
#else
    img->p = new float[width * height * depth];
#endif

    assert(img->p != NULL);
    
    memset (img->p, 0, width * height * depth) ;
    img->width = width;
    img->height = height;
    img->depth = depth;   
    img->stride = height;   
}

// move image to another struct
void MoveImage(FloatImage * src, FloatImage * dst)
{
    dst->p = src->p;
    dst->width = src->width;
    dst->height = src->height;
    dst->depth = src->depth;   
    dst->stride = src->height;   
    src->p = NULL;
}

// free image
void FreeImage(FloatImage * img)
{
    if(img->p != NULL)
#ifdef MATLAB_MEMORY
        mxFree(img->p);
#else
        delete[] img->p;
#endif
}

// image norm
void NormL2Image(FloatImage * img, FloatImage * dst)
{
    
}
        
#endif