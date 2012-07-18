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
    img->p = new float[width * height * depth];
    assert(img->p != NULL);
    
    img->width = width;
    img->height = height;
    img->depth = depth;   
    img->stride = height;   
}

// free image
void FreeImage(FloatImage * img)
{
    if(img->p != NULL)
        delete img->p;
}

// image norm
void NormL2Image(FloatImage * img, FloatImage * dst)
{
    
}
        
#endif