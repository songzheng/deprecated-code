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

// matlab style height x width x depth
void AllocateImage(FloatImage * img, int height, int width, int depth)
{
#ifdef MATLAB_MEMORY
    img->p = (float *)mxCalloc(width * height * depth, sizeof(float));
#else
    img->p = new float[width * height * depth];
#endif

    assert(img->p != NULL);
    
    memset (img->p, 0, width * height * depth * sizeof(float)) ;
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
void NormalizeColumn(FloatImage * img)
{
    for(int x=0; x<img->width; x++)
    {
        float * dst = img->p + x*img->height;
        
        float norm = 0.0f;
        for(int y=0; y<img->height; y++)
            norm += dst[y]*dst[y];
        
        norm = vl_fast_sqrt_f(norm);
        
        for(int y=0; y<img->height; y++)
            dst[y] /= norm;
    }
}

// rectangle structure
struct FloatRect
{
    float x1;// left
    float y1;// top
    float x2;// right
    float y2;// bottom
};

// for column blockwise sparse matrix
struct FloatSparseMatrix
{
    float ** p;
    int ** i;
    int * block_num;
    int width;
    int height;
    int block_stride;
};
        
void AllocateSparseMatrix(FloatSparseMatrix * mat, int height, int width, int block_stride)
{
#ifdef MATLAB_MEMORY
    mat->p = (float **)mxCalloc(width, sizeof(float*));
    mat->i = (int **)mxCalloc(width, sizeof(int*));
    mat->block_num = (int *)mxCalloc(width, sizeof(int));
#else
    mat->p = new (float*)[width];
    mat->i = new (int*)[width];
    mat->block_num = new int[width];
#endif
    
    memset(mat->p, 0, width * sizeof(float *)) ;
    memset(mat->i, 0, width * sizeof(int *)) ;
    memset(mat->block_num, 0, width * sizeof(int)) ;
    
    mat->height = height;
    mat->width = width;
    mat->block_stride = block_stride;
}

void AllocateColumnSparseMatrix(FloatSparseMatrix * mat, int col_idx, int length_sparse)
{
    
#ifdef MATLAB_MEMORY
    mat->p[col_idx] = (float *)mxCalloc(length_sparse*mat->block_stride, sizeof(float));
    mat->i[col_idx] = (int *)mxCalloc(length_sparse, sizeof(int));;
#else
    mat->p[col_idx] = new float[length_sparse*mat->block_stride];
    mat->i[col_idx] = new int[length_sparse];
#endif
    
    memset(mat->p[col_idx], 0, length_sparse * sizeof(float)) ;
    memset(mat->i[col_idx], 0, length_sparse * sizeof(int)) ;
    mat->block_num[col_idx] = length_sparse;
}

void FreeSparseMatrix(FloatSparseMatrix * mat)
{
    for(int i=0; i<mat->width; i++)
    {
        if (mat->block_num[i] != 0)
        {
#ifdef MATLAB_MEMORY
            mxFree(mat->p[i]);
            mxFree(mat->i[i]);
#else
            delete[] mat->p[i];
            delete[] mat->i[i];
#endif
        }
    }
    
#ifdef MATLAB_MEMORY
    mxFree(mat->p);
    mxFree(mat->i);
    mxFree(mat->block_num);
#else
    delete[] mat->p;
    delete[] mat->i;
    delete[] mat->block_num;
#endif   
            
}
#endif