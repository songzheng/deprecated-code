
#ifndef FEATURE_CODING_H
#define FEATURE_CODING_H
#include "image.h"

// learning based encoding
struct CodeBook;
typedef void (*CodingFunc) (float * src, float * dst, int dst_stride, CodeBook * codebook);

// code book struct:
//      name: name of code book [feature name] + [coding option]
//      f: coding function
//      mean, projection, centers, variances: code book content, null if N/A
//      length: origin featue length
//      codebook_size: number of code word
//      coded_length: coded featue length
struct CodeBook{
    char * name;
    CodingFunc f;
    float * mean;
    float * projection;
    float * centers;
    float * variances;
    
    int length;
    int codebook_size;
    int coded_length;
};

void EncodeRect(FloatImage *feat, FloatRect * rect, FloatImage * coor, float * dst, int dst_stride, CodeBook * codebook)
{
}


#endif