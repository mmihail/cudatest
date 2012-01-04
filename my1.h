
#pragma once

#include "vector_types.h"

typedef unsigned int         uint;

//int ret;
//__shared__ int       * d_ret;
__constant__ unsigned char    d_charset[256];
//__shared__ unsigned char    * d_charset;
//unsigned char    * d_charset;
//unsigned char* d_idata;
uint4                   htc;
int                             charset_length;
unsigned char   * charset;
int                             start, stop;
