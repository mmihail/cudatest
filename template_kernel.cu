/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <string.h>
//#include <string>

#define SDATA( index)      cutilBankChecker(sdata, index)

// Declare texture arrays
//texture<unsigned char, 1, cudaReadModeElementType> texRefCharset;
texture<unsigned char, 1, cudaReadModeElementType> texRefBaseKey;

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
//__constant__  const string d_charset="~!@#$%^&*()_-+ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"; 
//__constant__  const char d_charset[]  = "~!@#$%^&*()_-+ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"; 

template <int length> __global__ void testKernel(int *Ret, unsigned char* g_idata, int CharsetLength, uint4 HashToCrack)
{
  // shared memory
  // the size is determined by the host application
//__shared__  uint* sdata;
extern __shared__  unsigned char sdata[];
//unsigned int *shared_memory;

//const uint tidx=threadIdx.x + blockIdx.x * blockDim.x;
//const uint tidy=threadIdx.y + blockIdx.y * blockDim.y;
//const uint tidz=threadIdx.z + blockIdx.z * blockDim.z;

//const uint tid = tidx + tidy * gridDim.x * blockDim.x + tidz * gridDim.x * blockDim.x * gridDim.y * blockDim.y;


  // access thread id
//  const unsigned int tid = threadIdx.x;
    const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	

//idx = blockIdx.x * blockDim.x + threadIdx.x;
//  const int tid = (blockIdx.x+1) * (blockIdx.y+1) * threadIdx.x;	
  // access number of threads in this block
//  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  // use the bank checker macro to check for bank conflicts during host
  // emulation

//  SDATA(tid) = g_idata[0];
//  __syncthreads();

        unsigned int X[16];
//        const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
        int counter = tid;

        int oc;
        int a = 0, carry = 0;
    

	// rassovivaem simvoli v massiv X[16]    
        // These "if" statements are evaluated at compile time
        if (length >= 1)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 0) + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[0] = d_charset[a]; 

                counter = oc;
        }
        if (length >= 2)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 1) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[0] |= d_charset[a] << 8;

                counter = oc;
        }
        if (length >= 3)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 2) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[0] |= d_charset[a] << 16;

                counter = oc;
        }

        if (length >= 4)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 3) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[0] |= d_charset[a] << 24;

                counter = oc; 
        }

        if (length >= 5)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 4) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[1] = d_charset[a];

                counter = oc; 
        }

        if (length >= 6)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 5) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[1] |= d_charset[a] << 8;

                counter = oc;
        }

        if (length >= 7)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 6) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[1] |= d_charset[a] << 16;

                counter = oc;
        }

        if (length >= 8)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 7) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[1] |= d_charset[a] << 24;

                counter = oc;
        }

        if (length >= 9)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 8) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[2] = d_charset[a];

                counter = oc;
        }

        if (length >= 10)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 9) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[2] |= d_charset[a] << 8;

                counter = oc;
        }

        if (length >= 11)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 10) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[2] |= d_charset[a] << 16;

                counter = oc;
        }

        if (length >= 12)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 11) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[2] |= d_charset[a] << 24;

                counter = oc;
        }

        if (length >= 13)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 12) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[3] = d_charset[a];

                counter = oc;
        }

        if (length >= 14)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 13) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[3] |= d_charset[a] << 8;

                counter = oc;
        }

        if (length >= 15)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 14) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[3] |= d_charset[a] << 16;

                counter = oc;
        }

        if (length >= 16)
        {
                oc = counter / CharsetLength;
                a = tex1Dfetch(texRefBaseKey, 15) + carry + counter - oc * CharsetLength;
                if (a >= CharsetLength) { a -= CharsetLength; carry = 1; }
                else carry = 0;
                X[3] |= d_charset[a] << 24;

                counter = oc;
        }

	// PADDING
        switch(length)  // Evaluated at compile time
        {
                case 1:
                        X[ 0] |= (unsigned int)(0x00008000);
                        X[ 1] = X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = 
                        X[ 7] = X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 2:
                        X[ 0] |= (unsigned int)(0x00800000);
                        X[ 1] = X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = 
                        X[ 7] = X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 3:
                        X[ 0] |= (unsigned int)(0x80000000);
                        X[ 1] = X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = 
                        X[ 7] = X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 4:
                        X[ 1] = (unsigned int)(0x00000080);
                        X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 5:
                        X[ 1] |= (unsigned int)(0x00008000);
                        X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 6:
                        X[ 1] |= (unsigned int)(0x00800000);
                        X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 7:
                        X[ 1] |= (unsigned int)(0x80000000);
                        X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 8:
                        X[ 2] = (unsigned int)(0x00000080);
                        X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 9:
                        X[ 2] |= (unsigned int)(0x00008000);
                        X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 10:
                        X[ 2] |= (unsigned int)(0x00800000);
                        X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 11:
                        X[ 2] |= (unsigned int)(0x80000000);
                        X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] = 
                        X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 12:
                        X[ 3] = 128;
                        X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] = 
                        X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 13:
                        X[ 3] |= (unsigned int)(0x00008000);
                        X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] = 
                        X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 14:
                        X[ 3] |= (unsigned int)(0x00800000);
                        X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] = 
                        X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 15:
                        X[ 3] |= (unsigned int)(0x80000000);
                        X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] = 
                        X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
                case 16:
                        X[ 4] = (unsigned int)(0x00000080);
                        X[ 5] = X[ 6] = X[ 7] = X[ 8] = 
                        X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                        break;
        }

        X[14] = length << 3;
        X[15] = 0;

        unsigned int A, B, C, D;

#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))

#define P(a,b,c,d,k,s,t)                                \
{                                                                                                               \
        a += F(b,c,d) + X[k] + t; a = S(a,s) + b;                       \
}                                                                                                               \

    A = 0x67452301;
    B = 0xefcdab89;
    C = 0x98badcfe;
    D = 0x10325476;

#define F(x,y,z) (z ^ (x & (y ^ z)))

    P( A, B, C, D,  0,  7, 0xD76AA478 );
    P( D, A, B, C,  1, 12, 0xE8C7B756 );
    P( C, D, A, B,  2, 17, 0x242070DB );
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );
    P( A, B, C, D,  4,  7, 0xF57C0FAF );
    P( D, A, B, C,  5, 12, 0x4787C62A );
    P( C, D, A, B,  6, 17, 0xA8304613 );
    P( B, C, D, A,  7, 22, 0xFD469501 );
    P( A, B, C, D,  8,  7, 0x698098D8 );
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
    P( B, C, D, A, 11, 22, 0x895CD7BE );
    P( A, B, C, D, 12,  7, 0x6B901122 );
    P( D, A, B, C, 13, 12, 0xFD987193 );
    P( C, D, A, B, 14, 17, 0xA679438E );
    P( B, C, D, A, 15, 22, 0x49B40821 );

#undef F

#define F(x,y,z) (y ^ (z & (x ^ y)))

    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
        P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );

#undef F

#define F(x,y,z) (x ^ y ^ z)

    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );

#undef F

#define F(x,y,z) (y ^ (x | ~z))

    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );

#undef F

        if (A + 0x67452301 == HashToCrack.x &&
                B + 0xefcdab89 == HashToCrack.y &&
                C + 0x98badcfe == HashToCrack.z &&
                D + 0x10325476 == HashToCrack.w)
        {
                *Ret = tid;
        }

// __syncthreads();

//	g_odata[tid] = SDATA(tid);
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
