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
#include <stdio.h>

typedef unsigned int         uint;

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( uint* reference, uint* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold( uint* reference, uint* idata, const unsigned int len) 
{
//    const uint f_len = static_cast<uint>( len);
    for( unsigned int i = 0; i < len; ++i) 
    {
//        reference[i] = idata[i] * f_len;
        reference[i] = idata[i] * idata[i];
    }
}

