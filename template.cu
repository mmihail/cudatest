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
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "my1.h" 
#include <iostream>
#include <cutil.h>
#include "MyBignum.h"
#include <algorithm>


// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

// includes, kernels
#include <template_kernel.cu>

#define MAX_PASS_LENGTH 16 
#define MD5_INPUT_LENGTH 512 

using namespace std;

//int ret;
string        str_charset;
unsigned char	* h_idata,h_odata,h_zerodata;
unsigned int num_threads;
uint4 _threads,_block;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( uint* reference, uint* idata, const unsigned int len);

bool IsMD5Hash(string & hash)
{
        if (hash.length() != 32)
                return false;

        transform(hash.begin(), hash.end(), hash.begin(), ::tolower);

        for (int i = 0; i < 32; ++i)
                if (!((hash[i] >= '0' && hash[i] <= '9') || (hash[i] >= 'a' && hash[i] <= 'f')))
                        return false;

        return true;
}

void GetIntsFromHash(string hash)
{
        char* dummy = NULL;
        unsigned char bytes[16];
        for (int i = 0; i < 16; ++i)
                bytes[i] = (unsigned char)strtol(hash.substr(i * 2, 2).c_str(), &dummy, 16);

        htc.x = bytes[ 0] | bytes[ 1] << 8 | bytes[ 2] << 16 | bytes[ 3] << 24;
        htc.y = bytes[ 4] | bytes[ 5] << 8 | bytes[ 6] << 16 | bytes[ 7] << 24;
        htc.z = bytes[ 8] | bytes[ 9] << 8 | bytes[10] << 16 | bytes[11] << 24;
        htc.w = bytes[12] | bytes[13] << 8 | bytes[14] << 16 | bytes[15] << 24;
}


bool ParseCmdArgs(int argc, char ** argv)
{
        string temp;
        bool b = false;

        // defaults
	_threads.x=32;
	_block.x=64,_block.y=64,_block.z=1;

        for (int i = 1; i < argc; ++i)
        {
                temp = (string)argv[i];

                if (temp.substr(0, 8) == "--start=") { start = atoi(temp.substr(8).c_str()); }
                else if (temp.substr(0, 7) == "--stop=") { stop = atoi(temp.substr(7).c_str()); }
                else if (temp.substr(0, 10) == "--threads=") { _threads.x = atoi(temp.substr(10).c_str()); }
		else if (temp.substr(0, 9) == "--blocks=") { _block.x = _block.y = atoi(temp.substr(9).c_str()); }
                else if (IsMD5Hash(temp)) { GetIntsFromHash(temp); b = true; }
                else cerr << "#HOST WARNING#: " << temp << " is not a valid parameter!\n";
        }
//		  GetIntsFromHash("4ae71336e44bf9bf79d2752e234818a5");b=true;

        if (start < 1 || start > 16) { cerr << "#HOST WARNING#: Invalid \"start\" value! Using default(1).\n"; start = 1; }
        if (stop < 1 || stop > 16) { cerr << "#HOST WARNING#: Invalid \"stop\" value! Using default(16).\n"; stop = 16; }
        if (start > stop) { cerr << "#HOST WARNING#: \"start\" is bigger, than \"stop\". Using default \"start\"(1).\n"; start = 1; }

        num_threads = _threads.x*_block.x*_block.y;

        cout << "Charset: " << str_charset << ", start: "       << start << ", stop: " << stop << ", num_threads: " << num_threads << endl;
        cout << "block.x: " << _block.x << ", block.y: " << _block.y << ", block.z: " << _block.z << ", threads.x: " << _threads.x << endl;
	cout << "HTC= " << htc.x << htc.y << htc.z << htc.w << endl;
        if (!b) return false;
        return true;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{	
	// vremenno berem vse tut 
        str_charset="~!@#$%^&*()_-+ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";
        charset=NULL;
        start=3;
        stop=3;
	charset_length = str_charset.length();

// allocate host memory
//    float* h_idata = (float*) malloc( mem_size);
//        h_idata = new unsigned char[16];
    // allocate mem for the result on host side

//        ParseCmdArgs(argc, argv);
        if(!ParseCmdArgs(argc, argv))
        {
                cerr << "#HOST ERROR#: You have nost specified a hash. Nothing to do, exiting...\n";
                return 1;
        }

	charset = new unsigned char[charset_length];
        for (int i = 0; i < charset_length; ++i)
                charset[i] = str_charset[i];
	
 	   runTest( argc, argv);

		delete [] charset;
//		delete [] h_idata;
}


char *md5_pad(char *input) {
static char md5_padded[MD5_INPUT_LENGTH];
int x;
unsigned int orig_input_length;

        if (input == NULL) {
                return NULL;
        }

        // we store the length of the input (in bits) for later

        orig_input_length = strlen(input) * 8;

        // we would like to split the MD5 into 512 bit chunks with a special ending
        // the maximum input we support is currently 512 bits as we are not expecting a
        // string password to be larger than this

        memset(md5_padded, 0, MD5_INPUT_LENGTH);

        for(x = 0; x < strlen(input) && x < 56; x++) {
                md5_padded[x] = input[x];
        }

        md5_padded[x] = 0x80;

        // now we need to append the length in bits of the original message

        *((unsigned long *)md5_padded + 14) = orig_input_length;

        return md5_padded;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
// h_idata - host memory with input data
// h_odata - host memory with output data (result)
// d_idata - copy of h_idata to gpu memory
// d_odata - result
 
void
runTest( int argc, char** argv) 
{
//    bool bTestResult = true;

//    shrQAStart(argc, argv);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
//	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
//		cutilDeviceInit(argc, argv);
//	else
//		cutilSafeCall( cudaSetDevice( cutGetMaxGflopsDeviceId() ) );

//    unsigned int mem_size = sizeof( uint) * num_threads;
//    unsigned int mem_size = MAX_PASS_LENGTH * num_threads * sizeof(int);
//    unsigned int mem_size = MAX_PASS_LENGTH;
//    unsigned int mem_size1 = MAX_PASS_LENGTH * num_threads;
//    unsigned int mem_size = 32 * num_threads;

    // allocate host memory
    h_idata = (unsigned char*) malloc(charset_length);
//    h_zerodata = (unsigned char*) malloc( mem_size1);
    // initalize the memory
//    for( unsigned int i = 0; i < num_threads; ++i) 
//    {
//        h_idata[i] = (uint) i;
//    }

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    // allocate device memory
    unsigned char* d_idata;
    int       ret;
    int       * d_ret;
    int minusone = -1;

    cutilSafeCall( cudaMalloc( (void**) &d_idata, charset_length));
    cutilSafeCall( cudaMalloc( (void**)&d_ret, sizeof(int)));
    cutilSafeCall(cudaMemcpy(d_ret, &minusone, sizeof(int), cudaMemcpyHostToDevice));

//    cutilSafeCall(cudaMalloc((void**)&d_charset, sizeof(unsigned char) * charset_length));
//    cutilSafeCall(cudaMemcpy(d_charset, charset, sizeof(unsigned char) * charset_length, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpyToSymbol(d_charset, charset , charset_length));

    cudaBindTexture(0, texRefBaseKey, d_idata);
//    cudaBindTexture(0, texRefCharset, d_charset);

//testKernel<<< grid, threads, mem_size >>>(d_ret, d_idata, d_odata, i, charset_length, htc);

    // copy host memory to device
//    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );

    // allocate device memory for result
//    uint* d_odata;
//    unsigned char * d_odata;
//    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size1));

    // setup execution parameters
    // grid(x,y,1) x*y dolzno bit >= chem kol-vo procov u viduhi
    dim3  grid( _block.x, _block.y, 1);
    dim3  threads( _threads.x, 1, 1);


/*    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));


    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");
*/
    // allocate mem for the result on host side
//    h_odata = (unsigned char*) malloc( mem_size1);
//    unsigned char * h_odata = (unsigned char *) malloc( mem_size);
    // copy result from device to host
//    cutilSafeCall( cudaMemcpy( h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost) );

//    cutilCheckError( cutStopTimer( timer));
//    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
//    cutilCheckError( cutDeleteTimer( timer));

    timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

//===========================================================

//        const int maxthreadid = ((_block.y - 1) * _block.x + _block.x - 1) * _threads.x + _threads.x - 1;;
//        const uint maxthreadid = _threads.x*_block.x*_block.y - 1;
        const int maxthreadid = num_threads-1;

//        MyBignum bn_threadcount(threads.x * blocks.x * blocks.y);       // # of threads, that are executed during one kernel launch
        MyBignum bn_threadcount(num_threads);       // # of threads, that are executed during one kernel launch
//        MyBignum bn_threadcount(_threads.x*_block.x*_block.y);       // # of threads, that are executed during one kernel launch
//        MyBignum bn_base(charset_length);
//	  int bn_threadcount=num_threads;
	 unsigned int counter;

       for (int i = start; i < stop + 1; ++i)
        {
//                if (verbose) cout << "Actual length is " << i << ".\n";

                // Calculate the number of strings of length "i"
                MyBignum bn_full(1);

                for (int j = 0; j < i; ++j)
                        bn_full *= charset_length;

                MyBignum bn_counter(0);
//                memset(h_idata, 0, mem_size);
//                memset(h_odata, 0, mem_size1);
//                memset(h_zerodata, 0, mem_size1);
                memset(h_idata, 0, charset_length);
//                memset(h_odata, 0, charset_length);
                do
                {
                        bn_counter += bn_threadcount;

//			cudaThreadSynchronize();
//    			cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
    			cutilSafeCall( cudaMemcpy( d_idata, h_idata, charset_length, cudaMemcpyHostToDevice) );
//    			cutilSafeCall( cudaMemcpy( d_odata, h_zerodata, mem_size1, cudaMemcpyHostToDevice) );
//  		  	cutilSafeCall( cudaMemcpy( &ret, d_ret, sizeof(int) ,cudaMemcpyDeviceToHost) );
//			cudaThreadSynchronize();

    			// execute the kernel
//    			testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata, i, charset_length, htc);
//    			testKernel<<< grid, threads >>>(d_ret, d_idata, d_odata, i, charset_length, htc);
//    			testKernel<<< grid, threads >>>(d_ret, d_idata, i, charset_length, htc);

        switch (i)
        {
                case  1: testKernel <1> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  2: testKernel <2> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  3: testKernel <3> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  4: testKernel <4> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  5: testKernel <5> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  6: testKernel <6> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  7: testKernel <7> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  8: testKernel <8> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  9: testKernel <9> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  10: testKernel <10> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  11: testKernel <11> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  12: testKernel <12> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  13: testKernel <13> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  14: testKernel <14> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  15: testKernel <15> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
                case  16: testKernel <16> <<< grid, threads >>> (d_ret, d_idata, charset_length, htc); break;
        }



			cudaThreadSynchronize();
  		  	cutilSafeCall( cudaMemcpy( &ret, d_ret, sizeof(int) ,cudaMemcpyDeviceToHost) );
//			cudaThreadSynchronize();

//			cout << "Ret = " << ret << endl;

                        if (ret == -2) {
				cout << "Zhopa" << endl; 
				return; 
			} // CUDA error -> panic, end of world, abort cracking :(
                        else if (ret != -1)     //      Oh, yeah.. Keyword f.o.u.n.d. :)
                        {
				cout << "Ret = " << ret << " i= " << i << endl;
                                cout << "bn_counter= " << bn_counter.ToString() << " bn_full= " << bn_full.ToString() << endl;
				cout << "Nashel: "; 

                                // Recover the string from the return value
                                counter = ret;
                                for (int j = 0, a = 0, carry = 0; j < i; ++j, counter /= charset_length)
                                {
                                        a = h_idata[j] + carry + counter % charset_length;
                                        if (a >= charset_length) { carry = 1; a -= charset_length; }
                                        else carry = 0;
					cout << str_charset[a];
                                }
                                cout << endl;

				cutilCheckError( cutStopTimer( timer));
				printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
				cutilCheckError( cutDeleteTimer( timer));

                                return;
                        }
//                          cutilSafeCall( cudaMemcpy( h_odata, d_odata, mem_size1 ,cudaMemcpyDeviceToHost) );
//                        cutilSafeCall(cudaMemcpyFromSymbol(h_odata, d_charset, 16));
//                          for (int j = 0; j < maxthreadid; ++j) {
//                                printf("h_i=%i\n",(uint)(h_idata[j]));
//                                printf("h_o=%i %i\n",(uint)(h_odata[j]),j);
//                          }
//                          cout << endl;

                        // Advance texRefBaseKey
                        counter = maxthreadid;
                        for (int j = 0, a = 0, carry = 0; j < i; ++j, counter /= charset_length)
                        {
                                a = h_idata[j] + carry + counter % charset_length;
                                if (a >= charset_length) { carry = 1; a -= charset_length; }
                                else carry = 0;
                                h_idata[j] = a;
//                                cout << str_charset[h_idata[j]];
//                                printf("a=%i;\n",h_idata[j]);
                        }
//                              cout << endl;
//                              printf(" === count=%i;\n",counter);
//                              cout << "bn_counter= " << bn_counter.ToString() << " bn_full= " << bn_full.ToString() << endl;

                }while(bn_counter < bn_full);
	    cout << "pass len = " << i << " not found" << endl; 
        }
    cout << "Ne nashel." << endl; 

// ========================================================================

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

// ========================================================================

    // cleanup memory
    free( h_idata);
//    free( h_odata);
//    free( reference);

//        cudaUnbindTexture(texRefCharset);
        cudaUnbindTexture(texRefBaseKey);

    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_ret));
//    cutilSafeCall(cudaFree(d_charset));
//    cutilSafeCall(cudaFree(d_odata));

    cutilDeviceReset();
//    shrQAFinishExit(argc, (const char **)argv, (bTestResult ? QA_PASSED : QA_FAILED) );
}
