#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <inttypes.h>
#include <CL/cl.h>
#include "utils.h"

#define MAXGPU 3
#define MAXK 1024
#define MAXN 16777216
#define MAXWORKITEM 256 
uint32_t A[MAXN/MAXWORKITEM];

int main(int argc, char *argv[]) {
    // get platform
    cl_int status;
    cl_platform_id platform_id;
    cl_uint platform_amount;
    status = clGetPlatformIDs(1, &platform_id, &platform_amount);
    assert(status == CL_SUCCESS && platform_amount == 1);
    // get GPU
    cl_device_id GPU[MAXGPU];
    cl_uint GPU_amount;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, MAXGPU, GPU, &GPU_amount);
    assert(status == CL_SUCCESS);
    // create context
    cl_context context = clCreateContext(NULL, 1, GPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
    // create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, GPU[0], 0, &status);
    assert(status == CL_SUCCESS);   
    // create program with source
    FILE *kernelfp = fopen("vecdot.cl", "r");
    assert(kernelfp != NULL);
    char kernelBuffer[MAXK];
    const char *constKernelSource = kernelBuffer;
    size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
    cl_program program = clCreateProgramWithSource(context, 1, &constKernelSource, &kernelLength, &status);
    assert(status == CL_SUCCESS);
    // build program
    status = clBuildProgram(program, 1, GPU, NULL, NULL, NULL);
    assert(status == CL_SUCCESS);
    // create kernel
    cl_kernel kernel = clCreateKernel(program, "dotAndSum", &status);
    assert(status == CL_SUCCESS);
    // create buffer
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint)*MAXN/MAXWORKITEM, A, &status);
    assert(status == CL_SUCCESS);    
    // start
    int N;
    uint32_t key1, key2;
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        uint32_t padding = 0;
        while (N%MAXWORKITEM != 0) {
            padding += encrypt(N, key1) * encrypt(N, key2);
            N++;
        }
        // set kernel arg
        status = clSetKernelArg(kernel, 0, sizeof(cl_uint), (void*)&key1);
        assert(status == CL_SUCCESS);
        status = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void*)&key2);
        assert(status == CL_SUCCESS);
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferA);
        assert(status == CL_SUCCESS);
        // data shape
    	size_t globalThreads[] = {(size_t)N};
    	size_t localThreads[] = {(size_t)MAXWORKITEM};
    	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
        assert(status == CL_SUCCESS); 
        // get results
        clEnqueueReadBuffer(commandQueue, bufferA, CL_TRUE, 0, sizeof(cl_uint) * N/MAXWORKITEM, A, 0, NULL, NULL); 
	uint32_t sum = 0;
        for (int i = 0; i < N/MAXWORKITEM; i++)
        	sum += A[i];        
        printf("%u\n", sum-padding);
    }
    // release opencl preparation
    clReleaseContext(context);
    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    // release buffer
    clReleaseMemObject(bufferA);
    
    return 0;
}
