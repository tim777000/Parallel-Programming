#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <inttypes.h>
#include <CL/cl.h>
#include "utils.h"

#define MAXGPU 10
#define MAXK 10240
#define MAXN 1073741824
#define MAXWORKITEM 256 
#define BLOCK 512
#define GPUNUM 2
#define MAXLOG 4096

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
    assert(status == CL_SUCCESS && GPU_amount >= GPUNUM);
    // create context
    cl_context context = clCreateContext(NULL, GPUNUM, GPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
    // create command queue
    cl_command_queue commandQueue[GPUNUM];
    for (int device = 0; device < GPUNUM; device++) {
        commandQueue[device] = clCreateCommandQueue(context, GPU[device], CL_QUEUE_PROFILING_ENABLE, &status);
    }
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
    status = clBuildProgram(program, GPUNUM, GPU, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        char log[MAXLOG];
        size_t logLength;
        for (int device = 0; device < GPUNUM; device++) {
            clGetProgramBuildInfo(program, GPU[device], CL_PROGRAM_BUILD_LOG, MAXLOG, log, &logLength);
            puts(log);
        }
        exit(-1);
    }
    // create kernel
    cl_kernel kernel = clCreateKernel(program, "dotAndSum", &status);
    assert(status == CL_SUCCESS);
    // create buffer
    cl_mem bufferA[GPUNUM];
    for (int device = 0; device < GPUNUM; device++) {
        bufferA[device] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_uint)*16, NULL, &status);
    }
    assert(status == CL_SUCCESS);    
    // start
    int N;
    uint32_t key1, key2;
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        cl_event events[GPUNUM];
        int gpuHandlingNum = N;
        if (N%2 != 0) {
            gpuHandlingNum++;
        }
        gpuHandlingNum /= 2;
        int gpuHandlingNumBlock = (gpuHandlingNum+(MAXWORKITEM-gpuHandlingNum%MAXWORKITEM))/BLOCK;
        size_t globalThreads[] = {(size_t)gpuHandlingNumBlock+(MAXWORKITEM-gpuHandlingNumBlock%MAXWORKITEM)};
        size_t localThreads[] = {(size_t)MAXWORKITEM};
        //printf("N:%d\ngpuItem:%d\n", N, gpuHandlingNum); 
        for (int device = 0; device < GPUNUM; device++) {
            int start = device*gpuHandlingNum;
            //printf("device%d:%d\n", device, start);
            // set kernel arg
            status = clSetKernelArg(kernel, 0, sizeof(cl_uint), (void*)&key1);
            assert(status == CL_SUCCESS);
            status = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void*)&key2);
            assert(status == CL_SUCCESS);
            const uint32_t pattern = 0;
            status = clEnqueueFillBuffer(commandQueue[device], bufferA[device], &pattern, sizeof(cl_uint), 0, sizeof(cl_uint)*16, 0, NULL, NULL);
            assert(status == CL_SUCCESS);
            status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferA[device]);
            assert(status == CL_SUCCESS);
            status = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&start);
            assert(status == CL_SUCCESS);
            status = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&gpuHandlingNum);
            assert(status == CL_SUCCESS);
            status = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&N);
            assert(status == CL_SUCCESS);
            // data shape
    	    status = clEnqueueNDRangeKernel(commandQueue[device], kernel, 1, NULL, globalThreads, localThreads, 0, NULL, &events[device]);
            assert(status == CL_SUCCESS); 
        }
        clWaitForEvents(GPUNUM, events);
        // get results
        uint32_t A[GPUNUM][16];
        uint32_t sum = 0;
        for (int device = 0; device < GPUNUM; device++) {
            status = clEnqueueReadBuffer(commandQueue[device], bufferA[device], CL_TRUE, 0, sizeof(cl_uint)*16, A[device], 0, NULL, NULL);
            assert(status == CL_SUCCESS);
            for (int i = 0; i < 16; i++) {
                sum += A[device][i];
                //printf("A[%d][%d]:%d\n", device, i, A[device][i]);
            }
        }
        //printf("Sum:%u\n", sum);
        printf("%u\n", sum);
    }
    // release opencl preparation
    clReleaseContext(context);
    for (int device = 0; device < GPUNUM; device++) {
        clReleaseCommandQueue(commandQueue[device]);
    }
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    // release buffer
    for (int device = 0; device < GPUNUM; device++) {
        clReleaseMemObject(bufferA[device]);
    }
    
    return 0;
}
