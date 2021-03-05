#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <CL/cl.h>
 
#define MAXGPU 3
#define MAXK 1024
 
int main(void) {
    char input[32];
    scanf("%s", input);
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
    FILE *kernelfp = fopen(input, "r");
    assert(kernelfp != NULL);
    char kernelBuffer[MAXK];
    const char *constKernelSource = kernelBuffer;
    size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
    cl_program program = clCreateProgramWithSource(context, 1, &constKernelSource, &kernelLength, &status);
    assert(status == CL_SUCCESS);
    // build program
    status = clBuildProgram(program, 1, GPU, NULL, NULL, NULL);
    char program_log[1024*1024];
    size_t len;
    clGetProgramBuildInfo(program, GPU[0], CL_PROGRAM_BUILD_LOG, 1024*1024*sizeof(char), program_log, &len); 
    program_log[len] = '\0';
    printf("%s", program_log);
 
    return 0;
}