#include "labeling.h"
#include <stdio.h>
#include <ctype.h>
#include <thrust/tabulate.h>

template<class T> struct MM {
    const char *base;
    MM(const char *base): base(base) {}
    __host__ __device__ T operator()(const T& index) const { 
        if (base[index] == ' ')
            return index;
	else
            return -1;
    };
};

template<class T> struct getResults {
    const int *base;
    getResults(const int *base): base(base) {}
    __host__ __device__ T operator()(const T& index) const {
        return index-base[index];
    };
};

void labeling(const char *cuStr, int *cuPos, int strLen) {
    thrust::tabulate(thrust::device, cuPos, cuPos+strLen, MM<int32_t>(cuStr));
    thrust::inclusive_scan(thrust::device, cuPos, cuPos+strLen, cuPos, thrust::maximum<int>());
    thrust::tabulate(thrust::device, cuPos, cuPos+strLen, getResults<int32_t>(cuPos));
}
