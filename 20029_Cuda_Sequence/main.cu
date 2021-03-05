#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include <omp.h>
#define MAXN 1024

__global__ void matrixMul(int N, uint32_t A[], uint32_t B[], uint32_t C[]) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t sum = 0;
	int row = index/N;
	int column = index%N;
	for (int i = 0; i < N; i++) {
		sum += A[row*N + i] * B[i*N + column];
	}
	C[index] = sum;
}

__global__ void matrixAdd(int N, uint32_t A[], uint32_t B[], uint32_t C[]) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	C[index] = A[index] + B[index];
}

void rand_gen(uint32_t c, int N, uint32_t A[]) {
    uint32_t x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i*N + j] = x;
        }
    }
}

void signature(int N, uint32_t A[], uint32_t *B) {
    uint32_t h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i*N + j]) * 2654435761LU;
    }
    *B = h;
}

uint32_t IN[2][MAXN*MAXN];
uint32_t hostTmp[4][MAXN*MAXN];
uint32_t ret[2];
int main() {
	omp_set_num_threads(2);
	uint32_t N, S[2];
	uint32_t *gpuIn[2][2], *gpuOut[2][4];
	int size = sizeof(uint32_t) * MAXN * MAXN;
	for (int i = 0; i < 2; i++) {
		cudaSetDevice(i);
		for (int j = 0; j < 2; j++) {
			cudaMalloc((void **)&gpuIn[i][j], size);
		}
		for (int j = 0; j < 4; j++) {
			cudaMalloc((void **)&gpuOut[i][j], size);
		}
	}
	while (scanf("%d", &N)==1) {
		scanf("%d %d", &S[0], &S[1]);
		size = sizeof(uint32_t) * N * N;
		int BLOCK = 1;
		for (int i = 1; i <= 1024; i++) {
			if ((N*N) % i == 0) {
				BLOCK = i;
			}
		}
		dim3 block(BLOCK);
		dim3 grid(N*N / BLOCK);
#pragma omp parallel 
{
#pragma omp for
		for (int i = 0; i < 2; i++) {
			rand_gen(S[i], N, IN[i]);
		}
#pragma omp for
		for (int i = 0; i < 2; i++) {
			cudaSetDevice(i);
			for (int j = 0; j < 2; j++) {
				cudaMemcpy(gpuIn[i][j], IN[j], size, cudaMemcpyHostToDevice);
			}
			// AB & BA
			matrixMul <<< grid, block >>> (N, (uint32_t *)gpuIn[i][i], (uint32_t *)gpuIn[i][1-i], (uint32_t *)gpuOut[i][0]);
			// ABA & BAB
			matrixMul <<< grid, block >>> (N, (uint32_t *)gpuOut[i][0], (uint32_t *)gpuIn[i][i], (uint32_t *)gpuOut[i][1]);
			cudaMemcpy(hostTmp[i+2], gpuOut[i][1-i], size, cudaMemcpyDeviceToHost);
		}
#pragma omp for
		for (int i = 0; i < 2; i++) {
			cudaSetDevice(i);
			cudaMemcpy(gpuOut[i][2], hostTmp[1-i+2], size, cudaMemcpyHostToDevice);
			matrixAdd <<< grid, block >>> (N, (uint32_t *)gpuOut[i][i], (uint32_t *)gpuOut[i][2], (uint32_t *)gpuOut[i][3]);
			cudaMemcpy(hostTmp[i], gpuOut[i][3], size, cudaMemcpyDeviceToHost);
			signature(N, hostTmp[i], &ret[i]);
		}
}
		printf("%u\n", ret[0]);
		printf("%u\n", ret[1]);
	}
	return 0;
}
