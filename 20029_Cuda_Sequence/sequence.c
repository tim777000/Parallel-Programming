#include <stdio.h>
#include <stdint.h>
#define MAXN 1024
void multiply(int N, uint32_t A[][MAXN], uint32_t B[][MAXN], uint32_t C[][MAXN]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            uint32_t sum = 0;    // overflow, let it go.
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}
void add(int N, uint32_t A[][MAXN], uint32_t B[][MAXN], uint32_t C[][MAXN]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
        	C[i][j] = A[i][j] + B[i][j];
    }
}
void rand_gen(uint32_t c, int N, uint32_t A[][MAXN]) {
    uint32_t x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i][j] = x;
        }
    }
}
void print_matrix(int N, uint32_t A[][MAXN]) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i][j]);
        fprintf(stderr, " ]\n");
    }
}
uint32_t signature(int N, uint32_t A[][MAXN]) {
    uint32_t h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i][j]) * 2654435761LU;
    }
    return h;
}
uint32_t IN[2][MAXN][MAXN], TMP[6][MAXN][MAXN];
int main() {
	int N, S[2];
	while (scanf("%d", &N)==1) {
		for (int i = 0; i < 2; i++) {
			scanf("%d", &S[i]);
			rand_gen(S[i], N, IN[i]);
		}
		// AB
		multiply(N, IN[0], IN[1], TMP[0]);
		// BA
		multiply(N, IN[1], IN[0], TMP[1]);
		// AB+BA
		add(N, TMP[0], TMP[1], TMP[2]);
		//printf("%u\n", signature(N, TMP[2]));

		// ABA
		multiply(N, TMP[0], IN[0], TMP[3]);
		// BAB
		multiply(N, TMP[1], IN[1], TMP[4]);
		// ABA+BAB
		add(N, TMP[3], TMP[4], TMP[5]);
		//printf("%u\n", signature(N, TMP[5]));
		print_matrix(N, TMP[0]);
		printf("\n");
		print_matrix(N, TMP[1]);
		printf("\n");
		print_matrix(N, TMP[2]);
		printf("\n");
		print_matrix(N, TMP[3]);
		printf("\n");
		print_matrix(N, TMP[4]);
		printf("\n");
		print_matrix(N, TMP[5]);
	}
	return 0;
}
