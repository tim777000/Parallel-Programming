#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
 
#define MAXN 10240
 
int DP[2][1000000 + 5];
 
int max(int a, int b) {
    return a > b? a : b;
}
 
int main (void) {
    int N, M;
    int W[MAXN] = {0}, V[MAXN] = {0};
    scanf("%d %d", &N, &M);
    for (int i = 0; i < N; i++) {
        scanf("%d %d", &W[i], &V[i]);
    }
 
    int now = 1;
    int old = 1 - now;
    for (int i = 0; i < N; i++) {
        # pragma omp parallel
        {
            # pragma omp for
            for (int j = 1; j <= M; j++) {
                DP[now][j] = DP[old][j];
                if (j - W[i] >= 0) {
                    DP[now][j] = max(DP[old][j], DP[old][j - W[i]] + V[i]);
                }
            }
        }
        now = old;
        old = 1 - now;
    }
 
    printf("%d\n", DP[old][M]);
    return 0;
}
