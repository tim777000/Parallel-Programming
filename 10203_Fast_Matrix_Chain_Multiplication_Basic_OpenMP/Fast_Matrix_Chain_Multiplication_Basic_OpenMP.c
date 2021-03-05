#include <stdio.h>
#define MAXN 2048
#define INF (1LL<<60)
int N;
long long dp[MAXN*MAXN], SZ[MAXN+1];
int main() {
    while (scanf("%d", &N) == 1) {
        for (int i = 0; i <= N; i++)
            scanf("%lld", &SZ[i]);
        #pragma omp parallel
        {
            #pragma omp for
            for (int i=0; i<N; i++)
                dp[i*N + i] = 0;
            for (int i = 1; i < N; i++) {
                #pragma omp for
                for (int j = 0; j < N-i; j++) {
                    int l = j, r = j+i;
                    long long local = INF;
                    long long mul_temp = SZ[l] * SZ[r+1];
                    for (int k = l; k < r; k++) {
                        long long t = dp[l*N+k] + dp[r*N+k+1] + mul_temp * SZ[k+1];
                        local = (t < local)? t:local;
                    }
                    dp[l*N+r] = dp[r*N+l] = local;
                }
            }
        }
        printf("%lld\n", dp[0*N+N-1]);
    }
    return 0;
}