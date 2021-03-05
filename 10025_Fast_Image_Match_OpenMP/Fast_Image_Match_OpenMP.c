#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAXN 512

int main(void){
    int A[MAXN][MAXN], B[MAXN][MAXN], ans[MAXN][MAXN];
    int A_H, A_W, B_H, B_W;
    while (scanf("%d %d %d %d", &A_H, &A_W, &B_H, &B_W) == 4) {
        for (int i = 0; i < A_H; i++) {
            for (int j = 0; j < A_W; j++) {
                scanf("%d", &A[i][j]);
            }
        }
        for (int i = 0; i < B_H; i++) {
            for (int j = 0; j < B_W; j++) {
                scanf("%d", &B[i][j]);
            }
        }
        # pragma omp parallel for
        for (int i = 0; i < A_H*A_W; i++) {
            int j = i/A_W;
            int k = i%A_W;
            if ((j+B_H-1 >= A_H) || (k+B_W-1 >= A_W)) {
                ans[j][k] = -1;
                continue;
            }
            int diff = 0;
            for (int i_i = 0; i_i < B_H; i_i++) {
                for (int j_j = 0; j_j < B_W; j_j++) {
                    diff += (A[j+i_i][k+j_j] - B[i_i][j_j])*(A[j+i_i][k+j_j] - B[i_i][j_j]);
                }
            }
            ans[j][k] = diff;
        }
        int min_result, min_x, min_y;
        for (int i = 0; i < A_H; i++) {
            for (int j = 0; j < A_W; j++) {
                if (ans[i][j] == -1){
                    continue;
                }
                if (i == 0 && j == 0){
                    min_result = ans[i][j];
                    min_x = i;
                    min_y = j;
                }
                else if (min_result > ans[i][j]) {
                    min_result = ans[i][j];
                    min_x = i;
                    min_y = j;
                }
                else if (min_result == ans[i][j]) {
                    if (min_x > i) {
                        min_result = ans[i][j];
                        min_x = i;
                        min_y = j;
                    }
                    else if (min_x == i) {
                        if (min_y > j) {
                            min_result = ans[i][j];
                            min_x = i;
                            min_y = j;
                        }
                    }
                }
            }
        }
        printf("%d %d\n", min_x+1, min_y+1);
    }

    return 0;
}
