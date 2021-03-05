#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
 
#define MAXN 2048
char board[2][MAXN][MAXN] = {{{0}}};
 
int main (void) {
    int N, M;
    scanf("%d %d", &N, &M);
    for (int i = 1; i <= N; i++) {
        scanf("%s", &board[0][i][1]);
        for (int j = 1; j <= N; j++) {
            board[0][i][j] -= '0';
        }
    }
    int current = 1;
    int previous = 1 - current;
    for (int i = 0; i < M; i++) {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j <= N; j++) {
            for (int k = 1; k <= N; k++) {
                int alive = board[previous][j - 1][k - 1] \
                        + board[previous][j - 1][k] \
                        + board[previous][j - 1][k + 1] \
                        + board[previous][j][k - 1] \
                        + board[previous][j][k + 1] \
                        + board[previous][j + 1][k - 1] \
                        + board[previous][j + 1][k] \
                        + board[previous][j + 1][k + 1];
                if (board[previous][j][k] == 1) {
                    if (alive < 2) {
                        board[current][j][k] = 0;
                    }
                    else if (alive == 2 || alive == 3) {
                        board[current][j][k] = 1;
                    }
                    else if (alive > 3) {
                        board[current][j][k] = 0;
                    }
                }
                else if (board[previous][j][k] == 0) {
                    if (alive == 3) {
                        board[current][j][k] = 1;
                    }
                    else {
                        board[current][j][k] = 0;
                    }
                }
            }
        }
        current = previous;
        previous = 1 - current;
    }
 
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            board[previous][i][j] += '0';
        }
        printf("%s\n", &board[previous][i][1]);
    }
 
    return 0;
}
