#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
 
int N;
char map[18 + 5][18 + 5] = 0;
 
bool check (int queens_position[], int queen, int col) {
    for (int i = 0; i < queen; i++) {
        if (queens_position[i] == col || (abs(col - queens_position[i]) == \
            queen - i)) {
            return false;
        }
    }
    return true;
}
 
int queens (int queens_position[], int queen) {
    if (queen >= N) {
        /*printf("\n");
        for (int i = 0; i < N; i++){
            printf("%d:%d ", i, queens_position[i]);
        }
        printf("\n");*/
        return 1;
    }
    int sol = 0;
    for (int col = 0; col < N; col++) {
        if (map[queen][col] == '*') {
            continue;
        }
        if (check(queens_position, queen, col)) {
            queens_position[queen] = col;
            sol += queens(queens_position, queen + 1);
        }
    }
 
    return sol;
}
 
int main (void) {
    int count = 0;
    int queens_position[18 + 5] = {0};
    int ans = 0;
    while (scanf("%d", &N) == 1) {
        ans = 0;
        count++;
        for (int i = 0; i < N; i++) {
            scanf("%s", map[i]);
        }
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1) private (queens_position) reduction(+ : ans) collapse(2)
            for (int col_z = 0; col_z < N; col_z++) {
                for (int col_o = 0; col_o < N; col_o++) {
                    if (map[0][col_z] == '*' || map[1][col_o] == '*' || (abs(col_z -col_o) <= 1)) {
                        continue;
                    }
                    queens_position[0] = col_z;
                    queens_position[1] = col_o;
                    ans += queens(queens_position, 2);
                }
            }
        }
        printf("Case %d: %d\n", count, ans);
    }
 
    return 0;
}
