#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>
#include "utils.h"
 
#define MAXN 10000005
#define MAX_THREAD 6

uint32_t prefix_sum[MAXN];

void *processing(void *arg);
void *concluding(void *arg);

typedef struct process_arg {
    int start;
    int n;
    uint32_t key;
    uint32_t prev;
} P_ARG;

int main() {
    int n;
    uint32_t key;

    while (scanf("%d %" PRIu32, &n, &key) == 2) {
        pthread_t threads[MAX_THREAD];
        P_ARG p[MAX_THREAD];
        int thread_data_amount = n/MAX_THREAD;
        int thread_data_left = n%MAX_THREAD;
        int thread_amount = MAX_THREAD;
        if (thread_data_amount == 0){
            thread_amount = thread_data_left;
        }

        for (int i = 0; i < thread_amount; i++){
            p[i].key = key;
            if(thread_data_left > i){
                p[i].start = i*thread_data_amount+i+1;
                p[i].n = thread_data_amount+1;
            }
            else{
                p[i].start = i*thread_data_amount+thread_data_left+1;
                p[i].n = thread_data_amount;
            }
            pthread_create(&threads[i], NULL, processing, &p[i]);
        }
        for (int i = 0; i < thread_amount; i++) {
            pthread_join(threads[i], NULL);
        }

        for (int i = 1; i < thread_amount; i++) {
            prefix_sum[p[i].start + p[i].n-1] += prefix_sum[p[i-1].start + p[i-1].n-1];
            p[i].prev = prefix_sum[p[i-1].start + p[i-1].n-1];
            pthread_create(&threads[i], NULL, concluding, &p[i]);
        }
        for (int i = 1; i < thread_amount; i++) {
            pthread_join(threads[i], NULL);
        }
        output(prefix_sum, n);
    }
    return 0;
}


void *processing(void *arg) {
    P_ARG target = *(P_ARG *)arg;
    uint32_t sum = 0;
    for (int i = target.start; i < target.start+target.n; i++) {
        sum += encrypt(i, target.key);
        prefix_sum[i] = sum;
    }
}

void *concluding(void *arg){
    P_ARG target = *(P_ARG *)arg;
    for (int i = target.start; i < target.start+target.n-1; i++) {
        prefix_sum[i] += target.prev;
    }
}
