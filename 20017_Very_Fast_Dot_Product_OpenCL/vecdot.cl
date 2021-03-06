static inline unsigned int rotate_left(unsigned int x, unsigned int n) {
    return  (x << n) | (x >> (32-n));
}
static inline unsigned int encrypt(unsigned int m, unsigned int key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void dotAndSum(unsigned int key1, unsigned int key2, __global unsigned int* Sum, int N) {
    int globalId = get_global_id(0);
    int localSize = get_local_size(0);
    int localId = get_local_id(0);
    int groupId = get_group_id(0);

    unsigned int sum = 0;
    for (int i = 0; i < localSize && globalId*localSize+i < N; i++){
        sum += encrypt(globalId*localSize+i, key1) * encrypt(globalId*localSize+i, key2);
    }
    atomic_add(&Sum[groupId&15], sum);
}
