static inline unsigned int rotate_left(unsigned int x, unsigned int n) {
    return  (x << n) | (x >> (32-n));
}
static inline unsigned int encrypt(unsigned int m, unsigned int key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void dotAndSum(unsigned int key1, unsigned int key2, __global unsigned int* Sum) {
    __local unsigned int product[256];
    int globalId = get_global_id(0);
    int localSize = get_local_size(0);
    int localId = get_local_id(0);
    int groupId = get_group_id(0);

    product[localId] = encrypt(globalId, key1) * encrypt(globalId, key2);
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int offset = 1; offset < localSize; offset *= 2) {
        int mask = 2*offset - 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if((localId&mask)==0) {
            product[localId] += product[localId+offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localId == 0) {
        Sum[groupId] = product[0];
    }
}
