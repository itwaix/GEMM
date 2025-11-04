


template<const int WARP_SIZE>
__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE>> 1; mask >= 1; mask >> 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


template<const int WARP_SIZE>
__device__ __forceinline__ float warp_max(float val ) {
    #pragma unroll
    for(int mask = WARP_SIZE>>1; mask>=1; mask >> 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


_shfl_xor_sync