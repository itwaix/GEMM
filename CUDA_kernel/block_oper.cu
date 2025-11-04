#define int WARP_SIZE = 24;
template<const int NUM_THREADS=128>
__device__ __forceinline__ float Block_Sum(float val) {
    int WARP_NUM = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    static __shared__ float smem[WARP_NUM];

    var = warp_sum(val);

    if(lane == 0) smem = var;
      __syncthreads();

    val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;

    var = warp_sum(val);

    return val;
}

