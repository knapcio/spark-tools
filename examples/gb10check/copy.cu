// SPDX-License-Identifier: Apache-2.0
// A small standalone kernel: static compilation does not execute it.
extern "C" __global__ void sample_copy(const float* input, float* output, int n) {
    __shared__ float tile[128];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = i < n ? input[i] : 0.0f;
    __syncthreads();
    if (i < n) output[i] = tile[threadIdx.x];
}
