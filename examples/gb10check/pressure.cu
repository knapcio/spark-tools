// SPDX-License-Identifier: Apache-2.0
// Register pressure is compiler-dependent. The configured dynamic shared
// allocation deliberately exceeds the GB10 per-block user budget.
extern "C" __global__ void sample_pressure(const float* input, float* output, int n) {
    extern __shared__ float scratch[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float values[96];
    #pragma unroll
    for (int j = 0; j < 96; ++j) values[j] = input[(i + j) % n];
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 96; ++j) sum += values[j] * values[95 - j];
    scratch[threadIdx.x] = sum;
    if (i < n) output[i] = scratch[threadIdx.x];
}
