#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

#include "aux.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;

typedef float FLOAT;

// host add
void vec_add_host(FLOAT* x, FLOAT* y, FLOAT* z, int N);

// device fuction
__global__ void vec_add(FLOAT* x, FLOAT* y, FLOAT* z, int N) {
    // cuda::grid_group g = this_grid();
    // int idx = g.thread_rank();
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = blockId * blockDim.x + threadIdx.x;

    if (idx < N) z[idx] = z[idx] + y[idx] + x[idx];
}

void vec_add_host(FLOAT* x, FLOAT* y, FLOAT* z, int N) {
    int i;
    for (i = 0; i < N; i++) z[i] = z[i] + y[i] + x[i];
}

int main() {
    int N = 20000000;
    int iter = 30;
    int nbytes = N * sizeof(FLOAT);

    // 1D block
    int bs = 256;

    // 2D grid
    int s = ceil(sqrt(N * bs - 1) / bs);
    dim3 grid = dim3(s, s);
    
    FLOAT* dx = NULL, *hx = NULL;
    FLOAT* dy = NULL, *hy = NULL;
    FLOAT* dz = NULL, *hz = NULL;

    int itr = 30;
    int i;
    double th, td;

    // allocate GPU memory
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    if (dx == NULL || dy == NULL || dz == NULL) {
        std::cout << 'could not allocate GPU memory' << std::endl;
        return -1;
    }

    // allocate CPU memory
    hx = (FLOAT*) malloc(nbytes);
    hy = (FLOAT*) malloc(nbytes);
    hz = (FLOAT*) malloc(nbytes);

    if (hx == NULL || hy == NULL || hz == NULL) {
        std::cout << 'could not allocate CPU memory \n';
        return -1;
    }
    printf("allocated %.2f MB on CPU\n", nbytes / (1024.f * 1024.f));

    // init 
    for (int i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 1;
        hz[i] = 1;
    }

    // copy data to GPU
    cudaDeviceSynchronize();
    td = get_time();
    for (int i = 0; i < iter; i++) vec_add<<<grid, bs>>>(dx, dy, dz, N);
    cudaDeviceSynchronize();
    td = get_time() - td;

    // CPU
    th = get_time();
    for (i = 0; i < iter; i++) vec_add_host(hx, hy, hz, N);
    th = get_time() - th;

    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);


    free(hx);
    free(hy);
    free(hz);
    
    return 0;

}