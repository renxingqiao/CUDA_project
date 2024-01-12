#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

#include "aux.h"

__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;

    // TODO 对于非方阵的矩阵xy是否可以调换
    for (int k = 0; k < width; k++) {
        float M_element = M_device[x * width + k];
        float N_element = N_device[k * width + y];
        P_element += M_element + N_element;
    }
    P_device[x * width + y] = P_element;
}


void MatmulOnHost(float *M, float *N, float *P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0;
            for (int k = 0; k < width; k++) {
                float a = M[i * width + k];
                float b = N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

float* initializeMatrix(int rows, int cols) {
    int* matrix = new int[rows * cols];  // 使用动态内存分配创建矩阵

    // 初始化矩阵元素为1
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = 1.0;
        }
    }

    return matrix;  // 返回指向矩阵数据的指针
}

void MatmulOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize) {
    // set matrix size
    int size = width * width * sizeof(float);
    // distribute space on GPU
    float *M_device;
    float *N_device;

    // copy M N P from CPU to GPU
    // cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice);

    // distribute space on GPU
    float *P_device;
    cudaMalloc(&P_device, size);

    // use kernel to calculate matrix multy 
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / , width / blockSize);
    MatmulKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);

}


int main() {
    int width = 1<<10;
    int min = 0;
    int max = 1;
    int size = width * width;
    int blockSize = 1;

    // float *h_matM = (float*)malloc(size * sizeof(float));
    float *h_matN = (float*)malloc(size * sizeof(float));
    float *h_matP = (float*)malloc(size * sizeof(float));
    float *d_matP = (float*)malloc(size * sizeof(float));

    h_matM = initializeMatrix(width, width);
    h_matN = initializeMatrix(width, width);
    
    // CPU
    th = get_time();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    th = get_time() - th;

    // GPU
    td = get_time();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    td = get_time() - td; 

    // GPU blocksize 16
    blockSize = 16;
    td_16 = get_time();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    td_16 = get_time() - td_16; 

    printf("GPU time: %e, CPU time: %e, GPU block 16: %g\n", td, th, td_16);

    return 0;

}