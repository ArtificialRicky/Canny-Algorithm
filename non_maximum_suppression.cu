#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <cassert>

typedef unsigned char uchar;

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define SHMEM_WIDTH (BLOCK_WIDTH + 2)
#define SHMEM_HEIGHT (BLOCK_HEIGHT + 2)


__device__ bool is_45(float Angle) {
    return (Angle > 0 && Angle <= 45) || (Angle > 180 && Angle <= 225);
}

__device__ bool is_90(float Angle) {
    return (Angle > 45 && Angle <= 90) || (Angle > 225 && Angle <= 270);
}

__device__ bool is_135(float Angle) {
    return (Angle > 90 && Angle <= 135) || (Angle > 270 && Angle <= 315);
}

__device__ bool is_180(float Angle) {
    return (Angle == 0) || (Angle > 135 && Angle <= 180) || (Angle > 315 && Angle <= 360);
}

__global__ void non_maximum_suppression_kernel(
    unsigned char* img, const float* angle, int rows, int cols, int step)
{   
    __shared__ uchar sh_img[SHMEM_HEIGHT][SHMEM_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int i = by * BLOCK_HEIGHT + ty;
    int j = bx * BLOCK_WIDTH + tx;

    // load to shared memory
    if (i < rows && j < cols) {
        // center
        sh_img[ty + 1][tx + 1] = img[i * step + j];
        // left & right halo
        if (tx == 0 && j > 0)
            sh_img[ty + 1][0] = img[i * step + (j - 1)];
        if (tx == BLOCK_WIDTH - 1 && j < cols - 1)
            sh_img[ty + 1][SHMEM_WIDTH - 1] = img[i * step + (j + 1)];
        // up & down halo
        if (ty == 0 && i > 0)
            sh_img[0][tx + 1] = img[(i - 1) * step + j];
        if (ty == BLOCK_HEIGHT - 1 && i < rows - 1)
            sh_img[SHMEM_HEIGHT - 1][tx + 1] = img[(i + 1) * step + j];
        // the four corners
        if (tx == 0 && ty == 0 && i > 0 && j > 0)
            sh_img[0][0] = img[(i - 1) * step + (j - 1)];
        if (tx == 0 && ty == BLOCK_HEIGHT - 1 && i < rows - 1 && j > 0)
            sh_img[SHMEM_HEIGHT - 1][0] = img[(i + 1) * step + (j - 1)];
        if (tx == BLOCK_WIDTH - 1 && ty == 0 && i > 0 && j < cols - 1)
            sh_img[0][SHMEM_WIDTH - 1] = img[(i - 1) * step + (j + 1)];
        if (tx == BLOCK_WIDTH - 1 && ty == BLOCK_HEIGHT - 1 && i < rows - 1 && j < cols - 1)
            sh_img[SHMEM_HEIGHT - 1][SHMEM_WIDTH - 1] = img[(i + 1) * step + (j + 1)];
    }
    __syncthreads();

    if (i < 1 || j < 1 || i >= rows - 1 || j >= cols - 1) return;

    float Angle = angle[i * cols + j];
    uchar value = sh_img[ty + 1][tx + 1];
    uchar previous = 0, next = 0;

    if (is_45(Angle)) {
        previous = sh_img[ty][tx + 2];    // (i-1, j+1)
        next     = sh_img[ty + 2][tx];     // (i+1, j-1)
    } else if (is_90(Angle)) {
        previous = sh_img[ty][tx + 1];    // (i-1, j)
        next     = sh_img[ty + 2][tx + 1]; // (i+1, j)
    } else if (is_135(Angle)) {
        previous = sh_img[ty][tx];         // (i-1, j-1)
        next     = sh_img[ty + 2][tx + 2]; // (i+1, j+1)
    } else if (is_180(Angle)) {
        previous = sh_img[ty + 1][tx];     // (i, j-1)
        next     = sh_img[ty + 1][tx + 2]; // (i, j+1)
    }

    if (value < previous || value < next)
        img[i * step + j] = 0;
}

void non_maximum_suppression_cuda(cv::Mat &img_out, const cv::Mat_<float> &angle)
{
    int rows = img_out.rows;
    int cols = img_out.cols;
    int step = img_out.step;

    size_t img_size = rows * step * sizeof(uchar);
    size_t angle_size = rows * cols * sizeof(float);

    uchar *d_img;
    float *d_angle;

    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_angle, angle_size);

    cudaMemcpy(d_img, img_out.data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_angle, angle.ptr<float>(), angle_size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid((cols + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
              (rows + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    non_maximum_suppression_kernel<<<grid, block>>>(
        d_img, d_angle, rows, cols, step);
    cudaDeviceSynchronize();

    cudaMemcpy(img_out.data, d_img, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_angle);
}
