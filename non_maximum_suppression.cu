#include <cuda_runtime.h>
#include <opencv2/core.hpp>

typedef unsigned char uchar;

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
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1 || j < 1 || i >= rows - 1 || j >= cols - 1) return;

    int idx = i * step + j;
    float Angle = angle[i * cols + j];
    uchar value = img[idx];
    uchar previous = 0, next = 0;

    if (is_45(Angle)) {
        previous = img[(i - 1) * step + (j + 1)];
        next = img[(i + 1) * step + (j - 1)];
    } else if (is_90(Angle)) {
        previous = img[(i - 1) * step + j];
        next = img[(i + 1) * step + j];
    } else if (is_135(Angle)) {
        previous = img[(i - 1) * step + (j - 1)];
        next = img[(i + 1) * step + (j + 1)];
    } else if (is_180(Angle)) {
        previous = img[i * step + (j - 1)];
        next = img[i * step + (j + 1)];
    }

    if (value < previous || value < next)
        img[idx] = 0;
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

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    non_maximum_suppression_kernel<<<grid, block>>>(d_img, d_angle, rows, cols, step);
    cudaDeviceSynchronize();

    cudaMemcpy(img_out.data, d_img, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_angle);
}
