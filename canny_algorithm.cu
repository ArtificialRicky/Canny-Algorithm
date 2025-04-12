#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <assert.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void double_threshold_kernel(uchar *img, int width, int height, int low, int high, int step)
{
    // 注意：这个版本假设核函数只处理内部区域，不访问图像边界像素
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // 为了使用简单边界，分配 tile 比实际 block 大两列两行
    __shared__ uchar tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // 加载中心区域
    tile[local_y][local_x] = img[y * step + x];

    // 保证共享内存加载完毕
    __syncthreads();

    // 仅处理内部区域像素，不处理边界（已通过传参或 grid 配置排除）
    uchar value = tile[local_y][local_x];
    if (value < low) {
        value = 0;
    } else if (value > high) {
        value = 255;
    } else {
        bool has_strong_neighbor = false;
        #pragma unroll
        for (int m = -1; m <= 1 && !has_strong_neighbor; ++m) {
            #pragma unroll
            for (int n = -1; n <= 1; ++n) {
                if (m == 0 && n == 0) continue;
                if (tile[local_y + m][local_x + n] > high) {
                    value = 255;
                    has_strong_neighbor = true;
                    break;
                }
            }
        }
        if (!has_strong_neighbor)
            value = 0;
    }

    // 写回结果到全局内存
    img[y * step + x] = value;
}


__global__ void double_threshold_kernel_global(uchar* img, int width, int height, int low, int high, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

    int idx = y * step + x;
    uchar value = img[idx];
    bool changed = false;

    if (value < low) {
        img[idx] = 0;
        return;
    }

    if (value > high) {
        img[idx] = 255;
        return;
    }

    // 遍历 3x3 邻域（全局内存访问）
    for (int m = -1; m <= 1 && !changed; ++m) {
        for (int n = -1; n <= 1; ++n) {
            if (m == 0 && n == 0) continue;

            int neighbor_idx = (y + m) * step + (x + n);
            if (img[neighbor_idx] > high) {
                img[idx] = 255;
                changed = true;
                break;
            }
        }
    }

    if (!changed)
        img[idx] = 0;
}


void double_threshold(cv::Mat &img_out, const int &low, const int &high) {
    assert(low >= 0 && high >= 0 && low <= high);
    assert(img_out.type() == CV_8UC1);

    uchar *d_img;
    size_t img_size = img_out.rows * img_out.step;

    cudaMalloc(&d_img, img_size);
    cudaMemcpy(d_img, img_out.data, img_size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((img_out.cols + block.x - 1) / block.x,
              (img_out.rows + block.y - 1) / block.y);

    double_threshold_kernel<<<grid, block>>>(d_img, img_out.cols, img_out.rows, low, high, img_out.step);
    cudaDeviceSynchronize();

    cudaMemcpy(img_out.data, d_img, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
}
