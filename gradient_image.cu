// gradient_image.cu
#include "gradient_image.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 4
#define SHMEM_WIDTH (BLOCK_SIZE_X + 2)
#define SHMEM_HEIGHT (BLOCK_SIZE_Y + 2)

// 定义角度阈值判断函数
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

void Gradient_image(const cv::Mat &img_src,
    cv::Mat &img_out,      // an empty matrix to store result
    cv::Mat_<float> &angle)     // an empty matrix to store arctan(Gy / Gx)
{
    angle = cv::Mat_<float>::zeros(img_src.size());
    img_out = cv::Mat::zeros(img_src.size(), CV_8UC1);
    int row_minus_1 = img_src.rows - 1;
    int col_minus_1 = img_src.cols - 1;

    // int row = img_src.rows;
    // int col = img_src.cols;

    auto point = img_src.data;
    int step = img_src.step;

    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            uchar pixel_00 = point[(i - 1) * step + j - 1];
            uchar pixel_01 = point[(i - 1) * step + j];
            uchar pixel_02 = point[(i - 1) * step + j + 1];
            uchar pixel_10 = point[i * step + j - 1];
            // uchar pixel_11 = point[i * step + j];
            uchar pixel_12 = point[i * step + j + 1];
            uchar pixel_20 = point[(i + 1) * step + j - 1];
            uchar pixel_21 = point[(i + 1) * step + j];
            uchar pixel_22 = point[(i + 1) * step + j + 1];

            // float grad_x = (-1 * pixel_00) + (-2 * pixel_10) + (-1 * pixel_20) + (1 * pixel_02) + (2 * pixel_12) + (1 * pixel_22);
            float grad_x = pixel_02 + (2 * pixel_12) + pixel_22 - pixel_00 - (2 * pixel_10) - pixel_20;

            // float grad_y = (1 * pixel_00) + (2 * pixel_01) + (1 * pixel_02) + (-1 * pixel_20) + (-2 * pixel_21) + (-1 * pixel_22);
            float grad_y = pixel_00 + (2 * pixel_01) + pixel_02 - pixel_20 - (2 * pixel_21) - pixel_22;

            angle.at<float>(i, j) = atan(grad_y / (grad_x == 0 ? 0.00001 : grad_x));
            img_out.at<uchar>(i, j) = sqrt(grad_x * grad_x + grad_y * grad_y);
        }
    }
}

// CUDA 核函数：每个线程处理图像中一个非边界像素
__global__ void GradientImageKernel(const float* src, float* dst, float* angle, int rows, int cols, int step){
    // 全局坐标
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int j = bx * blockDim.x + tx;
    int i = by * blockDim.y + ty;
    // 边界上跳过
    if (i < 1 || i >= rows-1 || j < 1 || j >= cols-1) return;

    // tile 宽度（含两条 halo）
    // int Sx = blockDim.x + 2;
    int Sx = SHMEM_WIDTH;
    // shared 内存一维数组
    extern __shared__ float sh_src[];
    // tile 内局部坐标，加 1 是因为留出 halo
    int si = ty + 1;
    int sj = tx + 1;

    // 1) preload 自己的中心像素
    sh_src[si * Sx + sj] = src[i * step + j];

    // 2) preload 上下左右四条 halo
    if (ty == 0)                 
        sh_src[0 * Sx + sj] = src[(i-1) * step + j];
    if (ty == blockDim.y - 1)    
        sh_src[(Sx * (blockDim.y + 1)) + sj] = src[(i + 1) * step + j];
    if (tx == 0)                 
        sh_src[si * Sx + 0] = src[i * step + (j - 1)];
    if (tx == blockDim.x - 1)    
        sh_src[si * Sx + (blockDim.x+1)] = src[i * step + (j+1)];

    // 3) preload 四个角落
    if (ty==0 && tx==0)                         
        sh_src[0 * Sx + 0] = src[(i-1)*step + (j-1)];
    if (ty==0 && tx==blockDim.x-1) 
        sh_src[0 * Sx + (blockDim.x + 1)] = src[(i - 1) * step + (j + 1)];
    if (ty==blockDim.y-1 && tx==0)              
        sh_src[(blockDim.y + 1) * Sx + 0] = src[(i + 1) * step + (j - 1)];
    if (ty==blockDim.y-1 && tx==blockDim.x-1)   
        sh_src[(blockDim.y + 1) * Sx + (blockDim.x + 1)]  = src[(i + 1) * step + (j + 1)];

    // 等所有线程把 tile 和 halo 都搬进来
    __syncthreads();

    // 4) 从 shared memory 取 3×3 邻域
    auto at = [&](int di, int dj){
      return sh_src[(si+di)*Sx + (sj+dj)];
    };
    float grad_x = at(-1, +1) + 2 * at(0, +1) + at(+1, +1) - at(-1, -1) - 2 * at(0, -1) - at(+1, -1);
    float grad_y = at(-1, -1) + 2 * at(-1, 0) + at(-1, +1) - at(+1, -1) - 2 * at(+1, 0) - at(+1, +1);

    float g = sqrtf(grad_x * grad_x + grad_y * grad_y);
    int  idx = i * step + j;
    dst[idx]   = (float)g;
    angle[idx] = atan2f(grad_y, (grad_x==0?1e-5f:grad_x));
}

// 主机端 CUDA 包装函数：调用核函数计算梯度
// void Gradient_image_cuda(const cv::Mat &img_src,
//                          cv::Mat &img_out,
//                          cv::Mat_<float> &angle)
// {
//     int rows = img_src.rows;
//     int cols = img_src.cols;
//     size_t step = img_src.step; // 每行字节数

//     // 初始化输出图像和角度矩阵
//     img_out.create(rows, cols, CV_8UC1);
//     angle.create(rows, cols);

//     // 申请设备内存
//     unsigned char *d_src = nullptr, *d_dst = nullptr;
//     float *d_angle = nullptr;
//     size_t img_size = rows * step;

//     cudaMalloc((void**)&d_src, img_size);
//     cudaMalloc((void**)&d_dst, img_size);
//     cudaMalloc((void**)&d_angle, rows * cols * sizeof(float));

//     // 将输入图像数据复制到设备内存
//     cudaMemcpy(d_src, img_src.data, img_size, cudaMemcpyHostToDevice);

//     // 定义线程块和网格尺寸
//     dim3 block(16, 16);
//     dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

//     // // 启动 CUDA 核函数
//     // GradientImageKernel<<<grid, block>>>(d_src, d_dst, d_angle, rows, cols, step);
//     size_t shared_bytes = (block.y + 2) * (block.x + 2) * sizeof(unsigned char);

//     GradientImageKernel<<<grid, block, shared_bytes>>>(d_src, d_dst, d_angle, rows, cols, step);
//     cudaDeviceSynchronize();

//     // 将结果复制回主机内存
//     cudaMemcpy(img_out.data, d_dst, img_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(angle.data, d_angle, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

//     // 释放设备内存
//     cudaFree(d_src);
//     cudaFree(d_dst);
//     cudaFree(d_angle);
// }

// 非极大值抑制：仅保留局部极大值
__global__ void non_maximum_suppression_kernel(
    float* img, const float* angle, int rows, int cols, int step)
{   
    __shared__ float sh_img[SHMEM_HEIGHT][SHMEM_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int i = by * BLOCK_SIZE_Y + ty;
    int j = bx * BLOCK_SIZE_X + tx;

    // load to shared memory
    if (i < rows && j < cols) {
        // center
        sh_img[ty + 1][tx + 1] = img[i * step + j];
        // left & right halo
        if (tx == 0 && j > 0)
            sh_img[ty + 1][0] = img[i * step + (j - 1)];
        if (tx == BLOCK_SIZE_X - 1 && j < cols - 1)
            sh_img[ty + 1][SHMEM_WIDTH - 1] = img[i * step + (j + 1)];
        // up & down halo
        if (ty == 0 && i > 0)
            sh_img[0][tx + 1] = img[(i - 1) * step + j];
        if (ty == BLOCK_SIZE_Y - 1 && i < rows - 1)
            sh_img[SHMEM_HEIGHT - 1][tx + 1] = img[(i + 1) * step + j];
        // the four corners
        if (tx == 0 && ty == 0 && i > 0 && j > 0)
            sh_img[0][0] = img[(i - 1) * step + (j - 1)];
        if (tx == 0 && ty == BLOCK_SIZE_Y - 1 && i < rows - 1 && j > 0)
            sh_img[SHMEM_HEIGHT - 1][0] = img[(i + 1) * step + (j - 1)];
        if (tx == BLOCK_SIZE_X - 1 && ty == 0 && i > 0 && j < cols - 1)
            sh_img[0][SHMEM_WIDTH - 1] = img[(i - 1) * step + (j + 1)];
        if (tx == BLOCK_SIZE_X - 1 && ty == BLOCK_SIZE_Y - 1 && i < rows - 1 && j < cols - 1)
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

// void non_maximum_suppression(cv::Mat &img_out, const cv::Mat_<float> &angle)
// {
//     int rows = img_out.rows;
//     int cols = img_out.cols;
//     int step = img_out.step;

//     size_t img_size = rows * step * sizeof(uchar);
//     size_t angle_size = rows * cols * sizeof(float);

//     uchar *d_img;
//     float *d_angle;

//     cudaMalloc(&d_img, img_size);
//     cudaMalloc(&d_angle, angle_size);

//     cudaMemcpy(d_img, img_out.data, img_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_angle, angle.ptr<float>(), angle_size, cudaMemcpyHostToDevice);

//     dim3 block(16, 16);
//     dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
//     non_maximum_suppression_kernel<<<grid, block>>>(d_img, d_angle, rows, cols, step);
//     cudaDeviceSynchronize();

//     cudaMemcpy(img_out.data, d_img, img_size, cudaMemcpyDeviceToHost);
//     cudaFree(d_img);
//     cudaFree(d_angle);
// }

__global__ void double_threshold_kernel(float *img, int width, int height, float low, float high, int step)
{
    // 注意：这个版本假设核函数只处理内部区域，不访问图像边界像素
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // 为了使用简单边界，分配 tile 比实际 block 大两列两行
    __shared__ float tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 3];

    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // 加载中心区域
    tile[local_y][local_x] = img[y * step + x];

    // 保证共享内存加载完毕
    __syncthreads();

    // 仅处理内部区域像素，不处理边界（已通过传参或 grid 配置排除）
    float value = tile[local_y][local_x];
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

// 双阈值处理：根据低/高阈值确定边缘像素
// void double_threshold(cv::Mat &img_out, const int &low, const int &high) {
//     assert(low >= 0 && high >= 0 && low <= high);
//     assert(img_out.type() == CV_8UC1);

//     uchar *d_img;
//     size_t img_size = img_out.rows * img_out.step;

//     cudaMalloc(&d_img, img_size);
//     cudaMemcpy(d_img, img_out.data, img_size, cudaMemcpyHostToDevice);

//     dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//     dim3 grid((img_out.cols + block.x - 1) / block.x,
//               (img_out.rows + block.y - 1) / block.y);

//     double_threshold_kernel<<<grid, block>>>(d_img, img_out.cols, img_out.rows, low, high, img_out.step);
//     cudaDeviceSynchronize();

//     cudaMemcpy(img_out.data, d_img, img_size, cudaMemcpyDeviceToHost);
//     cudaFree(d_img);
// }

// Canny 边缘检测：调用 CUDA 计算梯度，然后进行非极大值抑制和双阈值处理
// ====================== 新增：统一 GPU Pipeline =========================
__global__
void cast_uchar_to_float(const unsigned char* in, float* out,
                         int rows, int cols, int step_bytes)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        // out 按 cols 紧凑存储
        out[y*cols + x] = float(in[y*step_bytes + x]);
    }
}

__global__
void cast_float_to_uchar(const float* in, unsigned char* out,
                         int rows, int cols, int step_bytes)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        // 截断回 uchar
        out[y*step_bytes + x] = (unsigned char)(in[y*cols + x]);
    }
}
// ------------------------------------------------------------------

// 原有 Canny_cuda 改动部分：
void Canny_cuda(const cv::Mat &img_src,
    cv::Mat &img_out,
    int low_threshold,
    int high_threshold)
{
CV_Assert(low_threshold <= high_threshold);
const int rows    = img_src.rows;
const int cols    = img_src.cols;
const int step_u  = img_src.step;           // uchar 每行字节数
const size_t img_bytes = size_t(rows)*step_u;
const size_t ang_bytes = size_t(rows)*cols*sizeof(float);

// —— 1. 申请所有显存 —— 
unsigned char *d_src_u = nullptr, *d_dst_u = nullptr;
float         *d_src_f = nullptr,
      *d_mag_f = nullptr,
      *d_ang   = nullptr;
cudaMalloc(&d_src_u,  img_bytes);
cudaMalloc(&d_dst_u,  img_bytes);
cudaMalloc(&d_src_f,  rows*cols*sizeof(float));
cudaMalloc(&d_mag_f,  rows*cols*sizeof(float));
cudaMalloc(&d_ang,    ang_bytes);

// 拷入原始 uchar 图
cudaMemcpy(d_src_u, img_src.data, img_bytes, cudaMemcpyHostToDevice);

// —— 2. block/grid 配置 & 共享内存字节数 —— 
dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
dim3 grid ((cols + block.x-1)/block.x,
   (rows + block.y-1)/block.y);
// shared 按 float 计算，为梯度 kernel 准备
size_t sh_bytes_f = size_t(block.y+2)*(block.x+2)*sizeof(float);

// —— 3. 创建计时事件 —— 
cudaEvent_t e0,e1,e2,e3;
cudaEventCreate(&e0);  cudaEventCreate(&e1);
cudaEventCreate(&e2);  cudaEventCreate(&e3);

// —— 4. uchar→float cast —— 
cast_uchar_to_float<<<grid, block>>>(d_src_u, d_src_f,
                             rows, cols, cols);
cudaDeviceSynchronize();

// —— 5-A. 梯度计算（开始计时） —— 
cudaEventRecord(e0);
GradientImageKernel<<<grid, block, sh_bytes_f>>>(
d_src_f, d_mag_f, d_ang, rows, cols, /*stride_f=*/cols);
cudaDeviceSynchronize();
cudaEventRecord(e1);

// —— 5-B. 非极大值抑制 —— 
non_maximum_suppression_kernel<<<grid, block>>>(
d_mag_f, d_ang, rows, cols, /*stride_f=*/cols);
cudaDeviceSynchronize();
cudaEventRecord(e2);

// —— 5-C. 双阈值 —— 
double_threshold_kernel<<<grid, block>>>(
d_mag_f, cols, rows,
float(low_threshold), float(high_threshold),
/*stride_f=*/step_u);
cudaDeviceSynchronize();
cudaEventRecord(e3);

// —— 6. float→uchar cast & 拷回 —— 
cast_float_to_uchar<<<grid, block>>>(d_mag_f, d_dst_u,
                             rows, cols, cols);
cudaDeviceSynchronize();
img_out.create(rows, cols, CV_8UC1);
cudaMemcpy(img_out.data, d_dst_u, img_bytes,
   cudaMemcpyDeviceToHost);

// —— 7. 打印各阶段耗时 —— 
float t_grad=0.f, t_nms=0.f, t_thr=0.f, t_total=0.f;
cudaEventElapsedTime(&t_grad , e0, e1);
cudaEventElapsedTime(&t_nms  , e1, e2);
cudaEventElapsedTime(&t_thr  , e2, e3);
cudaEventElapsedTime(&t_total, e0, e3);
std::cout << "Gradient kernel   : " << t_grad  << " ms\n";
std::cout << "NMS kernel        : " << t_nms   << " ms\n";
std::cout << "Threshold kernel  : " << t_thr   << " ms\n";
std::cout << "Total GPU stage   : " << t_total << " ms\n";

// —— 8. 释放显存 & 事件 —— 
cudaFree(d_src_u);
cudaFree(d_dst_u);
cudaFree(d_src_f);
cudaFree(d_mag_f);
cudaFree(d_ang);
cudaEventDestroy(e0);
cudaEventDestroy(e1);
cudaEventDestroy(e2);
cudaEventDestroy(e3);
}

// ================== **接口细节改动** ==================
// 1. non_maximum_suppression_kernel & double_threshold_kernel *不变*
//    只是把之前 host‑side 的包装函数删掉，直接在上面连续 launch。
// 2. 原来的 Gradient_image_cuda 仍可保留做 CPU/GPU 对照；
//    生产代码请直接调用 Canny_cuda() 以避免多次 memcpy。
// 3. 旧版 Canny() 可替换为：
void Canny(const cv::Mat &src, cv::Mat &dst,
      const int &low_th, const int &high_th)
{
Canny_cuda(src, dst, low_th, high_th);
}
