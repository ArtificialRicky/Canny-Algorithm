// gradient_image.cu
#include "gradient_image.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define SHMEM_WIDTH (BLOCK_SIZE_X + 2)
#define SHMEM_HEIGHT (BLOCK_SIZE_Y + 2)
// #define epsilon 0.00001f

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

// __device__ int get_angle(float Angle) {
//     return Angle<epsilon?3:(int(Angle-epsilon)/45)%4;
// }

__constant__ int DX_PREV[4]  = {  1, 0,  -1, -1 };
__constant__ int DY_PREV[4]  = { -1, -1, -1, 0 };
__constant__ int DX_NEXT[4]  = { -1, 0,  1,  1 };
__constant__ int DY_NEXT[4]  = {  1, 1,  1,  0 };

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

__global__ void GradientImageKernel(const float* src, float* dst, float* angle, int rows, int cols, int step){
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int j = bx * blockDim.x + tx;
    int i = by * blockDim.y + ty;
    if (i < 1 || i >= rows-1 || j < 1 || j >= cols-1) return;

    int Sx = SHMEM_WIDTH;
    extern __shared__ float sh_src[];
    int si = ty + 1;
    int sj = tx + 1;

    sh_src[si * Sx + sj] = src[i * step + j];

    if (ty == 0)                 
        sh_src[0 * Sx + sj] = src[(i-1) * step + j];
    if (ty == blockDim.y - 1)    
        sh_src[(Sx * (blockDim.y + 1)) + sj] = src[(i + 1) * step + j];
    if (tx == 0)                 
        sh_src[si * Sx + 0] = src[i * step + (j - 1)];
    if (tx == blockDim.x - 1)    
        sh_src[si * Sx + (blockDim.x+1)] = src[i * step + (j+1)];

    if (ty==0 && tx==0)                         
        sh_src[0 * Sx + 0] = src[(i-1)*step + (j-1)];
    if (ty==0 && tx==blockDim.x-1) 
        sh_src[0 * Sx + (blockDim.x + 1)] = src[(i - 1) * step + (j + 1)];
    if (ty==blockDim.y-1 && tx==0)              
        sh_src[(blockDim.y + 1) * Sx + 0] = src[(i + 1) * step + (j - 1)];
    if (ty==blockDim.y-1 && tx==blockDim.x-1)   
        sh_src[(blockDim.y + 1) * Sx + (blockDim.x + 1)]  = src[(i + 1) * step + (j + 1)];

    __syncthreads();

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
    
    int dir = (((int)Angle % 180)+179) / 45;
    uchar previous = sh_img[ty + 1 + DY_PREV[dir]][tx + 1 + DX_PREV[dir]];
    uchar next     = sh_img[ty + 1 + DY_NEXT[dir]][tx + 1 + DX_NEXT[dir]];
    // uchar previous = 0, next = 0;

    // if (is_45(Angle)) {
    //     previous = sh_img[ty][tx + 2];    // (i-1, j+1)
    //     next     = sh_img[ty + 2][tx];     // (i+1, j-1)
    // } else if (is_90(Angle)) {
    //     previous = sh_img[ty][tx + 1];    // (i-1, j)
    //     next     = sh_img[ty + 2][tx + 1]; // (i+1, j)
    // } else if (is_135(Angle)) {
    //     previous = sh_img[ty][tx];         // (i-1, j-1)
    //     next     = sh_img[ty + 2][tx + 2]; // (i+1, j+1)
    // } else if (is_180(Angle)) {
    //     previous = sh_img[ty + 1][tx];     // (i, j-1)
    //     next     = sh_img[ty + 1][tx + 2]; // (i, j+1)
    // }

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
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x >= width - 1 || y >= height - 1) return;
    __shared__ float tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 3];

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    tile[ly][lx] = img[y * step + x];

    if (threadIdx.y == 0) {
        tile[ly - 1][lx] = img[(y - 1) * step + x];
    }
    if (threadIdx.y == blockDim.y - 1) {
        tile[ly + 1][lx] = img[(y + 1) * step + x];
    }
    if (threadIdx.x == 0) {
        tile[ly][lx - 1] = img[y * step + (x - 1)];
    }
    if (threadIdx.x == blockDim.x - 1) {
        tile[ly][lx + 1] = img[y * step + (x + 1)];
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[ly - 1][lx - 1] = img[(y - 1) * step + (x - 1)];
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        tile[ly + 1][lx - 1] = img[(y + 1) * step + (x - 1)];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        tile[ly - 1][lx + 1] = img[(y - 1) * step + (x + 1)];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        tile[ly + 1][lx + 1] = img[(y + 1) * step + (x + 1)];
    }
    __syncthreads();

    float value = tile[ly][lx];
    if (value < low) {
        value = 0.0;
    } else if (value > high) {
        value = 255.0;
    } else {
        value = 0.0;
        bool has_strong_neighbor = false;
        #pragma unroll
        for (int m = -1; m <= 1 && !has_strong_neighbor; ++m) {
            #pragma unroll
            for (int n = -1; n <= 1; ++n) {
                if (tile[ly + m][lx + n] > high) {
                    value = 255.0;
                    has_strong_neighbor = true;
                    break;
                }
            }
        }
    }

    img[y * step + x] = value;
}

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


__global__
void cast_uchar_to_float(const unsigned char* in, float* out,
                         int rows, int cols, int step_bytes)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
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
        out[y*step_bytes + x] = (unsigned char)(in[y*cols + x]);
    }
}
// ------------------------------------------------------------------

void Canny_cuda(const cv::Mat &img_src,
    cv::Mat &img_out,
    int low_threshold,
    int high_threshold)
{
CV_Assert(low_threshold <= high_threshold);
const int rows    = img_src.rows;
const int cols    = img_src.cols;
const int step_u  = img_src.step;
const size_t img_bytes = size_t(rows)*step_u;
const size_t ang_bytes = size_t(rows)*cols*sizeof(float);

unsigned char *d_src_u = nullptr, *d_dst_u = nullptr;
float         *d_src_f = nullptr,
      *d_mag_f = nullptr,
      *d_ang   = nullptr;
cudaMalloc(&d_src_u,  img_bytes);
cudaMalloc(&d_dst_u,  img_bytes);
cudaMalloc(&d_src_f,  rows*cols*sizeof(float));
cudaMalloc(&d_mag_f,  rows*cols*sizeof(float));
cudaMalloc(&d_ang,    ang_bytes);

cudaMemcpy(d_src_u, img_src.data, img_bytes, cudaMemcpyHostToDevice);

dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
dim3 grid ((cols + block.x-1)/block.x,
   (rows + block.y-1)/block.y);
size_t sh_bytes_f = size_t(block.y+2)*(block.x+2)*sizeof(float);

cudaEvent_t e0,e1,e2,e3;
cudaEventCreate(&e0);  cudaEventCreate(&e1);
cudaEventCreate(&e2);  cudaEventCreate(&e3);

// —— 4. uchar→float cast —— 
cast_uchar_to_float<<<grid, block>>>(d_src_u, d_src_f,
                             rows, cols, cols);
cudaDeviceSynchronize();

cudaEventRecord(e0);
GradientImageKernel<<<grid, block, sh_bytes_f>>>(
d_src_f, d_mag_f, d_ang, rows, cols, /*stride_f=*/cols);
cudaDeviceSynchronize();
cudaEventRecord(e1);

non_maximum_suppression_kernel<<<grid, block>>>(
d_mag_f, d_ang, rows, cols, /*stride_f=*/cols);
cudaDeviceSynchronize();
cudaEventRecord(e2);

double_threshold_kernel<<<grid, block>>>(
d_mag_f, cols, rows,
float(low_threshold), float(high_threshold),
/*stride_f=*/step_u);
cudaDeviceSynchronize();
cudaEventRecord(e3);

cast_float_to_uchar<<<grid, block>>>(d_mag_f, d_dst_u,
                             rows, cols, cols);
cudaDeviceSynchronize();
img_out.create(rows, cols, CV_8UC1);
cudaMemcpy(img_out.data, d_dst_u, img_bytes,
   cudaMemcpyDeviceToHost);

float t_grad=0.f, t_nms=0.f, t_thr=0.f, t_total=0.f;
cudaEventElapsedTime(&t_grad , e0, e1);
cudaEventElapsedTime(&t_nms  , e1, e2);
cudaEventElapsedTime(&t_thr  , e2, e3);
cudaEventElapsedTime(&t_total, e0, e3);
std::cout << "Gradient kernel   : " << t_grad  << " ms\n";
std::cout << "NMS kernel        : " << t_nms   << " ms\n";
std::cout << "Threshold kernel  : " << t_thr   << " ms\n";
std::cout << "Total GPU stage   : " << t_total << " ms\n";

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

void Canny(const cv::Mat &src, cv::Mat &dst,
      const int &low_th, const int &high_th)
{
Canny_cuda(src, dst, low_th, high_th);
}
