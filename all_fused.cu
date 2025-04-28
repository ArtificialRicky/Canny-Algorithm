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

// __constant__ int DX_PREV[4]  = {  1, 0,  -1, -1 };
// __constant__ int DY_PREV[4]  = { -1, -1, -1, 0 };
// __constant__ int DX_NEXT[4]  = { -1, 0,  1,  1 };
// __constant__ int DY_NEXT[4]  = {  1, 1,  1,  0 };

__device__ __constant__ int DX_PREV[4] = {  1,  0, -1,  0 };
__device__ __constant__ int DY_PREV[4] = { -1, -1, -1, -1 };
__device__ __constant__ int DX_NEXT[4] = { -1,  0,  1,  0 };
__device__ __constant__ int DY_NEXT[4] = {  1,  1,  1,  1 };

// fused kernel
extern "C"
__global__ void canny_fused_kernel(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,  // output: rows×step_bytes uchar
    int rows, int cols, int step_bytes,
    float low_thresh, float high_thresh
) {
    extern __shared__ float shbuf[];
    float* shI = shbuf;                                 // SHMEM_HEIGHT×SHMEM_WIDTH
    float* shM = shbuf + SHMEM_HEIGHT*SHMEM_WIDTH;      // SHMEM_HEIGHT×SHMEM_WIDTH

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int j  = bx*BLOCK_SIZE_X + tx;
    int i  = by*BLOCK_SIZE_Y + ty;

    // 1) load intensities + halo
    if (i < rows && j < cols) {
        shI[ (ty+1)*SHMEM_WIDTH + (tx+1) ] = src[ i*cols + j ];
        if (tx==0   && j>0)        shI[(ty+1)*SHMEM_WIDTH + 0           ] = src[ i*cols + (j-1) ];
        if (tx==BLOCK_SIZE_X-1 && j<cols-1) shI[(ty+1)*SHMEM_WIDTH + SHMEM_WIDTH-1] = src[ i*cols + (j+1) ];
        if (ty==0   && i>0)        shI[ 0*SHMEM_WIDTH     + (tx+1) ] = src[(i-1)*cols + j];
        if (ty==BLOCK_SIZE_Y-1 && i<rows-1) shI[(SHMEM_HEIGHT-1)*SHMEM_WIDTH + (tx+1)] = src[(i+1)*cols + j];
        // 四角
        if (tx==0 && ty==0       && i>0       && j>0)       shI[ 0*SHMEM_WIDTH + 0           ] = src[(i-1)*cols+(j-1)];
        if (tx==BLOCK_SIZE_X-1 && ty==0       && i>0       && j<cols-1) shI[ 0*SHMEM_WIDTH + SHMEM_WIDTH-1] = src[(i-1)*cols+(j+1)];
        if (tx==0 && ty==BLOCK_SIZE_Y-1 && i<rows-1 && j>0)       shI[(SHMEM_HEIGHT-1)*SHMEM_WIDTH + 0] = src[(i+1)*cols+(j-1)];
        if (tx==BLOCK_SIZE_X-1 && ty==BLOCK_SIZE_Y-1 && i<rows-1 && j<cols-1) shI[(SHMEM_HEIGHT-1)*SHMEM_WIDTH + SHMEM_WIDTH-1] = src[(i+1)*cols+(j+1)];
    }
    __syncthreads();

    if (i>0 && i<rows-1 && j>0 && j<cols-1) {
        int base = ty*SHMEM_WIDTH + tx;
        float gx =  shI[base + 0*SHMEM_WIDTH + 2]    // (-1,+1)
                  + 2*shI[base + 1*SHMEM_WIDTH + 2]    // ( 0,+1)
                  +  shI[base + 2*SHMEM_WIDTH + 2]    // (+1,+1)
                  -  shI[base + 0*SHMEM_WIDTH + 0]    // (-1,-1)
                  - 2*shI[base + 1*SHMEM_WIDTH + 0]    // ( 0,-1)
                  -  shI[base + 2*SHMEM_WIDTH + 0];   // (+1,-1)
        float gy =  shI[base + 0*SHMEM_WIDTH + 0]    // (-1,-1)
                  + 2*shI[base + 0*SHMEM_WIDTH + 1]    // (-1, 0)
                  +  shI[base + 0*SHMEM_WIDTH + 2]    // (-1,+1)
                  -  shI[base + 2*SHMEM_WIDTH + 0]    // (+1,-1)
                  - 2*shI[base + 2*SHMEM_WIDTH + 1]    // (+1, 0)
                  -  shI[base + 2*SHMEM_WIDTH + 2];   // (+1,+1)
        float mag = sqrtf(gx*gx + gy*gy);
        shM[(ty+1)*SHMEM_WIDTH + (tx+1)] = mag;
        int dir = ((int)roundf(atan2f(gy, gx)*(4.0f/M_PI))) & 3;
    }
    __syncthreads();

    if (i>0 && i<rows-1 && j>0 && j<cols-1) {
        float v = shM[(ty+1)*SHMEM_WIDTH + (tx+1)];
        float gx =  shI[ty*SHMEM_WIDTH + tx+2]
                  + 2*shI[(ty+1)*SHMEM_WIDTH + tx+2]
                  +  shI[(ty+2)*SHMEM_WIDTH + tx+2]
                  -  shI[ty*SHMEM_WIDTH + tx]
                  - 2*shI[(ty+1)*SHMEM_WIDTH + tx]
                  -  shI[(ty+2)*SHMEM_WIDTH + tx];
        float gy =  shI[ty*SHMEM_WIDTH + tx]
                  + 2*shI[ty*SHMEM_WIDTH + tx+1]
                  +  shI[ty*SHMEM_WIDTH + tx+2]
                  -  shI[(ty+2)*SHMEM_WIDTH + tx]
                  - 2*shI[(ty+2)*SHMEM_WIDTH + tx+1]
                  -  shI[(ty+2)*SHMEM_WIDTH + tx+2];
        int dir = ((int)roundf(atan2f(gy, gx)*(4.0f/M_PI))) & 3;

        float prev = shM[(ty+1+DY_PREV[dir])*SHMEM_WIDTH + (tx+1+DX_PREV[dir])];
        float next = shM[(ty+1+DY_NEXT[dir])*SHMEM_WIDTH + (tx+1+DX_NEXT[dir])];

        float sup = (v >= prev && v >= next) ? v : 0.0f;

        unsigned char outv = 0;
        if      (sup > high_thresh) outv = 255;
        else if (sup >= low_thresh) {
            bool any_strong=false;
            #pragma unroll
            for(int dy=-1; dy<=1 && !any_strong; ++dy) {
              #pragma unroll
              for(int dx=-1; dx<=1; ++dx) {
                if (dy==0&&dx==0) continue;
                if (shM[(ty+1+dy)*SHMEM_WIDTH + (tx+1+dx)] > high_thresh) {
                  any_strong = true;
                  break;
                }
              }
            }
            outv = any_strong ? 255 : 0;
        }
        dst[i*step_bytes + j] = outv;
    }
}
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
void Canny_cuda(const cv::Mat &img_src,
    cv::Mat &img_out,
    int low_threshold,
    int high_threshold)
{
CV_Assert(low_threshold <= high_threshold);
int rows   = img_src.rows;
int cols   = img_src.cols;
int step_u = img_src.step;
size_t img_bytes = size_t(rows) * step_u;

unsigned char *d_src_u = nullptr, *d_dst_u = nullptr;
float         *d_src_f = nullptr;
cudaMalloc(&d_src_u,  img_bytes);
cudaMalloc(&d_dst_u,  img_bytes);
cudaMalloc(&d_src_f,  size_t(rows) * cols * sizeof(float));

cudaMemcpy(d_src_u, img_src.data, img_bytes, cudaMemcpyHostToDevice);

dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
dim3 grid((cols + block.x - 1) / block.x,
  (rows + block.y - 1) / block.y);

// uchar→float cast
cast_uchar_to_float<<<grid, block>>>(d_src_u, d_src_f,
                             rows, cols, cols);
cudaDeviceSynchronize();

cudaEvent_t ev_start, ev_stop;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_stop);

// 7. Fused kernel
size_t sh_bytes = 2 * SHMEM_WIDTH * SHMEM_HEIGHT * sizeof(float);
cudaEventRecord(ev_start);
canny_fused_kernel<<<grid, block, sh_bytes>>>(
d_src_f,              
d_dst_u,              
rows, cols,           
cols,                 // step_bytes for both in/out in elements
float(low_threshold),
float(high_threshold)
);
cudaEventRecord(ev_stop);
cudaDeviceSynchronize();

img_out.create(rows, cols, CV_8UC1);
cudaMemcpy(img_out.data, d_dst_u, img_bytes, cudaMemcpyDeviceToHost);

float t_fused = 0.f;
cudaEventElapsedTime(&t_fused, ev_start, ev_stop);
std::cout << "Canny fused kernel time: " << t_fused << " ms\n";

cudaFree(d_src_u);
cudaFree(d_dst_u);
cudaFree(d_src_f);
cudaEventDestroy(ev_start);
cudaEventDestroy(ev_stop);
}

void Canny(const cv::Mat &src, cv::Mat &dst,
      const int &low_th, const int &high_th)
{
Canny_cuda(src, dst, low_th, high_th);
}
