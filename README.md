# Parallel Canny Edge Detection on Modern GPU Architectures
This project demonstrates how to perform Canny edge detection on a 4K image using OpenCV in C++.


## Features

- Load a color image and convert it to grayscale  
- Apply Gaussian blur to reduce noise  
- Run the Canny algorithm to extract edges  
- Save the result to a specified output directory  
- Print image dimensions and total runtime

## Machine & Environment

- **Host**: ghc56  
- **GPU**: NVIDIA GeForce RTX 2080 (8 GiB)  
- **NVIDIA Driver**: 550.67  
- **CUDA Version**: 12.4  


## Change to your own images
By default the program reads `input/pic_4k.png`. To use your own image:

1. **Copy your image**  
   Put your new image file (e.g. `my_photo.jpg`) into the `input/` folder:
2. **Update the filename in code**  
In `canny_algorithm.cpp`, locate the line:
```cpp
cv::Mat img_src = cv::imread("input/pic_4k.png", cv::IMREAD_COLOR);
```
and change it to your file, for example:
```
cv::Mat img_src = cv::imread("input/your_photo.png", cv::IMREAD_COLOR);
```
## Build, run and clean
```
make
make run
make clean
```
