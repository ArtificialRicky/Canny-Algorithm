# === File paths ===
CPP_FILE = canny_algorithm.cpp
CU_FILE = canny_algorithm.cu
OUTPUT_FILE = canny_algorithm_cuda.out

# === Compilers ===
NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++11

# === OpenCV and library paths ===
OPENCV_INCLUDE = -I/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/include/opencv4/
LIB_PATHS = \
  -L/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
  -L/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
  -L/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/

LIBS = \
  -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
  -ljpeg -ljasper

# === RPATH ===
RPATHS = \
  -Xlinker -rpath -Xlinker /usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
  -Xlinker -rpath -Xlinker /usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
  -Xlinker -rpath -Xlinker /usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/

# === Build rules ===
build:
	$(NVCC) $(NVCC_FLAGS) $(CPP_FILE) $(CU_FILE) -o $(OUTPUT_FILE) \
	    $(OPENCV_INCLUDE) $(LIB_PATHS) $(LIBS) $(RPATHS)

run:
	./$(OUTPUT_FILE)

debug:
	$(NVCC) -g $(CPP_FILE) $(CU_FILE) -o $(OUTPUT_FILE) \
	    $(OPENCV_INCLUDE) $(LIB_PATHS) $(LIBS) $(RPATHS)

clean:
	rm -f $(OUTPUT_FILE)
