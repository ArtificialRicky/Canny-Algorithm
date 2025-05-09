CU_FILE = non_maximum_suppression.cu
CPP_FILE = canny_algorithm.cpp
OUTPUT_FILE = canny_algorithm.out

# build:
# 	nvcc $(CPP_FILE) $(CU_FILE) \
# 	    -I/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/include/opencv4/ \
# 	    -L/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
# 	    -L/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
# 	    -L/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/ \
# 	    -lopencv_core \
# 	    -lopencv_highgui \
# 	    -lopencv_imgcodecs \
# 	    -lopencv_imgproc \
# 	    -ljpeg \
# 	    -ljasper \
# 	    -Wl,-R/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
# 	    -Wl,-R/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
# 	    -Wl,-R/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/ \
# 	    -o $(OUTPUT_FILE)
build:
	nvcc $(CPP_FILE) $(CU_FILE) \
	    -I/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/include/opencv4/ \
	    -L/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
	    -L/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
	    -L/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/ \
	    -lopencv_core \
	    -lopencv_highgui \
	    -lopencv_imgcodecs \
	    -lopencv_imgproc \
	    -ljpeg \
	    -ljasper \
	    -o $(OUTPUT_FILE)


# run:
# 	./$(OUTPUT_FILE)
run:
	LD_LIBRARY_PATH=/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib:/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib:/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib:$$LD_LIBRARY_PATH ./$(OUTPUT_FILE)



debug:
	g++ -g $(CPP_FILE) \
	    -I/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/include/opencv4/ \
	    -L/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
	    -L/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
	    -L/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/ \
	    -lopencv_core \
	    -lopencv_highgui \
	    -lopencv_imgcodecs \
	    -lopencv_imgproc \
	    -ljpeg \
	    -ljasper \
	    -Wl,-R/usr/local/depot/fsl/fslpython/pkgs/libopencv-4.5.3-py38h5627943_1/lib/ \
	    -Wl,-R/usr/local/depot/fsl/fslpython/pkgs/jpeg-9e-h7f98852_0/lib/ \
	    -Wl,-R/usr/local/depot/fsl/fslpython/pkgs/jasper-1.900.1-h07fcdf6_1006/lib/ \
	    -o $(OUTPUT_FILE)