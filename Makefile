CPP_FILE ?= "canny_algorithm.cpp"
OUTPUT_FILE = "canny_algorithm.out"
build:
	g++ "$(CPP_FILE)" \
			-L/usr/lib \
				-lopencv_core \
				-lopencv_highgui \
				-lopencv_imgcodecs \
				-lopencv_imgproc \
			-I/usr/include/opencv4/ \
			-Wl,-R/usr/local/lib \
			-o "$(OUTPUT_FILE)"

run:
	./canny_algorithm.out


debug:
	g++ -g "$(CPP_FILE)" \
			-L/usr/lib \
				-lopencv_core \
				-lopencv_highgui \
				-lopencv_imgcodecs \
				-lopencv_imgproc \
			-I/usr/include/opencv4/ \
			-Wl,-R/usr/local/lib \
			-o "$(OUTPUT_FILE)"
