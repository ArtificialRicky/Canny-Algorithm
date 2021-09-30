CPP_FILE ?= "main.cpp"
OUTPUT_FILE = "main"
DIR ?= ""
build:
	g++ "$(DIR)$(CPP_FILE)" \
			-L/usr/local/lib \
				-lopencv_core \
				-lopencv_highgui \
				-lopencv_imgcodecs \
				-lopencv_imgproc \
			-std=c++20 \
			-I/usr/local/include/opencv4/ \
			-Wl,-R/usr/local/lib \
			-o "$(DIR)$(OUTPUT_FILE)"

debug:
	g++ -g "$(DIR)$(CPP_FILE)" \
			-L/usr/local/lib \
				-lopencv_core \
				-lopencv_highgui \
				-lopencv_imgcodecs \
				-lopencv_imgproc \
			-std=c++20 \
			-I/usr/local/include/opencv4/ \
			-Wl,-R/usr/local/lib \
			-o "$(DIR)$(OUTPUT_FILE)"
