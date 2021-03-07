# Canny edge detector algorithm, implement Canny algorithm by C++

@ Operating system: **Ubuntu 20.04.2 or any GNU linux distro**

@ Tools: **g++ 9.3.0, Opencv version 4.5.1, Microsoft Visual Studio code**.
- [Install opencv](#Install-Opencv-version-4\.5\.1)
- [Setup VScode](#Set-up-VS-code)
- [Build cpp file](#Build-the-cpp-file)

# Install Opencv version 4.5.1

 01. Install the required dependencies:
+ `$ sudo apt install build-essential cmake git pkg-config libgtk-3-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev`

02. Clone the OpenCV’s and OpenCV contrib repositories:
+ `$ mkdir ~/opencv_build && cd ~/opencv_build`
+ `$ git clone https://github.com/opencv/opencv.git`
+ `$ git clone https://github.com/opencv/opencv_contrib.git`

03. Once the download is complete, create a temporary build directory, and
switch to it:
+ `$ cd ~/opencv_build/opencv`
+ `$ mkdir build && cd build`

04. Set up the OpenCV build with CMake:
+ `$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..`

05. Start the compilation process:
+ `$ make -j8`

+ `-j8` mean you use 8 processors (included hyper threading) to compile, you can modify the -j flag according to your processor. If you do not know the
number of cores your processor, you can find it by typing `nproc`.

06. Install OpenCV with:
+ `$ sudo make install`

# Set up VS code:

- The `c_cpp_properties.json` file:
```
{
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/include/opencv4/",
                "/usr/local/include/",
                "/usr/include/"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "c++20",
            "intelliSenseMode": "gcc-x64"
        }
}
```


- The `task.json` file
```
{
    "tasks": [
        {
            "label": "C/C++: g++ build active file",
            "type": "shell",
            "command": "/usr/bin/g++",
            "args": [
                "-g",
                "${file}",
                "-std=c++2a",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}",
                "-I/usr/local/include/opencv4/",
                "-L/usr/local/lib",
                "-lopencv_shape",
                "-lopencv_stitching",
                "-lopencv_objdetect",
                "-lopencv_superres",
                "-lopencv_videostab",
                "-lopencv_calib3d",
                "-lopencv_features2d",
                "-lopencv_highgui",
                "-lopencv_videoio",
                "-lopencv_imgcodecs",
                "-lopencv_video",
                "-lopencv_photo",
                "-lopencv_ml",
                "-lopencv_imgproc",
                "-lopencv_flann",
                "-lopencv_core"
            ],
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": "build"
        }
    ],
    "version": "2.0.0"
}
```


- The `launch.json`file:
```
"version": "0.2.0",
    "configurations": [
        {
            "name": "C/C++ Debugger - Current File",
            "type": "cppdbg",
            "request": "launch",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "miDebuggerPath": "/usr/bin/gdb",
            "preLaunchTask": "C/C++: g++ build active file"
        }
    ]
}
```

# Build the cpp file
- Open terminal and type this command to compile .cpp file: ```cd "directory" && g++ canny_algorithm.cpp -o canny_algorithm -std=c++2a -pthread `pkg-config --libs --cflags opencv4` && ./canny_algorithm && rm canny_algorithm```
  - `"directory"` is the full path that `algorithm.cpp` is locating. **Example:** `/home/Canny/`.



