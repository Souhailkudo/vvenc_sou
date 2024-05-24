# A MAchine learning based approach to improve the complexity of Fraunhofer's Versatile Video Encoder (VVenC)

## Installation

- Download and build LightGBM 2.3.2 from Microsoft's git:
 ```sh
git clone https://github.com/microsoft/LightGBM.git
cd LightGBM/
git checkout 483a9bbad23adecf8db9b77c9f2caa69080ecf7e
mkdir build
cd build
cmake ..
make
```
For LightGBM to work with VVenC the correct directory needs to be specified in the CMakeLists.txt of the VVenC project. The directory is currently set locally to the VTM directory (../LightGBM), change it if needed. 
- install nlohmann's json version 3.10.5
```sh
git clone -b 'v3.10.5' --single-branch --depth 1 https://github.com/nlohmann/json
cd json
mkdir build
cd build
cmake -DBUILD_TESTING=OFF ..
make
make install
```
- To use ONNX for CNN inference in CPU or GPU, onnx-runtime needs to be installed. If you want to use docker, you can use my docker **"souhaiel7/onnx:1.0"** made public in dockerhub as it has everything ready to work. If you want to use your host computer please follow these steps:

  - Install onnx-runtime and everything that you would need to use ONNX for inference in C++, you can find everything in this link: https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/linux-x86_64.md
  You can try CPU inference which is simpler to setup then install the rest to try GPU inference.
  - Run these commands and change the onnc-runtime version to the version you installed:
    
 ```sh
ln -s onnxruntime-linux-x64-gpu-1.15.1/lib/libonnxruntime.so
cp onnxruntime-linux-x64-gpu-1.15.1/lib/* /usr/local/lib
mkdir -p /usr/local/include/onnxruntime/
cp -r onnxruntime-linux-x64-gpu-1.15.1/include/ /usr/local/include/onnxruntime/
```
  - Create these 2 cmake files (credit to jcarius) so that cmake can find onnx-runtime:
  
  First file: onnxruntimeVersion.cmake (change version inside the file if needed)
    ```
    # Custom cmake version file by jcarius

    set(PACKAGE_VERSION "1.15.1")
    
    # Check whether the requested PACKAGE_FIND_VERSION is compatible
    if("${PACKAGE_VERSION}" VERSION_LESS "${PACKAGE_FIND_VERSION}")
      set(PACKAGE_VERSION_COMPATIBLE FALSE)
    else()
      set(PACKAGE_VERSION_COMPATIBLE TRUE)
      if("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
        set(PACKAGE_VERSION_EXACT TRUE)
      endif()
    endif()
    ```
  Second file: onnxruntimeConfig.cmake
  ```
  # Custom cmake config file by jcarius to enable find_package(onnxruntime) without modifying LIBRARY_PATH and LD_LIBRARY_PATH
#
# This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against
#   onnxruntime_CXX_FLAGS    -- Additional (required) compiler flags

include(FindPackageHandleStandardArgs)

# Assume we are in <install-prefix>/share/cmake/onnxruntime/onnxruntimeConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(onnxruntime_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/include)
set(onnxruntime_LIBRARIES onnxruntime)
set(onnxruntime_CXX_FLAGS "") # no flags needed


find_library(onnxruntime_LIBRARY onnxruntime
    PATHS "${onnxruntime_INSTALL_PREFIX}/lib"
)

add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_COMPILE_OPTIONS "${onnxruntime_CXX_FLAGS}")

find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_LIBRARY onnxruntime_INCLUDE_DIRS)
```
  - Put the files where cmake can find them, in my case:
```sh
mkdir -p /usr/local/share/cmake/onnxruntime/
cp onnxruntimeVersion.cmake /usr/local/share/cmake/onnxruntime/
cp onnxruntimeConfig.cmake /usr/local/share/cmake/onnxruntime/
```
Just installing onnx-runtime and creating the cmakefiles should allow the CPU inference, for the GPU inference, TensorRT and other dependencies are needed and they are all mentioned in the MLDeploy link above. 

## Usage
Here is an example of encoding one file using our solution:
```sh
./vvencFFapp -i [path to raw video file] --preset slower -q 32 -s 416x240 -r 60 --InputBitDepth=8 -b bit.266 --useGPU=1 --predictionModes=1 --SplitNumber=0 --SplitNumberIntra=0 -t 16 --ModelFolder=[path to model_folder]
```
explanation of inputs that aren't original inputs of VVenC:
 - useGPU: Specifies whether you want the inference to be done in CPU (0) or GPU (1)
 - predictionModes: specifies whether you want to use the model the inter model only (1), intra only (2) both (3) or none (0) which would encode in original VVenC. Currently (1) gives the best results.
 - SplitNumber: Specifies the number of split modes you want the encoder to test in **inter** mode afer the model made its prediction, 0 is for custom numbers for each CU size which can be changed inside the jsonfile in model_folder. Currently 0 gives the best results.
 - SplitNumberIntra: Same as SplitNumber but for Intra. Currently (3) gives the best results.
 - ModelFolder: The folder "model_folder" provided in the git that contains all model files.  

