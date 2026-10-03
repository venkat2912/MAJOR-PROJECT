cmake_minimum_required(VERSION 3.18)
project(waste_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch comes from the pip-installed torch package:
#   -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(CUDAToolkit REQUIRED)

# Torch wheels pin the libstdc++ ABI; everything linked against them must match.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# The kernels have no LibTorch dependency.
add_library(waste_kernels STATIC src/gpu_pipeline.cu)
target_include_directories(waste_kernels PUBLIC include)
target_link_libraries(waste_kernels PUBLIC CUDA::cudart)

add_executable(waste_engine src/main.cpp)
target_include_directories(waste_engine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(waste_engine PRIVATE waste_kernels ${TORCH_LIBRARIES} ${OpenCV_LIBS})
