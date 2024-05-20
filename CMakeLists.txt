cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_ARCHITECTURES 70)    # for V100
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

project(MGPU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

find_package(MPI REQUIRED)

add_executable(mgpu
    src/helper.cpp
    src/multi_gpu.cu
    src/main.cpp
)

target_include_directories(mgpu
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_compile_definitions(mgpu PRIVATE MPI_ENABLED)
target_link_libraries(mgpu ${MPI_LIBRARIES})