cmake_minimum_required(VERSION 3.8)
project(CUDA_PROJECT)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

cuda_add_executable(vector_add src/vector_add.cu)
cuda_add_executable(matrix_mult src/matrix_mult.cu)

target_link_libraries(vector_add ${CUDA_LIBRARIES})
target_link_libraries(matrix_mult ${CUDA_LIBRARIES})
