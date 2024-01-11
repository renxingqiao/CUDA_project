cmake_minimum_required(VERSION 3.8)
project(VECTOR_ADD)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

cuda_add_executable(${PROJECT_NAME} src/vector_add.cu)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
