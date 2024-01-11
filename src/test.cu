#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <vector>

#include <cuda_runtime.h>

// For units of time such as h, min, s, ms, us, ns
using namespace std::literals::chrono_literals;

namespace {
// 1 << 20 ==> 1048576
constexpr size_t kDataSize = 1 << 20;
constexpr size_t kDataBytes = kDataSize * sizeof(float);
constexpr size_t kThreadPerBlock = 256;
}  // namespace

__global__ void AddVector(const float *const device_input1,
                             const float *const device_input2, const int num,
                             float *const device_result) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // printf("blockDim.x = %d, blockIdx.x = %d, threadIdx.x = %d, i = %d\n",
  // 	   blockDim.x, blockIdx.x, threadIdx.x, i);

  if (i < num) {
    // printf("device_input1[%d] = %.2f, device_input2[%d] = %.2f\n", i,
    // device_input1[i], i, device_input2[i]);
    device_result[i] = device_input1[i] + device_input2[i];
  }
}

int main() {
  auto global_start = std::chrono::high_resolution_clock::now();

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> host_input1(kDataSize, 0.0), host_input2(kDataSize, 0.0),
      host_result(kDataSize, 0.0);
  for (size_t i = 0; i < kDataSize; ++i) {
    host_input1[i] = static_cast<float>(i);
    host_input2[i] = static_cast<float>(2 * i);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = stop - start;
  std::cout << "The time taken to initialize data on the host is "
            << duration.count() << "ms. \n";

  // Allocate memories for the device pointers.
  start = std::chrono::high_resolution_clock::now();
  float *device_input1 = nullptr;
  float *device_input2 = nullptr;
  float *device_result = nullptr;
  cudaMalloc((void **)&device_input1, kDataBytes);
  cudaMalloc((void **)&device_input2, kDataBytes);
  cudaMalloc((void **)&device_result, kDataBytes);
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to allocate memories on the device is "
            << duration.count() << "ms. \n";

  // Copy data from host to device
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(device_input1, host_input1.data(), kDataBytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_input2, host_input2.data(), kDataBytes,
             cudaMemcpyHostToDevice);
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to copy data frome the host to the device is "
            << duration.count() << "ms. \n";

  // Free up host memories
  start = std::chrono::high_resolution_clock::now();
  host_input1.clear();
  host_input1.shrink_to_fit();
  host_input2.clear();
  host_input2.shrink_to_fit();
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to free up memories on the host is "
            << duration.count() << "ms. \n";

  const size_t block_per_grid =
      (kDataSize + kThreadPerBlock - 1) / kThreadPerBlock;
  std::cout << "block_per_grid = " << block_per_grid << std::endl;

  // Invoke the kernel function.
  start = std::chrono::high_resolution_clock::now();
  AddVector<<<block_per_grid, kThreadPerBlock>>>(
      device_input1, device_input2, kDataSize, device_result);
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to perform the kernel task is "
            << duration.count() << "ms. \n";

  // Copy data from device to host.
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(host_result.data(), device_result, kDataBytes,
             cudaMemcpyDeviceToHost);
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to copy data from the device to the host is "
            << duration.count() << "ms. \n";

  // Free up device memories.
  start = std::chrono::high_resolution_clock::now();
  cudaFree(device_input1);
  cudaFree(device_input2);
  cudaFree(device_result);
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to free up memories on the device is "
            << duration.count() << "ms. \n";

  // Show first 20 elements.
  start = std::chrono::high_resolution_clock::now();
  size_t display_num = 20;
  auto iter_end = host_result.begin();
  std::advance(iter_end, display_num);
  std::cout << "The first " << display_num << " results are:\n";
  std::copy(host_result.begin(), iter_end,
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "The time taken to output the results is " << duration.count()
            << "ms. \n";

  auto global_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> global_duration =
      global_stop - global_start;
  std::cout << "The whole time is " << global_duration.count() << "ms. \n";

  return 0;
}
