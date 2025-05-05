#include "MeasurementSeries.hpp"
#include "dtime.hpp"
#include "gpu-error.h"
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <cuda/barrier>

using namespace std;

constexpr int MAXThreadsPerSM = 2048;
constexpr int MAXWarpSize = 32;

const int64_t MAX_SHARED_MEM_ELEM = 1024;
const int64_t max_buffer_size = 128l * 1024 * 1024 + 2;
double *dA;

using kernel_ptr_type = void (*)(double *A, const size_t N);

template <typename T>
__global__ void init_kernel(T *A, const size_t N)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

    __shared__ T sh_mem_buf[MAX_SHARED_MEM_ELEM];
    __shared__ char bar_memory[256];
    cuda::barrier<cuda::thread_scope::thread_scope_block> *bar = (cuda::barrier<cuda::thread_scope::thread_scope_block> *)(bar_memory);

  if(threadIdx.x == 0)
    init(bar, blockDim.x);

  __syncthreads();
  for(int i = tidx; i < N; i += stride)
  {
     cuda::memcpy_async(&A[i],&sh_mem_buf[threadIdx.x],sizeof(T),*bar);
  }
  __syncthreads();
//  bar->arrive_and_wait(); // Wait for all copies to complete
}

template <typename T>
__global__ void read_kernel(T *A, const size_t N)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

    __shared__ T sh_mem_buf[MAX_SHARED_MEM_ELEM];
    __shared__ char bar_memory[256];
    cuda::barrier<cuda::thread_scope::thread_scope_block> *bar = (cuda::barrier<cuda::thread_scope::thread_scope_block> *)(bar_memory);
  
  if(threadIdx.x == 0)
    init(bar, blockDim.x);

  __syncthreads();
  for(int i = tidx; i < N; i += stride)
  {
     cuda::memcpy_async(&sh_mem_buf[threadIdx.x],&A[i],sizeof(T),*bar);
  }
  __syncthreads();
//  bar->arrive_and_wait(); // Wait for all copies to complete
}

void measureFunc(kernel_ptr_type func, int streamCount, int blockSize, int gridSize,
                 int blocksPerSM, bool warmup = true) {

#ifdef __NVCC__
  int maxActiveBlocks = 0;
  int currentCarveOut = 0;
  while (maxActiveBlocks < blocksPerSM) {
    GPU_ERROR(cudaFuncSetAttribute(
        func, cudaFuncAttributePreferredSharedMemoryCarveout, currentCarveOut));
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, func, blockSize, 0));
    // std::cout << maxActiveBlocks << " " << currentCarveOut << "\n";
    currentCarveOut++;
  }

#else

  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                          func, blockSize, 0));
  if (maxActiveBlocks > blocksPerSM)
    cout << "! " << maxActiveBlocks << " blocks per SM ";
#endif

  // NCU Profling
  if(!warmup)
  {
    func<<<gridSize, blockSize>>>(dA, max_buffer_size);
    return;
  }

  MeasurementSeries time;
  MeasurementSeries power;

  func<<<max_buffer_size / blockSize + 1, blockSize>>>(dA, max_buffer_size);

  for (int iter = 0; iter < 9; iter++)
  {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());
    func<<<gridSize, blockSize>>>(dA, max_buffer_size);
    func<<<gridSize, blockSize>>>(dA, max_buffer_size);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1) / 2);
  }

  cout << fixed << setprecision(0)
       << setw(6)
       //<< time.value() * 1000 << " "
       //<< setw(5) << time.spread() * 100
       //<< "   " << setw(5) << power.median() / 1000
       << " " << setw(5)
       << streamCount * max_buffer_size * sizeof(double) / time.median() * 1e-9;
  cout.flush();
}

void measureKernels(vector<pair<kernel_ptr_type, int>> kernels, int blockSize, int gridSize, int blocksPerSM)
{
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int threadsPerSM = prop.maxThreadsPerMultiProcessor;
  int threadsPerBlock = prop.maxThreadsPerBlock;

  if (blockSize * blocksPerSM > threadsPerSM || blockSize > threadsPerBlock) {
    cout << "Device Threads Per SM = " << threadsPerSM << endl;
    cout << "Device Threads Per Block = " << threadsPerBlock << endl;
    return;
  }

  int smCount = prop.multiProcessorCount;
  int threadBlocks = smCount * blocksPerSM;
  if(gridSize < threadBlocks)
    threadBlocks = gridSize;
  cout << setw(9) << gridSize << "   " << setw(9) << blockSize << "   " << setw(9) << threadBlocks * blockSize
       << "  " << setw(5) << setprecision(1)
       << (float)(threadBlocks * blockSize) / (smCount * threadsPerSM) * 100.0
       << " %  |  GB/s: ";

  for (auto kernel : kernels) {
    measureFunc(kernel.first, kernel.second, blockSize, gridSize, blocksPerSM, kernels.size() != 1);
  }

  cout << "\n";
}

int main(int argc, char **argv)
{
  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));

  vector<pair<kernel_ptr_type, int>> all_kernels = {
      {init_kernel<double>, 1},         {read_kernel<double>, 1}
  };
  vector<pair<kernel_ptr_type, int>> kernels;
  int gridSize = 0;
  int blockSize = 0;

  // check if ncu selected runs requested
  if(argc > 1)
  {
     int kernelType = std::atoi(argv[1]);
     gridSize = std::atoi(argv[2]);
     blockSize = std::atoi(argv[3]);
     kernels.push_back(all_kernels[kernelType]);
     cout << "gridsize   blockSize   threads       %occ  |                BW (GB/s)\n";
     measureKernels(kernels, blockSize, gridSize, 2);
  }
  else
  {
	gridSize = 1;
	blockSize = MAXWarpSize;
	kernels = all_kernels;


	cout << " ------------------------------ SINGLE GRID ------------------------------------------------------------ \n";
	cout << "gridsize   blockSize   threads       %occ  |                init"
	     << "       read       scale     triad       rmw\n";
	sleep(2);
	for (blockSize = MAXWarpSize; blockSize <= 1024; blockSize += MAXWarpSize)
	   measureKernels(kernels, blockSize, gridSize, 2);

  // Vary the Blocks/Per SM and the GridSize to measure which provides the best BW
  for(auto blocksPerSM = 2; blocksPerSM <=16; blocksPerSM *= 2)
  {
	   cout << " ------------------------- INCREASING GRID Blocks Per SM : " << blocksPerSM << " ------------------------- \n";
	   cout << "gridsize   blockSize   threads       %occ  |                init"
	        << "       read       scale     triad       rmw\n";
	   sleep(2);

	   blockSize = MAXThreadsPerSM / blocksPerSM;

	   for(gridSize = 2; gridSize <= 512; gridSize++)
     	   measureKernels(kernels, blockSize, gridSize, blocksPerSM);
	   sleep(2);
	   for(gridSize = 512; gridSize <= 1024; gridSize += 8)
   	     measureKernels(kernels, blockSize, gridSize, blocksPerSM);
 }

 for(auto blocksPerSM = 2; blocksPerSM <=16; blocksPerSM *= 2)
 {
	   cout << " ------------------------------ MAX GRID Blocks Per SM : " << blocksPerSM << " ------------------------- \n";
	   cout << "gridsize   blockSize   threads       %occ  |                init"
	        << "       read       scale     triad       rmw\n";
	   sleep(2);

	   for (blockSize = MAXWarpSize; blockSize <= MAXThreadsPerSM / blocksPerSM; blockSize += MAXWarpSize)
     {
	      gridSize = max_buffer_size / blockSize;
	      measureKernels(kernels, blockSize, gridSize, blocksPerSM);
	   }
  }
	}


  GPU_ERROR(cudaFree(dA));
}
