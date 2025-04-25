#include "MeasurementSeries.hpp"
#include "dtime.hpp"
#include "gpu-error.h"
#include <iomanip>
#include <iostream>
#include <unistd.h>

using namespace std;

constexpr int MAXThreadsPerSM = 2048;
constexpr int MAXWarpSize = 32;

const int64_t max_buffer_size = 128l * 1024 * 1024 + 2;
double *dA, *dB, *dC, *dD;

#ifdef __NVCC__
const int spoilerSize = 768;
#else
const int spoilerSize = 4 * 1024;
#endif

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 const double *__restrict__ C,
                                 const double *__restrict__ D, const size_t N,
                                 bool secretlyFalse);

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  for(int i = tidx; i < N; i += stride)
    A[i] = 0.23;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void read_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  double temp = 0.0;
  for(int i = tidx; i < N; i += stride)
    temp += B[i];

  if (secretlyFalse || temp == 123.0)
    A[tidx] = temp + spoiler[tidx];
}

template <typename T>
__global__ void scale_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  for(int i = tidx; i < N; i += stride)
    A[i] = B[i] * 1.2;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void rmw_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  for(int i = tidx; i < N; i += stride)
    A[i] = A[i] * 1.2;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void triad_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  for(int i = tidx; i < N; i += stride)
    A[i] = B[i] * 1.2 + C[i];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void stencil1d3pt_kernel(T *A, const T *__restrict__ B,
                                    const T *__restrict__ C,
                                    const T *__restrict__ D, const size_t N,
                                    bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N - 1 || tidx == 0)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  for(int i = tidx; i < N; i += stride)
    A[i] = 0.5 * B[i - 1] - 1.0 * B[i] + 0.5 * B[i + 1];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}
template <typename T>
__global__ void stencil1d5pt_kernel(T *A, const T *__restrict__ B,
                                    const T *__restrict__ C,
                                    const T *__restrict__ D, const size_t N,
                                    bool secretlyFalse) {
  __shared__ double spoiler[spoilerSize];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  if (tidx >= N - 2 || tidx < 2)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  for(int i = tidx; i < N; i += stride)
    A[i] = 0.25 * B[i - 2] + 0.25 * B[i - 1] - 1.0 * B[i] +
            0.5 * B[i + 1] + 0.5 * B[i + 2];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
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
    func<<<gridSize, blockSize>>>(dA, dB, dC, dD, max_buffer_size, false);
    return;
  }

  MeasurementSeries time;
  MeasurementSeries power;

  func<<<max_buffer_size / blockSize + 1, blockSize>>>(dA, dB, dC, dD, max_buffer_size, false);

  for (int iter = 0; iter < 9; iter++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());
    func<<<gridSize, blockSize>>>(
        dA, dB, dC, dD, max_buffer_size, false);
    func<<<gridSize, blockSize>>>(
        dA, dB, dC, dD, max_buffer_size, false);
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

void measureKernels(vector<pair<kernel_ptr_type, int>> kernels, int blockSize, int gridSize,
                    int blocksPerSM) {
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

int main(int argc, char **argv) {
  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));

  vector<pair<kernel_ptr_type, int>> all_kernels = {
      {init_kernel<double>, 1},         {read_kernel<double>, 1},
      {scale_kernel<double>, 2},        {triad_kernel<double>, 3},
      {rmw_kernel<double>, 1}
  };
  vector<pair<kernel_ptr_type, int>> kernels;
  int gridSize = 0;
  int blockSize = 0;

#ifdef ORG_VERSION
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dA, dA, dA, dA,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dB, dB, dB, dB,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dC, dC, dC, dC,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dD, dD, dD, dD,
                                                    max_buffer_size, false);
  GPU_ERROR(cudaDeviceSynchronize());

  vector<pair<kernel_ptr_type, int>> kernels = {
      {init_kernel<double>, 1},         {read_kernel<double>, 1},
      {scale_kernel<double>, 2},        {triad_kernel<double>, 3},
      {stencil1d3pt_kernel<double>, 2}, {stencil1d5pt_kernel<double>, 2}};

  cout << "blockSize   threads       %occ  |                init"
       << "       read       scale     triad       3pt        5pt\n";
#else
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

#endif

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
}
