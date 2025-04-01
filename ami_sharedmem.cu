#include <iostream>
#include <vector>
#include <math.h>

// Kernel function to perform FMA on 2 arrays
using functionDef = void (*)(long n, float *user_arg);

// AMI = OP x 2 OPS / 8 Bytes
// OP = 1: AMI = 0.25 OPs/B
// OP = 2: AMI = 0.5 OPs/B
// OP = 4: AMI = 1 OPs/B
// OP = 8: AMI = 2 OPs/B

template <typename T, int OP>
__global__
void fma(long n, float *user_arg)
{
  extern __shared__ T s[];
  int index = threadIdx.x;
  int stride = blockDim.x;
  T *x = reinterpret_cast<T *>(s);
  T *y = reinterpret_cast<T *>(s + n);

  for (auto i = index; i < n; i += stride)
  {
#pragma unroll
    for(auto o = 0; o < OP; o++)
       y[i] = x[i] * user_arg[o] + y[i];
  }
}

int main(int argc, char *argv[])
{
  if(argc < 4)
  {
    std::cerr << "Usage: ./ami_shmmeasure Mem-Size-in-KB Num-Blocks Block-Size Num-FMAs <Val-1> <Val-2> ...\n";
    return -1;
  }

  // Parse the command line
  long Mem_Size = std::atol(argv[1]);
  Mem_Size *= 1024;
  long N = (Mem_Size / 2) / sizeof(float);
  int numBlocks = std::atoi(argv[2]);
  int blockSize = std::atoi(argv[3]);
  int numFMAs = std::atoi(argv[4]);

  std::cout << "Memory Targetted: " << Mem_Size/1024 << " KB" << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "numBlocks: " << numBlocks << std::endl;
  std::cout << "blockSize: " << blockSize << std::endl;

  float *user_arg;
  cudaMallocManaged(&user_arg, 8*sizeof(float));
  for(auto i = 5; i < argc; ++i)
  {
    user_arg[i-5] = std::atof(argv[i]);
  }

  /******************* Print the address *********************/
  std::cout << "Memory Address of UserArg: 0x" << std::hex << user_arg << std::endl;
               
  std::vector<functionDef> run_fma_kernel {fma<float, 1>, fma<float, 2>, fma<float, 3>, fma<float, 4>, fma<float, 5>, fma<float, 6>, fma<float, 7>, fma<float, 8>};

  run_fma_kernel[numFMAs-1]<<<numBlocks, blockSize, (N*2)*sizeof(float)>>>(N, user_arg);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
	  std::cout << "Error:\t" << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Just use the result
  float sum = 0.0f;
  std::cout << std::dec << "Reduction Sum: " << sum << std::endl;

  return 0;
}
