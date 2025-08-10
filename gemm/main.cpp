/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasLt.h>

#define __NVCC__
#include "MeasurementSeries.hpp"
#include "dtime.hpp"
#include "gpu-error.h"

#include "sample_cublasLt_LtSgemm.h"
#include "helpers.h"

int main(int argc, char *argv[])
{
  if(argc < 5)
  {
    std::cerr << "Usage: MatMul M N K Profile\n";
    return -1;
  }

  // Parse the command line
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    int profile = std::atoi(argv[4]);

    std::cout << m <<","<< n << "," << k << std::endl;


    TestBench<float> props(m, n, k, 2.0f, 0.0f);

    if(profile)
    {
	    props.run([&props] {
		LtSgemm(props.ltHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			props.m,
			props.n,
			props.k,
			&props.alpha,
			props.Adev,
			props.m,
			props.Bdev,
			props.k,
			&props.beta,
			props.Cdev,
			props.m,
			props.workspace,
			props.workspaceSize);
	    });
	    return 0;
    }

  // warmup runs
  for (int iter = 0; iter < 3; iter++) {
    props.run([&props] {
        LtSgemm(props.ltHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                props.m,
                props.n,
                props.k,
                &props.alpha,
                props.Adev,
                props.m,
                props.Bdev,
                props.k,
                &props.beta,
                props.Cdev,
                props.m,
                props.workspace,
                props.workspaceSize);
    });
  }

  MeasurementSeries time;

  GPU_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  for (int iter = 0; iter < 31; iter++) {
    GPU_ERROR(cudaEventCreate(&start));
    GPU_ERROR(cudaEventCreate(&stop));
    GPU_ERROR(cudaEventRecord(start));

    props.run([&props] {
        LtSgemm(props.ltHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                props.m,
                props.n,
                props.k,
                &props.alpha,
                props.Adev,
                props.m,
                props.Bdev,
                props.k,
                &props.beta,
                props.Cdev,
                props.m,
                props.workspace,
                props.workspaceSize);
    });

    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "MS = " << milliseconds << std::endl;

    time.add(milliseconds);
  }

  std::cout << std::fixed << std::setprecision(4) << std::setw(5) << time.median() << std::endl;
  std::cout.flush();


    return 0;
}
