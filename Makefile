NVCC := nvcc

TEMP_NVCC := $(shell which nvcc)
CUDA_HOME :=  $(shell echo $(TEMP_NVCC) | rev |  cut -d'/' -f3- | rev)
MY_REPO := /home/ubuntu/GPU-Test/MyRepo/cuda-tests
#A100 - 80
#H100 - 90
ARCH := 80

# internal flags
NVCCFLAGS   := -std=c++20 -O3 -arch=sm_$(ARCH) --compiler-options="-O3 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" -Xcompiler "-Wall"
CCFLAGS     := 
NAME 		:= stream
LDFLAGS     := -L/opt/cuda/lib64 -lcuda -lnvidia-ml
PREFIX		:= hbm
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include -I$(MY_REPO)

# Source file (assumes there's only one .cu file in the directory)
SRC := $(wildcard *.cu)

# Target executable (replace .cu with .exe)
TARGET := $(SRC:.cu=.exe)

# Default rule
all: $(TARGET)

# Compile the .cu source file into an executable
%.exe: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

$(PREFIX)-$(NAME): hbm-stream.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
