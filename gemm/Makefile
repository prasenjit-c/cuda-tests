# CUDA compiler and flags
NVCC := nvcc
ARCH := 90

NVCCFLAGS := -O2 -std=c++14 -I.. -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

# Source files
SRC := sample_cublasLt_LtSgemm.cu helpers.cpp main.cpp

# Output binary
TARGET := sample_cublasLt_LtSgemm

# Default rule
all: $(TARGET)

# Linking everything with nvcc
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) -lcublasLt

# Clean rule
clean:
	rm -f $(TARGET)
