# Makefile for Julia Set CUDA project
# Tesla P100 is compute capability 6.0 (Pascal architecture)

NVCC = nvcc
CUDA_ARCH = sm_60
CFLAGS = -arch=$(CUDA_ARCH) -O3 -use_fast_math
TARGET = julia_set_cuda
SOURCE = julia_set_cuda.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)
	@echo "Compilation successful!"
	@echo "Run with: ./$(TARGET)"

clean:
	rm -f $(TARGET) julia_set.pgm julia_set.png
	@echo "Cleaned build artifacts"

run: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	@echo "Running benchmark..."
	./$(TARGET)

help:
	@echo "Available targets:"
	@echo "  make         - Compile the program"
	@echo "  make run     - Compile and run"
	@echo "  make clean   - Remove compiled files"
	@echo "  make benchmark - Run with timing"

.PHONY: all clean run benchmark help
