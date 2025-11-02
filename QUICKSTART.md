# Quick Start Guide - Julia Set CUDA

## On RHEL 9 System with Tesla P100

### 1. Go to the project julia

```bash

# Navigate to directory
cd julia_set_cuda/
```

### 2. Verify CUDA Setup

```bash
# Check CUDA compiler
nvcc --version

# Check GPU
nvidia-smi

# If nvcc not found, add CUDA to PATH:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Verify CUDA installation
- Check GPU compute capability
- Build the project
- Run test computation
- Create output image

### 4. Manual Build (Alternative)

```bash
# Compile
make

# Run
./julia_set_cuda

# View output
display julia_set.pgm

# Or convert to PNG
convert julia_set.pgm julia_set.png
```

## Expected Output

```
=== Julia Set CUDA Generator ===
Resolution: 4096x4096 pixels
Max iterations: 1000
Julia constant: -0.700 + 0.270i

Grid size: (256, 256)
Block size: (16, 16)
Total threads: 16777216

Performing warmup run...
Running 10 benchmark iterations...
  Run  1: 18.234 ms
  Run  2: 18.156 ms
  ...

=== Performance Results ===
Average computation time: 18.200 ms
Pixels computed: 16777216
Throughput: 922.00 Mpixels/sec
Estimated GFLOPS: 165.96

Image saved successfully!
```

## Customization

Edit `julia_set_cuda.cu` and change:

```c
// Try different Julia constants
#define C_REAL -0.4      // Change this
#define C_IMAG 0.6       // Change this

// Adjust resolution
#define WIDTH 8192       // Higher for more detail
#define HEIGHT 8192

// Adjust quality
#define MAX_ITER 2000    // Higher for smoother images
```

Then recompile:
```bash
make clean && make
```

## Troubleshooting

### "nvcc: command not found"
```bash
# Find CUDA installation
find /usr -name nvcc 2>/dev/null

# Add to PATH
export PATH=/path/to/cuda/bin:$PATH
```

### "CUDA Error: invalid device symbol"
Check that compute capability in Makefile matches your GPU:
```makefile
CUDA_ARCH = sm_60  # Tesla P100 is 6.0
```

### Low Performance
- Ensure GPU isn't throttling: `nvidia-smi`
- Check power mode: `nvidia-smi -q -d POWER`
- Verify no other processes using GPU

## Files in This Package

- `julia_set_cuda.cu` - Main CUDA C source code
- `Makefile` - Build configuration
- `README.md` - Project documentation
- `TECHNICAL_DETAILS.md` - Deep dive into implementation
- `setup.sh` - Automated setup and verification

## Next Steps

1. Run the default configuration
2. Try different Julia constants (see README)
3. Increase resolution to 8K
4. Modify coloring algorithm
5. Add your changes to your GitHub portfolio

## Performance Benchmarks

Record your results:
- Resolution: _______ × _______
- Time: _______ ms
- Throughput: _______ Mpixels/sec

Compare with:
- Different resolutions
- Different MAX_ITER values
- Different Julia constants

**Questions?** Check TECHNICAL_DETAILS.md for in-depth explanations.
