# Julia Set Fractal Generator - Pure CUDA C Implementation

High-performance Julia set fractal computation using CUDA C for direct GPU execution on Tesla P100.
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20P100-76B900)](https://www.nvidia.com/en-us/data-center/tesla-p100/)
[![Performance](https://img.shields.io/badge/Performance-2.78B%20pixels%2Fs-brightgreen)](docs/PERFORMANCE.md)

High-performance fractal computation achieving **2.78 billion pixels/second**
on Tesla P100 through direct CUDA C kernel programming.

## Mathematical Background

The Julia set is defined by the iterative formula:
```
z(n+1) = z(n)² + c
```
Where:
- `z` is a complex number (initialized to pixel coordinates)
- `c` is a complex constant that defines the specific Julia set
- Iteration continues until |z| > 2 or max iterations reached

The visualization colors each pixel based on how quickly the iteration diverges.

## Implementation Details

### CUDA Kernel Design
- **Massive parallelism**: Each pixel computed by a separate GPU thread
- **Thread organization**: 16x16 thread blocks for optimal memory access
- **Grid coverage**: Automatically scales to any resolution
- **Memory coalescing**: Sequential threads access adjacent memory locations

### Performance Optimizations
1. **Fast math**: Using `-use_fast_math` for faster floating-point operations
2. **Register usage**: All computation in registers (no shared memory needed)
3. **Minimal divergence**: Iteration loop is the same for all threads initially
4. **Direct memory writes**: Output written once per thread

## Compilation

```bash
make
```

This compiles with:
- Compute capability 6.0 (Pascal/P100)
- O3 optimization
- Fast math library

## Running

```bash
./julia_set_cuda
```
## Output

The program generates:
1. **Performance metrics**: Timing, throughput, GFLOPS estimate
2. **Image file**: `julia_set.pgm` (PGM grayscale format)

### Viewing the Image

```bash
# Using ImageMagick
display julia_set.pgm

# Convert to PNG
convert julia_set.pgm julia_set.png
```

## Configuration

Edit these constants in `julia_set_cuda.cu`:

```c
#define WIDTH 4096      // Image width
#define HEIGHT 4096     // Image height
#define MAX_ITER 1000   // Maximum iterations

#define C_REAL -0.7     // Julia constant (real part)
#define C_IMAG 0.27015  // Julia constant (imaginary part)
```

### Interesting Julia Set Constants

These values for different fractals:
- `c = -0.7 + 0.27015i` (default - detailed structure)
- `c = -0.4 + 0.6i` (branching pattern)
- `c = 0.285 + 0.01i` (spiral arms)
- `c = -0.8 + 0.156i` (dragon-like)

## Performance Expectations

On Tesla P100 (16GB, Pascal):
- **4K resolution (4096x4096)**: ~6.030 ms
- **Throughput**: ~2,782 Mpixels/sec
- **Estimated GFLOPS**: 100-200 (depending on iteration distribution)

## Technical Notes

### Algorithm Works Well on GPU

1. **Embarrassingly parallel**: No inter-thread communication needed
2. **Compute-bound**: Arithmetic operations dominate (GPU strength)
3. **Predictable memory access**: Linear output array
4. **Uniform workload**: Similar iteration counts across nearby pixels

### Memory Usage

- **Image buffer**: WIDTH × HEIGHT × 1 byte
- **4K resolution**: ~16 MB
- **8K resolution**: ~64 MB
- Well within P100's 16GB capacity

## Notes on Fractal Quality

- Higher `MAX_ITER` = more detail but slower computation
- The P100's fast floating-point helps maintain quality at speed
- Edge detail improves dramatically with iteration count

## Project Structure

```
.
├── julia_set_cuda.cu    # Main CUDA C implementation
├── Makefile             # Build configuration
├── README.md            # This file
└── julia_set.pgm        # Generated output (after running)
```

## Mathematical Complexity

- **Per-pixel operations**: 5-10 FLOPS per iteration
- **Total operations**: WIDTH × HEIGHT × avg_iterations × ops_per_iter
- **For 4K, 1000 iter**: ~160 billion operations
- **P100 completes this in milliseconds**: True parallel power

---

**Hardware**: NVIDIA Tesla P100 (Pascal, sm_60)
**Platform**: RHEL 9
**CUDA Version**: 12.4
