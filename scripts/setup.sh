#!/bin/bash
# Setup and verification script for Julia Set CUDA project
# For RHEL 9 with Tesla P100

set -e

echo "=== Julia Set CUDA Project Setup ==="
echo ""

# Check CUDA installation
echo "1. Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found!"
    echo "Please install CUDA toolkit or add it to PATH"
    echo "Example: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
echo "   Found CUDA version: $CUDA_VERSION"

# Check GPU
echo ""
echo "2. Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found"
else
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
fi

# Verify compute capability
echo ""
echo "3. Verifying compute capability..."
GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
echo "   GPU Compute Capability: $GPU_COMPUTE"
if [[ $GPU_COMPUTE != "6.0" ]]; then
    echo "   WARNING: Expected 6.0 for Tesla P100"
    echo "   You may need to adjust CUDA_ARCH in Makefile"
fi

# Build project
echo ""
echo "4. Building project..."
make clean
make

if [ -f "./julia_set_cuda" ]; then
    echo "   ✓ Build successful!"
else
    echo "   ✗ Build failed!"
    exit 1
fi

# Run test
echo ""
echo "5. Running test computation..."
./julia_set_cuda

# Check output
echo ""
echo "6. Verifying output..."
if [ -f "julia_set.pgm" ]; then
    SIZE=$(stat -c%s julia_set.pgm)
    echo "   ✓ Output file created: julia_set.pgm ($SIZE bytes)"
    
    # Check if ImageMagick is available
    if command -v convert &> /dev/null; then
        echo "   Converting to PNG..."
        convert julia_set.pgm julia_set.png
        echo "   ✓ PNG file created: julia_set.png"
    else
        echo "   Note: Install ImageMagick to convert PGM to PNG"
        echo "   Command: sudo dnf install ImageMagick"
    fi
else
    echo "   ✗ Output file not created!"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo "Your CUDA environment is ready!"
echo ""
echo "Next steps:"
echo "  - View image: display julia_set.pgm"
echo "  - Modify constants in julia_set_cuda.cu"
echo "  - Run: make && ./julia_set_cuda"
echo "  - Clean: make clean"
