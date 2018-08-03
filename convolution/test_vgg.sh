#!/usr/bin/env bash

#SBATCH -o test_vgg.txt

echo "NITER, batch_size, num_kernels, channels, imgsize, ksize, magma_kernel_conc, magma_conv, cublas, cudnn"

./conv 10 64 128 128 32 3 1 1
./conv 10 64 128 256 16 3 1 1
./conv 10 64 256 256 16 3 1 1
./conv 10 64 512 256 8  3 1 1
./conv 10 64 512 512 8  3 1 1
