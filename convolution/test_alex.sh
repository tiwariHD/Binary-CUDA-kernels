#!/usr/bin/env bash

#SBATCH -o test_alex.txt

echo "NITER, batch_size, num_kernels, channels, imgsize, ksize, magma_kernel_conc, magma_conv, cublas, cudnn"

./conv 10 128 256 96 27 5 1 2
./conv 10 128 384 256 13 3 1 1
./conv 10 128 384 384 13 3 1 1
./conv 10 128 256 384 13 3 1 1
