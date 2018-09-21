#!/usr/bin/env bash

#SBATCH -o test_dnn.txt

echo "NITER, num_filters, channels, imgsize, ksize, my_filter, my_input, my_conv, cublas_conv, cudnn"

#vgg
./conv 100 128 128 32 3 1 1
./conv 100 128 256 16 3 1 1
./conv 100 256 256 16 3 1 1
./conv 100 512 256 8  3 1 1
./conv 100 512 512 8  3 1 1
#alex
./conv 100 256 96 27 5 1 2
./conv 100 384 256 13 3 1 1
./conv 100 384 384 13 3 1 1
./conv 100 256 384 13 3 1 1
#vgg16
./conv 100 64 64 224 3 1 1
./conv 100 64 128 224 3 1 1
./conv 100 128 128 112 3 1 1
./conv 100 128 256 112 3 1 1
./conv 100 256 256 56 3 1 1
