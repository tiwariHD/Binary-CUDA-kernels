#!/usr/bin/env bash

#SBATCH -o test.txt

echo "size, xnor, my_xnor_conc, my_xnor_mul, magma, cublas, gemm"

for i in $(seq 1024 1024 16384); do
    ./bench $i
done
