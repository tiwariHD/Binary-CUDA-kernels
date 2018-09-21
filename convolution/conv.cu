#include <cudnn.h>
#include <cassert>
#include <cublas_v2.h>
#include <ctime>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <bitset>
#include "conv_kernels.cu"

//undef to exclude a particular method
#define _CUBCOV
#define _MAGCOV
#define _CUDCOV

#if defined _CUDCOV
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
#endif

/*
template<typename T>
void printAr(T* in, int row, int col) {

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << in[(i * col) + j] << " ";
        }
        std::cout << std::endl;
    }
}

void printArBinZ(unsigned int* in, int depth, int row, int col) {

  for(int k = 0; k < depth; k++) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << std::bitset<32>(in[(k * row * col) +
                (i * col) + j]).to_string() << " ";
            //convertToBinary(in[(i * col) + j]);
            //std::cout << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
  }
}
*/
int main(int argc, char* argv[]) {

    srand(time(NULL));

    int NITER = (argc > 1) ? std::atoi(argv[1]) : 1;
    int num_kernels = (argc > 2) ? std::atoi(argv[2]) : 128;
    int chans = (argc > 3) ? std::atoi(argv[3]) : 128;
    int IMG = (argc > 4) ? std::atoi(argv[4]) : 32;
    int ksize = (argc > 5) ? std::atoi(argv[5]) : 3;
    int stride = (argc > 6) ? std::atoi(argv[6]) : 1;
    int pad = (argc > 7) ? std::atoi(argv[7]) : 1;

    std::cout << NITER << ", " << num_kernels << ", " << chans
		<< ", " << IMG << ", " << ksize << ", ";

    int height = IMG;
    int width = IMG;

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    float *h_input = (float*)malloc(chans * height * width * sizeof(float));
    for (int i = 0; i < (chans * height * width); i++) {
        double x = (double)rand() / RAND_MAX;
        h_input[i] = (x > 0.5) ? 1.0 : -1.0;
    }

    float* d_input{nullptr};
    cudaMalloc(&d_input, chans * height * width * sizeof(float));
    cudaMemcpy(d_input, h_input, chans * height * width * sizeof(float),
               cudaMemcpyHostToDevice);


    float *kernel_template = (float*)malloc(ksize * ksize * sizeof(float));
    for (int vi = 0; vi < (ksize * ksize); vi ++) {
      double ix = (double)rand() / RAND_MAX;
      kernel_template[vi] = (ix > 0.5) ? 1.0 : -1.0;
    }

    float *h_kernel = (float*)malloc(num_kernels * chans * ksize * ksize
                    * sizeof(float));
    for (int kernel = 0; kernel < num_kernels; ++kernel) {
      for (int chan = 0; chan < chans; ++chan) {
        for (int row = 0; row < ksize; ++row) {
          for (int column = 0; column < ksize; ++column) {
            h_kernel[(chans * ksize * ksize * kernel) + (ksize * ksize * chan)
            + (ksize * row) + column] = kernel_template[(row * ksize) + column];
          }
        }
      }
    }

    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, num_kernels * chans * ksize * ksize * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    /*std::cout << "kernel: " << num_kernels << ", "  << chans << ", " <<
            ksize << ", " << ksize << "\n";
    //printArZ(h_kernel, 4, chans, ksize * ksize);*/

    float* d_output{nullptr};
    int image_bytes_out = num_kernels * height_col * width_col * sizeof(float);
    cudaMalloc(&d_output, image_bytes_out);


//----------magma
#if defined _MAGCOV
    auto magma_conv = [&](unsigned int pad_val) {

    cudaMemset(d_output, 0, image_bytes_out);

    unsigned int* d_input_conc{nullptr};
    cudaMalloc(&d_input_conc,chans * height * width * sizeof(unsigned int)/32);
    cudaMemset(d_input_conc, 0, chans * height * width*sizeof(unsigned int)/32);

    unsigned int* d_kernel_conc{nullptr};
    cudaMalloc(&d_kernel_conc, num_kernels * chans * ksize * ksize *
        sizeof(float) / 32);
    cudaMemset(d_kernel_conc, 0, num_kernels * chans * ksize * ksize *
        sizeof(float) /32);

    unsigned int* d_imColArr{nullptr};
    int imColArraySize = height_col * width_col * chans / 32 * ksize * ksize *
        sizeof(int);
    cudaMalloc(&d_imColArr, imColArraySize);

    int M1 = num_kernels;
    int N1 = chans / 32 * ksize * ksize;
    int K1 = height_col * width_col;

    auto start1 = std::chrono::high_resolution_clock::now();
    //for filter: (256 * 128 * 3 * 3); threads needed <= 9; blocks needed = 4
    int grid = num_kernels * chans / 32, block = ksize * ksize;
    concatenate_input_kernel<<< grid, block >>>(d_kernel, d_kernel_conc,
        ksize, ksize);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = end1 - start1;

    grid = chans / 32;
    block = min(height, 256);

    //for input:(128 * 32 * 32);threads needed <= 1024;blocks needed=128/32=4
    auto start2 = std::chrono::high_resolution_clock::now();
    concatenate_input_kernel<<< grid, block, 0 >>>(d_input, d_input_conc,
        height, width);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff2 = (end2 - start2);

    auto start3 = std::chrono::high_resolution_clock::now();
    im2col_gpu_int(d_input_conc, chans / 32, height, width, ksize, pad, stride,
        d_imColArr, pad_val);
    cudaDeviceSynchronize();

    dim3 blockDim1(16, 16);
    int gridSize1 = ceil(static_cast<float>(K1) / static_cast<float>(96));
    int gridSize2 = ceil(static_cast<float>(M1) / static_cast<float>(96));
    dim3 gridDim1(gridSize1, gridSize2);
    my_xnor_gemm_kernel<<< gridDim1, blockDim1, 0 >>>(K1, M1, N1, d_imColArr,
        K1, d_kernel_conc, N1, d_output, K1, 0, 0);
    cudaDeviceSynchronize();
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff3 = (end3 - start3);   
    
    float* h_output = (float*)malloc(image_bytes_out);
    memset(h_output, 0, image_bytes_out);
    cudaMemcpy(h_output, d_output, image_bytes_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_input_conc);
    cudaFree(d_kernel_conc);
    cudaFree(d_imColArr);
    delete[] h_output;

    double* tym = (double*)malloc(3*sizeof(double));
    tym[0] = diff1.count();
    tym[1] = diff2.count();
    tym[2] = diff3.count();

    return tym;

    };
    //magma_conv(0);
    double mtym[3];
    mtym[0] = mtym[1] = mtym[2] = 0;
    for (int i = 0; i < NITER; i++) {
        double* temp = magma_conv(0xF0F0F0F0);
        mtym[0] += temp[0];
        mtym[1] += temp[1];
        mtym[2] += temp[2];
        delete[] temp;
    }
    mtym[0] /= NITER;
    mtym[1] /= NITER;
    mtym[2] /= NITER;
    //std::cout << "--------magma time: " << std::fixed << mtym.first << " s,
        //<< mtym.second << " s\n";
    std::cout << std::fixed << mtym[0] << ", " << mtym[1] << ", " << mtym[2]
        << ", ";

#endif

//----------cublas
#if defined _CUBCOV
    cublasHandle_t handle;
    cublasCreate(&handle);

    auto cublas_conv = [&]() {

    cudaMemset(d_output, 0, image_bytes_out);

    float* d_imColArr{nullptr};
    int imColArraySize = height_col * width_col * chans * ksize * ksize
        * sizeof(int);
    cudaMalloc(&d_imColArr, imColArraySize);

    float alpha = 1.0, beta = 0.0;
    int M1 = num_kernels;
    int N1 = chans * ksize * ksize;
    int K1 = height_col * width_col;

    auto start = std::chrono::high_resolution_clock::now();
    
    im2col_gpu_float(d_input, chans, height, width,
        ksize, pad, stride, d_imColArr, 0);
    cudaDeviceSynchronize();
  
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K1, M1, N1, &alpha,
        d_imColArr, K1, d_kernel, N1, &beta, d_output, K1);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    float* h_output = (float*)malloc(image_bytes_out);
    memset(h_output, 0, image_bytes_out);
    cudaMemcpy(h_output, d_output, image_bytes_out, cudaMemcpyDeviceToHost);

    cudaFree(d_imColArr);
    delete[] h_output;

    return diff.count();

    };
    double cutym = 0;
    for (int i = 0; i < NITER; i++) {
        cutym += cublas_conv();
    }
    cutym /= NITER;
    //std::cout << "--------cublas time: " << std::fixed << cutym << " s\n";
    //std::cout << std::fixed << cutym << ", ";
    std::cout << std::fixed << cutym << "\n";

    cublasDestroy(handle);
#endif
//----------cudnn

#if defined _CUDCOV
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    auto cudnn_conv = [&]() {

    cudaMemset(d_output, 0, image_bytes_out);

    const float alpha = 1.0f, beta = 0.0f;
    int batchSize = 1;
    int batch_size{0}, channelC{0}, heightC{0}, widthC{0}; 
    size_t workspace_bytes = 0;
    void* d_workspace{nullptr};

    auto start = std::chrono::high_resolution_clock::now();

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batchSize,
                                        /*channels=*/chans,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/num_kernels,
                                        /*in_channels=*/chans,
                                        /*kernel_height=*/ksize,
                                        /*kernel_width=*/ksize));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/pad,
                                             /*pad_width=*/pad,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION));
                                             /*computeType=CUDNN_DATA_FLOAT));*/
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channelC,
                                                   &heightC,
                                                   &widthC));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batchSize,
                                        /*channels=*/num_kernels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));

    //std::cerr << "cudnn-convalgo-" << convolution_algorithm << ", ";
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
    //std::cerr << "Workspace size-" << (workspace_bytes / 1048576.0) << "MB, ";
    assert(workspace_bytes > 0);
    cudaMalloc(&d_workspace, workspace_bytes);

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));

   
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      //std::cout<< "--------cudnn time: " << std::fixed << diff.count() << "\n";

    int image_bytes1 = batch_size * channelC * heightC * widthC * sizeof(float);
    float* h_output = new float[image_bytes1];
    cudaMemcpy(h_output, d_output, image_bytes1, cudaMemcpyDeviceToHost);

    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    delete[] h_output;

    return diff.count();
  };
    double cdtym = 0.0;
    for (int i = 0; i < NITER; i++) {
        cdtym += cudnn_conv();
    }
    cdtym /= NITER;
    //std::cout << "--------cudnn time: " << std::fixed << cdtym << " s\n";
    std::cout << std::fixed << cdtym << "\n";

    cudnnDestroy(cudnn);
#endif

    delete[] h_input;
    delete[] h_kernel;
    delete[] kernel_template;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

return 0;
}
