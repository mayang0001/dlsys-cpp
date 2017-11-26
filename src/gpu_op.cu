#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void array_set_kernel(int nrow, int ncol, 
		                 float* input, 
				 float value) {
  int c_ = blockIdx.x * blockDim.x + threadIdx.x;
  int r_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (r_ >= nrow || c_ >= ncol)  
    return;
  input[r_ * ncol + c_] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
  int nrow = arr->shape[0];
  int ncol = arr->shape[1];

  float* input_data = (float*)arr->data;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x,
	       (nrow + dimBlock.y - 1) / dimBlock.y);
  array_set_kernel<<<dimGrid, dimBlock>>>(nrow, ncol, input_data, value);
  return 0;
}


__global__ void broadcast_to_kernel(int ntimes, int nnum,
				    const float* input,
				    float* output) {
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n >= nnum)
    return;
  for (int i = 0; i < ntimes; i++) {
    float* output_ = output + nnum * i;
    output_[n] = input[n];
  } 
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  int ntimes = output->shape[0];
  int nnum = 1;
  for (int i = 1; i < output->ndim; i++)
    nnum *= output->shape[i];
  
  const float* input_data = (const float*)input->data;
  float* output_data = (float*)output->data;

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((nnum + dimBlock.x - 1) / dimBlock.x);
  
  broadcast_to_kernel<<<dimGrid, dimBlock>>>(
      ntimes, nnum, input_data, output_data);
  return 0;
}


__global__ void reduce_sum_axis_zero_kernel(int reduce_n, int remain_n,
		                            const float* input,
					    float* output) {
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n >= remain_n)
    return;
  
  float sum = 0.0;
  for (int i = 0; i < reduce_n; i++) {
    const float* input_ = input + remain_n * i;
    sum += input_[n];
  }
  output[n] = sum;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  int reduce_n = input->shape[0];
  int remain_n = 1;
  for (int i = 1; i < input->ndim; i++) {
    remain_n *= input->shape[i];
  }
  const float* input_data = (const float*)input->data;
  float* output_data = (float*)output->data;

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((remain_n + dimBlock.x- 1) / dimBlock.x);
  reduce_sum_axis_zero_kernel<<<dimGrid, dimBlock>>>(
      reduce_n, remain_n, input_data, output_data);
  return 0;
}


__global__ void matrix_elementwise_add_kernel(int nrow, int ncol, 
					      const float* input_a, 
					      const float* input_b, 
					      float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;
  if (r_ >= nrow || c_ >= ncol)
    return;
  output[r_ * ncol + c_] = input_a[r_ * ncol + c_] + input_b[r_ * ncol + c_];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, 
			      DLArrayHandle output) {
  int nrow = matA->shape[0];
  int ncol = matA->shape[1];
 
  const float* input_data_a = (const float*)matA->data;
  const float* input_data_b = (const float*)matB->data;
  float* output_data = (float*)output->data; 

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, 
	       (nrow + dimBlock.y - 1) / dimBlock.y);
  
  matrix_elementwise_add_kernel<<<dimGrid, dimBlock>>>(
      nrow, ncol, input_data_a, input_data_b, output_data); 
  return 0;
}


__global__ void matrix_elementwise_add_const_kernel(int nrow, int ncol, 
						    const float* input, 
						    float val, 
						    float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;
  if (r_ >= nrow || c_ >= ncol)
    return;
  output[r_ * ncol + c_] = input[r_ * ncol + c_] + val; 
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, 
				     float val,
                                     DLArrayHandle output) {
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  
  const float* input_data = (const float*)input->data;
  float* output_data = (float*)output->data; 
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, 
	       (nrow + dimBlock.y - 1) / dimBlock.y);
  
  matrix_elementwise_add_const_kernel<<<dimGrid, dimBlock>>>(
    nrow, ncol, input_data, val, output_data);
  return 0;
}


__global__ void matrix_elementwise_multiply_kernel(int nrow, int ncol, 
						   const float* input_a, 
						   const float* input_b, 
						   float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;
  if (r_ >= nrow || c_ >= ncol)
    return;
  output[r_ * ncol + c_] = input_a[r_ * ncol + c_] * input_b[r_ * ncol + c_];
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  int nrow = matA->shape[0];
  int ncol = matA->shape[1];
 
  const float* input_data_a = (const float*)matA->data;
  const float* input_data_b = (const float*)matB->data;
  float* output_data = (float*)output->data;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, 
	       (nrow + dimBlock.y - 1) / dimBlock.y);
  
  matrix_elementwise_multiply_kernel<<<dimBlock, dimGrid>>>(
      nrow, ncol, input_data_a, input_data_b, output_data); 
  return 0;
}


__global__ void matrix_multiply_const(int nrow, int ncol, 
				      const float* input, 
				      float val, 
				      float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;
  if (r_ >= nrow || c_ >= ncol)
    return;
  output[r_ * ncol + c_] = input[r_ * ncol + c_] * val; 

}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, 
			       float val,
                               DLArrayHandle output) {
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  
  const float* input_data = (const float*)input->data;
  float* output_data = (float*)output->data;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, 
	       (nrow + dimBlock.y - 1) / dimBlock.y);

  matrix_multiply_const<<<dimGrid, dimBlock>>>(
      nrow, ncol, input_data, val, output_data);
  return 0;
}

__global__ void matrix_multiply_kernel(const float* input_a, 
				       const float* input_b,
				       bool transposeA,
				       bool transposeB,
				       int nrow_a, int ncol_a,
				       int nrow_b, int ncol_b, 
				       int nrow, int ncol, int nwidth,
				       float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;
  if (r_ >= nrow || c_ >= ncol)
    return;

  float a = 0.0, b = 0.0, sum = 0.0;
  for (int i = 0; i < nwidth; i++) {
      a = transposeA ? input_a[ncol_a * i + r_] : input_a[ncol_a * r_ + i];
      b = transposeB ? input_b[ncol_b * c_ + i] : input_b[ncol_b * i + c_];
      sum += a * b;
  }
  output[ncol * r_ + c_] = sum;
  return;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  int nrow_a = matA->shape[0], ncol_a = matA->shape[1];
  int nrow_b = matB->shape[0], ncol_b = matB->shape[1];

  // nrow is the number of row of result matrix;
  int nrow = transposeA ? ncol_a : nrow_a;
  // ncol is the number of col of result matrix;
  int ncol = transposeB ? nrow_b : ncol_b;
  int nwidth = transposeA ? nrow_a : ncol_a;

  const float* input_data_a = (const float*)matA->data;
  const float* input_data_b = (const float*)matB->data;
  float* output_data = (float*)matC->data;
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x,
	       (nrow + dimBlock.y - 1) / dimBlock.y);

  matrix_multiply_kernel<<<dimGrid, dimBlock>>>(
      input_data_a, input_data_b, 
      transposeA, transposeB,
      nrow_a, ncol_a, nrow_b, ncol_b, 
      nrow, ncol, nwidth,
      output_data);
  return 0;
}

__global__ void relu_kernel(int nrow, int ncol, 
			    const float* input, 
			    float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;

  if (r_ >= nrow || c_ >= ncol)
    return;
  float val = input[r_ * ncol + c_];
  output[r_ * ncol + c_] = val >= 0 ? val : 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  int nrow = input->shape[0];
  int ncol = input->shape[1];

  const float* input_data = (const float*)input->data;
  float* output_data = (float*)output->data;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol+ dimBlock.x - 1) / dimBlock.x, 
	       (nrow + dimBlock.y - 1) / dimBlock.y);
  relu_kernel<<<dimGrid, dimBlock>>>(
      nrow, ncol, input_data, output_data);
  return 0;
}

__global__ void relu_gradient_kernel(int nrow, int ncol, 
				     const float* input, 
				     const float* in_grad, 
				     float* output) {
  int r_ = blockDim.y * blockIdx.y + threadIdx.y;
  int c_ = blockDim.x * blockIdx.x + threadIdx.x;

  if (r_ >= nrow || c_ >= ncol)
    return;

  float input_val = input[r_ * ncol + c_];
  float in_grad_val = in_grad[r_ * ncol + c_];
  
  output[r_ * ncol + c_] = input_val >= 0 ? in_grad_val : 0;
}

int DLGpuReluGradient(const DLArrayHandle input, 
		      const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  int nrow = input->shape[0];
  int ncol = input->shape[1];

  const float* input_data =  (const float*)input->data;
  const float* in_grad_data =  (const float*)in_grad->data;
  float* output_data = (float*)output->data;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, 
	       (nrow + dimBlock.y - 1) / dimBlock.y);
  relu_gradient_kernel<<<dimGrid, dimBlock>>>(
      nrow, ncol, 
      input_data,
      in_grad_data,
      output_data);
  return 0;
}

__global__ void softmax_kernel(int nrow, int ncol,
		               const float* input,
			       float* output) {
  int r_ = blockDim.x * blockIdx.x + threadIdx.x;
  if (r_ >= nrow)
    return;

  input += r_ * ncol;
  float min_val = input[0];
  for (int i = 1; i < ncol; i++)
    min_val = min(min_val, input[i]);

  float sum = 0.0;
  for (int i = 0; i < ncol; i++) 
    sum += exp(input[i] - min_val);

  output += r_ * ncol;
  for (int i = 0; i < ncol; i++)
    output[i] = exp(input[i] - min_val) / sum;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float* input_data = (const float*)input->data; 
  float* output_data = (float*)output->data;
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((nrow + dimBlock.x - 1) / dimBlock.x);

  softmax_kernel<<<dimGrid, dimBlock>>>(
      nrow, ncol, input_data, output_data); 
  return 0;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
