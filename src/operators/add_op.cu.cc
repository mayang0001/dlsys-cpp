#include "add_op.h"

template <typename T>
__global__ void AddKernel(const T* in1, const T* in2, const int N, T* out) {
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < N;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in1[idx] + in2[idx];
  }
}

template <typename T>
void AddFunctor<GpuContext, T>::operator() (const T* in1, const T* in2, 
                                            const int N, T* out) {
  dim3 thread(256);
  dim3 block(20);
  AddKernel<<<block, thread>>>(in1, in2, N, out);
}
