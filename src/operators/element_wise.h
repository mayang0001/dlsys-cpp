#include <cmath>
#include <iostream>

#define CpuUnaryFunctor(name, op)                        \
template <typename T>                                    \
struct Cpu##name##Functor {                              \
  void operator() (const T* in1, const int N, T* out) {  \
    for (int i = 0; i < N; i++) {                        \
      out[i] = op(in1[i]);                               \
    }                                                    \
  }                                                      \
};

#define CpuBinaryFunctor(name, op)                                      \
template <typename T>                                                   \
struct Cpu##name##Functor {                                             \
  void operator() (const T* in1, const T* in2, const int N, T* out) {   \
    for (int i = 0; i < N; i++) {                                       \
      out[i] = in1[i] op in2[i];                                        \
    }                                                                   \
  }                                                                     \
};

#define GpuUnaryFunctor(name, op)                                 \
template <typename T>                                             \
__global__ void name##Kernel(const T* in, const int N, T* out) {  \
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x;           \
       idx < N;                                                   \
       idx += blockDim.x * gridDim.x) {                           \
    out[idx] = op(in[idx]);                                       \
  }                                                               \
}                                                                 \
                                                                  \
template <typename T>                                             \
struct Gpu##name##Functor {                                       \
  void operator() (const T* in, const int N, T* out) {            \
    dim3 thread(256);                                             \
    dim3 block((N + thread.x) / thread.x);                        \
    name##Kernel<<<block, thread>>>(in, N, out);                  \
  }                                                               \
};

#define GpuBinaryFunctor(name, op)                                     		  \
template <typename T>                                                             \
__global__ void name##Kernel(const T* in1, const T* in2, const int N, T* out) {   \
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x;               		  \
       idx < N;                                                       		  \
       idx += blockDim.x * gridDim.x) {                               		  \
    out[idx] = in1[idx] op in2[idx];                                  		  \
  }                                                                   		  \
}                                                                     		  \
                                                                      		  \
template <typename T>                                                 		  \
struct Gpu##name##Functor {                                           		  \
  void operator() (const T* in1, const T* in2, const int N, T* out) { 		  \
    dim3 thread(256);                                                 		  \
    dim3 block((N + thread.x) / thread.x);                            		  \
    name##Kernel<T><<<block, thread>>>(in1, in2, N, out);             		  \
  }                                                                   		  \
};

CpuBinaryFunctor(Add, +);
