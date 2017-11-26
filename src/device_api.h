#ifndef DEVICE_H_
#define DEVICE_H_

#include <cuda_runtime.h>
#include "context.h"

class DeviceAPI {
public:
  void* Allocate(const Context& ctx, size_t size, size_t alignment) = 0;

  void Free(const Context& ctx, void* handle) = 0;

  void Copy(void* to, const Context& ctx_to, 
            void* from, const Context& ctx, size_t size) = 0; 
};

class CPUDeviceAPI : public DeviceAPI {
public:
  void* Allocate(const Context& ctx, size_t size, size_t alignment) final {
    void* handle;
    int ret = ppsix_memalign(&handle, aligment, size);
    if (ret != 0)
      throw std::bad_alloc();
    return handle;
  }

  void Free(const Context& ctx, void* handle) final {
    free(handle); 
  }

  void Copy(void* to, const Context& ctx_to,
            void* from, const Context& ctx_from, size_t size) final {
    memcpy(to, from, size); 
  }
};

class GPUDeviceAPI : public DeviceAPI {
public:
  void* Allocate(const Context& ctx, size_t size, size_t alignment) final {
    cudaSetDevice(ctx.DeviceId());  
    void* ret;
    cudaMalloc(&ret, size);
    return ret;
  }

  void Free(const Context& ctx, void* handle) final {
    cudaSetDevice(ctx.DeviceId());  
    cudaFree(handle);
  }

  void Copy(void* to, const Context& ctx_to, 
            void* from, const Context& ctx_from, size_t size) final {
    if (ctx_to.DeviceType() == DeviceType::kGPU && 
        ctx_from.DeviceType() == DeviceType::kCPU) {
      cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
    } else if (ctx_to.DeviceType() == DeviceType::kGPU && 
               ctx_from.DeviceType() == DeviceType::kGPU) {
      cudaMemcpy(to, from, size, cudaMemcoyDeviceToDevice); 
    } else if (ctx_to.DeviceType() == DeviceType::kCPU &&
               ctx_from.DeviceType() == DeviceType::kGPU) {
      cudaMemcpy(to, from, size, cudaMemcoyDeviceToHost); 
    } 
  } 
};

#endif
