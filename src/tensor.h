#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <memory>
#include <vector>
#include "context.h"
#include "tensor_shape.h"

class Tensor {
public:
  Tensor() 
      : shape_(TensorShape(0)), ctx_(Context::cpu()) {
    handle_ = new float[shape_.NumElements()];
  };

  Tensor(const TensorShape& shape, const Context& ctx)
      : shape_(shape), ctx_(ctx) {
    handle_ = new float[shape_.NumElements()];
  }

  Tensor(const Tensor& tensor) 
      : shape_(tensor.shape_), ctx_(tensor.ctx_){
    handle_ = new float[shape_.NumElements()];     
    for (int i = 0; i < shape_.NumElements(); i++) {
      handle_[i]= tensor.handle_[i];
    }
  }

  Tensor& operator=(const Tensor& tensor) {
    if (this != &tensor) {
      delete[] handle_;
      shape_ = tensor.shape_;
      ctx_ = tensor.ctx_;
      handle_ = new float[shape_.NumElements()];
      for (int i = 0; i < shape_.NumElements(); i++) {
        handle_[i] = tensor.handle_[i];
      }
    }
    return *this;
  }

  ~Tensor() {
    delete[] handle_;
  }

  Tensor operator+(const Tensor& rhs) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] + rhs.handle_[i];
    }
    return result;
  }

  Tensor operator-(const Tensor& rhs) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] - rhs.handle_[i];
    }
    return result;
  }

  Tensor operator*(const Tensor& rhs) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] * rhs.handle_[i];
    }
    return result;
  }

  Tensor operator/(const Tensor& rhs) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] / rhs.handle_[i];
    }
    return result;
  }

  Tensor operator+(float val) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] + val;
    }
    return result;
  }

  Tensor operator-(float val) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] - val;
    }
    return result;
  }

  Tensor operator*(float val) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] * val;
    }
    return result;
  }

  Tensor operator/(float val) const {
    Tensor result(shape_, ctx_);
    for (int i = 0; i < NumElements(); i++) {
      result.handle_[i] = handle_[i] / val;
    }
    return result;
  }

  Tensor& operator+=(const Tensor& rhs) {
    for (int i = 0; i < NumElements(); i++) {
      handle_[i] += rhs.handle_[i];
    }
    return *this;
  }

  Tensor& operator-=(const Tensor& rhs) {
    for (int i = 0; i < NumElements(); i++) {
      handle_[i] -= rhs.handle_[i];
    }
    return *this;
  }

  Tensor& operator*=(const Tensor& rhs) {
    for (int i = 0; i < NumElements(); i++) {
      handle_[i] *= rhs.handle_[i];
    }
    return *this;
  }

  Tensor& operator/=(const Tensor& rhs) {
    for (int i = 0; i < NumElements(); i++) {
      handle_[i] /= rhs.handle_[i];
    }
    return *this;
  }

  void SyncFromCPU(const float* data, size_t size) {
    for (int i = 0; i < shape_.NumElements(); i++) {
      handle_[i] = data[i];
    } 
  }

  void SyncFromVector(const std::vector<float>& data, size_t size) {
    for (int i = 0; i < shape_.NumElements(); i++) {
      handle_[i] = data[i];
    } 
  }

  const TensorShape& GetTensorShape() const { return shape_; }
  const Context& GetContext() const { return ctx_; }

  float* GetHandle() { return handle_; }
  const float* GetHandle() const { return handle_; }

  int NumElements() const {
    return shape_.NumElements();
  }

  // For 2 dims tensor and 1 dim tensor
  void Debug() const {
    if (shape_.NumDims() == 2) {
      int dim_a = shape_.DimSize(0);
      int dim_b = shape_.DimSize(1);
      for (int i = 0; i < dim_a; i++) {
        for (int j = 0; j < dim_b; j++) {
          std::cout << handle_[dim_b * i + j] << " ";
        }
        std::cout << std::endl;
      } 
    } else {
      for (int i = 0; i < shape_.DimSize(0); i++) {
        std::cout << handle_[i] << " ";
      }
      std::cout << std::endl;
    }
  }

private:
  float* handle_;
  Context ctx_;
  TensorShape shape_;
};


Tensor operator+(float val, const Tensor& rhs);

Tensor operator*(float val, const Tensor& rhs);

#endif
