#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <memory>
#include "context.h"
#include "tensor_shape.h"

class Tensor {
public:
  Tensor() : shape_(TensorShape(0)), ctx_(Context::cpu()) {
    handle_ = new float[shape_.num_elements()];
  };

  Tensor(const TensorShape& shape, const Context& ctx)
      : shape_(shape), ctx_(ctx) {
    handle_ = new float[shape_.num_elements()];
  }

  Tensor(const Tensor& tensor) 
      : shape_(tensor.shape_), ctx_(tensor.ctx_){
    handle_ = new float[shape_.num_elements()];     
    for (int i = 0; i < shape_.num_elements(); i++) {
      handle_[i]= tensor.handle_[i];
    }
  }

  Tensor& operator=(const Tensor& tensor) {
    if (this != &tensor) {
      delete[] handle_;
      shape_ = tensor.shape_;
      ctx_ = tensor.ctx_;
      handle_ = new float[shape_.num_elements()];
      for (int i = 0; i < shape_.num_elements(); i++) {
        handle_[i] = tensor.handle_[i];
      }
    }
    return *this;
  }

  ~Tensor() {
    delete[] handle_;
  }

  void SyncFromCPU(const float* data, size_t size) {
    for (int i = 0; i < shape_.num_elements(); i++) {
      handle_[i] = data[i];
    } 
  }

  const TensorShape& GetTensorShape() const { return shape_; }
  const Context& GetContext() const { return ctx_; }

  int num_elements() const {
    return shape_.num_elements();
  }

  // Just for 2 dim Tensor
  void Debug() const {
    int dim_a = shape_.dim_size(0);
    int dim_b = shape_.dim_size(1);
    for (int i = 0; i < dim_a; i++) {
      for (int j = 0; j < dim_b; j++) {
        std::cout << handle_[dim_b * i + j] << " ";
      }
      std::cout << std::endl;
    } 
  }

public:
  float* handle_;
private:
  Context ctx_;
  TensorShape shape_;
};

#endif
