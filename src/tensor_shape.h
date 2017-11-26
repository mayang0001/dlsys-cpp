#ifndef TENSOR_SHAPE_
#define TENSOR_SHAPE_

#include <iostream>
#include <string>

class TensorShape {
public:
  TensorShape() {
  }

  ~TensorShape() {
    delete[] dim_size_;  
  }

  TensorShape(int x) {
    dim_size_ = new int;
    dim_size_[0] = x;
    num_dims_ = 1; 
  }

  TensorShape(int x, int y) {
    dim_size_ = new int[2];
    dim_size_[0] = x;
    dim_size_[1] =y;
    num_dims_ = 2;
  }

  TensorShape(int x, int y, int z) {
    dim_size_ = new int[3];
    dim_size_[0] = x;
    dim_size_[1] = y;
    dim_size_[2] = x;
    num_dims_ = 3;
  }

  TensorShape(const TensorShape& shape) {
    num_dims_ = shape.dims(); 
    dim_size_ = new int[num_dims_];
    for (int i = 0; i < num_dims_; i++) {
      dim_size_[i] = shape.dim_size(i);
    }
  }

  TensorShape& operator=(const TensorShape& shape) {
    if (this != &shape) {
      delete[] dim_size_;
      num_dims_ = shape.num_dims_;
      dim_size_ = new int[shape.dims()];
      for (int i = 0; i < shape.dims(); i++) {
        dim_size_[i] = shape.dim_size(i);
      }
    }
    return *this;
  }

  int dims() const { return num_dims_; }
  int dim_size(int d) const { 
    if (d >= num_dims_) {
      return -1;
    } else {
      return dim_size_[d];
    }
  } 

  int num_elements() const {
    int num_elements = 1;
    for (int i = 0; i < num_dims_; i++) {
      num_elements *= dim_size_[i]; 
    } 
    return num_elements;
  }

  std::string DebugString() const {
    std::string shape_str = "[";
    for (int i = 0; i < num_dims_; i++) {
      shape_str += (std::to_string(dim_size_[i]) + ",");
    }
    shape_str.resize(shape_str.size() - 1);
    shape_str += "]";
    return shape_str;
  }
private:
  int* dim_size_;
  int num_dims_;
};

#endif
