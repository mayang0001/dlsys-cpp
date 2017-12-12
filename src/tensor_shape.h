#ifndef TENSOR_SHAPE_
#define TENSOR_SHAPE_

#include <iostream>
#include <string>

class TensorShape {
public:
  TensorShape()
      : num_dims_(0) {
  }

  explicit TensorShape(int x) 
      : num_dims_(1) {
    dim_size_ = new int;
    dim_size_[0] = x;
  }

  TensorShape(int x, int y) 
      : num_dims_(2) {
    dim_size_ = new int[2];
    dim_size_[0] = x;
    dim_size_[1] =y;
  }

  TensorShape(int x, int y, int z)
      : num_dims_(3) {
    dim_size_ = new int[3];
    dim_size_[0] = x;
    dim_size_[1] = y;
    dim_size_[2] = x;
  }

  TensorShape(const TensorShape& shape) 
      : num_dims_(shape.num_dims_) {
    dim_size_ = new int[num_dims_];
    for (int i = 0; i < num_dims_; i++) {
      dim_size_[i] = shape.DimSize(i);
    }
  }

  TensorShape& operator=(const TensorShape& shape) {
    if (this != &shape) {
      delete[] dim_size_;
      num_dims_ = shape.num_dims_;
      dim_size_ = new int[shape.NumDims()];
      for (int i = 0; i < shape.NumDims(); i++) {
        dim_size_[i] = shape.DimSize(i);
      }
    }
    return *this;
  }

  TensorShape(TensorShape&& shape) 
      : num_dims_(shape.num_dims_) {
    dim_size_ = shape.dim_size_;
    shape.dim_size_ = nullptr;
  }

  TensorShape& operator=(TensorShape&& shape) {
    if (this != &shape) {
      num_dims_ = shape.num_dims_;
      dim_size_ = shape.dim_size_;
      shape.dim_size_ = nullptr;
    } 
    return *this;
  }

  bool operator==(const TensorShape& rhs) {
    if (num_dims_ != rhs.num_dims_) {
      return false;
    } else {
      for (int i = 0; i < num_dims_; i++) {
        if (dim_size_[i] != rhs.dim_size_[i]) {
          return false;
        }
      }
    }
    return true;
  }

  bool operator!=(const TensorShape& rhs) {
    return !(*this == rhs);
  }

  ~TensorShape() {
    delete[] dim_size_;  
  }

  void AppendDim(int dim) {
    if (num_dims_ == 0) {
      dim_size_ = new int[3];      
    }
    dim_size_[num_dims_++] = dim;
  }

  int NumDims() const { 
    return num_dims_; 
  }

  int DimSize(int d) const { 
    if (d >= num_dims_) {
      return -1;
    } else {
      return dim_size_[d];
    }
  } 
  
  int NumElements() const {
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
  int num_dims_;
  int* dim_size_;
};

#endif
