#ifndef OP_H_
#define OP_H_

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include "tensor.h"

class Op {
public:
  Op(const std::string& op_type) : op_type_(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors, 
                       std::vector<Tensor>& out_tensors) = 0;
    
  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) = 0;

  virtual void Gradient() = 0;

  std::string GetOpType() { return op_type_; }

  static std::shared_ptr<Op> Create(const std::string& name);

private:
  std::string op_type_;
};

class AddOp : public Op {
public:
  AddOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override {
    const TensorShape& out_shape = out_tensors[0].GetTensorShape();
    for (int i = 0; i < out_shape.num_elements(); i++) {
      out_tensors[0].GetHandle()[i] = 
          in_tensors[0].GetHandle()[i] + in_tensors[1].GetHandle()[i];
    }
  }

  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override {
    out_shapes.push_back(in_shapes[0]);
  }

  virtual void Gradient() override {
  
  }
};

class MinusOp : public Op {
public:
  MinusOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override {
    const TensorShape& out_shape = out_tensors[0].GetTensorShape();
    for (int i = 0; i < out_shape.num_elements(); i++) {
      out_tensors[0].GetHandle()[i] = 
          in_tensors[0].GetHandle()[i] - in_tensors[1].GetHandle()[i];
    }
  }

  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override {
    out_shapes.push_back(in_shapes[0]);
  }

  virtual void Gradient() override {
  
  }
};

class MultiplyOp : public Op {
public:
  MultiplyOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override {
    const TensorShape& out_shape = out_tensors[0].GetTensorShape();
    for (int i = 0; i < out_shape.num_elements(); i++) {
      out_tensors[0].GetHandle()[i] = 
          in_tensors[0].GetHandle()[i] * in_tensors[1].GetHandle()[i];
    }
  }

  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override {
    out_shapes.push_back(in_shapes[0]);
  }

  virtual void Gradient() override {
  
  }
};

class DevideOp : public Op {
public:
  DevideOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override {
    const TensorShape& out_shape = out_tensors[0].GetTensorShape();
    for (int i = 0; i < out_shape.num_elements(); i++) {
      out_tensors[0].GetHandle()[i] = 
          in_tensors[0].GetHandle()[i] / in_tensors[1].GetHandle()[i];
    }
  }

  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override {
    out_shapes.push_back(in_shapes[0]);
  }

  virtual void Gradient() override {
  
  }
};

class MatMulOp : public Op {
public:
  MatMulOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override {
    int num_m = in_tensors[0].GetTensorShape().dim_size(0);
    int num_n = in_tensors[0].GetTensorShape().dim_size(1);
    int num_k = in_tensors[1].GetTensorShape().dim_size(1);
    for (int i = 0; i < num_m; i++) {
      for (int j = 0; j < num_k; j++) {
        int sum = 0;
        for (int k = 0; k < num_n; k++) {
          sum += in_tensors[0].GetHandle()[num_n * i + k] * 
                 in_tensors[1].GetHandle()[num_k * k + j];
        }
        out_tensors[0].GetHandle()[num_k * i + j] = sum;
      }
    }
  }

  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override {
    TensorShape out_shape(in_shapes[0].dim_size(0), in_shapes[1].dim_size(1));
    out_shapes.push_back(out_shape);
  }

  virtual void Gradient() override {
  
  }
};

class SoftmaxOp : public Op {
public:
  SoftmaxOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override {
  }

  virtual void Infer(const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override {
  }

  virtual void Gradient() override {
  
  }
};
#endif
