#ifndef OP_H_
#define OP_H_

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include "tensor.h"
#include "operator.h"

class Node;

class Op {
public:
  Op(const std::string& op_type) : op_type_(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors, 
                       std::vector<Tensor>& out_tensors) = 0;
    
  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) = 0;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) = 0;

  std::string GetOpType() { return op_type_; }

  static std::shared_ptr<Op> Create(const std::string& name);

private:
  std::string op_type_;
};

class AddOp : public Op {
public:
  AddOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node,
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node,
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class AddByConstOp : public Op {
public:
  AddByConstOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class MinusOp : public Op {
public:
  MinusOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node,
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class MinusByConstOp : public Op {
public:
  MinusByConstOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class MultiplyOp : public Op {
public:
  MultiplyOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class MultiplyByConstOp : public Op {
public:
  MultiplyByConstOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class DevideOp : public Op {
public:
  DevideOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node,
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class DevideByConstOp : public Op {
public:
  DevideByConstOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
};

class MatMulOp : public Op {
public:
  MatMulOp(const std::string& op_type) : Op(op_type) {}

  // TODO Matrix transpose
  virtual void Compute(const Node& node,
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
  
};

class ZerosOp : public Op {
public:
  ZerosOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node,
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
  
};

class OnesOp : public Op {
public:
  OnesOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
  
};

class ReduceSumAxisZeroOp : public Op {
public:
  ReduceSumAxisZeroOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node,
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
  
};

class BroadCastToOp : public Op {
public:
  BroadCastToOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node,
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
  
};

class SoftmaxOp : public Op {
public:
  SoftmaxOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;

};

class SoftmaxCrossEntropyOp : public Op {
public:
  SoftmaxCrossEntropyOp(const std::string& op_type) : Op(op_type) {}

  virtual void Compute(const Node& node, 
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) override;

  virtual void Infer(const Node& node, 
                     const std::vector<TensorShape>& in_shapes, 
                     std::vector<TensorShape>& out_shapes) override;

  virtual void Gradient(const Node& ndoe, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) override;
  
};

#endif
