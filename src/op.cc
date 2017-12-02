#include <memory>
#include "node.h"
#include "op.h"

void AddOp::Compute(const Node& node,
                    const std::vector<Tensor>& in_tensors, 
                    std::vector<Tensor>& out_tensors) {
  const TensorShape& out_shape = out_tensors[0].GetTensorShape();
  for (int i = 0; i < out_shape.num_elements(); i++) {
    out_tensors[0].GetHandle()[i] = 
        in_tensors[0].GetHandle()[i] + in_tensors[1].GetHandle()[i];
  }
}

void AddOp::Infer(const Node& node,
                  const std::vector<TensorShape>& in_shapes,
                  std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void AddOp::Gradient(const Node& node, 
                     const Node& in_grad, 
                     std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}

void MinusOp::Compute(const Node& node,
                      const std::vector<Tensor>& in_tensors, 
                      std::vector<Tensor>& out_tensors) {
  const TensorShape& out_shape = out_tensors[0].GetTensorShape();
  for (int i = 0; i < out_shape.num_elements(); i++) {
    out_tensors[0].GetHandle()[i] = 
        in_tensors[0].GetHandle()[i] - in_tensors[1].GetHandle()[i];
  }
}

void MinusOp::Infer(const Node& node,
                    const std::vector<TensorShape>& in_shapes,
                    std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void MinusOp::Gradient(const Node& node,
                       const Node& in_grad, 
                       std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}

void MultiplyOp::Compute(const Node& node,
                         const std::vector<Tensor>& in_tensors, 
                         std::vector<Tensor>& out_tensors) {
  const TensorShape& out_shape = out_tensors[0].GetTensorShape();
  for (int i = 0; i < out_shape.num_elements(); i++) {
    out_tensors[0].GetHandle()[i] = 
        in_tensors[0].GetHandle()[i] * in_tensors[1].GetHandle()[i];
  }
}

void MultiplyOp::Infer(const Node& node,
                       const std::vector<TensorShape>& in_shapes,
                       std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void MultiplyOp::Gradient(const Node& node, 
                          const Node& in_grad, 
                          std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {in_grad * inputs[0], in_grad * inputs[1]};
}

void MultiplyByConstOp::Compute(const Node& node,
                                const std::vector<Tensor>& in_tensors, 
                                std::vector<Tensor>& out_tensors) {
  const TensorShape& out_shape = out_tensors[0].GetTensorShape();
  float const_val;
  node.GetAttr("const_val", const_val);
  for (int i = 0; i < out_shape.num_elements(); i++) {
    out_tensors[0].GetHandle()[i] = in_tensors[0].GetHandle()[i] * const_val;
  }
}

void MultiplyByConstOp::Infer(const Node& node,
                       const std::vector<TensorShape>& in_shapes,
                       std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void MultiplyByConstOp::Gradient(const Node& node, 
                          const Node& in_grad, 
                          std::vector<Node>& out_grads) {
  float const_val;
  node.GetAttr("const_val", const_val);
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {in_grad * const_val};
}

void DevideOp::Compute(const Node& node,
                       const std::vector<Tensor>& in_tensors, 
                       std::vector<Tensor>& out_tensors) {
  const TensorShape& out_shape = out_tensors[0].GetTensorShape();
  for (int i = 0; i < out_shape.num_elements(); i++) {
    out_tensors[0].GetHandle()[i] = 
        in_tensors[0].GetHandle()[i] / in_tensors[1].GetHandle()[i];
  }
}

void DevideOp::Infer(const Node& node,
                    const std::vector<TensorShape>& in_shapes,
                    std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void DevideOp::Gradient(const Node& node, 
                        const Node& in_grad, 
                        std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs);
  Node lhs_node = in_grad * node / inputs[1];
  Node rhs_node = in_grad / inputs[1];
  out_grads = {lhs_node, rhs_node};
}

void MatMulOp::Compute(const Node& node,
                       const std::vector<Tensor>& in_tensors, 
                       std::vector<Tensor>& out_tensors) {
  bool trans_a;
  node.GetAttr("trans_a", trans_a);
  bool trans_b;
  node.GetAttr("trans_b", trans_b);

  const float* a = in_tensors[0].GetHandle();
  const float* b = in_tensors[1].GetHandle();

  TensorShape shape_a = in_tensors[0].GetTensorShape();
  TensorShape shape_b = in_tensors[1].GetTensorShape();

  int num_m = trans_a ? shape_a.dim_size(1) : shape_a.dim_size(0);
  int num_n = trans_b ? shape_b.dim_size(0) : shape_b.dim_size(1);
  int num_k = trans_a ? shape_a.dim_size(0) : shape_b.dim_size(1);

  for (int i = 0; i < num_m; i++) {
    for (int j = 0; j < num_n; j++) {
      float sum = 0.0;
      for (int k = 0; k < num_k; k++) {
        float a_val = a[num_k * i + k];
        float b_val = b[num_k * j + k];
        sum += a_val * b_val;
      }
      out_tensors[0].GetHandle()[num_m * i + j] = sum;
    }
  }
}

void MatMulOp::Infer(const Node& node,
                     const std::vector<TensorShape>& in_shapes,
                     std::vector<TensorShape>& out_shapes) {
  bool trans_a;
  node.GetAttr("trans_a", trans_a);
  bool trans_b;
  node.GetAttr("trans_b", trans_b);
  int m = trans_a ? in_shapes[0].dim_size(1) : in_shapes[0].dim_size(0);
  int n = trans_b ? in_shapes[1].dim_size(0) : in_shapes[1].dim_size(1);
  out_shapes = {TensorShape(m, n)};
}

void MatMulOp::Gradient(const Node& node,
                        const Node& in_grad,
                        std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  bool trans_a;
  node.GetAttr("trans_a", trans_a);
  bool trans_b;
  node.GetAttr("trans_b", trans_a);
  
  if (trans_a == false && trans_b == false) {
    Node lhs_grad = MatMulOperator(in_grad, inputs[1], false, true);
    Node rhs_grad = MatMulOperator(inputs[0], in_grad, true, false);
    out_grads = {lhs_grad, rhs_grad};
  } else if (trans_a == true && trans_b == false) {
    Node lhs_grad = MatMulOperator(inputs[1], in_grad, false, true);
    Node rhs_grad = MatMulOperator(inputs[0], in_grad, true, false);
    out_grads = {lhs_grad, rhs_grad};
  } else if (trans_a == false && trans_b == true) {
    Node lhs_grad = MatMulOperator(in_grad, inputs[1], false, true);
    Node rhs_grad = MatMulOperator(in_grad, inputs[0], true, false);
    out_grads = {lhs_grad, rhs_grad};
  } else {
    Node lhs_grad = MatMulOperator(inputs[1], in_grad, false, true); 
    Node rhs_grad = MatMulOperator(in_grad, inputs[0], true, false);
    out_grads = {lhs_grad, rhs_grad};
  }
}

void ZerosOp::Compute(const Node& node,
                      const std::vector<Tensor>& in_tensors,
                      std::vector<Tensor>& out_tensors) {
  for (int i = 0; i < in_tensors[0].NumElements(); i++) {
    out_tensors[0].GetHandle()[i] = 0;
  }
}

void ZerosOp::Infer(const Node& node,
                    const std::vector<TensorShape>& in_shapes,
                    std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void ZerosOp::Gradient(const Node& node, 
                       const Node& in_grad,
                       std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {ZerosOperator(inputs[0])};
}

void OnesOp::Compute(const Node& node,
                     const std::vector<Tensor>& in_tensors,
                     std::vector<Tensor>& out_tensors) {
  for (int i = 0; i < in_tensors[0].NumElements(); i++) {
    out_tensors[0].GetHandle()[i] = 1;
  }
}

void OnesOp::Infer(const Node& node,
                   const std::vector<TensorShape>& in_shapes,
                   std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void OnesOp::Gradient(const Node& node,
                      const Node& in_grad,
                      std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {ZerosOperator(inputs[0])};
}

void ReduceSumAxisZeroOp::Compute(const Node& node,
                                  const std::vector<Tensor>& in_tensors,
                                  std::vector<Tensor>& out_tensors) {
  const float* in = in_tensors[0].GetHandle();
  int num_elements = in_tensors[0].NumElements();
  int reduce_elements = num_elements / in_tensors[0].GetTensorShape().dim_size(0);
  float* out = out_tensors[0].GetHandle();
  for (int i = 0; i < num_elements; i++) {
    out[i % reduce_elements] = in[i];
  }
}

void ReduceSumAxisZeroOp::Infer(const Node& node,
                    const std::vector<TensorShape>& in_shapes,
                    std::vector<TensorShape>& out_shapes) {
  for (int i = 1; i < in_shapes[0].dims(); i++) {
    out_shapes[0].AppendDim(in_shapes[0].dim_size(i));
  }
}

void ReduceSumAxisZeroOp::Gradient(const Node& node, 
                                   const Node& in_grad,
                                   std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {BroadCastToOperator(in_grad, inputs[0])};
}

void BroadCastToOp::Compute(const Node& node,
                            const std::vector<Tensor>& in_tensors,
                            std::vector<Tensor>& out_tensors) {
  // we need to duplicate one elements n_times time
  int n_times = in_tensors[1].GetTensorShape().dim_size(0);
  int num_elements = in_tensors[0].NumElements();
  
  float* base = out_tensors[0].GetHandle();
  for (int i = 0; i < n_times; i++) {
    for (int j = 0; j < num_elements; i++) {
      base[j] = in_tensors[0].GetHandle()[j];
    } 
    base += num_elements;
  }
}

void BroadCastToOp::Infer(const Node& node,
                          const std::vector<TensorShape>& in_shapes,
                          std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[1]};
}

void BroadCastToOp::Gradient(const Node& node,
                             const Node& in_grad, 
                             std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  Node lhs_node = ReduceSumAxisZeroOperator(in_grad);
  Node rhs_node = ZerosOperator(inputs[1]);
  out_grads = {lhs_node, rhs_node};
}

void SoftmaxOp::Compute(const Node& node,
                        const std::vector<Tensor>& in_tensors,
                        std::vector<Tensor>& out_tensors) {
  Tensor y = in_tensors[0];
  int m = y.GetTensorShape().dim_size(0);
  int n = y.GetTensorShape().dim_size(1);

  for (int i = 0; i < m; i++) {
    float sum = 0.0;
    for (int j = 0; j < n; j++) {
      float val = exp(y.GetHandle()[n * i + j]);
      sum += val; 
      out_tensors[0].GetHandle()[n * i + j] = val;
    }
    for (int j = 0; j < n; j++) {
      out_tensors[0].GetHandle()[n * i + j] /= sum;
    }
  }
}

void SoftmaxOp::Infer(const Node& node,
                      const std::vector<TensorShape>& in_shapes,
                      std::vector<TensorShape>& out_shapes) {
  out_shapes = {in_shapes[0]};
}

void SoftmaxOp::Gradient(const Node& node,
                         const Node& in_grad,
                         std::vector<Node>& out_grads) {
  std::cout << "Softmax Op has no gradient function" << std::endl;
}

void SoftmaxCrossEntropyOp::Compute(const Node& node,
                                    const std::vector<Tensor>& in_tensors,
                                    std::vector<Tensor>& out_tensors) {
  Tensor y = in_tensors[0];
  Tensor y_ = in_tensors[1];

  int m = y.GetTensorShape().dim_size(0);
  int n = y.GetTensorShape().dim_size(1);
  float total_sum = 0.0;
  std::vector<float> tmp(n);
  for (int i = 0; i < m; i++) {
    float sum = 0.0;
    for (int j = 0; j < n; j++) {
       float val = exp(y.GetHandle()[n * i + j]);
       tmp[j] = val;
       sum += val;
    }
    for (int j = 0; j < n; j++) {
      tmp[j] /= sum;
      total_sum += (-1) * y_.GetHandle()[n * i + j] * log(tmp[j]);
    }
  }
  out_tensors[0].GetHandle()[0] = (total_sum / m / n);
}

void SoftmaxCrossEntropyOp::Infer(const Node& node,
                                  const std::vector<TensorShape>& in_shapes,
                                  std::vector<TensorShape>& out_shapes) {
  out_shapes = {TensorShape(1)};
}

void SoftmaxCrossEntropyOp::Gradient(const Node& node, 
                                     const Node& in_grad,
                                     std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs);
  Node lhs_grad = (SoftmaxOperator(inputs[0]) + inputs[1] * -1) * in_grad;
  Node rhs_grad = ZerosOperator(inputs[1]);
  out_grads = {lhs_grad, rhs_grad};
}

std::shared_ptr<Op> Op::Create(const std::string& name) {
  if (name == "Add") {
    return std::make_shared<AddOp>(name);
  } else if (name == "Minus") {
    return std::make_shared<MinusOp>(name);
  } else if (name == "Multiply"){
    return std::make_shared<MultiplyOp>(name);
  } else if (name == "MultiplyByConst"){
    return std::make_shared<MultiplyByConstOp>(name);
  } else if (name == "Devide"){
    return std::make_shared<DevideOp>(name);
  } else if (name == "MatMul") {
    return std::make_shared<MatMulOp>(name);
  } else if (name == "Zeros"){
    return std::make_shared<ZerosOp>(name);
  } else if (name == "Ones"){
    return std::make_shared<OnesOp>(name);
  } else if (name == "ReduceSumAxisZero"){
    return std::make_shared<ReduceSumAxisZeroOp>(name);
  } else if (name == "BroadCastTo"){
    return std::make_shared<BroadCastToOp>(name);
  } else if (name == "Softmax"){
    return std::make_shared<SoftmaxOp>(name);
  } else if (name == "SoftmaxCrossEntropy"){
    return std::make_shared<SoftmaxCrossEntropyOp>(name);
  } else {
    return std::make_shared<DevideOp>(nullptr);
  }
}

