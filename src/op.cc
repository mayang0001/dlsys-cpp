#include <memory>
#include "node.h"
#include "op.h"

void AddOp::Gradient(const Node& node, 
                     const Node& in_grad, 
                     std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}

void MinusOp::Gradient(const Node& node,
                       const Node& in_grad, 
                       std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}

// TODO
void MultiplyOp::Gradient(const Node& node, 
                          const Node& in_grad, 
                          std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {in_grad * inputs[0], in_grad * inputs[1]};
}

void MatMulOp::Gradient(const Node& node,
                        const Node& in_grad,
                        std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  bool trans_a;
  node.GetParam("trans_a", trans_a);
  bool trans_b;
  node.GetParam("trans_b", trans_a);
  
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

void ZerosOp::Gradient(const Node& node, 
                       const Node& in_grad,
                       std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {ZerosOperator(inputs[0])};
}

void OnesOp::Gradient(const Node& node,
                      const Node& in_grad,
                      std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {ZerosOperator(inputs[0])};
}

void ReduceSumAxisZeroOp::Gradient(const Node& node, 
                                   const Node& in_grad,
                                   std::vector<Node>& out_grads) {
  std::vector<Node> inputs;
  node.GetInputNodes(inputs); 
  out_grads = {BroadCastToOperator(in_grad, inputs[0])};
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

void SoftmaxOp::Gradient(const Node& node,
                         const Node& in_grad,
                         std::vector<Node>& out_grads) {

}

void SoftmaxCrossEntropyOp::Gradient(const Node& node, 
                                     const Node& in_grad,
                                     std::vector<Node>& out_grads) {

}

std::shared_ptr<Op> Op::Create(const std::string& name) {
  if (name == "Add") {
    return std::make_shared<AddOp>(name);
  } else if (name == "Minus") {
    return std::make_shared<MinusOp>(name);
  } else if (name == "Multiply"){
    return std::make_shared<MultiplyOp>(name);
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

