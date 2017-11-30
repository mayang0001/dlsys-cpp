#include <memory>
#include "node.h"
#include "op.h"

void AddOp::Gradient(const Node& in_grad, std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}

void MinusOp::Gradient(const Node& in_grad, std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}

// TODO
void MultiplyOp::Gradient(const Node& in_grad, std::vector<Node>& out_grads) {
  //out_grads = {in_grad * node.input(1), in_grad * node.input(0)};
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

