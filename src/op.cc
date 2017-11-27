#include <memory>
#include "op.h"

std::shared_ptr<Op> Op::Create(const std::string& name) {
  if (name == "Add") {
    return std::make_shared<AddOp>(name);
  } else if (name == "Minus") {
    return std::make_shared<MinusOp>(name);
  } else if (name == "MatMul") {
    return std::make_shared<MatMulOp>(name);
  } else if (name == "Softmax") {
    return std::make_shared<SoftmaxOp>(name);
  } else if (name == "Multiply"){
    return std::make_shared<MultiplyOp>(name);
  } else if (name == "Devide"){
    return std::make_shared<DevideOp>(name);
  } else if (name == "Zeros"){
    return std::make_shared<ZerosOp>(name);
  } else if (name == "Ones"){
    return std::make_shared<OnesOp>(name);
  } else {
    return std::make_shared<DevideOp>(nullptr);
  }
}
