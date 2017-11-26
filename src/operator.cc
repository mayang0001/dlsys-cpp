#include "operator.h"
#include "node.h"

inline Operator::Operator(const std::string& name) {
  std::unordered_map<std::string, std::shared_ptr<Op>> name_to_op;
  if (name_to_op.find(name) == name_to_op.end()) {
    name_to_op[name] = Op::Create(name);
  }
  op_ = name_to_op[name];
}

template <typename T>
inline Operator Operator::SetParam(const std::string& name, const T& val) {
  // TODO
  return *this;
}

template <>
inline Operator Operator::SetParam<Node>(const std::string& name, 
                                         const Node& value) {
  // TODO
  return *this;
}

template <>
inline Operator Operator::SetParam<Tensor>(const std::string& name, 
                                           const Tensor& value) {
  // TODO
  return *this;
}

template <typename... Args>
inline Node Operator::CreateNode(Args... args) {
  Node node(op_->GetOpType());
  node.SetOp(op_);
  node.PushInput(args...);
  return node; 
}

Node AddOperator(const Node& lhs, const Node& rhs) {
  return Operator("Add").CreateNode(lhs, rhs);
}

Node MinusOperator(const Node& lhs, const Node& rhs) {
  return Operator("Minus").CreateNode(lhs, rhs);
}

Node MultiplyOperator(const Node& lhs, const Node& rhs) {
  return Operator("Multiply").CreateNode(lhs, rhs);
}
Node DevideOperator(const Node& lhs, const Node& rhs) {
  return Operator("Devide").CreateNode(lhs, rhs);
}

Node MatMulOperator(const Node& lhs, const Node& rhs, 
                    bool tanspose_a = false, bool transpose_b = false) {
  Node node = Operator("MatMul").CreateNode(lhs, rhs);
  node.SetParam("transpose_a", transpose_a);
  node.SetParam("transpose_b", transpose_b);
}

Node SoftmaxOperator(const Node& lhs, const Node& rhs) {
  return Operator("Softmax").CreateNode(lhs, rhs);
}
