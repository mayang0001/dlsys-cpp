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
inline Operator Operator::PushInput(T val) {
  inputs_.push_back(val); 
  return *this;
} 

// TODO if typename... can be Node...
template <typename T, typename... Args>
inline Operator Operator::PushInput(T val, Args... args) {
  inputs_.push_back(val);
  if (sizeof...(args) > 0) {
    PushInput(args...); 
  }
  return *this;
}

template <typename T>
inline Operator Operator::SetParam(const std::string& name, const T& val) {
  std::stringstream ss;
  ss << val;
  std::string val_str;
  ss >> val_str;
  attrs_[name] = val_str;
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

Node MultiplyByConstOperator(const Node& lhs, float const_val) {
  Node node = Operator("MultiplyByConst").CreateNode(lhs);
  node.SetAttr("const_val", const_val);
  return node;
}

Node DevideOperator(const Node& lhs, const Node& rhs) {
  return Operator("Devide").CreateNode(lhs, rhs);
}

Node MatMulOperator(const Node& lhs, const Node& rhs, 
                    bool trans_a, bool trans_b) {
  Node node = Operator("MatMul").CreateNode(lhs, rhs);
  node.SetAttr("trans_a", trans_a);
  node.SetAttr("trans_b", trans_b);
  return node;
}

Node SoftmaxOperator(const Node& lhs, const Node& rhs) {
  return Operator("Softmax").CreateNode(lhs, rhs);
}

Node ZerosOperator(const Node& node) {
  return Operator("Zeros").CreateNode(node);
}

Node OnesOperator(const Node& node) {
  return Operator("Ones").CreateNode(node);
}

Node ReduceSumAxisZeroOperator(const Node& node) {
  return Operator("ReduceSumAxisZero").CreateNode(node);
}

Node BroadCastToOperator(const Node& from, const Node& to) {
  return Operator("BroadCastTo").CreateNode(from, to);
}

Node SoftmaxOperator(const Node& node) {
  return Operator("Softmax").CreateNode(node);
}

Node SoftmaxCrossEntropyOperator(const Node& lhs, const Node& rhs) {
  return Operator("SoftmaxCrossEntropy").CreateNode(lhs, rhs);
}
