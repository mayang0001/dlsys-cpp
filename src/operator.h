#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

class Op;
class Node;

class Operator {
public:
  explicit Operator(const std::string& name); 

  template <typename T>
  Operator PushInput(T val);

  template <typename T, typename... Args>
  Operator PushInput(T val, Args... args);

  template <typename T>
  Operator SetParam(const std::string& name, const T& val);

  template <typename... Args>
  Node CreateNode(Args... args);

private:
  std::vector<Node> inputs_;
  std::unordered_map<std::string, std::string> attrs_;
  std::shared_ptr<Op> op_;
};

Node AddOperator(const Node& lhs, const Node& rhs);

Node AddByConstOperator(const Node& lhs, float const_val);

Node MinusOperator(const Node& lhs, const Node& rhs);

Node MinusByConstOperator(const Node& lhs, float const_val);

Node MultiplyOperator(const Node& lhs, const Node& rhs);

Node MultiplyByConstOperator(const Node& lhs, float const_val);

Node DevideOperator(const Node& lhs, const Node& rhs);

Node DevideByConstOperator(const Node& lhs, float const_val);

Node MatMulOperator(const Node& lhs, const Node& rhs, 
                    bool trans_a = false, bool trans_b = false);

Node SoftmaxOperator(const Node& lhs, const Node& rhs);

Node ZerosOperator(const Node& node);

Node OnesOperator(const Node& node);

Node ReduceSumAxisZeroOperator(const Node& node);

Node BroadCastToOperator(const Node& from, const Node& to);

Node SoftmaxOperator(const Node& node);

Node SoftmaxCrossEntropyOperator(const Node& lhs, const Node& rhs);

Node ReluOperator(const Node& node);

#endif
