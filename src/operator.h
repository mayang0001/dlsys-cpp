#ifndef OPERATOR_H_
#define OPERATOR_H_

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

Node MinusOperator(const Node& lhs, const Node& rhs);

Node MultiplyOperator(const Node& lhs, const Node& rhs);

Node DevideOperator(const Node& lhs, const Node& rhs);

Node MatMulOperator(const Node& lhs, const Node& rhs, 
                    bool trans_a = false, bool trans_b = false);

Node SoftmaxOperator(const Node& lhs, const Node& rhs);

Node ZerosOperator(const Node& node);

Node OnesOperator(const Node& node);

Node ReduceSumAxisZeroOperator(const Node& node);

Node BroadCastToOperator(const Node& node);

Node SoftmaxOperator(const Node& node);

Node SoftmaxCrossEntropyOperator(const Node& node);

#endif
