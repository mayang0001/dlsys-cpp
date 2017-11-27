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
  Operator SetParam(const std::string& name, const T& val);

  template <typename... Args>
  Node CreateNode(Args... args);

private:
  std::unordered_map<std::string, std::string> attrs_;
  std::shared_ptr<Op> op_;
};

Node AddOperator(const Node& lhs, const Node& rhs);

Node MinusOperator(const Node& lhs, const Node& rhs);

Node MultiplyOperator(const Node& lhs, const Node& rhs);

Node DevideOperator(const Node& lhs, const Node& rhs);

Node MatMulOperator(const Node& lhs, const Node& rhs);

Node SoftmaxOperator(const Node& lhs, const Node& rhs);

#endif
