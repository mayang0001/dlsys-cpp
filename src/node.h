#ifndef NODE_H_
#define NODE_H_

#include <functional>
#include <unordered_map>
#include <iostream>
#include <string>
#include "op.h"

class Node {
public:
  explicit Node(const std::string& name) : name_(name) {}

  Node(const Node& rhs) 
      : name_(rhs.name_), inputs_(rhs.inputs_), op_(rhs.op_) {
  }

  Node& operator=(const Node& rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      inputs_ = rhs.inputs_;
      op_ = rhs.op_;
    }
    return *this;
  }

  ~Node() {
  
  }

  Node operator+(const Node& rhs) const;
  Node operator-(const Node& rhs) const;
  Node operator*(const Node& rhs) const;
  Node operator/(const Node& rhs) const;

  void GetInputNodes(std::vector<Node>& input_nodes) const {
    input_nodes = inputs_;
  }

  template <typename T>
  void PushInput(const T& t) {
    inputs_.push_back(t);
  }

  template <typename T, typename... Args>
  void PushInput(const T& t, const Args&... args) {
    inputs_.push_back(t); 
    if (sizeof...(args) != 0) {
      PushInput(args...);
    }
  }

  template<typename T>
  void SetParam(const std::string& key, const T& val) {
  
  }

  void SetOp(std::shared_ptr<Op> op) {
    op_ = op;
  }

  const std::string& name() const { return name_; }

  std::shared_ptr<Op> GetOp() const { return op_; }

  bool IsVariable() const { return inputs_.size() == 0; }

private:
  std::string name_;
  std::vector<Node> inputs_;
  std::unordered_map<std::string, std::string> attrs_;
  std::shared_ptr<Op> op_;
};

namespace std {

template <>
struct hash<Node> {
  size_t operator()(const Node& node) const {
    return std::hash<std::string>()(node.name());
  }
};

template <>
struct equal_to<Node> {
  bool operator()(const Node& lhs, const Node& rhs) const {
    return lhs.name() == rhs.name();
  }
};

}
#endif
