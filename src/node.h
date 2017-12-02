#ifndef NODE_H_
#define NODE_H_

#include <functional>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <string>
#include "op.h"

class Node {
public:
  Node() = default;

  explicit Node(const std::string& name) : name_(name) {}

  Node(const Node& rhs) 
      : name_(rhs.name_), 
        inputs_(rhs.inputs_), 
        attrs_(rhs.attrs_), 
        op_(rhs.op_) {
  }

  Node& operator=(const Node& rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      inputs_ = rhs.inputs_;
      attrs_ = rhs.attrs_;
      op_ = rhs.op_;
    }
    return *this;
  }

  ~Node() {}

  Node operator+(const Node& rhs) const;
  Node operator-(const Node& rhs) const;
  Node operator*(const Node& rhs) const;
  Node operator/(const Node& rhs) const;
  
  Node operator+(float const_val) const;
  Node operator-(float const_val) const;
  Node operator*(float const_val) const;
  Node operator/(float const_val) const;

  Node& operator+=(const Node& rhs);
  Node& operator-=(const Node& rhs);
  Node& operator*=(const Node& rhs);
  Node& operator/=(const Node& rhs);

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

  template <typename T>
  void SetAttr(const std::string& key, const T& val) {
    std::string val_str;
    std::stringstream ss;
    ss << val;
    ss >> val_str;   
    attrs_[key] = val_str;
  }

  template <typename T>
  bool GetAttr(const std::string& key, T& val) const {
    auto iter = attrs_.find(key);
    if (iter != attrs_.end()) {
      std::stringstream ss;
      ss << iter->second;
      ss >> val; 
      return true;
    } else {
      return false;
    }
  }

  void SetOp(std::shared_ptr<Op> op) {
    op_ = op;
  }

  void SetName() {
    name_ = op_->GetOpType() + "(";
    for (auto input : inputs_) {
      name_ += (input.name() + ",");
    }
    name_.resize(name_.size() - 1);
    name_ += ")";
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
