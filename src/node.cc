#include "node.h"
#include "operator.h"

Node Node::operator+(const Node& rhs) const {
  return AddOperator(*this, rhs);
} 

Node Node::operator-(const Node& rhs) const {
  return MinusOperator(*this, rhs);
}

Node Node::operator*(const Node& rhs) const {
  return MultiplyOperator(*this, rhs);
}

Node Node::operator/(const Node& rhs) const {
  return DevideOperator(*this, rhs);
}
