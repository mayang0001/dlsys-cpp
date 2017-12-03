#include "node.h"
#include "operator.h"

Node Node::operator+(const Node& rhs) const {
  return AddOperator(*this, rhs);
} 

Node Node::operator+(float const_val) const {
  return AddByConstOperator(*this, const_val);
}

Node Node::operator-(const Node& rhs) const {
  return MinusOperator(*this, rhs);
}

Node Node::operator-(float const_val) const {
  return MinusByConstOperator(*this, const_val);
}

Node Node::operator*(const Node& rhs) const {
  return MultiplyOperator(*this, rhs);
}

Node Node::operator*(float const_val) const {
  return MultiplyByConstOperator(*this, const_val);
}

Node Node::operator/(const Node& rhs) const {
  return DevideOperator(*this, rhs);
}

Node Node::operator/(float const_val) const {
  return DevideByConstOperator(*this, const_val);
}

Node& Node::operator+=(const Node& rhs) {
  *this = *this + rhs;
  return *this; 
}

Node& Node::operator-=(const Node& rhs) {
  *this = *this - rhs;
  return *this; 
}

Node& Node::operator*=(const Node& rhs) {
  *this = *this * rhs;
  return *this; 
}

Node& Node::operator/=(const Node& rhs) {
  *this = *this / rhs;
  return *this; 
}

Node operator+(float val, const Node& node) {
  return node + val;
}

Node operator*(float val, const Node& node) {
  return node * val;
}
