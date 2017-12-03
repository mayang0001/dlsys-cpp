#include "tensor.h"

Tensor operator+(float val, const Tensor& rhs) {
  return rhs + val;
}

Tensor operator*(float val, const Tensor& rhs) {
  return rhs * val;
}
