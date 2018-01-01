#include <iostream>

#include "../tensor.h"
#include "../tensor_shape.h"
#include "add_op.h"

int main() {
  AddOp<CPUContext, float> add_op;
  std::vector<float> vals(100);
  for (int i = 0; i < 100; i++) vals[i] = 100;

  Tensor a(TensorShape(100));
  a.SyncFromVector(vals, 100);
  Tensor b(TensorShape(100));
  b.SyncFromVector(vals, 100);
  std::vector<Tensor> out_tensors{Tensor(TensorShape(100))};
  add_op.Compute(Node("add"), {a, b}, out_tensors);
  std::cout << out_tensors[0].Debug() << std::endl;
}
