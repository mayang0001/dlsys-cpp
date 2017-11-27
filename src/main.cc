#include <iostream>
#include <unordered_map>
#include "executor.h"
#include "operator.h"

int main() {
  float* src = new float[10 * 10];
  for (int i = 0; i < 100; i++) {
    src[i] = 1;
  }

  Context ctx = Context::cpu();
  Tensor tensor_a(TensorShape(10, 10), ctx);
  tensor_a.SyncFromCPU(src, tensor_a.GetTensorShape().num_elements());
  Tensor tensor_b(TensorShape(10, 10), ctx);
  tensor_b.SyncFromCPU(src, tensor_b.GetTensorShape().num_elements());
  Tensor tensor_c(TensorShape(10, 10), ctx);
  tensor_c.SyncFromCPU(src, tensor_c.GetTensorShape().num_elements());

  Node node_a("a");
  Node node_b("b");
  Node node_c("c");
  Node node_d = AddOperator(node_a, node_b);
  Node node_e = MatMulOperator(node_c, node_d);
  Executor exec({node_e}, ctx);
  std::unordered_map<Node, Tensor> feed_dicts;
  feed_dicts[node_a] = tensor_a;
  feed_dicts[node_b] = tensor_b;
  feed_dicts[node_c] = tensor_c;
  exec.Run(feed_dicts);
  feed_dicts[node_e].Debug();
  delete[] src;
}
