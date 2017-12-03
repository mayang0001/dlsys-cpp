
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "executor.h"
#include "operator.h"


int main() {
  float* src = new float[4 * 2];
  for (int i = 0; i < 8; i++) {
    src[i] = 1;
  }

  Context ctx = Context::cpu();
  Tensor tensor_a(TensorShape(4, 2), ctx);
  tensor_a.SyncFromCPU(src, tensor_a.NumElements());
  Tensor tensor_b(TensorShape(4, 2), ctx);
  tensor_b.SyncFromCPU(src, tensor_b.NumElements());
  Tensor tensor_c(TensorShape(4, 2), ctx);
  tensor_c.SyncFromCPU(src, tensor_c.NumElements());

  Node node_a("a");
  Node node_b("b");

  // Test AddOperator
  std::cout << "test add operator" << std::endl;
  Node node_c = AddOperator(node_a, node_b);
  Executor exec_add({node_c}, ctx);
  std::unordered_map<Node, Tensor> feed_dicts;
  feed_dicts[node_a] = tensor_a;
  feed_dicts[node_b] = tensor_b;
  std::unordered_map<Node, Tensor> dicts = feed_dicts;
  exec_add.Run(dicts);
  dicts[node_c].Debug();

  // Test AddByConstOperator
  std::cout << "test add by const operator" << std::endl;
  node_c = AddByConstOperator(node_a, 2.5);
  Executor exec_add_by_const({node_c}, ctx);
  dicts = feed_dicts;
  exec_add_by_const.Run(dicts);
  dicts[node_c].Debug();

  // Test MinusOperator 
  std::cout << "test minus operator" << std::endl;
  node_c = MinusOperator(node_a, node_b);
  Executor exec_minus({node_c}, ctx);
  dicts = feed_dicts;
  exec_minus.Run(dicts);
  dicts[node_c].Debug();

  // Test MinusByConstOperator
  std::cout << "test minus by const operator" << std::endl;
  node_c = MinusByConstOperator(node_a, 2.5);
  Executor exec_minus_by_const({node_c}, ctx);
  dicts = feed_dicts;
  exec_minus_by_const.Run(dicts);
  dicts[node_c].Debug();

  // Test MultiplyOperator
  std::cout << "test multiply operator" << std::endl;
  node_c = MultiplyOperator(node_a, node_b);
  Executor exec_multiply({node_c}, ctx);
  dicts = feed_dicts;
  exec_multiply.Run(dicts);
  dicts[node_c].Debug();

  // Test MultiplyByConstOperator
  std::cout << "test multiply by const operator" << std::endl;
  node_c = MultiplyByConstOperator(node_a, 2.5);
  Executor exec_multiply_by_const({node_c}, ctx);
  dicts = feed_dicts;
  exec_multiply_by_const.Run(dicts);
  dicts[node_c].Debug();

  // Test DevideOperator
  std::cout << "test devide operator" << std::endl;
  node_c = DevideOperator(node_a, node_b);
  Executor exec_devide({node_c}, ctx);
  dicts = feed_dicts;
  exec_devide.Run(dicts);
  dicts[node_c].Debug();

  // Test DevideByConstOperator
  std::cout << "test multiply by const operator" << std::endl;
  node_c = DevideByConstOperator(node_a, 2.5);
  Executor exec_devide_by_const({node_c}, ctx);
  dicts = feed_dicts;
  exec_devide_by_const.Run(dicts);
  dicts[node_c].Debug();

  // Test MatMulOperator 
  std::cout << "test matmul operator" << std::endl;
  node_c = MatMulOperator(node_a, node_b, true, false);
  Executor exec_matmul({node_c}, ctx);
  dicts = feed_dicts;
  exec_matmul.Run(dicts);
  dicts[node_c].Debug();

  // Test ZerosOperator 
  std::cout << "test zeros operator" << std::endl;
  node_c = ZerosOperator(node_a);
  Executor exec_zeros({node_c}, ctx);
  dicts = feed_dicts;
  exec_zeros.Run(dicts);
  dicts[node_c].Debug();

  // Test OnesOperator 
  std::cout << "test ones operator" << std::endl;
  node_c = OnesOperator(node_a);
  Executor exec_ones({node_c}, ctx);
  dicts = feed_dicts;
  exec_ones.Run(dicts);
  dicts[node_c].Debug();

  // Test SoftmaxOperator 
  std::cout << "test softmax operator" << std::endl;
  node_c = SoftmaxOperator(node_a);
  Executor exec_softmax({node_c}, ctx);
  dicts = feed_dicts;
  exec_softmax.Run(dicts);
  dicts[node_c].Debug();

  // Test SoftmaxCrossEntropyOperator 
  //float* y_src = new float[6];
  //float* y_src_ = new float[6];
  float y_src[6] = {1, 0, 0.5, 0.5, 0.9, 0.1};
  float y_src_[6] = {1, 0, 0, 1, 1, 0};
  Tensor y(TensorShape(3, 2), ctx);
  y.SyncFromCPU(y_src, y.NumElements());
  Tensor y_(TensorShape(3, 2), ctx);
  y_.SyncFromCPU(y_src_, y_.NumElements());
  std::cout << "test softmax cross entropy operator" << std::endl;
  node_c = SoftmaxCrossEntropyOperator(node_a, node_b);
  Executor exec_softmax_cross_entropy({node_c}, ctx);
  dicts[node_a] = y;
  dicts[node_b] = y_;
  exec_softmax_cross_entropy.Run(dicts);
  dicts[node_c].Debug();

  // Test ReduceSumAxisZeroOperator 
  std::cout << "test reduce sum axis zero operator" << std::endl;
  node_c = ReduceSumAxisZeroOperator(node_a);
  Executor exec_reduce_sum_axis_zero({node_c}, ctx);
  dicts = feed_dicts;
  exec_reduce_sum_axis_zero.Run(dicts);
  dicts[node_c].Debug();

  // Test BroadCastToOperator 
  std::cout << "test broad cast to operator" << std::endl;
  Node node_d = BroadCastToOperator(node_c, node_a);
  Executor exec_broad_cast_to({node_d}, ctx);
  dicts = feed_dicts;
  exec_broad_cast_to.Run(dicts);
  dicts[node_d].Debug();

  delete[] src;
}
