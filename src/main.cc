#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "executor.h"
#include "operator.h"


int main() {
  Node weights("weights");
  Node bias("bias");

  Node x("x");
  Node y_("y_");
  Node z = MatMulOperator(x, weights);
  Node logit = z + BroadCastToOperator(bias, z);
  Node loss = SoftmaxCrossEntropyOperator(logit, y_);
  Context ctx = Context::cpu();
  Executor exec(ctx, loss, {weights, bias});

  int size = 10;

  Tensor x_val(TensorShape(size, size), ctx);
  Tensor y_val(TensorShape(size, size), ctx);
  Tensor w_val(TensorShape(size, size), ctx);
  Tensor b_val(TensorShape(size, size), ctx);
  std::unordered_map<Node, Tensor> feed_dicts;
  feed_dicts[x] = x_val;
  feed_dicts[y_] = y_val;
  feed_dicts[weights] = w_val;
  feed_dicts[bias] = b_val;

  std::vector<Tensor> out_vals;
  std::vector<Tensor> grad_vals;
  int iter_num = 1;
  for (int i = 0; i < iter_num; i++) {
    std::cout << i << std::endl;
    exec.Run({loss}, out_vals, {weights, bias}, grad_vals, feed_dicts);
  }
}
