#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "data_reader.h"
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

  int batch_size = 1000;

  Tensor x_val(TensorShape(batch_size, 784), ctx);
  Tensor y_val(TensorShape(batch_size, 10), ctx);

  Tensor w_val(TensorShape(784, 10), ctx);
  std::vector<float> ws(784 * 10, 0.0);
  w_val.SyncFromVector(ws, 784 * 10);
  
  Tensor b_val(TensorShape(10), ctx);
  std::vector<float> bs(10, 0.0);
  b_val.SyncFromVector(bs, 10);

  std::vector<Tensor> out_vals;
  std::vector<Tensor> grad_vals;
  int iter_num = 200;
  float lr = 0.0001;
  MnistReader train_x("../mnist/test_x.txt", batch_size);
  MnistReader train_y("../mnist/test_y.txt", batch_size);

  std::unordered_map<Node, Tensor> feed_dicts;
  for (int i = 0; i < iter_num; i++) {
    std::cout << "iter num " << i << std::endl;
    std::vector<float> xs;
    std::vector<float> ys;
    train_x.NextBatch(xs);
    train_y.NextBatch(ys);
    x_val.SyncFromVector(xs, batch_size * 784);
    y_val.SyncFromVector(ys, batch_size * 10);

    feed_dicts[x] = x_val;
    feed_dicts[y_] = y_val;
    feed_dicts[weights] = w_val;
    feed_dicts[bias] = b_val;

    exec.Run({loss}, out_vals, {weights, bias}, grad_vals, feed_dicts);
    w_val -= lr * grad_vals[0];
    b_val -= lr * grad_vals[1];

    out_vals[0].Debug();
  }
}
