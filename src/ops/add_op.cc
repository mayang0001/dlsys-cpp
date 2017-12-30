#include "add_op.h"

CpuUnaryFunctor(Add, +);

template <>
void Compute<CpuContext, float>::Compute(const Node& node,
                                         const std::vector<Tensor>& in_tensors,
                                         std::vector<Tensor>& out_tensors) {
  static_assert(in_tensors.size() == 2, "add op must have two inputs");
  const int N = out_tensors[0].NumElements();
  CpuAddFunctor<float>()(in_tensors[0], in_tensors[1], N, out_tensors[0]);
}


template <typename Context, typename T>
void AddOp<Context, T>::Infer(const Node& node,
                              const std::vector<TensorShape>& in_shapes,
                              std::vector<TensorShape>& out_shapes) {
  static_assert(in_shapes.size() == 2);
  out_shapes = {in_shapes[0]};
}

template <typename Context, typename T>
void AddOp<Context, T>::Gradient(const Node& node,
                                 const Node& in_grad,
                                 std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}
