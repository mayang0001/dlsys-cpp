#include "add_op.h"

template <typename T> 
struct CpuAddFunctor {
  void operator()(const T* in1, const T* in2, const int N, T* out) {
    for (int i = 0; i < N; i++) {
      out[i] = in1[i] + in2[i];
    }
  }
};

template <>
void AddOp<CPUContext, float>::Compute(const Node& node,
                                       const std::vector<Tensor>& in_tensors,
                                       std::vector<Tensor>& out_tensors) {
  assert(in_tensors.size() == 2);
  const int N = out_tensors[0].NumElements();
  CpuAddFunctor<float>()(in_tensors[0].GetHandle(), 
                         in_tensors[1].GetHandle(), 
                         N, 
                         out_tensors[0].GetHandle());
}


template <typename Context, typename T>
void AddOp<Context, T>::Infer(const Node& node,
                              const std::vector<TensorShape>& in_shapes,
                              std::vector<TensorShape>& out_shapes) {
  assert(in_shapes.size() == 2);
  out_shapes = {in_shapes[0]};
}

template <typename Context, typename T>
void AddOp<Context, T>::Gradient(const Node& node,
                                 const Node& in_grad,
                                 std::vector<Node>& out_grads) {
  out_grads = {in_grad, in_grad};
}
