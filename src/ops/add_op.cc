#include "add_op.h"

template <typename T>
struct AddFunctor<CpuContext, T> {
  void operator()(const T* in1, const T* in2, const int N, T* out) {
    for (int i = 0; i < N; i++) {
      out[i] = in1[i] + in2[i];
    } 
  }
};

template <typename Context, typename T>
class AddOp {
  void Compute(const Node& node, 
               const std::vector<Tensor>& in_tensors,
               std::vector<Tensor>& out_tensors) {
    assert(in_tensors.size() == 2);
    
    AddFunctor<Context, T>()(
        in_tensors[0].data<T>(),
        in_tensors[1].data<T>(),
        out_tensors[0].NumElements(),
        out_tensors[0].data<T>());
  }

}
