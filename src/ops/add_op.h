#ifndef ADD_OP_H_
#define ADD_OP_H_

#include "op.h"

template <typename Context, typename T>
class AddOp final : public Op<Context> {
  AddOp();

  void Compute(const Node& node, 
               const std::vector<Tensor>& in_tensors,
               std::vector<Tensor>& out_tensors) override;

  void Infer(const Node& node,
             const std::vector<TensorShape>& in_shapes,
             std::vector<TensorShape>& out_shapes) override;

  void Gradient(const Node& node,
                const Node& in_grad,
                std::vector<Node>& out_grads) override;
};

#endif //ADD_OP_H_

