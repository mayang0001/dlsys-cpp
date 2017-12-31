#ifndef OP_H_
#define OP_H_

#include <cassert>
#include <vector>
#include "../context.h"
#include "../node.h"
#include "../tensor.h"

template <typename Context>
class Op {
 public:
  virtual void Compute(const Node& node,
                       const std::vector<Tensor>& in_tensors,
                       std::vector<Tensor>& out_tensors) = 0;

  virtual void Infer(const Node& node,
                     const std::vector<TensorShape>& in_shapes,
                     std::vector<TensorShape>& out_shapes) = 0;

  virtual void Gradient(const Node& node,
                        const Node& in_node,
                        std::vector<Node>& out_grads) = 0;
};

#endif // OP_H_
