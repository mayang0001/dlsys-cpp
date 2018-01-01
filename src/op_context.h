#ifndef OP_CONTEXT_
#define OP_CONTEXT_

#include <vector>

class OpContext {
 public:
  const Tensor& input(int idx) {
    assert(idx < inputs_.size());
    return inputs_[i];
  }

  void Allocate(const TensorShape& shape, Tensor* tensor);

  Node& Node() { return *node; }

  template <typename T>
  void GetAttr(const std::string& name, const T& val) const;

 private:
  const unique_ptr<const Node> node_;
  std::vector<Tensor> inputs_;
};

#endif  // OP_CONTEXT_
