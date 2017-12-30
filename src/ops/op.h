
template <typename Context>
class Op {
 public:
  virtual void Compute() = 0;

  virtual void Infer() = 0;

  virtual void Gradient() = 0;
};
