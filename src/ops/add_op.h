#ifndef ADD_OP_H_
#define ADD_OP_H_

template <>
struct AddFunctor<CpuContext, T> {
  void operator()(const T* in1, const T* in2, const int N, T* out);
};

#endif //ADD_OP_H_

