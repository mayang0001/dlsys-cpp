#ifndef CONTEXT_H_
#define CONTEXT_H_

class CPUContext {
 public:
  explicit CPUContext(int device_id = 0) : device_id_(device_id) {};

 private:
  int device_id_;
};

class GPUContext {
 public:
  explicit GPUContext(int device_id = 0) : device_id_(device_id) {};

 private:
  int device_id_;
};

#endif
