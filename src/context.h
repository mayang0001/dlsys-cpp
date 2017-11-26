#ifndef CONTEXT_H_
#define CONTEXT_H_

typedef enum {
  kCPU = 1,
  kGPU = 2
} DeviceType; 

class Context {
public:
  Context(DeviceType device_type, int device_id)
      : device_type_(device_type), device_id_(device_id) {}

  DeviceType GetDeviceType() { return device_type_; }
  int GetDeviceId() { return device_id_; }

  static Context cpu() {
    Context ctx(DeviceType::kCPU, 0);
    return ctx; 
  }
  static Context gpu() { 
    Context ctx(DeviceType::kGPU, 0);
    return ctx; 
  }
  
private:
  DeviceType device_type_;
  int device_id_;
};

#endif
