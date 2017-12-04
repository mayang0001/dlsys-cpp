#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

class MnistReader {
 public:
  MnistReader(const std::string& path, int batch_size) 
      : idx_(0), file_path_(path), batch_size_(batch_size) {
    std::ifstream file(path);
    std::string line;
    while (getline(file, line)) {
      std::istringstream words(line);
      float feature;
      std::vector<float> features;
      while (words >> feature) {
        features.push_back(feature); 
      }
      data_.push_back(features);
    }
    std::cout << "read finished" << std::endl;
  };

  bool NextBatch(std::vector<float>& features) {
    if (idx_ + batch_size_ < data_.size()) {
      features.clear();
      for (int i = 0; i < batch_size_; i++) {
        for (int j = 0; j < data_[idx_].size(); j++) {
          features.push_back(data_[idx_][j]);
        }
        idx_++;
      }
      return true;
    } else {
      return false;
    }
  }

 private:
  std::vector<std::vector<float>> data_;
  int idx_;
  std::string file_path_;
  int batch_size_;
};
