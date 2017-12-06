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
    std::vector<float> features;
    while (getline(file, line)) {
      std::istringstream words(line);
      float feature;
      features.clear();
      while (words >> feature) {
        features.push_back(feature); 
      }
      data_.push_back(features);
    }
    std::cout << "read finished" << std::endl;
  };

  int NextBatch(std::vector<float>& features) {
    if (idx_ + batch_size_ >= data_.size()) {
      idx_ = 0;
    }

    int num_samples = std::min(batch_size_, (int)data_.size() - idx_);
    features.clear();
    for (int i = 0; i < num_samples; i++) {
      for (int j = 0; j < data_[idx_].size(); j++) {
        features.push_back(data_[idx_][j]);
      }
      idx_++;
    }
    return num_samples;
  }

 private:
  std::vector<std::vector<float>> data_;
  int idx_;
  std::string file_path_;
  int batch_size_;
};
