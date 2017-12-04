#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

class MnistReader {
 public:
  MnistReader(const std::string& path, int batch_size) 
      : file_path_(path), batch_size_(batch_size) {
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
  };

  bool NextBatch(std::vector<std::vector<float>>& data) {
  
  }

 private:
  std::vector<std::vector<float>> data_;
  std::string file_path_;
  int batch_size_;
};
