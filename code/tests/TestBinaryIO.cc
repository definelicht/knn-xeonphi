#include "knn/BinaryIO.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
  const std::string path("TestLoadBinary.data");
  std::vector<double> data;
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist;
  for (int i = 0; i < 1000; ++i) {
    data.emplace_back(dist(rng));
  }
  knn::WriteBinaryFile(path, data.cbegin(), data.cend());
  auto test = knn::LoadBinaryFile<double>(path);
  assert(test.size() == data.size());
  for (int i = 0, iEnd = data.size(); i < iEnd; ++i) {
    assert(test[i] == data[i]);
  }
  return 0;
}
