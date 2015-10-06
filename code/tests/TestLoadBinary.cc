#include "knn/LoadBinary.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <cstdio>

int main() {
  const std::string path("TestLoadBinary.data");
  const std::vector<double> data{1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
  {
    std::ofstream outFile(path, std::ios::binary);
    outFile.write(reinterpret_cast<char const *>(data.data()),
                  sizeof(double) * data.size());
  }
  auto test = LoadBinary<double>(path);
  assert(test.size() == data.size());
  for (int i = 0, iEnd = data.size(); i < iEnd; ++i) {
    assert(test[i] == data[i]);
  }
  std::cout << "TestLoadBinary ran successfully." << std::endl;
  return 0;
}
