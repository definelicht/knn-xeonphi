#include "KDTree.h"
#include <iostream>

int main() {
  std::vector<float> data = {1,  2,  1,  2,  -2, -1, -2, -1, -2, -1, -2, -1,
                             1,  2,  1,  2,  2,  2,  1,  1,  2,  2,  1,  1,
                             -1, -1, -2, -2, -1, -1, -2, -2, 5,  5};
  std::vector<int> labels = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0};
  KDTree<2, float, int> kdTree(data, labels);
  std::cout << kdTree;
  return 0;
}
