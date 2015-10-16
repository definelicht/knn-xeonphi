#include "KDTree.h"
#include "Knn.h"
#include <iostream>

int main() {
  KDTree<float, int> kdTree(
      {{1, 2, 1, 2, -1, -2, -1, -2, -1, -2, -1, -2, 1, 2, 1, 2},
       {1, 1, 2, 2, 1, 1, 2, 2, -1, -1, -2, -2, -1, -1, -2, -2}},
      {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3});
  std::cout << kdTree;
  return 0;
}
