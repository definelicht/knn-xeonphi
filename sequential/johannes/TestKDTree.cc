#include "KDTree.h"
#include <iostream>

int main() {
  KDTree<float, int> kdtree(
      {{1, 2, 1, 2, -1, -2, -1, -2, -1, -2, -1, -2, 1, 2, 1, 2},
       {1, 1, 2, 2, 1, 1, 2, 2, -1, -1, -2, -2, -1, -1, -2, -2}},
      {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3});
  std::cout << kdtree;
  return 0;
}
