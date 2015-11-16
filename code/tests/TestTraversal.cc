#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "knn/KDTree.h"
#include "knn/Random.h"

using namespace knn;

void Traverse(typename KDTree<8, true, float>::NodeItr itr,
              std::vector<size_t> &indices) {
  if (!itr.inBounds()) {
    return;
  }
  auto left = itr.Left();
  auto right = itr.Right();
  assert(left.inBounds() == right.inBounds());
  if (!left.inBounds() && !right.inBounds()) {
    indices.push_back(itr.index());
    return;
  }
  if (left.inBounds()) {
    Traverse(left, indices);
  }
  if (right.inBounds()) {
    Traverse(right, indices);
  }
}

int main() {
  constexpr size_t n = 1024;
  constexpr size_t nDims = 8;
  std::vector<float> train(n*nDims);
  Uniform(train.begin(), train.end());
  KDTree<nDims, true, float> kdTree(train);
  std::vector<size_t> indices;
  indices.reserve(n);
  Traverse(kdTree.Root(), indices);
  assert(indices.size() == n);
  std::sort(indices.begin(), indices.end());
  for (size_t i = 0; i < n; ++i) {
    assert(indices[i] == i);
  }
  return 0;
}
