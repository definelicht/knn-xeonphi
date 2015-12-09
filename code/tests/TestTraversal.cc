#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "knn/KDTree.h"
#include "knn/Random.h"

using namespace knn;

constexpr size_t n = 1<<14;
constexpr size_t nDims = 128;
using TreeType = KDTree<nDims, false, float>;

void Traverse(typename TreeType::NodeItr itr, std::vector<size_t> &indices) {
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
  std::vector<float> train(n*nDims);
  random::FillUniform(train.begin(), train.end());
  TreeType kdTree(train);
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
