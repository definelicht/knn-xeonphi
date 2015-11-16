#include "knn/Common.h"
#include <cassert>
#include <iostream>
#include <numeric>

using namespace knn;

int main() {
  std::vector<float> arr = {0.5, 5, 50,
                            1, 10, 100,
                            1, 10, 100,
                            1.5, 15, 150};
  std::vector<size_t> indices(4);
  std::iota(indices.begin(), indices.end(), 0);
  auto meanAndVariance =
      MeanAndVariance<float, 3>(arr, 0, indices.cbegin(), indices.cend());
  assert(meanAndVariance.first[0] == 1);
  assert(meanAndVariance.first[1] == 10);
  assert(meanAndVariance.first[2] == 100);
  // Result is not divided by number of elements
  assert(meanAndVariance.second[0] == 0.125*4); 
  assert(meanAndVariance.second[1] == 12.5*4);
  assert(meanAndVariance.second[2] == 1250*4);
  return 0;
}
