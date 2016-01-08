#include "knn/Common.h"
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
      MeanAndVariance<typename std::vector<float>::const_iterator, 3>(
          arr.cbegin(), indices.cbegin(), indices.cend(), 0);
  KNN_ASSERT(meanAndVariance.first[0] == 1);
  KNN_ASSERT(meanAndVariance.first[1] == 10);
  KNN_ASSERT(meanAndVariance.first[2] == 100);
  // Result is not divided by number of elements
  KNN_ASSERT(meanAndVariance.second[0] == 0.125*4); 
  KNN_ASSERT(meanAndVariance.second[1] == 12.5*4);
  KNN_ASSERT(meanAndVariance.second[2] == 1250*4);
  return 0;
}
