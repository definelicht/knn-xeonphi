#pragma once

#include <algorithm>
#include <array>
#include <stdint.h>
#include <vector>

namespace knn {

template <typename T>
using DataItr = typename std::vector<T>::const_iterator;

template <typename T>
using DataContainer = std::vector<T>;

template <typename T> int log2(T x) {
  int highestBit = 0;
  while (x >>= 1) {
    ++highestBit;
  }
  return highestBit;
}

template <typename T, unsigned Dim>
std::array<T, Dim> Variance(std::vector<T> const &dataMatrix,
                            const int nSamples,
                            std::vector<size_t>::const_iterator begin,
                            std::vector<size_t>::const_iterator const &end) {
  std::array<T, Dim> sum;
  std::array<T, Dim> sumOfSquares;
  std::fill(sum.begin(), sum.end(), 0);
  std::fill(sumOfSquares.begin(), sumOfSquares.end(), 0);
  const int iMax = nSamples > 0
                       ? std::min(nSamples,
                                  static_cast<int>(std::distance(begin, end)))
                       : std::distance(begin, end);
  for (int i = 0; i < iMax; ++i, ++begin) {
    size_t index = Dim * (*begin);
    for (unsigned j = 0; j < Dim; ++j, ++index) {
      const T val = dataMatrix[index];
      sum[j] += val;
      sumOfSquares[j] += val * val;
    }
  }
  const float iMaxInv = 1. / iMax;
  for (unsigned j = 0; j < Dim; ++j) {
    // Reuse sumOfSquares vector for computing the variance. Don't divide by
    // (N - 1) as this will only be used for intercomparison.
    sumOfSquares[j] =
        sumOfSquares[j] - (sum[j] * sum[j]) * iMaxInv; // / (iMax - 1);
  }
  return sumOfSquares;
}

} // End namespace knn
