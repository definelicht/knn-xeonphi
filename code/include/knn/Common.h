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
std::pair<std::array<T, Dim>, std::array<T, Dim>>
MeanAndVariance(std::vector<T> const &dataMatrix, const int nSamples,
                std::vector<size_t>::const_iterator begin,
                const std::vector<size_t>::const_iterator end) {
  std::pair<std::array<T, Dim>, std::array<T, Dim>> output;
  std::fill(output.first.begin(), output.first.end(), 0);
  std::fill(output.second.begin(), output.second.end(), 0);
  const int iMax = nSamples > 0
                       ? std::min(nSamples,
                                  static_cast<int>(std::distance(begin, end)))
                       : std::distance(begin, end);
  for (int i = 0; i < iMax; ++i, ++begin) {
    size_t index = Dim * (*begin);
    for (unsigned j = 0; j < Dim; ++j, ++index) {
      const T val = dataMatrix[index];
      output.first[j] += val;
      output.second[j] += val * val;
    }
  }
  const float iMaxInv = 1. / iMax;
  for (unsigned j = 0; j < Dim; ++j) {
    // Don't divide by (N - 1) as this will only be used for intercomparison.
    output.second[j] =
        output.second[j] - (output.first[j] * output.first[j]) * iMaxInv;
    output.first[j] *= iMaxInv;
  }
  return output;
}

} // End namespace knn
