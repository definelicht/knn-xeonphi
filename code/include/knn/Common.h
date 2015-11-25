#pragma once

#include <algorithm>
#include <array>
#include <stdint.h>
#include <vector>
#ifdef KNN_USE_VC
#include <Vc/Vc>
#endif

namespace knn {

template <typename T>
using DataItr = T const*;

template <typename T>
using DataContainer = std::vector<T>;

template <typename T> int log2(T x) {
  int highestBit = 0;
  while (x >>= 1) {
    ++highestBit;
  }
  return highestBit;
}

template <typename T, unsigned Dim, typename IteratorType>
std::pair<std::array<T, Dim>, std::array<T, Dim>>
MeanAndVariance(std::vector<T> const &dataMatrix, const int nSamples,
                IteratorType begin, const IteratorType end) {
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

template <typename T, int Dim>
T SquaredEuclidianDistance(T const *__restrict__ const a,
                           T const *__restrict__ const b) {
  T dist = 0;
#ifdef KNN_USE_VC
  static constexpr int iMaxVec = Dim / Vc::Vector<T>::Size;
  for (int i = 0; i < iMaxVec; i += Vc::Vector<T>::Size) {
    Vc::Vector<T> lhs(a + i);
    const Vc::Vector<T> rhs(b + i);
    lhs -= rhs; 
    lhs *= lhs;
    dist += lhs.sum();
  }
  // Explicitly loop over the tail here instead of using the scalar
  // implementation to avoid that the compiler unneccesarily autovectorizes the
  // tail
  for (int i = iMaxVec; i < Dim; ++i) {
    T diff = a[i] - b[i];
    dist += diff*diff;
  }
#else
  for (int i = 0; i < Dim; ++i) {
    T diff = a[i] - b[i];
    dist += diff*diff;
  }
#endif
  return dist;
}

} // End namespace knn
