#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <type_traits>
#include <stdint.h>
#include <vector>
#ifdef KNN_USE_VC
#include <Vc/Vc>
#endif

namespace knn {

#define KNN_FORCE_INLINE __attribute__((always_inline)) inline

#ifdef NDEBUG
#define KNN_ASSERT(condition) if (!(condition)) exit(1);
#else
#define KNN_ASSERT(condition) assert(condition);
#endif

template <typename IteratorType>
using CheckRandomAccess = typename std::enable_if<std::is_base_of<
    std::random_access_iterator_tag,
    typename std::iterator_traits<IteratorType>::iterator_category>::value>::
    type;

template <typename IteratorType>
constexpr bool HasRandomAccess() {
  return std::is_base_of<
      std::random_access_iterator_tag,
      typename std::iterator_traits<IteratorType>::iterator_category>::value;
}

template <typename T> int log2(T x) {
  int highestBit = 0;
  while (x >>= 1) {
    ++highestBit;
  }
  return highestBit;
}

template <typename DataIterator, unsigned Dim, typename IndexIterator>
std::pair<
    std::array<typename std::iterator_traits<DataIterator>::value_type, Dim>,
    std::array<typename std::iterator_traits<DataIterator>::value_type, Dim>>
MeanAndVariance(const DataIterator data, IndexIterator begin,
                const IndexIterator end, const int nSamples) {
  using T = typename std::iterator_traits<DataIterator>::value_type;
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
      const T val = data[index];
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

template <typename DataIterator, int Dim>
KNN_FORCE_INLINE
typename std::iterator_traits<DataIterator>::value_type
SquaredEuclidianDistance(const DataIterator a, const DataIterator b) {
  static_assert(HasRandomAccess<DataIterator>(),
                "Data iterator must support random access.");
  using T = typename std::iterator_traits<DataIterator>::value_type;
  T dist = 0;
#ifdef KNN_USE_VC
  T const *aPtr = &(*a);
  T const *bPtr = &(*b);
  static constexpr int iMaxVec = Dim / Vc::Vector<T>::Size;
  for (int i = 0; i < iMaxVec; i += Vc::Vector<T>::Size) {
    Vc::Vector<T> lhs(aPtr + i);
    const Vc::Vector<T> rhs(bPtr + i);
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
