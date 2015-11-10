#pragma once

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
}
