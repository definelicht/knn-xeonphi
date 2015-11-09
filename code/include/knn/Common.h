#pragma once

#include <vector>

namespace knn {

template <typename T>
using DataItr = typename std::vector<T>::const_iterator;

template <typename T>
using DataContainer = std::vector<T>;

}
