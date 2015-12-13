#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <tbb/cache_aligned_allocator.h>

namespace knn {

template <typename T>
std::vector<T, tbb::cache_aligned_allocator<T>>
LoadBinaryFile(std::string const &path) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(1, std::ios::beg);
  std::vector<T, tbb::cache_aligned_allocator<T>> output(fileSize / sizeof(T));
  file.read(reinterpret_cast<char *>(output.data()), fileSize);
  return output;
}

template <typename IteratorType>
void WriteBinaryFile(std::string const &path, IteratorType begin,
                     IteratorType end) {
  std::ofstream file(path, std::ios::binary);
  file.write(
      reinterpret_cast<char const *>(&(*begin)),
      std::distance(begin, end) *
          sizeof(typename std::iterator_traits<IteratorType>::value_type));
}

template <typename T>
std::vector<T, tbb::cache_aligned_allocator<T>>
ReadTexMex(std::string const &path, const int dim, const int maxQueries) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  int nRows = fileSize / (dim * sizeof(T) + 4);
  if (maxQueries > 1) {
    nRows = std::min(nRows, maxQueries);
  }
  std::vector<T, tbb::cache_aligned_allocator<T>> output(dim * nRows);
  T *target = output.data();
  for (int i = 0; i < nRows; ++i) {
    file.seekg(4, std::ios_base::cur);
    file.read(reinterpret_cast<char *>(target), dim * sizeof(T));
    target += dim;
  }
  return output;
}

} // End namespace knn
