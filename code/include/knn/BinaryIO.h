#pragma once

#include <string>
#include <vector>

namespace knn {

template <typename T>
std::vector<T> LoadBinaryFile(std::string const &path);

template <typename T>
void WriteBinaryFile(std::string const &path, std::vector<T> const &data);

} // End namespace knn
