#include "knn/LoadBinary.h"
#include <fstream>
#include <iostream>

template <typename T>
std::vector<T> LoadBinaryBackend(std::string const &path) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> output(fileSize / sizeof(T));
  file.read(reinterpret_cast<char *>(output.data()), fileSize);
  return output;
}

#define KNN_LOADBINARY_DUPLICATE(TYPE) \
template <> \
std::vector<TYPE> LoadBinary<TYPE>(std::string const &path) { \
  return LoadBinaryBackend<TYPE>(path); \
}
KNN_LOADBINARY_DUPLICATE(char)
KNN_LOADBINARY_DUPLICATE(unsigned char)
KNN_LOADBINARY_DUPLICATE(int)
KNN_LOADBINARY_DUPLICATE(unsigned int)
KNN_LOADBINARY_DUPLICATE(short)
KNN_LOADBINARY_DUPLICATE(unsigned short)
KNN_LOADBINARY_DUPLICATE(long)
KNN_LOADBINARY_DUPLICATE(unsigned long)
KNN_LOADBINARY_DUPLICATE(float)
KNN_LOADBINARY_DUPLICATE(double)
#undef KNN_LOADBINARY_DUPLICATE
