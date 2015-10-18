#include "knn/BinaryIO.h"
#include <fstream>

namespace knn {

template <typename T>
std::vector<T> LoadBinaryFileBackend(std::string const &path) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> output(fileSize / sizeof(T));
  file.read(reinterpret_cast<char *>(output.data()), fileSize);
  return output;
}

template <typename T>
void WriteBinaryFileBackend(std::string const &path,
                            std::vector<T> const &data) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<char const *>(data.data()),
             data.size() * sizeof(T));
}

#define KNN_BINARYIO_INSTANTIATIONS(TYPE) \
template <> \
std::vector<TYPE> LoadBinaryFile<TYPE>(std::string const &path) { \
  return LoadBinaryFileBackend<TYPE>(path); \
} \
template <> \
void WriteBinaryFile<TYPE>(std::string const &path,\
                           std::vector<TYPE> const &data) { \
  WriteBinaryFileBackend<TYPE>(path, data); \
}
KNN_BINARYIO_INSTANTIATIONS(char)
KNN_BINARYIO_INSTANTIATIONS(unsigned char)
KNN_BINARYIO_INSTANTIATIONS(int)
KNN_BINARYIO_INSTANTIATIONS(unsigned int)
KNN_BINARYIO_INSTANTIATIONS(short)
KNN_BINARYIO_INSTANTIATIONS(unsigned short)
KNN_BINARYIO_INSTANTIATIONS(long)
KNN_BINARYIO_INSTANTIATIONS(unsigned long)
KNN_BINARYIO_INSTANTIATIONS(float)
KNN_BINARYIO_INSTANTIATIONS(double)
#undef KNN_BINARYIO_INSTANTIATIONS

} // End namespace knn
