#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Timer.h"
#include <iostream>
#include <string>

using namespace knn;

template <typename DataType, typename LabelType>
std::vector<LabelType> Classify(std::string const &trainPath,
                                std::string const &labelPath,
                                std::string const &testPath);

int main(int argc, char const *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: <training data file> <label file> <test data file> "
                 "<data type> <label type>"
              << std::endl;
    return 1;
  }
  std::string trainPath(argv[1]);
  std::string labelPath(argv[2]);
  std::string testPath(argv[3]);
  std::string dataTypeName(argv[4]);
  std::string labelTypeName(argv[5]);
  Timer timer;
  if (dataTypeName == "double" && labelTypeName == "int") {
    auto classes = Classify<double, int>(trainPath, labelPath, testPath);  
  } else {
    std::cerr << "Unsupported types \"" << dataTypeName << "\" and \""
              << labelTypeName << "\".\n";
    return 1;
  }
  double elapsed = timer.Stop();
  std::cout << "Classification done in " << elapsed << " seconds.\n";
  return 0;
}

template <typename DataType, typename LabelType>
std::vector<LabelType> Classify(std::string const &trainPath,
                                std::string const &labelPath,
                                std::string const &testPath) {
  auto train = LoadBinaryFile<DataType>(trainPath); 
  auto label = LoadBinaryFile<LabelType>(labelPath);
  auto test = LoadBinaryFile<DataType>(testPath);
  return {};
}
