#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/KnnLinear.h"
#include "knn/Timer.h"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>

using namespace knn;

template <typename T>
std::vector<T> ReadTexMex(std::string const &path, const int dim) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  const int nRows = fileSize / (dim*sizeof(T) + 4);
  std::vector<T> output(dim*nRows);
  T* target = output.data();
  for (int i = 0; i < nRows; ++i) {
    file.seekg(4, std::ios_base::cur);
    file.read(reinterpret_cast<char *>(target), dim*sizeof(T));
    target += dim;
  }
  return output;
}

int main(int argc, char const *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: <training data file> <label file> <test data file> "
                 "<k> [<shave at 3 dimensions>]"
              << std::endl;
    return 1;
  }
  bool shave = false;
  if (argc >= 6) {
    shave = std::stoi(argv[5]);
  }
  const int k = std::stoi(argv[4]);
  Timer timer;
  auto distFunc = [](std::vector<float>::const_iterator const &a,
                     std::vector<float>::const_iterator const &b) {
    float dist = 0;
    for (int i = 0; i < 128; ++i) {
      float distDim = a[i] - b[i];
      dist += distDim*distDim;
    }
    return dist;
  };
  if (!shave) {

    std::cout << "Reading data... ";
    timer.Start();
    auto train = ReadTexMex<float>(argv[1], 128);
    auto groundTruth = ReadTexMex<int>(argv[2], k);
    auto test = ReadTexMex<float>(argv[3], 128);
    const int nTrain = train.size()/128;
    const int nTest = test.size()/128;
    assert(nTest == static_cast<int>(groundTruth.size()/k));
    std::vector<int> labels(nTrain);
    std::iota(labels.begin(), labels.end(), 0);
    double elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";

    KnnLinear<float, int, float, 128> linear(train, labels, distFunc);
    std::cout << "Nearest neighbors using linear search... ";
    timer.Start();
    auto resultLinear = linear.Knn(k, test);
    elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";
    
    std::cout << "Building kd-tree... ";
    timer.Start();
    KDTree<128, false, float, int, float> kdTree(train, labels, distFunc);
    elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";
    std::cout << "Classifying using kd-tree... ";
    timer.Start();
    auto resultKdTree = kdTree.Knn(k, test);
    elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";

    int equalLinear = 0;
    int equalKdTree = 0;

    timer.Start();
    #pragma omp parallel for reduction(+:equalLinear, equalKdTree) 
    for (int i = 0; i < nTest; ++i) {
      std::sort(groundTruth.begin()+i*k, groundTruth.begin()+(i+1)*k);
      std::sort(resultLinear[i].begin(), resultLinear[i].end());
      std::sort(resultKdTree[i].begin(), resultKdTree[i].end());
      equalLinear += std::equal(groundTruth.cbegin() + i * k,
                                groundTruth.cbegin() + (i + 1) * k,
                                resultLinear[i].cbegin());
      equalKdTree += std::equal(groundTruth.cbegin() + i * k,
                                groundTruth.cbegin() + (i + 1) * k,
                                resultKdTree[i].cbegin());
    }
    elapsed = timer.Stop();
    std::cout << "Verification done in " << elapsed << " seconds.\n";
    std::cout << "Linear: " << equalLinear << " / " << nTest << " ("
              << static_cast<float>(equalLinear) / nTest
              << ") classified correctly.\n";
    std::cout << "kd-tree: " << equalKdTree << " / " << nTest << " ("
              << static_cast<float>(equalKdTree) / nTest
              << ") classified correctly.\n";
  } else {
    std::cerr << "Shave NYI.\n";
    return 1;
  }
  return 0;
}

