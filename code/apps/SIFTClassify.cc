#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Knn.h"
#include "knn/Timer.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <unordered_map>

using namespace knn;

template <typename T>
std::vector<T> ReadTexMex(std::string const &path, const int dim,
                          const int maxQueries) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  int nRows = fileSize / (dim * sizeof(T) + 4);
  if (maxQueries > 1) {
    nRows = std::min(nRows, maxQueries);
  }
  std::vector<T> output(dim * nRows);
  T *target = output.data();
  for (int i = 0; i < nRows; ++i) {
    file.seekg(4, std::ios_base::cur);
    file.read(reinterpret_cast<char *>(target), dim * sizeof(T));
    target += dim;
  }
  return output;
}

int main(int argc, char const *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: <training data file> <label file> <test data file> "
                 "<k> [<max number of queries>]"
              << std::endl;
    return 1;
  }
  const int k = std::stoi(argv[4]);
  int nQueries = -1;
  if (argc >= 6) {
    nQueries = std::stoi(argv[5]);
  }

  Timer timer;
  auto distFunc = [](std::vector<float>::const_iterator const &a,
                     std::vector<float>::const_iterator const &b) {
    float dist = 0;
    for (int i = 0; i < 128; ++i) {
      float distDim = a[i] - b[i];
      dist += distDim * distDim;
    }
    return dist;
  };

  std::cout << "Reading data... ";
  timer.Start();
  auto train = ReadTexMex<float>(argv[1], 128, -1);
  auto groundTruth = ReadTexMex<int>(argv[2], k, nQueries);
  auto test = ReadTexMex<float>(argv[3], 128, nQueries);
  const int nTrain = train.size() / 128;
  const int nTest = test.size() / 128;
  assert(nTest == static_cast<int>(groundTruth.size() / k));
  std::vector<int> labels(nTrain);
  std::iota(labels.begin(), labels.end(), 0);
  double elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

  std::cout << "Nearest neighbors using linear search... ";
  timer.Start();
  auto resultLinear = KnnLinear<128, float>(train, k, test, distFunc);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

  std::cout << "Building kd-tree... ";
  timer.Start();
  KDTree<128, false, float> kdTree(train);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";
  std::cout << "Nearest neighbor search using one exact tree... ";
  timer.Start();
  auto resultKdTree = KnnExact<128, false, float>(kdTree, k, test, distFunc);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

  std::cout << "Building randomized trees... ";
  timer.Start();
  auto trees = KDTree<128, true, float>::BuildRandomizedTrees(
      train, 5, KDTree<128, true, float>::Pivot::median, 100);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";
  std::cout
      << "Nearest neighbor search using 5 randomized approximate trees... ";
  timer.Start();
  auto resultRandomized =
      KnnApproximate<128, float>(trees, k, 1000, test, distFunc);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

  float equalLinear = 0;
  float equalKdTree = 0;
  float equalRand = 0;

  timer.Start();
  auto sortByIndex = [](std::pair<float, size_t> const &a,
                        std::pair<float, size_t> const &b) {
    return a.second < b.second;
  };
  #pragma omp parallel for reduction(+ : equalLinear, equalKdTree)
  for (int i = 0; i < nTest; ++i) {
    std::sort(groundTruth.begin() + i * k, groundTruth.begin() + (i + 1) * k);
    std::sort(resultLinear[i].begin(), resultLinear[i].end(), sortByIndex);
    std::sort(resultKdTree[i].begin(), resultKdTree[i].end(), sortByIndex);
    std::sort(resultRandomized[i].begin(), resultRandomized[i].end(),
              sortByIndex);
    auto iLinear = resultLinear[i].cbegin();
    auto iLinearEnd = resultLinear[i].cend();
    auto iKd = resultKdTree[i].cbegin();
    auto iKdEnd = resultKdTree[i].cend();
    auto iRand = resultRandomized[i].cbegin();
    auto iRandEnd = resultRandomized[i].cend();
    for (auto iGt = groundTruth.cbegin() + i * k,
              iGtEnd = groundTruth.cend() + (i + 1) * k;
         iGt < iGtEnd; ++iGt) {
      while (iLinear < iLinearEnd) {
        if (static_cast<int>(iLinear->second) == *iGt) {
          ++iLinear;
          equalLinear += 1;
          break;
        } else if (static_cast<int>(iLinear->second) > *iGt) {
          break;
        }
        ++iLinear;
      }
      while (iKd < iKdEnd) {
        if (static_cast<int>(iKd->second) == *iGt) {
          ++iKd;
          equalKdTree += 1;
          break;
        } else if (static_cast<int>(iKd->second) > *iGt) {
          break;
        }
        ++iKd;
      }
      while (iRand < iRandEnd) {
        if (static_cast<int>(iRand->second) == *iGt) {
          ++iRand;
          equalRand += 1;
          break;
        } else if (static_cast<int>(iRand->second) > *iGt) {
          break;
        }
        ++iRand;
      }
    }
  }
  equalLinear /= nTest;
  equalKdTree /= nTest;
  equalRand   /= nTest;
  elapsed = timer.Stop();
  std::cout << "Verification done in " << elapsed << " seconds.\n";
  std::cout << "Linear: " << equalLinear << " / " << nTest << " ("
            << static_cast<float>(equalLinear) / nTest
            << ") classified correctly on average.\n";
  std::cout << "kd-tree: " << equalKdTree << " / " << nTest << " ("
            << static_cast<float>(equalKdTree) / nTest
            << ") classified correctly on average.\n";
  std::cout << "Randomized kd-trees: " << equalRand << " / " << nTest << " ("
            << static_cast<float>(equalRand) / nTest
            << ") classified correctly on average.\n";

  return 0;
}

