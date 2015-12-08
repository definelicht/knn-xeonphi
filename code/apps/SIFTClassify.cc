#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Knn.h"
#include "knn/Timer.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <unordered_map>
#ifdef KNN_USE_FLANN
#include <flann/flann.h>
#endif

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
                 "<k> <max leaves to check> [<max number of queries>]"
              << std::endl;
    return 1;
  }
  const int k = std::stoi(argv[4]);
  const int maxChecks = std::stoi(argv[5]);
  int nQueries = -1;
  if (argc >= 7) {
    nQueries = std::stoi(argv[6]);
  }

  knn::Timer timer;

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

  using DataItr = typename decltype(train)::const_iterator;

  std::cout << "Nearest neighbors using linear search... ";
  timer.Start();
  auto resultLinear = knn::KnnLinear<128, float, DataItr>(
      train.cbegin(), train.cend(), test.cbegin(), test.cend(), k,
      knn::SquaredEuclidianDistance<DataItr, 128>);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

  std::cout << "Building kd-tree... ";
  timer.Start();
  knn::KDTree<float, 128, false> kdTree(train.cbegin(), train.cend());
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";
  std::cout << "Nearest neighbor search using one exact tree... ";
  timer.Start();
  auto resultKdTree = knn::KnnExact<128, float, DataItr>(
      kdTree, train.cbegin(), test.cbegin(), test.cend(), k,
      knn::SquaredEuclidianDistance<DataItr, 128>);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

#ifdef KNN_USE_FLANN
  std::cout << "Building FLANN randomized kd-tree trees... ";
  flann::Matrix<float> flannTrain(train.data(), nTrain, 128);
  flann::Matrix<float> flannTest(test.data(), nTest, 128);
  flann::Matrix<int> flannIndices(new int[nTest * k], flannTest.rows, k);
  flann::Matrix<float> flannDists(new float[nTest * k], flannTest.rows, k);
  timer.Start();
  flann::Index<flann::L2<float>> index(flannTrain, flann::KDTreeIndexParams(5));
  index.buildIndex();
  double elapsedFlannBuild = timer.Stop();
  std::cout << "Done in " << elapsedFlannBuild << " seconds.\n";
  std::cout << "Nearest neighbor search using 5 randomized approximate FLANN "
               "trees... ";
  index.knnSearch(flannTest, flannIndices, flannDists, k,
                  flann::SearchParams(maxChecks));
  double elapsedFlannSearch = timer.Stop();
  std::vector<std::vector<std::pair<float, size_t>>> resultKdTreeFlann(nTest);
  for (int i = 0; i < nTest; ++i) {
    for (int j = 0; j < k; ++j) {
      resultKdTreeFlann[i].emplace_back(
          std::make_pair(flannDists[i][j], flannIndices[i][j]));
    }
  }
  std::cout << "Done in " << elapsedFlannSearch << " seconds.\n";
#endif

  std::cout << "Building randomized trees... ";
  timer.Start();
  auto trees = knn::KDTree<float, 128, true>::BuildRandomizedTrees(
      train.cbegin(), train.cend(), 5,
      knn::KDTree<float, 128, true>::Pivot::median, 100);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";
#ifdef KNN_USE_FLANN
  std::cout << "Build speedup over FLANN: " << elapsedFlannBuild / elapsed
            << ".\n";
#endif
  std::cout
      << "Nearest neighbor search using 5 randomized approximate trees... ";
  timer.Start();
  auto resultRandomized = knn::KnnApproximate<128, float, DataItr>(
      trees, train.cbegin(), test.cbegin(), test.cend(), k, maxChecks,
      knn::SquaredEuclidianDistance<DataItr, 128>);
  elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";
#ifdef KNN_USE_FLANN
  std::cout << "Search speedup over FLANN: " << elapsedFlannSearch / elapsed
            << ".\n";
#endif

  float equalLinear = 0;
  float equalKdTree = 0;
  float equalRand = 0;
#ifdef KNN_USE_FLANN
  float equalKdTreeFlann = 0;
#endif

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
#ifdef KNN_USE_FLANN
    std::sort(resultKdTreeFlann[i].begin(), resultKdTreeFlann[i].end(),
              sortByIndex);
#endif
    auto iLinear = resultLinear[i].cbegin();
    auto iLinearEnd = resultLinear[i].cend();
    auto iKd = resultKdTree[i].cbegin();
    auto iKdEnd = resultKdTree[i].cend();
    auto iRand = resultRandomized[i].cbegin();
    auto iRandEnd = resultRandomized[i].cend();
#ifdef KNN_USE_FLANN
    auto iKdFlann = resultKdTreeFlann[i].cbegin();
    auto iKdFlannEnd = resultKdTreeFlann[i].cend();
#endif
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
#ifdef KNN_USE_FLANN
      while (iKdFlann < iKdFlannEnd) {
        if (static_cast<int>(iKdFlann->second) == *iGt) {
          ++iKdFlann;
          equalKdTreeFlann += 1;
          break;
        } else if (static_cast<int>(iKdFlann->second) > *iGt) {
          break;
        }
        ++iKdFlann;
      }
#endif
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
#ifdef KNN_USE_FLANN
  equalKdTreeFlann /= nTest;
  std::cout << "FLANN kd-tree: " << equalKdTreeFlann << " / " << nTest << " ("
            << static_cast<float>(equalKdTreeFlann) / nTest
            << ") classified correctly on average.\n";
#endif

  return 0;
}

