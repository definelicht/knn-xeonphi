#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Knn.h"
#include "knn/ParseArguments.h"
#include "knn/Timer.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <unordered_map>
#ifdef KNN_USE_FLANN
#include <flann/flann.h>
#endif
#ifdef KNN_USE_OMP
#include <omp.h>
#endif

int main(int argc, char const *argv[]) {

  if (argc < 4) {
    std::cerr << "Usage: <training data file> <label file> <test data file> "
                 "[-methods=<[linear,kdtree,randomtrees,flann,all]>] [-k=<k>] "
                 "[-trees=<number of randomized trees>] [-checks=<max "
                 "leaves to check>] [-querues=<max number of queries>] "
                 "[-output=<path to output benchmark file>]"
              << std::endl;
    return 1;
  }
  int k = 100;
  int nTrees = 4;
  int maxChecks = 1000;
  int nQueries = -1;
  std::string methods = "all";
  std::string outputFile = "";
  {
    ParseArguments args(argc, argv);
    args("k", k);
    args("trees", nTrees);
    args("checks", maxChecks);
    args("queries", nQueries);
    args("methods", methods);
    args("output", outputFile);
  }
  bool runAll = methods.find("all") != std::string::npos;
  bool runLinear = methods.find("linear") != std::string::npos;
  bool runKdTree = methods.find("kdtree") != std::string::npos;
#ifdef KNN_USE_FLANN
  bool runFlann = methods.find("flann") != std::string::npos;
#endif
  bool runRandomized = methods.find("randomized") != std::string::npos;
  std::ofstream output;
  if (outputFile != "") {
    output.open(outputFile, std::ios_base::app);
  }
  double elapsedLinear = 0;
  auto reportBenchmark = [&elapsedLinear, &runAll, &runLinear](double elapsed) {
    std::cout << " Done in " << elapsed << " seconds";
    if (elapsedLinear > 0) {
      std::cout << " (" << elapsedLinear / elapsed << " speedup)";
    }
    std::cout << ".\n";
  };

  knn::Timer timer;

  int nThreads = std::thread::hardware_concurrency();

  std::cout << "Available hardware concurrency: "
            << std::thread::hardware_concurrency() << "\n";
#ifdef KNN_USE_OMP
  nThreads = omp_get_max_threads();
  std::cout << "Available OMP threads (only used by FLANN): "
            << omp_get_max_threads() << "\n";
#endif

  auto writeBenchmark = [&output, &nThreads](std::string const &method,
                                             double elapsed) {
    if (output.is_open()) {
      output << method << "," << nThreads << "," << elapsed << "\n";
    }
  };

  std::cout << "Reading data... " << std::flush;
  timer.Start();
  auto train = knn::ReadTexMex<float>(argv[1], 128, -1);
  auto groundTruth = knn::ReadTexMex<int>(argv[2], k, nQueries);
  auto test = knn::ReadTexMex<float>(argv[3], 128, nQueries);
  const int nTrain = train.size() / 128;
  const int nTest = test.size() / 128;
  std::vector<int> labels(nTrain);
  std::iota(labels.begin(), labels.end(), 0);
  double elapsed = timer.Stop();
  std::cout << "Done in " << elapsed << " seconds.\n";

  using DataItr = typename decltype(train)::const_iterator;

  std::vector<std::pair<float, int>,
              tbb::scalable_allocator<std::pair<float, int>>> resultLinear;
  if (runAll || runLinear) {
    std::cout << "Nearest neighbors using linear search..." << std::flush;
    timer.Start();
    resultLinear = knn::KnnLinear<128, float, DataItr>(
        train.cbegin(), train.cend(), test.cbegin(), test.cend(), k,
        knn::SquaredEuclidianDistance<DataItr, 128>);
    elapsedLinear = timer.Stop();
    std::cout << " Done in " << elapsedLinear << " seconds.\n";
    writeBenchmark("linear", elapsedLinear);
  }
  
  std::vector<std::pair<float, int>,
              tbb::scalable_allocator<std::pair<float, int>>> resultKdTree;
  if (runAll || runKdTree) {
    std::cout << "Building kd-tree...";
    timer.Start();
    knn::KDTree<float, 128, false> kdTree(train.cbegin(), train.cend());
    elapsed = timer.Stop();
    std::cout << " Done in " << elapsed << " seconds.\n";
    std::cout << "Nearest neighbor search using one exact tree...";
    timer.Start();
    resultKdTree = knn::KnnExact<128, float, DataItr>(
        kdTree, train.cbegin(), test.cbegin(), test.cend(), k,
        knn::SquaredEuclidianDistance<DataItr, 128>);
    elapsed = timer.Stop();
    reportBenchmark(elapsed);
    writeBenchmark("kdtree", elapsed);
  }

#ifdef KNN_USE_FLANN
  std::vector<std::pair<float, int>,
              tbb::scalable_allocator<std::pair<float, int>>> resultFlann;
  double elapsedFlannBuild, elapsedFlannSearch;
  if (runAll || runFlann) {
    std::cout << "Building FLANN randomized kd-tree trees..." << std::flush;
    flann::Matrix<float> flannTrain(train.data(), nTrain, 128);
    flann::Matrix<float> flannTest(test.data(), nTest, 128);
    flann::Matrix<int> flannIndices(new int[nTest * k], flannTest.rows, k);
    flann::Matrix<float> flannDists(new float[nTest * k], flannTest.rows, k);
    timer.Start();
    flann::Index<flann::L2<float>> index(flannTrain,
                                         flann::KDTreeIndexParams(nTrees));
    index.buildIndex();
    elapsedFlannBuild = timer.Stop();
    std::cout << " Done in " << elapsedFlannBuild << " seconds.\n";
    std::cout << "Nearest neighbor search using 5 randomized approximate FLANN "
                 "trees..." << std::flush;
    flann::SearchParams params(maxChecks);
#ifdef KNN_USE_OMP
    params.cores = omp_get_max_threads();
#endif
    timer.Start();
    index.knnSearch(flannTest, flannIndices, flannDists, k, params);
    elapsedFlannSearch = timer.Stop();
    for (int i = 0; i < nTest; ++i) {
      for (int j = 0; j < k; ++j) {
        resultFlann.emplace_back(
            std::make_pair(flannDists[i][j], flannIndices[i][j]));
      }
    }
    reportBenchmark(elapsedFlannSearch);
    writeBenchmark("kdtree", elapsedFlannSearch);
  }
#endif

  std::vector<std::pair<float, int>,
              tbb::scalable_allocator<std::pair<float, int>>> resultRandomized;
  if (runAll || runRandomized) {
    std::cout << "Building randomized trees... " << std::flush;
    timer.Start();
    auto trees = knn::KDTree<float, 128, true>::BuildRandomizedTrees(
        train.cbegin(), train.cend(), nTrees,
        knn::KDTree<float, 128, true>::Pivot::median, 100);
    elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";
#ifdef KNN_USE_FLANN
    if (runAll || runFlann) {
      std::cout << "Build speedup over FLANN: " << elapsedFlannBuild / elapsed
                << ".\n";
    }
#endif
    std::cout << "Nearest neighbor search using " << nTrees
              << " randomized approximate trees..." << std::flush;
    timer.Start();
    resultRandomized = knn::KnnApproximate<128, float, DataItr>(
        trees, train.cbegin(), test.cbegin(), test.cend(), k, maxChecks,
        knn::SquaredEuclidianDistance<DataItr, 128>);
    elapsed = timer.Stop();
    reportBenchmark(elapsed);
    writeBenchmark("randomized", elapsed);
#ifdef KNN_USE_FLANN
    if (runAll || runFlann) {
      std::cout << "Search speedup over FLANN: " << elapsedFlannSearch / elapsed
                << ".\n";
    }
#endif
  }

  float equalLinear = 0;
  float equalKdTree = 0;
  float equalRand = 0;
#ifdef KNN_USE_FLANN
  float equalFlann = 0;
#endif

  timer.Start();
  auto sortByIndex = [](std::pair<float, size_t> const &a,
                        std::pair<float, size_t> const &b) {
    return a.second < b.second;
  };
#pragma omp parallel for reduction(+ : equalLinear, equalKdTree)
  for (int i = 0; i < nTest; ++i) {
    std::sort(groundTruth.begin() + i * k, groundTruth.begin() + (i + 1) * k);
    if (runAll || runLinear) {
      std::sort(resultLinear.begin() + i * k,
                resultLinear.begin() + (i + 1) * k, sortByIndex);
    }
    if (runAll || runKdTree) {
      std::sort(resultKdTree.begin() + i * k,
                resultKdTree.begin() + (i + 1) * k, sortByIndex);
    }
    if (runAll || runRandomized) {
      std::sort(resultRandomized.begin() + i * k,
                resultRandomized.begin() + (i + 1) * k, sortByIndex);
    }
#ifdef KNN_USE_FLANN
    if (runAll || runFlann) {
      std::sort(resultFlann.begin() + i * k,
                resultFlann.begin() + (i + 1) * k, sortByIndex);
    }
#endif
    auto iLinear = resultLinear.cbegin() + i * k;
    auto iLinearEnd = resultLinear.cbegin() + (i + 1) * k;
    auto iKd = resultKdTree.cbegin() + i * k;
    auto iKdEnd = resultKdTree.cbegin() + (i + 1) * k;
    auto iRand = resultRandomized.cbegin() + i * k;
    auto iRandEnd = resultRandomized.cbegin() + (i + 1) * k;
#ifdef KNN_USE_FLANN
    auto iKdFlann = resultFlann.cbegin() + i * k;
    auto iKdFlannEnd = resultFlann.cbegin() + (i + 1) * k;
#endif
    for (auto iGt = groundTruth.cbegin() + i * k,
              iGtEnd = groundTruth.cend() + (i + 1) * k;
         iGt < iGtEnd; ++iGt) {
      if (runAll || runLinear) {
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
      }
      if (runAll || runKdTree) {
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
      }
      if (runAll || runRandomized) {
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
#ifdef KNN_USE_FLANN
      if (runAll || runFlann) {
        while (iKdFlann < iKdFlannEnd) {
          if (static_cast<int>(iKdFlann->second) == *iGt) {
            ++iKdFlann;
            equalFlann += 1;
            break;
          } else if (static_cast<int>(iKdFlann->second) > *iGt) {
            break;
          }
          ++iKdFlann;
        }
      }
#endif
    }
  }
  equalLinear /= nTest;
  equalKdTree /= nTest;
  equalRand /= nTest;
  elapsed = timer.Stop();
  std::cout << "Verification done in " << elapsed << " seconds.\n";
  if (runAll || runLinear) {
    std::cout << "Linear: " << equalLinear << " / " << nTest << " ("
              << static_cast<float>(equalLinear) / nTest
              << ") classified correctly on average.\n";
  }
  if (runAll || runKdTree) {
    std::cout << "kd-tree: " << equalKdTree << " / " << nTest << " ("
              << static_cast<float>(equalKdTree) / nTest
              << ") classified correctly on average.\n";
  }
  if (runAll || runRandomized) {
    std::cout << "Randomized kd-trees: " << equalRand << " / " << k << " ("
              << static_cast<float>(equalRand) / k
              << ") classified correctly on average.\n";
  }
#ifdef KNN_USE_FLANN
  equalFlann /= nTest;
  if (runAll || runFlann) {
    std::cout << "FLANN kd-tree: " << equalFlann << " / " << k << " ("
              << static_cast<float>(equalFlann) / k
              << ") classified correctly on average.\n";
  }
#endif

  return 0;
}

