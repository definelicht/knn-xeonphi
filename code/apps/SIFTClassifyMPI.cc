#include <iostream>
#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Knn.h"
#include "knn/Mpi.h"
#include "knn/ParseArguments.h"
#include "knn/Timer.h"

// This application is intended to be used with only one MPI worker on each
// device. Rank 0 will build the tree using tbb multithreading and will scatter
// it to the other workers, e.g. one worker responsible for data transfer on the
// xeon phi, who will then run search jobs using tbb.

int main(int argc, char *argv[]) {

  knn::mpi::Context context(argc, argv);
  const auto mpiRank = knn::mpi::rank();
  const auto mpiSize = knn::mpi::size();

  if (argc < 5) {
    if (mpiRank == 0) {
      std::cerr << "Usage: <training data file> <label file> <test data file> "
                   "<max leaves to check> "
                   "[-k=<k> -nTrees=<number of random trees> -nQueries=<max "
                   "number of queries> -iterations=<number of iterations to "
                   "run search>]"
                << std::endl;
    }
    return 0;
  }

  std::cout << "Rank " << mpiRank << " hardware concurrency: "
            << std::thread::hardware_concurrency() << "\n";

  const std::string trainPath(argv[1]);
  const std::string labelPath(argv[2]);
  const std::string testPath(argv[3]);
  const int maxChecks = std::stoi(argv[4]);
  int nQueries = -1;
  int k = 100;
  int nTrees = 4;
  int nIterations = 1;
  {
    ParseArguments args(argc, argv);
    args("k", k);
    args("nTrees", nTrees);
    args("nQueries", nQueries);
    args("iterations", nIterations);
  }

  knn::Timer timer;

  if (mpiRank == 0) {
    std::cout << "Reading data... " << std::flush;
  }
  timer.Start();
  std::vector<float, tbb::cache_aligned_allocator<float>> train, test;
  std::vector<int, tbb::cache_aligned_allocator<int>> groundTruth;
  using DataItr = typename decltype(train)::const_iterator;
  std::array<int, 2> dataSizes;
  if (mpiRank == 0) {
    train = knn::ReadTexMex<float>(trainPath, 128, -1);
    groundTruth = knn::ReadTexMex<int>(labelPath, k, nQueries);
    test = knn::ReadTexMex<float>(testPath, 128, nQueries);
    dataSizes[0] = train.size();
    dataSizes[1] = test.size();
  }
  knn::mpi::Barrier();
  if (mpiRank == 0) {
    std::cout << "Done in " << timer.Stop() << " seconds.\n" << std::flush;
    std::cout << "Broadcasting data... " << std::flush;
  }
  timer.Start();
  knn::mpi::Broadcast(dataSizes.begin(), dataSizes.end(), 0);
  train.resize(dataSizes[0]);
  test.resize(dataSizes[1]);
  knn::mpi::Broadcast(train.begin(), train.end(), 0);
  knn::mpi::Broadcast(test.begin(), test.end(), 0);
  const int nTrain = train.size() / 128;
  const int nTest = test.size() / 128;
  knn::mpi::Barrier();
  if (mpiRank == 0) {
    std::cout << "Done in " << timer.Stop() << " seconds.\n" << std::flush;
    std::cout << "Building trees... " << std::flush;
  }
  timer.Start();
  std::vector<knn::KDTree<float, 128, true>> kdTrees;
  if (mpiRank == 0) {
    kdTrees = knn::KDTree<float, 128, true>::BuildRandomizedTrees(
        train.begin(), train.end(), nTrees);
  }
  knn::mpi::Barrier();
  if (mpiRank == 0) {
    std::cout << "Done in " << timer.Stop() << " seconds.\n" << std::flush;
    std::cout << "Broadcasting trees... " << std::flush;
  }
  timer.Start();
  knn::KDTree<float, 128, true>::BroadcastTreesMPI(kdTrees, nTrees,
                                                   2 * nTrain - 1, 0);
  knn::mpi::Barrier();
  if (mpiRank == 0) {
    std::cout << "Done in " << timer.Stop() << " seconds.\n" << std::flush;
  }
  std::vector<int> begin(mpiSize);
  std::vector<int> end(mpiSize);
  std::vector<int> outputSizes(mpiSize);
  begin[0] = 0;
  end[0] = 0;
  outputSizes[0] = 0;
  for (int i = 1; i < mpiSize; ++i) {
    begin[i] = nTest * (i - 1) / (mpiSize - 1);
    end[i] = nTest * i / (mpiSize - 1);
    // TODO: horrible hack
    outputSizes[i] = sizeof(std::pair<float, int>) * k * (end[i] - begin[i]);
  }
  std::vector<std::pair<float, int>,
              tbb::scalable_allocator<std::pair<float, int>>> results;
  if (mpiRank == 0) {
    results.resize(k * nTest);
  }
  for (int i = 0; i < nIterations; ++i) {
    knn::mpi::Barrier();
    if (mpiRank != 0) {
      if (nIterations > 1) {
        std::cout << "Running iteration " << i + 1 << " / " << nIterations
                  << "..." << std::flush;
      } else {
        std::cout << "Performing search..." << std::flush;
      }
      timer.Start();
      results = knn::KnnApproximate<128, float, DataItr>(
          kdTrees, train.cbegin(), test.cbegin() + 128 * begin[mpiRank],
          test.cbegin() + 128 * end[mpiRank], k, maxChecks,
          knn::SquaredEuclidianDistance<DataItr, 128>);
      std::cout << " Done in " << timer.Stop()
                << " seconds.\nCopying back results..." << std::flush;
    }
    knn::mpi::Barrier();
    // const auto mpiType = knn::mpi::CreateDataType<2>(
    //     {sizeof(float), sizeof(int)},
    //     {offsetof(PairType, first), offsetof(PairType, second)},
    //     {MPI_FLOAT, MPI_INT});
    // MPI_Gatherv(results.data(), outputSizes[mpiRank], mpiType, results.data(),
    //             outputSizes.data(), begin.data(), mpiType, 0, MPI_COMM_WORLD);
    timer.Start();
    // TODO: dreadful hack
    MPI_Gatherv(results.data(), outputSizes[mpiRank], MPI_CHAR, results.data(),
                outputSizes.data(), begin.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
    knn::mpi::Barrier();
    if (mpiRank != 0) {
      std::cout << " Done in " << timer.Stop() << " seconds.\n";
    }
  }
  if (mpiRank == 0) {
    int equal = 0;
    for (int i = 0; i < nTest; ++i) {
      std::sort(groundTruth.begin() + i * k, groundTruth.begin() + (i + 1) * k);
      std::sort(
          results.begin() + i * k, results.begin() + (i + 1) * k,
          [](std::pair<float, int> const &a, std::pair<float, int> const &b) {
            return a.second < b.second;
          });
      auto iResult = results.begin() + i * k;
      const auto iResultEnd = results.begin() + (i + 1) * k;
      for (auto iGt = groundTruth.cbegin() + i * k,
                iGtEnd = groundTruth.cend() + (i + 1) * k;
           iGt < iGtEnd; ++iGt) {
        while (iResult < iResultEnd) {
          if (static_cast<int>(iResult->second) == *iGt) {
            ++iResult;
            ++equal;
            break;
          } else if (static_cast<int>(iResult->second) > *iGt) {
            break;
          }
          ++iResult;
        }
      }
    }
    std::cout << static_cast<float>(equal) / nTest
              << "% classified correctly.\n";
  }

  return 0;
}
