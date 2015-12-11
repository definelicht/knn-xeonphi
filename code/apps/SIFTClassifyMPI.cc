#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Knn.h"
#include "knn/Mpi.h"
#include "knn/Timer.h"

// This application is intended to be used with only one MPI worker on each
// device. Rank 0 will build the tree using tbb multithreading and will scatter
// it to the other workers, e.g. one worker responsible for data transfer on the
// xeon phi, who will then run search jobs using tbb.

int main(int argc, char *argv[]) {

  knn::mpi::Context context(argc, argv);
  const auto mpiRank = knn::mpi::rank();
  const auto mpiSize = knn::mpi::size();

  if (argc < 7) {
    if (mpiRank == 0) {
      std::cerr << "Usage: <training data file> <label file> <test data file> "
                   "<k> <number of random trees> <max leaves to check> [<max "
                   "number of queries>]"
                << std::endl;
    }
    return 0;
  }

  const std::string trainPath(argv[1]);
  const std::string labelPath(argv[2]);
  const std::string testPath(argv[3]);
  const int k = std::stoi(argv[4]);
  const int nTrees = std::stoi(argv[5]);
  const int maxChecks = std::stoi(argv[6]);
  int nQueries = -1;
  if (argc >= 8) {
    nQueries = std::stoi(argv[7]);
  }

  knn::Timer timer;

  if (mpiRank == 0) {
    std::cout << "Reading data... " << std::flush;
  }
  timer.Start();
  std::vector<float> train, test;
  std::vector<int> groundTruth;
  using DataItr = typename decltype(train)::const_iterator;
  std::array<int, 2> dataSizes;
  if (mpiRank == 0) {
    train = knn::ReadTexMex<float>(trainPath, 128, -1);
    groundTruth = knn::ReadTexMex<int>(labelPath, k, nQueries);
    test = knn::ReadTexMex<float>(testPath, 128, nQueries);
    dataSizes[0] = train.size();
    dataSizes[1] = test.size();
  }
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
  if (mpiRank == 0) {
    std::cout << "Done in " << timer.Stop() << " seconds.\n" << std::flush;
    std::cout << "Scattering trees... " << std::flush;
  }
  timer.Start();
  knn::KDTree<float, 128, true>::ScatterTreesMPI(kdTrees, nTrees,
                                                 2 * nTrain - 1, 0);
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
    outputSizes[i] = end[i] - begin[i]; 
  }
  // Rank 0 will return immediately as begin == end
  std::vector<std::pair<float, int>,
              tbb::scalable_allocator<std::pair<float, int>>> results;
  if (mpiRank == 0) {
    results.resize(k * nTest);
  }
  timer.Start();
  if (mpiRank != 0) {
    results = knn::KnnApproximate<128, float, DataItr>(
        kdTrees, train.cbegin(), test.cbegin() + begin[mpiRank],
        test.cbegin() + end[mpiRank], k, maxChecks,
        knn::SquaredEuclidianDistance<DataItr, 128>);
  }
  return 0;
  double elapsed = timer.Stop();
  using PairType = std::pair<float, int>;
  const auto mpiType = knn::mpi::CreateDataType<2>(
      {sizeof(float), sizeof(int)},
      {offsetof(PairType, first), offsetof(PairType, second)},
      {MPI_FLOAT, MPI_INT});
  MPI_Gatherv(results.data(), outputSizes[mpiRank], mpiType, results.data(),
              outputSizes.data(), begin.data(), mpiType, 0, MPI_COMM_WORLD);
  double elapsedCopy = timer.Stop();
  if (mpiRank != 0) {
    std::cout << "Search done in " << elapsedCopy << " seconds (" << elapsed
              << " excluding memory transfer).\n";
  }

  return 0;
}
