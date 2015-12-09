#include "knn/BinaryIO.h"
#include "knn/Mpi.h"
#include "knn/Timer.h"

int main(int argc, char *argv[]) {

  knn::mpi::Context context(argc, argv);
  const auto mpiRank = knn::mpi::rank();

  if (mpiRank == 0 && argc < 7) {
    std::cerr << "Usage: <training data file> <label file> <test data file> "
                 "<k> <number of random trees> <max leaves to check> [<max "
                 "number of queries>]"
              << std::endl;
  }

  const std::string trainPath(argv[1]);
  const std::string labelPath(argv[2]);
  const std::string testPath(argv[3]);
  const int k = std::stoi(argv[4]);
  const int maxChecks = std::stoi(argv[5]);
  int nQueries = -1;
  if (argc >= 7) {
    nQueries = std::stoi(argv[6]);
  }

  knn::Timer timer;

  std::cout << "Reading data... " << std::flush;
  timer.Start();
  const auto train = knn::ReadTexMex<float>(trainPath, 128, -1);
  const auto groundTruth = knn::ReadTexMex<int>(labelPath, k, nQueries);
  const auto test = knn::ReadTexMex<float>(testPath, 128, nQueries);
  std::cout << "Done.\n";

  return 0;
}
