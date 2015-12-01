/* File:   TestFlann.cpp */
/* Date:   Tue Dec  1 20:53:22 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    FLANN Test */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include "knn/BinaryIO.h"
#include "knn/Timer.h"
#include "flann/flann.hpp"

using namespace std;
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


int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: <training data file>" << std::endl;
        return 1;
    }

    Timer timer;

    std::cout << "Reading data... ";
    timer.Start();
    auto train = ReadTexMex<float>(argv[1], 128, -1);
    double elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";

    flann::Matrix<float> flannTrain(train.data(), train.size()/128, 128);
    flann::Index<flann::L2<float> > index(flannTrain, flann::KDTreeIndexParams(1));

    std::cout << "Building randomized flann kd-tree with "
    << std::thread::hardware_concurrency()
    << " available hardware threads... ";
    timer.Start();
    index.buildIndex();                                                                                               
    double elapsedParallel = timer.Stop();
    std::cout << "Done in " << elapsedParallel << " seconds.\nSpeedup: " << std::endl;


    return 0;
}
