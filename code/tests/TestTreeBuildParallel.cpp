#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>
#include "knn/BinaryIO.h"
#include "knn/KDTree.h"
#include "knn/Random.h"
#include "knn/Timer.h"

using namespace std;
using namespace knn;

/* constexpr size_t n = 1<<1; */
constexpr size_t n = 4;
/* constexpr size_t nDims = 128; */
constexpr size_t nDims = 2;

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

using myTreeType = KDTree<nDims, false, float>;

void printTree(const myTreeType::NodeItr& a, const myTreeType::NodeItr& b)
{
    cout << *a.value() << endl;
    cout << *b.value() << endl;
    cout << endl;
    if (a.inBounds())
    {
        cout << "Left" << endl;
        printTree(a.Left(), b.Left());
    }
    if (a.inBounds())
    {
        cout << "Right" << endl;
        printTree(a.Right(), b.Right());
    }
}

void compareTree(const myTreeType::NodeItr& a, const myTreeType::NodeItr& b, const size_t level=0)
{
    cout << "Level = " << level << endl;
    cout << "Bounds a:   " << a.inBounds() << endl;
    cout << "Bounds b:   " << b.inBounds() << endl;
    cout << "Index a:    " << a.index() << endl;
    cout << "Index b:    " << b.index() << endl;
    cout << "SplitDim a: " << a.splitDim() << endl;
    cout << "SplitDim b: " << b.splitDim() << endl;
    cout << "Value a:    " << *a.value() << endl;
    cout << "Value b:    " << *b.value() << endl;
    cout << endl;
    assert(a.inBounds() == b.inBounds());
    assert(a.index() == b.index());
    assert(a.splitDim() == b.splitDim());
    assert(*a.value() == *b.value());
    if (a.inBounds())
        compareTree(a.Left(), b.Left(), level+1);
    if (a.inBounds())
        compareTree(a.Right(), b.Right(), level+1);
}


int main(int argc, char** argv)
{
#if 0
    vector<float> train(n*nDims);
    Uniform(train.begin(), train.end());
    myTreeType kdTreeS(train, myTreeType::Pivot::median, 1, false);
    myTreeType kdTreeP(train, myTreeType::Pivot::median, 1, true);

    /* printTree(kdTreeS.Root(), kdTreeP.Root()); */
    compareTree(kdTreeS.Root(), kdTreeP.Root());
    return 0;
#else

    if (argc < 2) {
        std::cerr << "Usage: <training data file>" << std::endl;
        return 1;
    }

    int nThreads, maxThreads;
#pragma omp parallel
    {
#pragma omp single
        {
            nThreads   = omp_get_num_threads();
            maxThreads = omp_get_max_threads();
        }
    }

    Timer timer;

    cout << "Threads = " << nThreads << " out of " << maxThreads << endl;

    std::cout << "Reading data... ";
    timer.Start();
    auto train = ReadTexMex<float>(argv[1], 128, -1);
    double elapsed = timer.Stop();
    std::cout << "Done in " << elapsed << " seconds.\n";
    {
        using myTreeType = KDTree<128, false, float>;
        std::cout << "Building kd-tree seqential... ";
        timer.Start();
        myTreeType kdTreeS(train, myTreeType::Pivot::median, -1, false);
        elapsed = timer.Stop();
        std::cout << "Done in " << elapsed << " seconds.\n";
    }
    {
        using myTreeType = KDTree<128, false, float>;
        std::cout << "Building kd-tree parallel... ";
        timer.Start();
        myTreeType kdTreeP(train, myTreeType::Pivot::median, -1, true);
        elapsed = timer.Stop();
        std::cout << "Done in " << elapsed << " seconds.\n";
    }
    return 0;
#endif /* 0 */
}
