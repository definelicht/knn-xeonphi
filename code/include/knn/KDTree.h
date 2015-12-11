#pragma once

#include <algorithm>  // std::nth_element
#include <functional> // std::function
#include <numeric>    // std::iota
#include <ostream>
#include <random>
#include <stdexcept>  // std::invalid_argument
#include <string>
#include <thread>     // std::thread::hardware_concurrency
#include <vector>
#include <tbb/parallel_for_each.h>
#include <tbb/task_group.h>
#include <tbb/scalable_allocator.h>

#include "knn/BoundedHeap.h"
#include "knn/Common.h"
#ifdef KNN_USE_MPI
#include "knn/Mpi.h"
#endif
#include "knn/Random.h"

namespace knn {

template <typename T, unsigned Dim, bool Randomized>
class alignas(64) KDTree {

public:
  struct alignas(16) Node {
    int index;
    int splitDim;
    int left, right;
    Node();
    Node(int _index, int _splitDim, int _left = -1, int _right = -1);
  };

  class NodeItr {
  public:
    NodeItr();
    NodeItr(Node const *current, Node const *begin);
    NodeItr Left() const;
    NodeItr Right() const;
    bool inBounds() const;
    size_t index() const;
    size_t splitDim() const;

  private:
    Node const *current_, *begin_;
  };

  enum class Pivot { median, mean };

  KDTree();

  template <typename DataIterator>
  KDTree(DataIterator begin, DataIterator end, Pivot pivot = Pivot::median,
         int nVarianceSamples = -1, int nHighestVariances = -1,
         const bool parallel = true);

  KDTree(KDTree<T, Dim, Randomized> const &) = default;

  KDTree(KDTree<T, Dim, Randomized> &&) = default;

  KDTree<T, Dim, Randomized> &
  operator=(KDTree<T, Dim, Randomized> const &) = default;

  KDTree<T, Dim, Randomized> &
  operator=(KDTree<T, Dim, Randomized> &&) = default;

  size_t nLeaves() const;

  constexpr size_t nDims() const;

  NodeItr Root() const;

  Node const *data() const;

  Node const &operator[](size_t i) const;

  template <typename DataIterator>
  static std::vector<KDTree<T, Dim, true>>
  BuildRandomizedTrees(DataIterator begin, DataIterator end, int nTrees,
                       Pivot pivot = Pivot::median, int nVarianceSamples = 100,
                       int nHighestVariances = Dim > 10 ? 5 : Dim / 2,
                       bool parallel = true);

#ifdef KNN_USE_MPI
  static void BroadcastTreesMPI(std::vector<KDTree<T, Dim, true>> &trees,
                              int nTrees, int treeSize, int root = 0);
#endif

private:
  using IndexItr =
      typename std::vector<size_t, tbb::scalable_allocator<size_t>>::iterator;

  template <typename DataIterator>
  int BuildTree(DataIterator data, IndexItr begin, IndexItr end, int myIndex,
                Pivot pivot, int nVarianceSamples, int nHighestVariances,
                int treeWidth, int splitWidth);

  std::vector<Node> tree_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace {

template <bool Randomized, typename T, unsigned Dim>
class GetSplitDim;

template <typename T, unsigned Dim>
class GetSplitDim<false, T, Dim> {
public:
  static size_t Get(const size_t, std::array<T, Dim> const &variances) {
    return std::distance(
        variances.cbegin(),
        std::max_element(variances.cbegin(), variances.cend()));
  }
};

template <typename T, unsigned Dim>
class GetSplitDim<true, T, Dim> {

public:
  static size_t Get(const size_t nHighestVariances,
                    std::array<T, Dim> const &variances) {
    std::array<size_t, Dim> indices;
    std::iota(indices.begin(), indices.end(), 0);
    auto begin = indices.begin();
    std::nth_element(begin, begin + nHighestVariances, indices.end(),
                     [&variances](size_t a, size_t b) {
                       return variances[a] > variances[b];
                     });
    return indices[random::Uniform<size_t>() % nHighestVariances];
  }
};

} // End anonymous namespace

template <typename T, unsigned Dim, bool Randomized>
KDTree<T, Dim, Randomized>::KDTree()
    : tree_() {}

template <typename T, unsigned Dim, bool Randomized>
template <typename DataIterator>
KDTree<T, Dim, Randomized>::KDTree(const DataIterator begin,
                                   const DataIterator end, Pivot pivot,
                                   int nVarianceSamples, int nHighestVariances,
                                   const bool parallel)
    : tree_{} {
  static_assert(Dim > 0, "kd-tree data cannot be zero-dimensional.");
  static_assert(
      HasRandomAccess<DataIterator>(),
      "Input iterator to kd-tree constructor must support random access.");
  const size_t dataSize = std::distance(begin, end);
  const size_t nLeaves = Dim > 0 ? std::distance(begin, end) / Dim : 0;
  if (nLeaves == 0) {
    throw std::invalid_argument("kd-tree received empty data set.");
  }
  if (dataSize % Dim != 0) {
    throw std::invalid_argument("kd-tree received unbalanced data.");
  }
  const size_t nNodes = 2 * nLeaves - 1;
  tree_.resize(nNodes); // exact number of nodes if points are not consumed
                        // until leaf is reached, is tree is evenly split,
                        // always odd
  std::vector<size_t, tbb::scalable_allocator<size_t>> indices(nLeaves);
  std::iota(indices.begin(), indices.end(), 0);
  if (!parallel) {
    BuildTree<DataIterator>(begin, indices.begin(), indices.end(), 0, pivot,
                            nVarianceSamples, nHighestVariances, 1, 1);
  } else {
    BuildTree<DataIterator>(begin, indices.begin(), indices.end(), 0, pivot,
                            nVarianceSamples, nHighestVariances, 1,
#ifndef __MIC__
                            std::thread::hardware_concurrency());
#else
                            240);
#endif
  }
}

template <typename T, unsigned Dim, bool Randomized>
KDTree<T, Dim, Randomized>::Node::Node()
    : index(-1), splitDim(-1), left(-1), right(-1) {}

template <typename T, unsigned Dim, bool Randomized>
KDTree<T, Dim, Randomized>::Node::Node(const int _index, const int _splitDim,
                                       const int _left, const int _right)
    : index(_index), splitDim(_splitDim), left(_left), right(_right) {}

template <typename T, unsigned Dim, bool Randomized>
KDTree<T, Dim, Randomized>::NodeItr::NodeItr()
    : current_(nullptr), begin_(nullptr) {}

template <typename T, unsigned Dim, bool Randomized>
KDTree<T, Dim, Randomized>::NodeItr::NodeItr(Node const *const current,
                                             Node const *const begin)
    : current_(current), begin_(begin) {}

template <typename T, unsigned Dim, bool Randomized>
typename KDTree<T, Dim, Randomized>::NodeItr
KDTree<T, Dim, Randomized>::NodeItr::Left() const {
  return {current_->left >= 0 ? begin_ + current_->left : nullptr, begin_};
}

template <typename T, unsigned Dim, bool Randomized>
typename KDTree<T, Dim, Randomized>::NodeItr
KDTree<T, Dim, Randomized>::NodeItr::Right() const {
  return {current_->right >= 0 ? begin_ + current_->right : nullptr, begin_};
}

template <typename T, unsigned Dim, bool Randomized>
bool KDTree<T, Dim, Randomized>::NodeItr::inBounds() const {
  return current_ != nullptr;
}

template <typename T, unsigned Dim, bool Randomized>
size_t KDTree<T, Dim, Randomized>::NodeItr::index() const {
  return current_->index;
}

template <typename T, unsigned Dim, bool Randomized>
size_t KDTree<T, Dim, Randomized>::NodeItr::splitDim() const {
  return current_->splitDim;
}

template <typename T, unsigned Dim, bool Randomized>
typename KDTree<T, Dim, Randomized>::NodeItr
KDTree<T, Dim, Randomized>::Root() const {
  return {tree_.data(), tree_.data()};
}

template <typename T, unsigned Dim, bool Randomized>
size_t KDTree<T, Dim, Randomized>::nLeaves() const {
  return tree_.size();
}

template <typename T, unsigned Dim, bool Randomized>
constexpr size_t KDTree<T, Dim, Randomized>::nDims() const {
  return Dim;
}

template <typename T, unsigned Dim, bool Randomized>
typename KDTree<T, Dim, Randomized>::Node const *
KDTree<T, Dim, Randomized>::data() const {
  return tree_.data();
}

template <typename T, unsigned Dim, bool Randomized>
typename KDTree<T, Dim, Randomized>::Node const &KDTree<T, Dim, Randomized>::
operator[](size_t i) const {
  return tree_[i];
}

template <typename T, unsigned Dim, bool Randomized>
template <typename DataIterator>
int KDTree<T, Dim, Randomized>::BuildTree(
    const DataIterator data, const IndexItr begin, const IndexItr end,
    const int myIndex, const Pivot pivot, const int nVarianceSamples,
    const int nHighestVariances, const int treeWidth, const int splitWidth) {

  // Check if recursion has hit the bottom
  if (std::distance(begin, end) > 1) {
    const auto meanAndVariance = MeanAndVariance<DataIterator, Dim, IndexItr>(
        data, begin, end, nVarianceSamples);
    const int splitDim = GetSplitDim<Randomized, T, Dim>::Get(
        nHighestVariances, meanAndVariance.second);
    int splitPivot;
    if (pivot == Pivot::median) {
      splitPivot = std::distance(begin, end) / 2;
      std::nth_element(
          begin, begin + splitPivot, end, [&data, &splitDim](int a, int b) {
            return data[Dim * a + splitDim] < data[Dim * b + splitDim];
          });
    } else {
      const T splitMean = meanAndVariance.first[splitDim];
      splitPivot = std::distance(
          begin,
          std::partition(begin, end, [&data, &splitDim, &splitMean](int i) {
            return data[Dim * i + splitDim] < splitMean;
          }));
    }
    const int splitIndex = begin[splitPivot];

    tree_[myIndex] = Node(splitIndex, splitDim);

    // Points are not consumed before a leaf is reached
    const int offset =
        2 * std::distance(begin, begin + splitPivot); // (2*nSubleaves - 1) + 1

    if (treeWidth < splitWidth) {
      // Keep spawning more tasks
      tbb::task_group myGroup;
      myGroup.run([&] {
        tree_[myIndex].left = BuildTree(
            data, begin, begin + splitPivot, myIndex + 1, pivot,
            nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
      });
      myGroup.run([&] {
        tree_[myIndex].right = BuildTree(
            data, begin + splitPivot, end, myIndex + offset, pivot,
            nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
      });
      myGroup.wait();
    } else {
      // Enough tasks, recurse the subtree alone
      tree_[myIndex].left = BuildTree(
          data, begin, begin + splitPivot, myIndex + 1, pivot, nVarianceSamples,
          nHighestVariances, 2 * treeWidth, splitWidth);
      tree_[myIndex].right = BuildTree(
          data, begin + splitPivot, end, myIndex + offset, pivot,
          nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
    }
  } else {
    // Leaf node, no babies
    tree_[myIndex] = Node(*begin, 0);
  }
  return myIndex;
}

template <typename T, unsigned Dim, bool Randomized>
template <typename DataIterator>
std::vector<KDTree<T, Dim, true>>
KDTree<T, Dim, Randomized>::BuildRandomizedTrees(
    const DataIterator begin, const DataIterator end, const int nTrees,
    const Pivot pivot, const int nVarianceSamples, const int nHighestVariances,
    const bool parallel) {
  static_assert(
      HasRandomAccess<DataIterator>(),
      "BuildRandomizedTrees: input iterators must support random access.");
  if (nTrees < 1) {
    throw std::invalid_argument(
        "BuildRandomizedTrees: number of trees must be > 0.");
  }
  std::vector<KDTree<T, Dim, true>> trees(nTrees);
  if (parallel) {
    tbb::parallel_for_each(
        trees.begin(), trees.end(), [&](KDTree<T, Dim, true> &tree) {
          tree = KDTree<T, Dim, true>(begin, end, pivot, nVarianceSamples,
                                      nHighestVariances, true);
        });
  } else {
    std::for_each(trees.begin(), trees.end(), [&](KDTree<T, Dim, true> &tree) {
      tree = KDTree<T, Dim, true>(begin, end, pivot, nVarianceSamples,
                                  nHighestVariances, false);
    });
  }
  return trees;
}

#ifdef KNN_USE_MPI
template <typename T, unsigned Dim, bool Randomized>
void KDTree<T, Dim, Randomized>::BroadcastTreesMPI(
    std::vector<KDTree<T, Dim, true>> &trees, const int nTrees,
    const int treeSize, const int root) {
  // const auto nodeMpiType = mpi::CreateDataType<4>(
  //     {sizeof(int), sizeof(int), sizeof(int), sizeof(int)},
  //     {offsetof(Node, index), offsetof(Node, splitDim), offsetof(Node, left),
  //      offsetof(Node, right)},
  //     {MPI_INT, MPI_INT, MPI_INT, MPI_INT});
  if (mpi::rank() != root) {
    trees.resize(nTrees);
    for (int i = 0; i < nTrees; ++i) {
      trees[i].tree_.resize(treeSize);
    }
  }
  std::vector<MPI_Request> requests(nTrees);
  for (int i = 0; i < nTrees; ++i) {
    // Distribute built trees to all ranks
    // TODO: HACK HACK HACK HACK
    MPI_Ibcast(trees[i].tree_.data(), sizeof(Node) * trees[i].tree_.size(),
               MPI_CHAR, root, MPI_COMM_WORLD, &requests[i]);
    //  MPI_Ibcast(trees[i].tree_.data(), trees[i].tree_.size(), nodeMpiType, root,
    //             MPI_COMM_WORLD, &requests[i]);
  }
  mpi::WaitAll(requests);
}
#endif

} // End namespace knn
