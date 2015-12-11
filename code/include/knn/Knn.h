#pragma once

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <tbb/parallel_for.h>
#include "knn/BoundedHeap.h"
#include "knn/Common.h"
#include "knn/KDTree.h"

namespace knn {

////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename DistType, typename T, bool Negate = false>
bool CompareNeighbors(std::pair<DistType, T> const &a,
                      std::pair<DistType, T> const &b) {
  if (Negate) {
    return a.first >= b.first;
  } else {
    return a.first < b.first;
  }
}

template <unsigned Dim, typename DistType, typename DataIterator,
          typename NeighborIterator>
void KnnExactRecurse(
    typename KDTree<typename std::iterator_traits<DataIterator>::value_type,
                    Dim, false>::Node const *const tree,
    const int treeIndex, const DataIterator data, const DataIterator query,
    const size_t k, BoundedHeap<NeighborIterator, true> &neighbors,
    std::function<DistType(DataIterator, DataIterator)> distFunc) {

  if (treeIndex < 0) {
    return;
  }

  const auto &node = tree[treeIndex];
  const auto dataIndex = node.index;
  const auto value = data + Dim * dataIndex;

  if (node.left < 0 && node.right < 0) {
    // This is a leaf
    neighbors.TryPush(std::make_pair(distFunc(query, value), dataIndex));
    return;
  }
  const auto leftChild = node.left;
  const auto rightChild = node.right;

  const auto splitDim = node.splitDim;
  const bool doLeft = query[splitDim] < value[splitDim];
  const auto bestChild = doLeft ? leftChild : rightChild;
  const auto otherChild = doLeft ? rightChild : leftChild;

  // Explore best branch first
  KnnExactRecurse<Dim, DistType, DataIterator>(tree, bestChild, data, query, k,
                                               neighbors, distFunc);

  // Now explore the other branch if necessary
  if (std::abs(value[splitDim] - query[splitDim]) < neighbors.Max().first ||
      neighbors.size() < k) {
    KnnExactRecurse<Dim, DistType, DataIterator>(tree, otherChild, data, query,
                                                 k, neighbors, distFunc);
  }
}

template <unsigned Dim, typename DistType, typename DataIterator,
          typename BranchIterator, typename NeighborIterator>
void KnnApproximateRecurse(
    typename KDTree<typename std::iterator_traits<DataIterator>::value_type,
                    Dim, true>::Node const *const tree,
    const int treeIndex, const DataIterator data, const DataIterator query,
    const int k, const int maxLeaves,
    std::function<DistType(DataIterator, DataIterator)> const &distFunc,
    BoundedHeap<BranchIterator, false> &branchesToCheck,
    DistType minDistToBoundary,
    std::vector<bool, tbb::scalable_allocator<bool>> &binsChecked,
    BoundedHeap<NeighborIterator, true> &neighbors, int &nSearched) {

  if (neighbors.isFull() &&
      (nSearched >= maxLeaves || minDistToBoundary >= neighbors.Max().first)) {
    return;
  }

  const auto &node = tree[treeIndex];
  const auto dataIndex = node.index;
  const auto value = data + Dim * dataIndex;

  // Check if this is a leaf node
  if (node.left < 0 && node.right < 0) {
    if (!binsChecked[dataIndex]) {
      neighbors.TryPush(std::make_pair(distFunc(query, value), dataIndex));
      binsChecked[dataIndex] = true;
      ++nSearched;
    }
    return;
  }
  const auto leftChild = node.left;
  const auto rightChild = node.right;

  const auto splitDim = node.splitDim;
  const bool doLeft = query[splitDim] < value[splitDim];
  const auto bestChild = doLeft ? leftChild : rightChild;
  const auto otherChild = doLeft ? rightChild : leftChild;
  const DistType minDistToBoundaryBranch =
      minDistToBoundary + std::abs(value[splitDim] - query[splitDim]);
  // Add the branch not taken to the list of potential branches by the
  // accumulated minimum distance to that branch's boundary
  branchesToCheck.TryPush(
      std::make_tuple(minDistToBoundaryBranch, tree, otherChild));
  // Now continue recurse the most promising branch
  KnnApproximateRecurse<Dim, DistType>(
      tree, bestChild, data, query, k, maxLeaves, distFunc, branchesToCheck,
      minDistToBoundary, binsChecked, neighbors, nSearched);
}

} // End anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <unsigned Dim, typename DistType, typename DataIterator,
          typename OutputIterator>
void KnnLinear(
    const DataIterator trainingBegin, const DataIterator trainingEnd,
    const DataIterator query, const int k,
    const std::function<DistType(DataIterator, DataIterator)> distFunc,
    const OutputIterator neighborsBegin) {
  const size_t nPoints = std::distance(trainingBegin, trainingEnd) / Dim;
  BoundedHeap<OutputIterator, true> neighbors(
      neighborsBegin, neighborsBegin + k, CompareNeighbors<DistType, int>);
  DataIterator currComp = trainingBegin;
  // Compare to all training points
  for (size_t i = 0; i < nPoints; ++i, currComp += Dim) {
    neighbors.TryPush(std::make_pair(distFunc(query, currComp), i));
  }
}

template <unsigned Dim, typename DistType, typename DataIterator>
std::vector<std::pair<DistType, int>,
            tbb::scalable_allocator<std::pair<DistType, int>>>
KnnLinear(const DataIterator trainingBegin, const DataIterator trainingEnd,
          const DataIterator query, const int k,
          const std::function<DistType(DataIterator, DataIterator)> distFunc) {
  std::vector<std::pair<DistType, int>,
              tbb::scalable_allocator<std::pair<DistType, int>>> neighbors(k);
  KnnLinear<Dim, DistType>(trainingBegin, trainingEnd, query, k, distFunc,
                           neighbors.begin());
  return neighbors;
}

template <unsigned Dim, typename DistType, typename DataIterator>
std::vector<std::pair<DistType, int>,
            tbb::scalable_allocator<std::pair<DistType, int>>>
KnnLinear(const DataIterator trainingBegin, const DataIterator trainingEnd,
          const DataIterator queryBegin, const DataIterator queryEnd,
          const int k,
          const std::function<DistType(DataIterator, DataIterator)> distFunc) {
  const int nQueries = std::distance(queryBegin, queryEnd) / Dim;
  std::vector<std::pair<DistType, int>,
              tbb::scalable_allocator<std::pair<DistType, int>>>
      neighbors(nQueries * k);
  tbb::parallel_for(0, nQueries, [&](int i) {
    KnnLinear<Dim, DistType>(trainingBegin, trainingEnd, queryBegin + i * Dim,
                             k, distFunc, neighbors.begin() + i * k);
  });
  return neighbors;
}
template <unsigned Dim, typename DistType, typename DataIterator,
          typename OutputIterator>
void KnnExact(
    KDTree<typename std::iterator_traits<DataIterator>::value_type, Dim,
           false> const &kdTree,
    const DataIterator data, const DataIterator query, const int k,
    const std::function<DistType(DataIterator, DataIterator)> distFunc,
    const OutputIterator outputBegin) {
  BoundedHeap<OutputIterator, true> neighbors(outputBegin, outputBegin + k,
                                              CompareNeighbors<DistType, int>);
  KnnExactRecurse<Dim, DistType>(kdTree.data(), 0, data, query, k, neighbors,
                                 distFunc);
}

template <unsigned Dim, typename DistType, typename DataIterator>
std::vector<std::pair<DistType, int>,
            tbb::scalable_allocator<std::pair<DistType, int>>>
KnnExact(KDTree<typename std::iterator_traits<DataIterator>::value_type, Dim,
                false> const &kdTree,
         const DataIterator data, const DataIterator query, const int k,
         const std::function<DistType(DataIterator, DataIterator)> distFunc) {
  std::vector<std::pair<DistType, int>,
              tbb::scalable_allocator<std::pair<DistType, int>>> neighbors(k);
  KnnExact(kdTree, data, query, k, distFunc, neighbors.begin());
  return neighbors;
}

template <unsigned Dim, typename DistType, typename DataIterator>
std::vector<std::pair<DistType, int>,
            tbb::scalable_allocator<std::pair<DistType, int>>>
KnnExact(KDTree<typename std::iterator_traits<DataIterator>::value_type, Dim,
                false> const &kdTree,
         const DataIterator data, const DataIterator queryBegin,
         const DataIterator queryEnd, const int k,
         const std::function<DistType(DataIterator, DataIterator)> distFunc) {
  const int nQueries = std::distance(queryBegin, queryEnd) / Dim;
  std::vector<std::pair<DistType, int>,
              tbb::scalable_allocator<std::pair<DistType, int>>>
      neighbors(k * nQueries);
  tbb::parallel_for(0, nQueries, [&](int i) {
    KnnExact(kdTree, data, queryBegin + i * Dim, k, distFunc,
             neighbors.begin() + i * k);
  });
  return neighbors;
}

template <unsigned Dim, typename DistType, typename DataIterator,
          typename OutputIterator>
std::vector<std::pair<DistType, int>,
            tbb::scalable_allocator<std::pair<DistType, int>>>
KnnApproximate(
    std::vector<KDTree<typename std::iterator_traits<DataIterator>::value_type,
                       Dim, true>> const &randTrees,
    const DataIterator data, const DataIterator query, const int k,
    const int maxLeaves,
    const std::function<DistType(DataIterator, DataIterator)> distFunc) {
  std::vector<std::pair<DistType, int>,
              tbb::scalable_allocator<std::pair<DistType, int>>> neighbors(k);
  KnnApproximate(randTrees, data, query, k, maxLeaves, distFunc,
                 neighbors.begin());
  return neighbors;
}

template <unsigned Dim, typename DistType, typename DataIterator,
          typename OutputIterator>
void KnnApproximate(
    std::vector<KDTree<typename std::iterator_traits<DataIterator>::value_type,
                       Dim, true>> const &randTrees,
    const DataIterator data, const DataIterator query, const int k,
    const int maxLeaves,
    const std::function<DistType(DataIterator, DataIterator)> distFunc,
    const OutputIterator neighborsBegin) {

  // A branch records the minimum distance from the point to any leaf in that
  // branch, as well as an iterator to the node entry in the tree
  using Tree =
      typename KDTree<typename std::iterator_traits<DataIterator>::value_type,
                      Dim, true>::Node const *;
  using Branch = std::tuple<DistType, Tree, int>;
  // Use negated comparison in heap so lowest distances are at the top.
  // When the heap is full, simply reject new values.
  std::vector<Branch, tbb::scalable_allocator<Branch>> branchVec(maxLeaves);
  BoundedHeap<typename decltype(branchVec)::iterator, false> branchesToCheck(
      branchVec.begin(), branchVec.end(), [](Branch const &a, Branch const &b) {
        return std::get<0>(a) >= std::get<0>(b);
      });
  // Keep track of leaf nodes already inspected
  std::vector<bool, tbb::scalable_allocator<bool>> binsChecked(
      randTrees[0].nLeaves(), false);
  // Heap of nearest neighbors. Top of the heap will be the highest distance
  // found, but will keep track of the k lowest distances found.
  // An entry is the distance to the neighbor and the neighbor's index in the
  // training data set.
  BoundedHeap<OutputIterator, true> neighbors(
      neighborsBegin, neighborsBegin + k, CompareNeighbors<DistType, int>);
  // Count number of leaves searched as termination criterion
  int nSearched = 0;

  // Recurse to the bottom of each randomized tree once to find the best bins in
  // each, as well as the most promising branches to search next
  for (auto &randTree : randTrees) {
    KnnApproximateRecurse<Dim, DistType>(randTree.data(), 0, data, query, k,
                                         maxLeaves, distFunc, branchesToCheck,
                                         0, binsChecked, neighbors, nSearched);
  }

  // Now search throughout the proposed branches until search is exhausted or
  // the maximum number of leaves is reached
  Branch branchToSearch;
  while (nSearched < maxLeaves && branchesToCheck.TryPopMax(branchToSearch)) {
    KnnApproximateRecurse<Dim, DistType>(
        std::get<1>(branchToSearch), std::get<2>(branchToSearch), data, query,
        k, maxLeaves, distFunc, branchesToCheck, std::get<0>(branchToSearch),
        binsChecked, neighbors, nSearched);
  }
}

template <unsigned Dim, typename DistType, typename DataIterator>
std::vector<std::pair<DistType, int>,
            tbb::scalable_allocator<std::pair<DistType, int>>>
KnnApproximate(
    std::vector<KDTree<typename std::iterator_traits<DataIterator>::value_type,
                       Dim, true>> const &randTrees,
    const DataIterator data, const DataIterator queryBegin,
    const DataIterator queryEnd, const int k, const int maxLeaves,
    std::function<DistType(DataIterator, DataIterator)> distFunc) {
  const int nQueries = std::distance(queryBegin, queryEnd) / Dim;
  std::vector<std::pair<DistType, int>,
              tbb::scalable_allocator<std::pair<DistType, int>>>
      neighbors(k * nQueries);
  tbb::parallel_for(0, nQueries, [&](int i) {
    KnnApproximate(randTrees, data, queryBegin + i * Dim, k, maxLeaves,
                   distFunc, neighbors.begin() + i * k);
  });
  return neighbors;
}

} // End namespace knn
