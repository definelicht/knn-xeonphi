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
// Declarations
////////////////////////////////////////////////////////////////////////////////

/// \brief O(n^2) KNN for a single query.
template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnLinear(DataContainer<DataType> const &trainingPoints, int k,
          DataItr<DataType>query,
          std::function<DistType(DataItr<DataType>,
                                 DataItr<DataType>)> distFunc);

/// \brief O(N) KNN for multiple queries, accelerated using OpenMP if
/// available.
template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnLinear(DataContainer<DataType> const &trainingPoints, int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType>,
                                 DataItr<DataType>)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a single query.
template <size_t Dim, bool Randomized, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnExact(KDTree<Dim, Randomized, DataType> const &kdTree, int k,
         DataItr<DataType>query,
         std::function<DistType(DataItr<DataType>,
                                DataItr<DataType>)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a multiple query.
/// Accelerated using OpenMP if available.
template <size_t Dim, bool Randomized, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnExact(KDTree<Dim, Randomized, DataType> const &kdTree, int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType>,
                                DataItr<DataType>)> distFunc);

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnApproximate(std::vector<KDTree<Dim, true, DataType>> const &randTrees, int k,
               int maxLeaves, DataItr<DataType>query,
               std::function<DistType(DataItr<DataType>,
                                      DataItr<DataType>)> distFunc);

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnApproximate(std::vector<KDTree<Dim, true, DataType>> const &randTrees, int k,
               int maxLeaves, DataContainer<DataType> const &queries,
               std::function<DistType(DataItr<DataType>,
                                      DataItr<DataType>)> distFunc);

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

template <size_t Dim, bool Randomized, typename DistType, typename DataType>
void KnnExactRecurse(
    typename KDTree<Dim, Randomized, DataType>::NodeItr node, const size_t k,
    DataItr<DataType> query,
    BoundedHeap<std::pair<DistType, size_t>, true> &neighbors,
    std::function<DistType(DataItr<DataType>, DataItr<DataType>)> distFunc) {

  const auto value = node.value();

  const auto leftChild = node.Left();
  const auto rightChild = node.Right();
  if (!leftChild.inBounds() && !rightChild.inBounds()) {
    // This is a leaf
    neighbors.TryPush(std::make_pair(distFunc(query, value), node.index()));
    return;
  }

  const auto splitDim = node.splitDim();
  const bool doLeft = query[splitDim] < value[splitDim];
  const auto &bestChild = doLeft ? leftChild : rightChild;
  const auto &otherChild = doLeft ? rightChild : leftChild;

  // Explore best branch first
  KnnExactRecurse<Dim, Randomized, DistType, DataType>(bestChild, k, query,
                                                       neighbors, distFunc);

  // Now explore the other branch if necessary
  if (std::abs(value[splitDim] - query[splitDim]) < neighbors.Max().first ||
      neighbors.size() < k) {
    KnnExactRecurse<Dim, Randomized, DistType, DataType>(otherChild, k, query,
                                                         neighbors, distFunc);
  }
}

template <size_t Dim, typename DistType, typename DataType>
void KnnApproximateRecurse(
    typename KDTree<Dim, true, DataType>::NodeItr const &node, const int k,
    const int maxLeaves, DataItr<DataType>query,
    std::function<DistType(DataItr<DataType>,
                           DataItr<DataType>)> const &distFunc,
    BoundedHeap<
        std::pair<DistType, typename KDTree<Dim, true, DataType>::NodeItr>,
        false> &branchesToCheck,
    DistType minDistToBoundary, std::vector<bool> &binsChecked,
    BoundedHeap<std::pair<DistType, size_t>, true> &neighbors, int &nSearched) {

  if (neighbors.isFull() &&
      (nSearched >= maxLeaves || minDistToBoundary >= neighbors.Max().first)) {
    return;
  }

  const auto value = node.value();

  // Check if this is a leaf node
  const auto leftChild = node.Left();
  const auto rightChild = node.Right();
  if (!leftChild.inBounds() && !rightChild.inBounds()) {
    const auto index = node.index();
    if (!binsChecked[index]) {
      neighbors.TryPush(std::make_pair(distFunc(query, value), index));
      binsChecked[index] = true;
      ++nSearched;
    }
    return;
  }

  const size_t splitDim = node.splitDim();
  const bool doLeft = query[splitDim] < value[splitDim];
  const auto &bestChild = doLeft ? leftChild : rightChild;
  const auto &otherChild = doLeft ? rightChild : leftChild;
  const DistType minDistToBoundaryBranch =
      minDistToBoundary + std::abs(value[splitDim] - query[splitDim]);
  // Add the branch not taken to the list of potential branches by the
  // accumulated minimum distance to that branch's boundary
  branchesToCheck.TryPush(std::make_pair(minDistToBoundaryBranch, otherChild));
  // Now continue recurse the most promising branch
  KnnApproximateRecurse<Dim, DistType, DataType>(
      bestChild, k, maxLeaves, query, distFunc, branchesToCheck,
      minDistToBoundary, binsChecked, neighbors, nSearched);
}

} // End anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>> KnnLinear(
    DataContainer<DataType> const &trainingPoints, const int k,
    DataItr<DataType> query,
    std::function<DistType(DataItr<DataType>, DataItr<DataType>)> distFunc) {
  const size_t nPoints = trainingPoints.size() / Dim;
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(
      k, CompareNeighbors<DistType, size_t>);
  const DataItr<DataType> begin = trainingPoints.data();
  DataItr<DataType> currComp = begin;
  // Compare to all training points
  for (size_t i = 0; i < nPoints; ++i, currComp += Dim) {
    neighbors.TryPush(std::make_pair(distFunc(query, currComp), i));
  }
  auto heapContent = neighbors.Destroy();
  return heapContent;
}

template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>> KnnLinear(
    DataContainer<DataType> const &trainingPoints, const int k,
    DataContainer<DataType> const &queries,
    std::function<DistType(DataItr<DataType>, DataItr<DataType>)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<std::pair<DistType, size_t>>> neighbors(nQueries);
  tbb::parallel_for(0, nQueries, [&](int i) {
    neighbors[i] = KnnLinear<Dim, DistType, DataType>(
        trainingPoints, k, queries.data() + i * Dim, distFunc);
  });
  return neighbors;
}

template <size_t Dim, bool Randomized, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnExact(KDTree<Dim, Randomized, DataType> const &kdTree, const int k,
         DataItr<DataType>query,
         std::function<DistType(DataItr<DataType>,
                                DataItr<DataType>)> distFunc) {
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(
      k, CompareNeighbors<DistType, size_t>);
  KnnExactRecurse<Dim, Randomized, DistType, DataType>(kdTree.Root(), k, query,
                                                       neighbors, distFunc);
  auto heapContent = neighbors.Destroy();
  return heapContent;
}

template <size_t Dim, bool Randomized, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnExact(KDTree<Dim, Randomized, DataType> const &kdTree, const int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType>,
                                DataItr<DataType>)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<std::pair<DistType, size_t>>> neighbors(nQueries);
  tbb::parallel_for(0, nQueries, [&](int i) {
    neighbors[i] = KnnExact(kdTree, k, queries.data() + i * Dim, distFunc);
  });
  return neighbors;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnApproximate(std::vector<KDTree<Dim, true, DataType>> const &randTrees,
               const int k, const int maxLeaves, DataItr<DataType>query,
               std::function<DistType(DataItr<DataType>,
                                      DataItr<DataType>)> distFunc) {

  // Iterator to nodes in the tree 
  using TreeItr = typename KDTree<Dim, true, DataType>::NodeItr;
  // A branch records the minimum distance from the point to any leaf in that
  // branch, as well as an iterator to the node entry in the tree
  using Branch = std::pair<DistType, TreeItr>;
  // Use negated comparison in heap so lowest distances are at the top.
  // When the heap is full, simply reject new values.
  BoundedHeap<Branch, false> branchesToCheck(
      maxLeaves, CompareNeighbors<DistType, TreeItr, true>);
  // Keep track of leaf nodes already inspected
  std::vector<bool> binsChecked(randTrees[0].nLeaves(), false);
  // Heap of nearest neighbors. Top of the heap will be the highest distance
  // found, but will keep track of the k lowest distances found.
  // An entry is the distance to the neighbor and the neighbor's index in the
  // training data set.
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(k);
  // Count number of leaves searched as termination criterion
  int nSearched = 0;

  // Recurse to the bottom of each randomized tree once to find the best bins in
  // each, as well as the most promising branches to search next
  for (auto &randTree : randTrees) {
    KnnApproximateRecurse<Dim, DistType, DataType>(
        randTree.Root(), k, maxLeaves, query, distFunc, branchesToCheck, 0,
        binsChecked, neighbors, nSearched);
  }

  // Now search throughout the proposed branches until search is exhausted or
  // the maximum number of leaves is reached
  auto branchToSearch =
      std::make_pair<float, typename KDTree<Dim, true, DataType>::NodeItr>(
          0, randTrees[0].Root());
  while (nSearched < maxLeaves && branchesToCheck.TryPopMax(branchToSearch)) {
    KnnApproximateRecurse<Dim, DistType, DataType>(
        branchToSearch.second, k, maxLeaves, query, distFunc, branchesToCheck,
        branchToSearch.first, binsChecked, neighbors, nSearched);
  }
  
  auto heapContent = neighbors.Destroy();
  return heapContent;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>> KnnApproximate(
    std::vector<KDTree<Dim, true, DataType>> const &randTrees, const int k,
    const int maxLeaves, DataContainer<DataType> const &queries,
    std::function<DistType(DataItr<DataType>, DataItr<DataType>)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<std::pair<DistType, size_t>>> neighbors(nQueries);
  tbb::parallel_for(0, nQueries, [&](int i) {
    neighbors[i] = KnnApproximate(randTrees, k, maxLeaves,
                                  queries.data() + i * Dim, distFunc);
  });
  return neighbors;
}

} // End namespace knn
