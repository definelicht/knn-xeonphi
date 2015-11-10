#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <unordered_map>
#include <vector>
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
          DataItr<DataType> const &query,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc);

/// \brief O(N) KNN for multiple queries, accelerated using OpenMP if
/// available.
template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnLinear(DataContainer<DataType> const &trainingPoints, int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a single query.
template <size_t Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, int k,
         DataItr<DataType> const &query,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a multiple query.
/// Accelerated using OpenMP if available.
template <size_t Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc);

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnApproximate(std::vector<KDTree<Dim, true, DataType>> const &randTrees, int k,
               int maxLeaves, DataItr<DataType> const &query,
               std::function<DistType(DataItr<DataType> const &,
                                      DataItr<DataType> const &)> distFunc);

////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename DistType, typename T>
bool CompareNeighbors(std::pair<DistType, T> const &a,
                      std::pair<DistType, T> const &b) {
  return a.first < b.first;
}

template <size_t Dim, typename DistType, typename DataType>
void KnnExactRecurse(
    typename KDTree<Dim, false, DataType>::NodeItr const &node, const size_t k,
    DataItr<DataType> const &query,
    BoundedHeap<std::pair<DistType, size_t>, true> &neighbors,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc) {

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
  KnnExactRecurse<Dim, DistType, DataType>(bestChild, k, query, neighbors,
                                           distFunc);

  // Now explore the other branch if necessary
  if (std::abs(value[splitDim] - query[splitDim]) <
          neighbors.PeekFront().first ||
      neighbors.size() < k) {
    KnnExactRecurse<Dim, DistType, DataType>(otherChild, k, query, neighbors,
                                             distFunc);
  }
}

template <size_t Dim, typename DistType, typename DataType>
void KnnApproximateRecurse(
    typename KDTree<Dim, true, DataType>::NodeItr const &node, const int k,
    const int maxLeaves, DataItr<DataType> const &query,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc,
    BoundedHeap<
        std::pair<DistType, typename KDTree<Dim, true, DataType>::NodeItr>,
        true> &bestBins,
    std::vector<bool> &binsChecked,
    BoundedHeap<std::pair<DistType, size_t>, true> &neighbors,
    int &nSearched) {

  if (nSearched >= maxLeaves) {
    return;
  }

  // Extract node content
  const auto value = node.value();
  const auto index = node.index();

  // Check if this is a leaf node
  const auto leftChild = node.Left();
  const auto rightChild = node.Right();
  if (!leftChild.inBounds() && !rightChild.inBounds()) {
    if (!binsChecked[index]) {
      neighbors.TryPush(distFunc(query, value), index);
      binsChecked[index] = true;
      ++nSearched;
    }
    return;
  }
  // Recurse the best child and add the other child to the priority queue by its
  // distance in the split dimension
  const size_t splitDim = node.splitDim();
  const bool doLeft = query[splitDim] < value[splitDim];
  const auto &bestChild = doLeft ? leftChild : rightChild;
  const auto &otherChild = doLeft ? rightChild : leftChild;
  if (otherChild.inBounds()) {
    bestBins.TryPush(std::make_pair(std::abs(value[splitDim] - query[splitDim]),
                                    otherChild));
  }
  if (bestChild.inBounds()) {
    KnnApproximateRecurse<Dim, DistType, DataType>(
        bestChild, k, maxLeaves, query, distFunc, bestBins, binsChecked,
        neighbors, nSearched);
  }

}

} // End anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnLinear(DataContainer<DataType> const &trainingPoints, const int k,
          DataItr<DataType> const &query,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc) {
  const size_t nPoints = trainingPoints.size() / Dim;
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(
      k, CompareNeighbors<DistType, size_t>);
  const DataItr<DataType> begin = trainingPoints.cbegin();
  DataItr<DataType> currComp = begin;
  // Compare to all training points
  for (size_t i = 0; i < nPoints; ++i, currComp += Dim) {
    neighbors.TryPush(std::make_pair(distFunc(query, currComp), i));
  }
  auto heapContent = neighbors.Destroy();
  return heapContent;
}

template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnLinear(DataContainer<DataType> const &trainingPoints, const int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<std::pair<DistType, size_t>>> neighbors(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    neighbors[i] =
        KnnLinear<Dim>(trainingPoints, k, queries.cbegin() + i * Dim, distFunc);
  }
  return neighbors;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, const int k,
         DataItr<DataType> const &query,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc) {
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(
      k, CompareNeighbors<DistType, size_t>);
  KnnExactRecurse<Dim, DistType, DataType>(kdTree.Root(), k, query, neighbors,
                                           distFunc);
  auto heapContent = neighbors.Destroy();
  return heapContent;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::vector<std::pair<DistType, size_t>>>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, const int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<std::pair<DistType, size_t>>> neighbors(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    neighbors[i] = KnnExact(kdTree, k, queries.cbegin() + i * Dim, distFunc);
  }
  return neighbors;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::pair<DistType, size_t>>
KnnApproximate(std::vector<KDTree<Dim, true, DataType>> const &randTrees,
               const int k, const int maxLeaves, DataItr<DataType> const &query,
               std::function<DistType(DataItr<DataType> const &,
                                      DataItr<DataType> const &)> distFunc) {

  // Initialization
  BoundedHeap<
      std::pair<DistType, typename KDTree<Dim, true, DataType>::NodeItr>, false>
      bestBins(maxLeaves,
               CompareNeighbors<DistType,
                                typename KDTree<Dim, true, DataType>::NodeItr>);
  // Keep track of leaf nodes already inspected
  std::vector<bool> binsChecked(randTrees[0].nLeaves(), true);
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(k);
  int nSearched = 0;

  // Recurse to the bottom of each tree once
  for (auto &randTree : randTrees) {
    KnnApproximateRecurse<Dim, DistType>(randTree.Root(), k, maxLeaves, query,
                                         distFunc, bestBins, binsChecked,
                                         neighbors, nSearched);
  }

  // Now search throughout the proposed branches until search is exhausted or
  // the maximum number of leaves is reached
  auto branchToSearch = randTrees[0].Root(); 
  while (nSearched < maxLeaves && bestBins.TryPop(branchToSearch)) {
    KnnApproximateRecurse<Dim, DistType>(branchToSearch, k, maxLeaves, query,
                                         distFunc, bestBins, binsChecked,
                                         neighbors, nSearched);
  }

  return neighbors;
}

} // End namespace knn
