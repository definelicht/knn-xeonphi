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
std::vector<size_t>
KnnLinear(DataContainer<DataType> const &trainingPoints, int k,
          DataItr<DataType> const &query,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc);

/// \brief O(N) KNN for multiple queries, accelerated using OpenMP if
/// available.
template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::vector<size_t>>
KnnLinear(DataContainer<DataType> const &trainingPoints, int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a single query.
template <size_t Dim, typename DistType, typename DataType>
std::vector<size_t>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, int k,
         DataItr<DataType> const &query,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a multiple query.
/// Accelerated using OpenMP if available.
template <size_t Dim, typename DistType, typename DataType>
std::vector<std::vector<size_t>>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc);

template <size_t Dim, typename DistType, typename DataType>
std::vector<size_t>
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

  auto value = node.value();
  neighbors.TryPush(
      std::make_pair(distFunc(query, value), node.index()));

  auto traverseSubtree = [&](bool doLeft) {
    if (doLeft) {
      const auto left = node.Left();
      if (left.inBounds()) {
        KnnExactRecurse<Dim, DistType, DataType>(left, k, query, neighbors,
                                                 distFunc);
      }
    } else {
      const auto right = node.Right();
      if (right.inBounds()) {
        KnnExactRecurse<Dim, DistType, DataType>(right, k, query, neighbors,
                                                 distFunc);
      }
    }
  };

  // Recurse the subtree with the highest intersection
  const size_t splitDim = node.splitDim();
  const bool doLeft = query[splitDim] < value[splitDim];
  traverseSubtree(doLeft);

  // If the longest nearest neighbor hypersphere crosses the splitting
  // hyperplane after traversing the subtree with the highest intersection, we
  if (std::abs(value[splitDim] - query[splitDim]) <
          neighbors.PeekFront().first ||
      neighbors.size() < k) {
    traverseSubtree(!doLeft);
  }
}

// template <size_t Dim, typename DistType, typename DataType, typename LabelType>
// std::vector<LabelType> KnnApproximateRecurse(
//     typename KDTree<Dim, true, DataType, LabelType>::NodeItr const &root,
//     const int k, const int maxLeaves, DataItr<DataType> const &query,
//     std::function<DistType(DataItr<DataType> const &,
//                            DataItr<DataType> const &)> distFunc,
//     BoundedHeap<true, std::pair<DistType, typename KDTree<Dim, true, DataType,
//                                                           LabelType>::NodeItr>>
//         &bestBins,
//     std::vector<bool> &binsChecked, BoundedHeap<DistType, LabelType> &neighbors,
//     int &leavesSearched) {
// 
//   // Check if this branch has already been searched
//   if (checked[std::distance(
//   
// }

} // End anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <unsigned Dim, typename DistType, typename DataType>
std::vector<size_t>
KnnLinear(DataContainer<DataType> const &trainingPoints, const int k,
          DataItr<DataType> const &query,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc) {
  const size_t nPoints = trainingPoints.size()/Dim;
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(
      k, CompareNeighbors<DistType, size_t>);
  const DataItr<DataType> begin = trainingPoints.cbegin();
  DataItr<DataType> currComp = begin;
  // Compare to all training points
  for (size_t i = 0; i < nPoints; ++i, currComp += Dim) {
    neighbors.TryPush(std::make_pair(distFunc(query, currComp), i));
  }
  auto heapContent = neighbors.Destroy();
  // Sort neighbors by closest distance
  std::sort_heap(heapContent.begin(), heapContent.end(),
                 CompareNeighbors<DistType, size_t>);
  std::vector<size_t> neighborIndices(k);
  for (int i = 0; i < k; ++i) {
    neighborIndices[i] = heapContent[i].second;
  }
  return neighborIndices;
}

template <unsigned Dim, typename DistType, typename DataType>
std::vector<std::vector<size_t>>
KnnLinear(DataContainer<DataType> const &trainingPoints, const int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<size_t>> neighbors(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    neighbors[i] =
        KnnLinear<Dim>(trainingPoints, k, queries.cbegin() + i * Dim, distFunc);
  }
  return neighbors;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<size_t>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, const int k,
         DataItr<DataType> const &query,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc) {
  BoundedHeap<std::pair<DistType, size_t>, true> neighbors(
      k, CompareNeighbors<DistType, size_t>);
  KnnExactRecurse<Dim, DistType, DataType>(kdTree.Root(), k, query, neighbors,
                                           distFunc);
  // Sort according to lowest distance
  auto heapContent = neighbors.Destroy();
  std::sort_heap(heapContent.begin(), heapContent.end(),
                 CompareNeighbors<DistType, size_t>);
  std::vector<size_t> neighborIndices(k);
  for (int i = 0; i < k; ++i) {
    neighborIndices[i] = heapContent[i].second;
  }
  return neighborIndices;
}

template <size_t Dim, typename DistType, typename DataType>
std::vector<std::vector<size_t>>
KnnExact(KDTree<Dim, false, DataType> const &kdTree, const int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<size_t>> neighbors(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    neighbors[i] = KnnExact(kdTree, k, queries.cbegin() + i * Dim, distFunc);
  }
  return neighbors;
}

// template <size_t Dim, typename DistType, typename DataType, typename LabelType>
// std::vector<LabelType> KnnApproximate(
//     std::vector<KDTree<Dim, true, DataType, LabelType>> const &randTrees,
//     const int k, const int maxLeaves, DataItr<DataType> const &query,
//     std::function<DistType(DataItr<DataType> const &,
//                            DataItr<DataType> const &)> distFunc) {
// 
//   // Initialization
//   BoundedHeap<false, std::pair<DistType, typename KDTree<Dim, true, DataType,
//                                                          LabelType>::NodeItr>>
//       bestBins(maxLeaves,
//                CompareNeighbors<DistType, typename KDTree<Dim, true, DataType,
//                                                           LabelType>::NodeItr>);
//   std::vector<bool> checked(randTrees[0].size(), true); 
//   BoundedHeap<true, std::pair<DistType, LabelType>> neighbors(k);
//   int leavesSearched = 0;
// 
//   // Recurse to the bottom of each tree once
//   for (int i = 0, iMax = randTrees.size(); i < iMax; ++i) {
//     KnnApproximateRecurse<Dim, DistType>(randTrees, k, maxLeaves, query,
//                                          distFunc, bestBins, checked, neighbors,
//                                          leavesSearched);
//   }
// }

} // End namespace knn
