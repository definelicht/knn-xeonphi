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
template <unsigned Dim, typename DistType, typename DataType,
          typename LabelType>
std::vector<LabelType>
KnnLinear(DataContainer<DataType> const &trainingPoints,
          DataContainer<LabelType> const &trainingLabels, const int k,
          DataItr<DataType> const &query,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc);

/// \brief O(N) classification using KNN for a single query.
template <unsigned Dim, typename DistType, typename DataType,
          typename LabelType>
LabelType KnnLinearClassify(
    DataContainer<DataType> const &trainingPoints,
    DataContainer<LabelType> const &trainingLabels, const int k,
    DataItr<DataType> const &query,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc);

/// \brief O(N) KNN for multiple queries, accelerated using OpenMP if
/// available.
template <unsigned Dim, typename DistType, typename DataType, typename LabelType>
std::vector<std::vector<LabelType>>
KnnLinear(DataContainer<DataType> const &trainingPoints,
          DataContainer<LabelType> const &trainingLabels, const int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc);

/// \brief O(N) classification using KNN for multiple queries, accelerated
/// using OpenMP if available.
template <unsigned Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType> KnnLinearClassify(
    DataContainer<DataType> const &trainingPoints,
    DataContainer<LabelType> const &trainingLabels, const int k,
    DataContainer<DataType> const &queries,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a single query.
template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType>
KnnExact(KDTree<Dim, false, DataType, LabelType> const &kdTree, const int k,
         DataItr<DataType> const &query,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc);

/// \brief Classification using exact KNN using a kd-tree for search for a
/// single query.
template <size_t Dim, typename DistType, typename DataType, typename LabelType>
LabelType KnnClassifyExact(
    KDTree<Dim, false, DataType, LabelType> const &kdTree, const int k,
    DataItr<DataType> const &query,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc);

/// \brief Exact KNN using a kd-tree for search for a multiple query.
/// Accelerated using OpenMP if available.
template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<std::vector<LabelType>>
KnnExact(KDTree<Dim, false, DataType, LabelType> const &kdTree, const int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc);

/// \brief Classification using exact KNN using a kd-tree for search for
/// multiple queries. Accelerated using OpenMP if available.
template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType> KnnClassifyExact(
    KDTree<Dim, false, DataType, LabelType> const &kdTree, const int k,
    DataContainer<DataType> &queries,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc);

template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType> KnnApproximate(
    std::vector<KDTree<Dim, true, DataType, LabelType>> const &randTrees,
    const int k, DataItr<DataType> const &query,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc);

////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename DistType, typename LabelType>
bool CompareNeighbors(std::pair<DistType, LabelType> const &a,
                      std::pair<DistType, LabelType> const &b) {
  return a.first < b.first;
}

template <size_t Dim, typename DistType, typename DataType, typename LabelType>
void KnnExactRecurse(
    typename KDTree<Dim, false, DataType, LabelType>::NodeItr const &node,
    const size_t k, DataItr<DataType> const &query,
    BoundedHeap<std::pair<DistType, LabelType>, true> &neighbors,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc) {

  neighbors.TryPush(
      std::make_pair(distFunc(query, node.value()), *node.label()));

  auto traverseSubtree = [&](bool doLeft) {
    if (doLeft) {
      const auto left = node.Left();
      if (left.inBounds()) {
        KnnExactRecurse<Dim, DistType, DataType, LabelType>(
            left, k, query, neighbors, distFunc);
      }
    } else {
      const auto right = node.Right();
      if (right.inBounds()) {
        KnnExactRecurse<Dim, DistType, DataType, LabelType>(
            right, k, query, neighbors, distFunc);
      }
    }
  };

  // Recurse the subtree with the highest intersection
  const size_t splitDim = node.splitDim();
  const bool doLeft = query[splitDim] < node.value()[splitDim];
  traverseSubtree(doLeft);

  // If the longest nearest neighbor hypersphere crosses the splitting
  // hyperplane after traversing the subtree with the highest intersection, we
  if (std::abs(node.value()[splitDim] - query[splitDim]) <
          neighbors.PeekFront().first ||
      neighbors.size() < k) {
    traverseSubtree(!doLeft);
  }
}

} // End anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <unsigned Dim, typename DistType, typename DataType,
          typename LabelType>
std::vector<LabelType>
KnnLinear(DataContainer<DataType> const &trainingPoints,
          DataContainer<LabelType> const &trainingLabels, const int k,
          DataItr<DataType> const &query,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc) {
  auto sizeDiv =
      std::div(static_cast<int>(trainingPoints.size()), static_cast<int>(Dim));
  assert(sizeDiv.rem == 0);
  const int nPoints = sizeDiv.quot;
  assert(k <= nPoints);
  BoundedHeap<std::pair<DistType, LabelType>, true> neighbors(
      k, CompareNeighbors<DistType, LabelType>);
  DataItr<DataType> currComp = trainingPoints.cbegin();
  // Compare to all training points
  for (int i = 0; i < nPoints; ++i, currComp += Dim) {
    neighbors.TryPush(
        std::make_pair(distFunc(query, currComp), trainingLabels[i]));
  }
  auto heapContent = neighbors.Destroy();
  // Sort neighbors by closest distance
  std::sort_heap(heapContent.begin(), heapContent.end(),
                 CompareNeighbors<DistType, LabelType>);
  std::vector<LabelType> neighborLabels(k);
  for (int i = 0; i < k; ++i) {
    neighborLabels[i] = heapContent[i].second;
  }
  return neighborLabels;
}

template <unsigned Dim, typename DistType, typename DataType,
          typename LabelType>
LabelType KnnLinearClassify(
    DataContainer<DataType> const &trainingPoints,
    DataContainer<LabelType> const &trainingLabels, const int k,
    DataItr<DataType> const &query,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc) {
  auto neighbors =
      KnnLinear(trainingPoints, trainingLabels, k, query, distFunc);
  std::unordered_map<LabelType, int> count;
  for (auto &label : neighbors) {
    ++count[label];
  }
  LabelType classification =
      std::max_element(count.cbegin(), count.cend(),
                       [](std::pair<LabelType, int> const &a,
                          std::pair<LabelType, int> const &b) {
                         return a.second < b.second;
                       })
          ->first;
  return classification;
}

template <unsigned Dim, typename DistType, typename DataType, typename LabelType>
std::vector<std::vector<LabelType>>
KnnLinear(DataContainer<DataType> const &trainingPoints,
          DataContainer<LabelType> const &trainingLabels, const int k,
          DataContainer<DataType> const &queries,
          std::function<DistType(DataItr<DataType> const &,
                                 DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<LabelType>> neighbors(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    neighbors[i] = KnnLinear<Dim>(trainingPoints, trainingLabels, k,
                                  queries.cbegin() + i * Dim, distFunc);
  }
  return neighbors;
}

template <unsigned Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType> KnnLinearClassify(
    DataContainer<DataType> const &trainingPoints,
    DataContainer<LabelType> const &trainingLabels, const int k,
    DataContainer<DataType> const &queries,
    std::function<DistType(DataItr<DataType> const &,
                           DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<LabelType> classification(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    classification[i] = KnnLinearClassify(trainingPoints, trainingLabels, k,
                                          queries.cbegin() + i * Dim, distFunc);
  }
  return classification;
}

template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType>
KnnExact(KDTree<Dim, false, DataType, LabelType> const &kdTree, const int k,
         DataItr<DataType> const &query,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc) {
  BoundedHeap<std::pair<DistType, LabelType>, true> neighbors(
      k, CompareNeighbors<DistType, LabelType>);
  KnnExactRecurse<Dim, DistType, DataType, LabelType>(kdTree.Root(), k, query,
                                                      neighbors, distFunc);
  // Sort according to lowest distance
  auto heapContent = neighbors.Destroy();
  std::sort_heap(heapContent.begin(), heapContent.end(),
                 CompareNeighbors<DistType, LabelType>);
  std::vector<LabelType> neighborLabels(k);
  for (int i = 0; i < k; ++i) {
    neighborLabels[i] = heapContent[i].second;
  }
  return neighborLabels;
}

template <size_t Dim, typename DistType, typename DataType, typename LabelType>
LabelType
KnnClassifyExact(KDTree<Dim, false, DataType, LabelType> const &kdTree,
                 const int k, DataItr<DataType> const &query,
                 std::function<DistType(DataItr<DataType> const &,
                                        DataItr<DataType> const &)> distFunc) {
  auto neighbors = KnnExact(kdTree, k, query, distFunc);
  std::unordered_map<LabelType, int> count;
  for (auto &label : neighbors) {
    ++count[label];
  }
  LabelType classification =
      std::max_element(count.cbegin(), count.cend(),
                       [](std::pair<LabelType, int> const &a,
                          std::pair<LabelType, int> const &b) {
                         return a.second < b.second;
                       })
          ->first;
  return classification;
}

template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<std::vector<LabelType>>
KnnExact(KDTree<Dim, false, DataType, LabelType> const &kdTree, const int k,
         DataContainer<DataType> const &queries,
         std::function<DistType(DataItr<DataType> const &,
                                DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<LabelType>> labels(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    labels[i] = KnnExact(kdTree, k, queries.cbegin() + i * Dim, distFunc);
  }
  return labels;
}

template <size_t Dim, typename DistType, typename DataType, typename LabelType>
std::vector<LabelType>
KnnClassifyExact(KDTree<Dim, false, DataType, LabelType> const &kdTree,
                 const int k, DataContainer<DataType> &queries,
                 std::function<DistType(DataItr<DataType> const &,
                                        DataItr<DataType> const &)> distFunc) {
  const int nQueries = queries.size() / Dim;
  std::vector<LabelType> labels(nQueries);
#pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    labels[i] = KnnClassify(kdTree, k, queries.cbegin() + i * Dim, distFunc);
  }
  return labels;
}

} // End namespace knn
