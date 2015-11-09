#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace knn {

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
class KnnLinear {

public:
  using DataItr = typename std::vector<DataType>::const_iterator;

  KnnLinear(std::vector<DataType> const &points,
            std::vector<LabelType> const &labels,
            std::function<DistType(DataItr const &, DataItr const &)> const
                &distFunc);

  std::vector<LabelType> Knn(int k, DataItr const &query) const;

  std::vector<std::vector<LabelType>>
  Knn(int k, std::vector<DataType> const &queries) const;

  LabelType KnnClassify(int k, DataItr const &query) const;

  std::vector<LabelType>
  KnnClassify(int k, std::vector<DataType> const &queries) const;

private:
  static bool CompareNeighbors(std::pair<DistType, LabelType> const &a,
                               std::pair<DistType, LabelType> const &b);

  std::vector<DataType> const &points_;
  std::vector<LabelType> const &labels_;
  const int nPoints_;
  std::function<DistType(DataItr const &, DataItr const &)> distFunc_;
};

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
KnnLinear<DataType, LabelType, DistType, Dim>::KnnLinear(
    std::vector<DataType> const &points, std::vector<LabelType> const &labels,
    std::function<DistType(DataItr const &, DataItr const &)> const &distFunc)
    : points_(points), labels_(labels), nPoints_(points_.size() / Dim),
      distFunc_(distFunc) {
  assert(points_.size() % Dim == 0);
}

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
std::vector<LabelType>
KnnLinear<DataType, LabelType, DistType, Dim>::Knn(const int k,
                                                   DataItr const &query) const {
  assert(k <= static_cast<int>(nPoints_));
  std::vector<std::pair<DistType, LabelType>> neighbors(k);
  DataItr currComp = points_.cbegin();
  // Initialize closest neighbors with the first k entries
  for (int i = 0; i < k; ++i, currComp += Dim) {
    neighbors[i] = std::make_pair(distFunc_(query, currComp), labels_[i]);
  }
  // The maximum element will be replaced when a better distance is found
  int maxElement = std::distance(
      neighbors.cbegin(),
      std::max_element(neighbors.cbegin(), neighbors.cend(), CompareNeighbors));
  // Scan all other points
  for (int i = k; i < nPoints_; ++i, currComp += Dim) {
    DistType dist = distFunc_(query, currComp);
    if (dist >= neighbors[maxElement].first) {
      continue;
    }
    neighbors[maxElement] = std::make_pair(dist, labels_[i]);
    maxElement =
        std::distance(neighbors.cbegin(),
                      std::max_element(neighbors.cbegin(), neighbors.cend(),
                                       CompareNeighbors));
  }
  // Sort neighbors by closest distance
  std::sort(neighbors.begin(), neighbors.end(), CompareNeighbors);
  std::vector<LabelType> neighborLabels(k);
  for (int i = 0; i < k; ++i) {
    neighborLabels[i] = neighbors[i].second;
  }
  return neighborLabels;
}

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
LabelType KnnLinear<DataType, LabelType, DistType, Dim>::KnnClassify(
    const int k, DataItr const &query) const {
  auto neighbors = Knn(k, query);
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

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
std::vector<std::vector<LabelType>>
KnnLinear<DataType, LabelType, DistType, Dim>::Knn(
    int k, std::vector<DataType> const &queries) const {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<LabelType>> neighbors(nQueries);
  #pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    neighbors[i] = Knn(k, queries.cbegin() + i * Dim);
  }
  return neighbors;
}

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
std::vector<LabelType>
KnnLinear<DataType, LabelType, DistType, Dim>::KnnClassify(
    int k, std::vector<DataType> const &queries) const {
  const int nQueries = queries.size() / Dim;
  std::vector<LabelType> classification(nQueries);
  #pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    classification[i] = KnnClassify(k, queries.cbegin() + i * Dim);
  }
  return classification;
}

template <typename DataType, typename LabelType, typename DistType,
          unsigned Dim>
bool KnnLinear<DataType, LabelType, DistType, Dim>::CompareNeighbors(
    std::pair<DistType, LabelType> const &a,
    std::pair<DistType, LabelType> const &b) {
  return a.first < b.first;
}

} // End namespace knn
