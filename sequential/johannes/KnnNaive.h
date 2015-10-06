#pragma once

#include <iostream>

#include <algorithm>
#include <cassert>
#include <functional>
#include <unordered_map>
#include <vector>

template <typename T, typename U> class KnnNaive {

public:
  KnnNaive(std::vector<T> const &points, std::vector<U> const &labels);

  template <typename V>
  std::vector<U>
  Classify(int k, std::vector<T> const &data,
           std::function<V(T const &, T const &)> const &distFunc) const;

private:
  std::vector<T> points_;
  std::vector<int> labels_;
  std::unordered_map<int, U> labelMapping_;
};

template <typename T, typename U>
KnnNaive<T, U>::KnnNaive(std::vector<T> const &points,
                         std::vector<U> const &labels)
    : points_(points), labels_(labels.size()) {
  int index = 0;
  std::unordered_map<U, int> invMap;
  for (int i = 0, iEnd = labels.size(); i < iEnd; ++i) {
    auto insertion = invMap.emplace(std::make_pair(labels[i], index));
    labels_[i] = insertion.first->second;
    if (insertion.second == true) {
      labelMapping_.emplace(std::make_pair(index, labels[i]));
      ++index;
    }
  }
}

template <typename T, typename U>
template <typename V>
std::vector<U> KnnNaive<T, U>::Classify(
    const int k, std::vector<T> const &data,
    std::function<V(T const &, T const &)> const &distFunc) const {
  assert(k <= static_cast<int>(points_.size()));
  std::vector<U> classification(data.size());
  #pragma omp parallel for
  for (int i = 0, iEnd = data.size(); i < iEnd; ++i) {
    auto bestDist = std::make_pair(std::vector<V>(k), std::vector<int>(k));
    for (int j = 0; j < k; ++j) {
      bestDist.first[j] = distFunc(data[i], points_[j]);
      bestDist.second[j] = j;
    }
    int maxElement = std::distance(
        bestDist.first.cbegin(),
        std::max_element(bestDist.first.cbegin(), bestDist.first.cend()));
    for (int j = k, jEnd = points_.size(); j < jEnd; ++j) {
      V dist = distFunc(data[i], points_[j]);
      if (dist >= bestDist.first[maxElement]) continue;
      bestDist.first[maxElement] = dist;
      bestDist.second[maxElement] = j;
      maxElement = std::distance(
          bestDist.first.cbegin(),
          std::max_element(bestDist.first.cbegin(), bestDist.first.cend()));
    }
    std::vector<int> count(labelMapping_.size(), 0);
    for (auto &j : bestDist.second) {
      ++count[labels_[j]];
    }
    classification[i] =
        labelMapping_.find(std::distance(
                               count.cbegin(),
                               std::max_element(count.cbegin(), count.cend())))
            ->second;
  }
  return classification;
}
