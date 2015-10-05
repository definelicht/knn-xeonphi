#pragma once

#include <functional>
#include <vector>

template <typename T, typename U> class KnnNaive {

public:
  KnnNaive(std::vector<T> const &points, std::vector<U> const &labels);

  template <typename V>
  std::vector<U>
  Classify(std::vector<T> const &data,
           std::function<V(T const &, T const &)> const &distFunc) const;

private:
  std::vector<T> points_;
  std::vector<U> labels_;
};

template <typename T, typename U>
KnnNaive<T, U>::KnnNaive(std::vector<T> const &points,
                         std::vector<U> const &labels)
    : points_(points), labels_(labels) {}

template <typename T, typename U>
template <typename V>
std::vector<U> KnnNaive<T, U>::Classify(
    std::vector<T> const &data,
    std::function<V(T const &, T const &)> const &distFunc) const {
  std::vector<U> classification(data.size());
  #pragma omp parallel for
  for (int i = 0, iEnd = data.size(); i < iEnd; ++i) {
    std::pair<V, int> bestDist =
        std::make_pair(distFunc(data[i], points_[0]), 0);
    for (int j = 1, jEnd = points_.size(); j < jEnd; ++j) {
      V dist = distFunc(data[i], points_[j]);
      bestDist = bestDist.first < dist ? bestDist : std::make_pair(dist, j);
    }
    classification[i] = labels_[bestDist.second];
  }
  return classification;
}
