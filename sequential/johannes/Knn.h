#pragma once

#include "KDTree.h"
#include <functional>
#include <stdexcept>
#include <vector>
#include <unordered_map>

template <typename T, typename U> class Knn {

public:
  Knn(std::vector<std::vector<T>> const &points, std::vector<U> const &labels);

  template <typename V>
  U Classify(int k, std::vector<T> const &dataPoint,
             std::function<V(std::vector<T> const &,
                             std::vector<T> const &)> const &distFunc) const;

private:
  template <typename V>
  void Recurse(size_t k, std::vector<T> const &point,
               std::function<V(std::vector<T> const &,
                               std::vector<T> const &)> const &distFunc,
               typename KDTree<T, int>::Ptr const &node,
               std::vector<V> &bestDistances, std::vector<U> &labels,
               size_t &maxDist) const;

  KDTree<T, int> kdTree{};
  std::unordered_map<int, U> labelMapping_{};
};

template <typename T, typename U>
Knn<T, U>::Knn(std::vector<std::vector<T>> const &points,
               std::vector<U> const &labels) {
  int index = 0;
  std::unordered_map<U, int> invMap;
  std::vector<int> labelsInt;
  labelsInt.reserve(labels.size());
  for (int i = 0, iEnd = labels.size(); i < iEnd; ++i) {
    auto insertion = invMap.emplace(std::make_pair(labels[i], index));
    labelsInt.emplace_back(insertion.first->second);
    if (insertion.second == true) {
      labelMapping_.emplace(std::make_pair(index, labels[i]));
      ++index;
    }
  }
  kdTree = KDTree<T, int>(points, labelsInt);
}

template <typename T, typename U>
template <typename V>
void Knn<T, U>::Recurse(
    const size_t k, std::vector<T> const &point,
    std::function<V(std::vector<T> const &, std::vector<T> const &)> const
        &distFunc,
    typename KDTree<T, int>::Ptr const &node, std::vector<V> &bestDistances,
    std::vector<U> &labels, size_t &maxDist) const {

  const V thisDist = distFunc(point, node.value());

  // If current distance is better than the current longest current candidate,
  // replace the previous candidate with the current distance
  if (bestDistances.size() == k) {
    if (thisDist < bestDistances[maxDist]) {
      bestDistances[maxDist] = thisDist;
      labels[maxDist] = node.label();
      auto distBegin = bestDistances.cbegin();
      maxDist = std::distance(
          distBegin, std::max_element(distBegin, bestDistances.cend()));
    }
  } else {
    bestDistances.emplace_back(thisDist);
    labels.emplace_back(node.label());
    auto distBegin = bestDistances.cbegin();
    maxDist = std::distance(distBegin,
                            std::max_element(distBegin, bestDistances.cend()));
  }

  auto traverseSubtree = [&](bool doLeft) {
    if (doLeft) {
      const auto left = node.Left();
      if (left.inBounds()) {
        Recurse(k, point, distFunc, left, bestDistances, labels, maxDist); 
      }
    } else {
      const auto right = node.Right();
      if (right.inBounds()) {
        Recurse(k, point, distFunc, right, bestDistances, labels, maxDist);
      }
    }
  };

  // Recurse the subtree with the highest intersection
  const size_t splitDim = node.dim();
  const bool doLeft = point[splitDim] < node.value()[splitDim];
  traverseSubtree(doLeft);

  // If the longest nearest neighbor hypersphere crosses the splitting
  // hyperplane after traversing the subtree with the highest intersection, we
  // have to traverse the other subtree too
  if (std::abs(node.value()[splitDim] - point[splitDim]) <
      bestDistances[maxDist] || bestDistances.size() < k) {
    traverseSubtree(!doLeft);
  }
}

template <typename T, typename U>
template <typename V>
U Knn<T, U>::Classify(
    const int k, std::vector<T> const &point,
    std::function<V(std::vector<T> const &, std::vector<T> const &)> const
        &distFunc) const {
  std::vector<V> bestDistances;
  std::vector<U> labels; // U must have a default constructor...
  size_t maxDist = 0;
  Recurse<V>(k, point, distFunc, kdTree.Root(), bestDistances, labels, maxDist);
  std::vector<int> vote(labelMapping_.size(), 0);
  U label{};
  int highest = -1;
  for (auto &l : labels) {
    if (++vote[l] > highest) {
      highest = vote[l];
      label = labelMapping_.find(l)->second;
    }
  }
  return label;
}
