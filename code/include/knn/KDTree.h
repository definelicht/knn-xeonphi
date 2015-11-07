#pragma once
#include <cassert>
#include <iostream>

#include <algorithm>  // std::nth_element
#include <functional> // std::function
#include <numeric>    // std::iota
#include <ostream>
#include <random>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <unordered_map>
#include <vector>

namespace {
template <bool Randomized, size_t Dim, typename DataType> class GetSplitDimImpl;
} // End anonymous namespace

template <size_t Dim, bool Randomized, typename DataType, typename LabelType,
          typename DistType>
class KDTree {

public:
  using DataContainer = std::vector<DataType>;
  using LabelContainer = std::vector<LabelType>;
  using DataItr = typename DataContainer::const_iterator;
  using LabelItr = typename LabelContainer::const_iterator;

private:
  struct Node {
    const DataItr value;
    const LabelItr label;
    const size_t splitDim;
    typename std::vector<Node>::const_iterator parent, left, right;
    Node(DataItr const &_value, LabelItr const &_label, size_t _splitDim,
         typename std::vector<Node>::const_iterator const &_parent,
         typename std::vector<Node>::const_iterator const &_end);
  };
  using TreeItr = typename std::vector<Node>::const_iterator;

public:
  class NodeItr {
  public:
    NodeItr(TreeItr const &node, TreeItr const &end);
    NodeItr Left() const;
    NodeItr Right() const;
    NodeItr Parent() const;
    bool TryLeft();
    bool TryRight();
    bool TryParent();
    bool inBounds() const;
    DataItr value() const;
    LabelItr label() const;
    size_t splitDim() const;

  private:
    TreeItr node_;
    const TreeItr end_;
  };

  KDTree();

  KDTree(DataContainer const &points, LabelContainer const &labels,
         std::function<DistType(DataItr const &, DataItr const &)> const
             &distFunc);

  KDTree(KDTree<Dim, Randomized, DataType, LabelType, DistType> const &other) =
      default;

  KDTree(KDTree<Dim, Randomized, DataType, LabelType, DistType> &&other) =
      default;

  KDTree<Dim, Randomized, DataType, LabelType, DistType> &
  operator=(KDTree<Dim, Randomized, DataType, LabelType, DistType> const &rhs) =
      default;

  KDTree<Dim, Randomized, DataType, LabelType, DistType> &operator=(
      KDTree<Dim, Randomized, DataType, LabelType, DistType> &&rhs) = default;

  size_t size() const;

  constexpr size_t nDims() const;

  int nVarianceSamples() const;

  void set_nVarianceSamples(int nVarianceSamples);

  int maxLeafVisits() const;

  void set_maxLeafVisits(int maxLeafVisits);

  NodeItr Root() const;

  std::vector<LabelType> Knn(int k, DataItr const &point) const;

  std::vector<std::vector<LabelType>> Knn(int k,
                                          DataContainer const &points) const;

  LabelType KnnClassify(int k, DataItr const &point) const;

  std::vector<LabelType> KnnClassify(int k, DataContainer const &points) const;

private:
  TreeItr BuildTree(DataContainer const &data, LabelContainer const &labels,
                    std::vector<size_t>::iterator begin,
                    std::vector<size_t>::iterator const &end,
                    TreeItr const &parent);

  std::vector<DataType>
  Variance(DataContainer const &data, std::vector<size_t>::const_iterator begin,
           std::vector<size_t>::const_iterator const &end);

  static bool CompareNeighbors(std::pair<DistType, LabelType> const &a,
                               std::pair<DistType, LabelType> const &b);

  void KnnRecurse(size_t k, DataItr const &point, NodeItr const &node,
                  std::vector<std::pair<DistType, LabelType>> &neighbors,
                  size_t &maxDist, int &leavesVisited) const;

  size_t size_;
  int nVarianceSamples_{0};
  int maxLeafVisits_{0};
  std::vector<Node> tree_{};
  std::function<DistType(DataItr const &, DataItr const &)> distFunc_;
  GetSplitDimImpl<Randomized, Dim, DataType> getSplitDimImpl_{};
}; // End class KDTree

namespace {
template <size_t Dim, typename DataType>
class GetSplitDimImpl<false, Dim, DataType> {
public:
  size_t operator()(std::vector<DataType> const &variances) {
    return std::distance(
        variances.cbegin(),
        std::max_element(variances.cbegin(), variances.cend()));
  }
};
template <size_t Dim, typename DataType>
class GetSplitDimImpl<true, Dim, DataType> {
public:
  GetSplitDimImpl()
      : nHighestVariances_{Dim > 10 ? 5 : Dim / 2}, rng(std::random_device{}()),
        dist(0, nHighestVariances_ - 1) {}
  // TODO: how to set this from the templated main class?
  void set_nHighestVariances(int nHighestVariances) {
    nHighestVariances_ = nHighestVariances > 0 && nHighestVariances < Dim
                             ? nHighestVariances
                             : Dim;
  }
  int nHighestVariances() const { return nHighestVariances_; }
  size_t operator()(std::vector<DataType> const &variances) {
    std::vector<size_t> indices(variances.size());
    auto begin = indices.begin();
    std::nth_element(begin, begin + nHighestVariances_, variances.end(),
                     [&variances](size_t a, size_t b) {
                       return variances[a] < variances[b];
                     });
    return indices[dist(rng)];
  }

private:
  int nHighestVariances_;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist;
};
} // End anonymous namespace

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::KDTree()
    : size_(0) {}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::KDTree(
    DataContainer const &points, LabelContainer const &labels, std::function<DistType(DataItr const &, DataItr const &)> const &distFunc)
    : size_(Dim > 0 ? points.size() / Dim : 0), distFunc_(distFunc) {
  static_assert(Dim > 0, "KDTree data cannot be zero-dimensional.");
  if (size_ == 0) {
    throw std::invalid_argument("KDTree received empty data set.");
  }
  if (points.size() % Dim != 0) {
    throw std::invalid_argument("KDTree received unbalanced data.");
  }
  if (labels.size() != size_) {
    throw std::invalid_argument(
        "KDTree received mismatched point and label size.");
  }
  tree_.reserve(size_);
  std::vector<size_t> indices(size_);
  std::iota(indices.begin(), indices.end(), 0);
  BuildTree(points, labels, indices.begin(), indices.end(),
            tree_.begin() + size_);
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::Node::Node(
    DataItr const &_value, LabelItr const &_label, const size_t _splitDim,
    TreeItr const &_parent, TreeItr const &_end)
    : value(_value), label(_label), splitDim(_splitDim), parent(_parent),
      left(_end), right(_end) {}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::NodeItr(
    TreeItr const &node, TreeItr const &end)
    : node_(node), end_(end) {}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::Left() const {
  return {node_->left, end_};
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::Right() const {
  return {node_->right, end_};
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::Parent()
    const {
  return {node_->parent, end_};
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
bool KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::TryLeft() {
  if (node_->left != end_) {
    node_ = node_->left;
    return true;
  }
  return false;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
bool KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::TryRight() {
  if (node_->right != end_) {
    node_ = node_->right;
    return true;
  }
  return false;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
bool KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::TryParent() {
  if (node_->parent != end_) {
    node_ = node_->parent;
    return true;
  }
  return false;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
bool KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::inBounds()
    const {
  return node_ != end_;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::DataItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::value() const {
  return node_->value;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::LabelItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::label() const {
  return node_->label;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
size_t KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr::splitDim() const {
  return node_->splitDim;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::Root() const {
  return {tree_.cbegin(), tree_.cend()};
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
size_t KDTree<Dim, Randomized, DataType, LabelType, DistType>::size() const {
  return size_;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
constexpr size_t
KDTree<Dim, Randomized, DataType, LabelType, DistType>::nDims() const {
  return Dim;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
int KDTree<Dim, Randomized, DataType, LabelType, DistType>::nVarianceSamples()
    const {
  return nVarianceSamples_;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
void KDTree<Dim, Randomized, DataType, LabelType, DistType>::set_nVarianceSamples(int nVarianceSamples) {
  nVarianceSamples_ = nVarianceSamples;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
int KDTree<Dim, Randomized, DataType, LabelType, DistType>::maxLeafVisits()
    const {
  return maxLeafVisits_;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
void KDTree<Dim, Randomized, DataType, LabelType, DistType>::set_maxLeafVisits(
    int maxLeafVisits) {
  maxLeafVisits_ = maxLeafVisits;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::TreeItr
KDTree<Dim, Randomized, DataType, LabelType, DistType>::BuildTree(
    DataContainer const &points, LabelContainer const &labels,
    std::vector<size_t>::iterator begin,
    std::vector<size_t>::iterator const &end, TreeItr const &parent) {
  if (begin >= end) {
    return tree_.cbegin() + size_;
  }
  const auto variances = Variance(points, begin, end);
  const size_t splitDim = getSplitDimImpl_(variances);
  const size_t middle = std::distance(begin, end) / 2;
  std::nth_element(
      begin, begin + middle, end, [&points, splitDim](size_t a, size_t b) {
        return points[Dim * a + splitDim] < points[Dim * b + splitDim];
      });
  const size_t medianIndex = begin[middle];
  auto treeItr = tree_.end();
  tree_.emplace_back(points.cbegin() + Dim * medianIndex,
                     labels.cbegin() + medianIndex, splitDim, parent,
                     tree_.cbegin() + size_);
  treeItr->left = BuildTree(points, labels, begin, begin + middle, treeItr);
  treeItr->right = BuildTree(points, labels, begin + middle + 1, end, treeItr);
  return treeItr;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
void KDTree<Dim, Randomized, DataType, LabelType, DistType>::KnnRecurse(
    const size_t k, DataItr const &point, NodeItr const &node,
    std::vector<std::pair<DistType, LabelType>> &neighbors,
    size_t &maxDist, int &leavesVisited) const {

  // Approximate version
  if (neighbors.size() >= k && maxLeafVisits_ > 0 &&
      leavesVisited > maxLeafVisits_) {
    return;
  }
  ++leavesVisited;

  const DistType thisDist = distFunc_(point, node.value());

  // If current distance is better than the current longest current candidate,
  // replace the previous candidate with the new one
  if (neighbors.size() == k) {
    if (thisDist < neighbors[maxDist].first) {
      neighbors[maxDist] = std::make_pair(thisDist, *node.label());;
      auto distBegin = neighbors.cbegin();
      maxDist = std::distance(
          distBegin,
          std::max_element(distBegin, neighbors.cend(), CompareNeighbors));
    }
  } else {
    neighbors.emplace_back(thisDist, *node.label());
    auto distBegin = neighbors.cbegin();
    maxDist =
        std::distance(distBegin, std::max_element(distBegin, neighbors.cend(),
                                                  CompareNeighbors));
  }

  auto traverseSubtree = [&](bool doLeft) {
    if (doLeft) {
      const auto left = node.Left();
      if (left.inBounds()) {
        KnnRecurse(k, point, left, neighbors, maxDist,
                   leavesVisited);
      }
    } else {
      const auto right = node.Right();
      if (right.inBounds()) {
        KnnRecurse(k, point, right, neighbors, maxDist,
                   leavesVisited);
      }
    }
  };

  // Recurse the subtree with the highest intersection
  const size_t splitDim = node.splitDim();
  const bool doLeft = point[splitDim] < node.value()[splitDim];
  traverseSubtree(doLeft);

  // If the longest nearest neighbor hypersphere crosses the splitting
  // hyperplane after traversing the subtree with the highest intersection, we
  if (std::abs(node.value()[splitDim] - point[splitDim]) <
          neighbors[maxDist].first ||
      neighbors.size() < k) {
    traverseSubtree(!doLeft);
  }
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
std::vector<LabelType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::Knn(
    const int k, DataItr const &point) const {
  std::vector<std::pair<DistType, LabelType>> neighbors;
  size_t maxDist = 0;
  int leavesVisited = 0;
  KnnRecurse(k, point, Root(), neighbors, maxDist, leavesVisited);
  // Sort according to lowest distance
  std::sort(neighbors.begin(), neighbors.end(), CompareNeighbors);
  std::vector<LabelType> neighborLabels(k);
  for (int i = 0; i < k; ++i) {
    neighborLabels[i] = neighbors[i].second;
  }
  return neighborLabels;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
LabelType KDTree<Dim, Randomized, DataType, LabelType, DistType>::KnnClassify(
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

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
std::vector<std::vector<LabelType>>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::Knn(
    const int k, DataContainer const &queries) const {
  const int nQueries = queries.size() / Dim;
  std::vector<std::vector<LabelType>> labels(nQueries);
  #pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    labels[i] = Knn(k, queries.cbegin() + i * Dim);
  }
  return labels;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
std::vector<LabelType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::KnnClassify(
    const int k, DataContainer const &queries) const {
  const int nQueries = queries.size() / Dim;
  std::vector<LabelType> labels(nQueries);
  #pragma omp parallel for
  for (int i = 0; i < nQueries; ++i) {
    labels[i] = KnnClassify(k, queries.cbegin() + i * Dim);
  }
  return labels;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
std::vector<DataType>
KDTree<Dim, Randomized, DataType, LabelType, DistType>::Variance(
    DataContainer const &data, std::vector<size_t>::const_iterator begin,
    std::vector<size_t>::const_iterator const &end) {
  std::vector<DataType> sum(Dim, 0);
  std::vector<DataType> sumOfSquares(Dim, 0);
  const int iMax = nVarianceSamples_ > 0
                       ? std::min(nVarianceSamples_,
                                  static_cast<int>(std::distance(begin, end)))
                       : std::distance(begin, end);
  for (int i = 0; i < iMax; ++i) {
    const size_t pointIndex = Dim * (*begin);
    for (unsigned j = 0; j < Dim; ++j) {
      const DataType val = data[pointIndex + j];
      sum[j] += val;
      sumOfSquares[j] += val * val;
    }
    ++begin;
  }
  for (unsigned j = 0; j < Dim; ++j) {
    // Reuse sumOfSquares vector for computing the variance. Don't divide by
    // (N - 1) as this will only be used for intercomparison.
    sumOfSquares[j] =
        sumOfSquares[j] - (sum[j] * sum[j]) / iMax; // / (iMax - 1);
  }
  return sumOfSquares;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType,
          typename DistType>
bool KDTree<Dim, Randomized, DataType, LabelType, DistType>::CompareNeighbors(
    std::pair<DistType, LabelType> const &a,
    std::pair<DistType, LabelType> const &b) {
  return a.first < b.first;
}

template <size_t Dim, bool Randomized, typename DataType, typename LabelType, typename DistType>
std::ostream &
operator<<(std::ostream &os,
           KDTree<Dim, Randomized, DataType, LabelType, DistType> const &tree) {
  std::function<void(
      typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr,
      std::string indent)> printRecursive =
      [&os, &printRecursive](typename KDTree<Dim, Randomized, DataType, LabelType, DistType>::NodeItr root,
                             std::string indent) {
        indent += "  ";
        auto val = root.value();
        os << indent << "(" << (*val);
        for (size_t i = 1; i < Dim; ++i) {
          os << ", " << (*(val + i));
        }
        os << ") -> " << (*root.label()) << "\n";
        auto left = root.Left();
        if (left.inBounds()) {
          os << indent << "Left:\n";
          printRecursive(left, indent);
        }
        auto right = root.Right();
        if (right.inBounds()) {
          os << indent << "Right:\n";
          printRecursive(right, indent);
        }
      };
  os << "KDTree of size " << tree.size() << " and dimensionality "
     << tree.nDims() << ":\n";
  printRecursive(tree.Root(), "");
  return os;
}
