#pragma once

#include <algorithm>  // std::nth_element
#include <functional> // std::function
#include <numeric>    // std::iota
#include <iostream>
#include <ostream>
#include <random>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <unordered_map>
#include <vector>

namespace {
template <bool Randomized, size_t Dim, typename DataType>
class GetSplitDimImpl;
} // End anonymous namespace

template <size_t Dim, typename DataType, typename LabelType,
          bool Randomized = false>
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

  KDTree(DataContainer const &points, LabelContainer const &labels);

  KDTree(KDTree<Dim, DataType, LabelType, Randomized> const &other) = default;

  KDTree(KDTree<Dim, DataType, LabelType, Randomized> &&other) = default;

  KDTree<Dim, DataType, LabelType, Randomized> &
  operator=(KDTree<Dim, DataType, LabelType, Randomized> const &rhs) = default;

  KDTree<Dim, DataType, LabelType, Randomized> &
  operator=(KDTree<Dim, DataType, LabelType, Randomized> &&rhs) = default;

  size_t size() const;

  constexpr size_t nDims() const;

  int nVarianceSamples() const;

  void set_nVarianceSamples(int nVarianceSamples);

  int maxLeafVisits() const;

  void set_maxLeafVisits(int maxLeafVisits);

  NodeItr Root() const;

  template <typename DistType>
  LabelType Knn(int k, DataItr const &point,
                std::function<DistType(DataItr const &, DataItr const &)> const
                    &distFunc) const;


private:
  TreeItr BuildTree(DataContainer const &data, LabelContainer const &labels,
                    std::vector<size_t>::iterator begin,
                    std::vector<size_t>::iterator const &end,
                    TreeItr const &parent);

  std::vector<DataType>
  Variance(DataContainer const &data, std::vector<size_t>::const_iterator begin,
           std::vector<size_t>::const_iterator const &end);

  template <typename DistType>
  void KnnRecurse(
      size_t k, DataItr const &point,
      std::function<DistType(DataItr const &, DataItr const &)> const &distFunc,
      NodeItr const &node, std::vector<DistType> &bestDistances,
      std::vector<int> &labels, size_t &maxDist, int &leavesVisited) const;

  size_t size_;
  int nVarianceSamples_{0};
  int maxLeafVisits_{0};
  std::vector<Node> tree_{};
  std::unordered_map<int, LabelType> labelMapping_{};
  std::unordered_map<LabelType, int> labelMappingInv_{};
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

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
KDTree<Dim, DataType, LabelType, Randomized>::KDTree()
    : size_(0) {}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
KDTree<Dim, DataType, LabelType, Randomized>::KDTree(
    DataContainer const &points, LabelContainer const &labels)
    : size_(Dim > 0 ? points.size() / Dim : 0) {
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
  int index = 0;
  std::vector<int> labelsInt;
  labelsInt.reserve(labels.size());
  for (int i = 0, iEnd = labels.size(); i < iEnd; ++i) {
    auto insertion = labelMappingInv_.emplace(std::make_pair(labels[i], index));
    labelsInt.emplace_back(insertion.first->second);
    if (insertion.second == true) {
      labelMapping_.emplace(std::make_pair(index, labels[i]));
      ++index;
    }
  }
  tree_.reserve(size_);
  std::vector<size_t> indices(size_);
  std::iota(indices.begin(), indices.end(), 0);
  BuildTree(points, labels, indices.begin(), indices.end(),
            tree_.begin() + size_);
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
KDTree<Dim, DataType, LabelType, Randomized>::Node::Node(DataItr const &_value,
                                                         LabelItr const &_label,
                                                         const size_t _splitDim,
                                                         TreeItr const &_parent,
                                                         TreeItr const &_end)
    : value(_value), label(_label), splitDim(_splitDim), parent(_parent),
      left(_end), right(_end) {}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::NodeItr(
    TreeItr const &node, TreeItr const &end)
    : node_(node), end_(end) {}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::NodeItr
KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::Left() const {
  return {node_->left, end_};
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::NodeItr
KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::Right() const {
  return {node_->right, end_};
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::NodeItr
KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::Parent() const {
  return {node_->parent, end_};
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
bool KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::TryLeft() {
  if (node_->left != end_) {
    node_ = node_->left;
    return true;
  }
  return false;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
bool KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::TryRight() {
  if (node_->right != end_) {
    node_ = node_->right;
    return true;
  }
  return false;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
bool KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::TryParent() {
  if (node_->parent != end_) {
    node_ = node_->parent;
    return true;
  }
  return false;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
bool KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::inBounds() const {
  return node_ != end_;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::DataItr
KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::value() const {
  return node_->value;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::LabelItr
KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::label() const {
  return node_->label;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
size_t KDTree<Dim, DataType, LabelType, Randomized>::NodeItr::splitDim() const {
  return node_->splitDim;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::NodeItr
KDTree<Dim, DataType, LabelType, Randomized>::Root() const {
  return {tree_.cbegin(), tree_.cend()};
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
size_t KDTree<Dim, DataType, LabelType, Randomized>::size() const {
  return size_;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
constexpr size_t KDTree<Dim, DataType, LabelType, Randomized>::nDims() const {
  return Dim;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
int KDTree<Dim, DataType, LabelType, Randomized>::nVarianceSamples() const {
  return nVarianceSamples_;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
void KDTree<Dim, DataType, LabelType, Randomized>::set_nVarianceSamples(
    int nVarianceSamples) {
  nVarianceSamples_ = nVarianceSamples;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
int KDTree<Dim, DataType, LabelType, Randomized>::maxLeafVisits() const {
  return maxLeafVisits_;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
void KDTree<Dim, DataType, LabelType, Randomized>::set_maxLeafVisits(
    int maxLeafVisits) {
  maxLeafVisits_ = maxLeafVisits;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
typename KDTree<Dim, DataType, LabelType, Randomized>::TreeItr
KDTree<Dim, DataType, LabelType, Randomized>::BuildTree(
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
  treeItr->left =
      BuildTree(points, labels, begin, begin + middle, treeItr);
  treeItr->right =
      BuildTree(points, labels, begin + middle + 1, end, treeItr);
  return treeItr;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
template <typename DistType>
void KDTree<Dim, DataType, LabelType, Randomized>::KnnRecurse(
    const size_t k, DataItr const &point,
    std::function<DistType(DataItr const &, DataItr const &)> const &distFunc,
    NodeItr const &node, std::vector<DistType> &bestDistances,
    std::vector<int> &labels, size_t &maxDist, int &leavesVisited) const {

  // Approximate version
  if (maxLeafVisits_ > 0 && leavesVisited > maxLeafVisits_) {
    return;
  }
  ++leavesVisited;

  const DistType thisDist = distFunc(point, node.value());

  // If current distance is better than the current longest current candidate,
  // replace the previous candidate with the new one
  if (bestDistances.size() == k) {
    if (thisDist < bestDistances[maxDist]) {
      bestDistances[maxDist] = thisDist;
      labels[maxDist] = labelMappingInv_.find(*node.label())->second;
      auto distBegin = bestDistances.cbegin();
      maxDist = std::distance(
          distBegin, std::max_element(distBegin, bestDistances.cend()));
    }
  } else {
    bestDistances.emplace_back(thisDist);
    labels.emplace_back(labelMappingInv_.find(*node.label())->second);
    auto distBegin = bestDistances.cbegin();
    maxDist = std::distance(distBegin,
                            std::max_element(distBegin, bestDistances.cend()));
  }

  auto traverseSubtree = [&](bool doLeft) {
    if (doLeft) {
      const auto left = node.Left();
      if (left.inBounds()) {
        KnnRecurse(k, point, distFunc, left, bestDistances, labels, maxDist,
                   leavesVisited);
      }
    } else {
      const auto right = node.Right();
      if (right.inBounds()) {
        KnnRecurse(k, point, distFunc, right, bestDistances, labels, maxDist,
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
          bestDistances[maxDist] ||
      bestDistances.size() < k) {
    traverseSubtree(!doLeft);
  }
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
template <typename DistType>
LabelType KDTree<Dim, DataType, LabelType, Randomized>::Knn(
    const int k, DataItr const &point,
    std::function<DistType(DataItr const &, DataItr const &)> const &distFunc)
    const {
  std::vector<DistType> bestDistances;
  std::vector<int> labels;
  size_t maxDist = 0;
  int leavesVisited = 0;
  KnnRecurse<DistType>(k, point, distFunc, Root(), bestDistances, labels,
                       maxDist, leavesVisited);
  std::vector<int> vote(labelMapping_.size(), 0);
  LabelType label{};
  int highest = -1;
  for (int l : labels) {
    if (++vote[l] > highest) {
      highest = vote[l];
      label = labelMapping_.find(l)->second;
    }
  }
  return label;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
std::vector<DataType> KDTree<Dim, DataType, LabelType, Randomized>::Variance(
    DataContainer const &data, std::vector<size_t>::const_iterator begin,
    std::vector<size_t>::const_iterator const &end) {
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
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
    // Reuse sumOfSquars vector for computing the variance
    sumOfSquares[j] = (sumOfSquares[j] - sum[j] * sum[j] / iMax) / (iMax - 1);
  }
  return sumOfSquares;
}

template <size_t Dim, typename DataType, typename LabelType, bool Randomized>
std::ostream &
operator<<(std::ostream &os,
           KDTree<Dim, DataType, LabelType, Randomized> const &tree) {
  std::function<void(
      typename KDTree<Dim, DataType, LabelType, Randomized>::NodeItr,
      std::string indent)> printRecursive = [&os,
                                             &printRecursive](
      typename KDTree<Dim, DataType, LabelType, Randomized>::NodeItr root,
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
