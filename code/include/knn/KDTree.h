#pragma once

#include <algorithm>  // std::nth_element
#include <functional> // std::function
#include <numeric>    // std::iota
#include <ostream>
#include <random>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>
#include "knn/BoundedHeap.h"
#include "knn/Common.h"

namespace knn {

namespace {
template <bool Randomized, size_t Dim, typename DataType> class GetSplitDimImpl;
} // End anonymous namespace

template <size_t Dim, bool Randomized, typename DataType>
class KDTree {

private:
  struct Node {
    DataItr<DataType> value;
    size_t index;
    const size_t splitDim;
    typename std::vector<Node>::const_iterator parent, left, right;
    Node(DataItr<DataType> const &_value, size_t _index, size_t _splitDim,
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
    DataItr<DataType> value() const;
    size_t index() const;
    size_t splitDim() const;

  private:
    TreeItr node_;
    const TreeItr end_;
  };

  KDTree(DataContainer<DataType> const &points);

  size_t size() const;

  constexpr size_t nDims() const;

  int nVarianceSamples() const;

  void set_nVarianceSamples(int nVarianceSamples);

  NodeItr Root() const;

private:
  TreeItr BuildTree(DataContainer<DataType> const &data,
                    std::vector<size_t>::iterator begin,
                    std::vector<size_t>::iterator const &end,
                    TreeItr const &parent);

  std::vector<DataType>
  Variance(DataContainer<DataType> const &data,
           std::vector<size_t>::const_iterator begin,
           std::vector<size_t>::const_iterator const &end);

  size_t size_;
  int nVarianceSamples_{0};
  std::vector<Node> tree_{};
  GetSplitDimImpl<Randomized, Dim, DataType> getSplitDimImpl_{};
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

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

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::KDTree(DataContainer<DataType> const &points)
    : size_(Dim > 0 ? points.size() / Dim : 0) {
  static_assert(Dim > 0, "KDTree data cannot be zero-dimensional.");
  if (size_ == 0) {
    throw std::invalid_argument("KDTree received empty data set.");
  }
  if (points.size() % Dim != 0) {
    throw std::invalid_argument("KDTree received unbalanced data.");
  }
  tree_.reserve(size_);
  std::vector<size_t> indices(size_);
  std::iota(indices.begin(), indices.end(), 0);
  BuildTree(points, indices.begin(), indices.end(), tree_.begin() + size_);
}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::Node::Node(DataItr<DataType> const &_value,
                                              size_t _index,
                                              const size_t _splitDim,
                                              TreeItr const &_parent,
                                              TreeItr const &_end)
    : value(_value), index(_index), splitDim(_splitDim), parent(_parent),
      left(_end), right(_end) {}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::NodeItr::NodeItr(
    TreeItr const &node, TreeItr const &end)
    : node_(node), end_(end) {}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::NodeItr
KDTree<Dim, Randomized, DataType>::NodeItr::Left() const {
  return {node_->left, end_};
}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::NodeItr
KDTree<Dim, Randomized, DataType>::NodeItr::Right() const {
  return {node_->right, end_};
}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::NodeItr
KDTree<Dim, Randomized, DataType>::NodeItr::Parent() const {
  return {node_->parent, end_};
}

template <size_t Dim, bool Randomized, typename DataType>
bool KDTree<Dim, Randomized, DataType>::NodeItr::TryLeft() {
  if (node_->left != end_) {
    node_ = node_->left;
    return true;
  }
  return false;
}

template <size_t Dim, bool Randomized, typename DataType>
bool KDTree<Dim, Randomized, DataType>::NodeItr::TryRight() {
  if (node_->right != end_) {
    node_ = node_->right;
    return true;
  }
  return false;
}

template <size_t Dim, bool Randomized, typename DataType>
bool KDTree<Dim, Randomized, DataType>::NodeItr::TryParent() {
  if (node_->parent != end_) {
    node_ = node_->parent;
    return true;
  }
  return false;
}

template <size_t Dim, bool Randomized, typename DataType>
bool KDTree<Dim, Randomized, DataType>::NodeItr::inBounds() const {
  return node_ != end_;
}

template <size_t Dim, bool Randomized, typename DataType>
DataItr<DataType>
KDTree<Dim, Randomized, DataType>::NodeItr::value() const {
  return node_->value;
}

template <size_t Dim, bool Randomized, typename DataType>
size_t KDTree<Dim, Randomized, DataType>::NodeItr::index() const {
  return node_->index;
}

template <size_t Dim, bool Randomized, typename DataType>
size_t KDTree<Dim, Randomized, DataType>::NodeItr::splitDim() const {
  return node_->splitDim;
}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::NodeItr
KDTree<Dim, Randomized, DataType>::Root() const {
  return {tree_.cbegin(), tree_.cend()};
}

template <size_t Dim, bool Randomized, typename DataType>
size_t KDTree<Dim, Randomized, DataType>::size() const {
  return size_;
}

template <size_t Dim, bool Randomized, typename DataType>
constexpr size_t KDTree<Dim, Randomized, DataType>::nDims() const {
  return Dim;
}

template <size_t Dim, bool Randomized, typename DataType>
int KDTree<Dim, Randomized, DataType>::nVarianceSamples() const {
  return nVarianceSamples_;
}

template <size_t Dim, bool Randomized, typename DataType>
void KDTree<Dim, Randomized, DataType>::set_nVarianceSamples(
    int nVarianceSamples) {
  nVarianceSamples_ = nVarianceSamples;
}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::TreeItr
KDTree<Dim, Randomized, DataType>::BuildTree(
    DataContainer<DataType> const &points, std::vector<size_t>::iterator begin,
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
  tree_.emplace_back(points.cbegin() + Dim * medianIndex, medianIndex, splitDim,
                     parent, tree_.cbegin() + size_);
  treeItr->left = BuildTree(points, begin, begin + middle, treeItr);
  treeItr->right = BuildTree(points, begin + middle + 1, end, treeItr);
  return treeItr;
}

template <size_t Dim, bool Randomized, typename DataType>
std::vector<DataType> KDTree<Dim, Randomized, DataType>::Variance(
    DataContainer<DataType> const &data,
    std::vector<size_t>::const_iterator begin,
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

template <size_t Dim, bool Randomized, typename DataType>
std::ostream &
operator<<(std::ostream &os,
           KDTree<Dim, Randomized, DataType> const &tree) {
  std::function<void(
      typename KDTree<Dim, Randomized, DataType>::NodeItr,
      std::string indent)> printRecursive = [&os,
                                             &printRecursive](
      typename KDTree<Dim, Randomized, DataType>::NodeItr root,
      std::string indent) {
    indent += "  ";
    auto val = root.value();
    os << indent << "(" << (*val);
    for (size_t i = 1; i < Dim; ++i) {
      os << ", " << (*(val + i));
    }
    os << ")\n";
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

} // End namespace knn
