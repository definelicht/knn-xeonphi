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
  struct Node;
  using TreeItr = typename std::vector<Node>::const_iterator;

  struct Node {
    DataItr<DataType> value;
    size_t index;
    const size_t splitDim;
    TreeItr left, right;
    Node(DataItr<DataType> const &_value, size_t _index, size_t _splitDim,
         TreeItr const &_left, TreeItr const &_right);
  };

public:
  class NodeItr {
  public:
    NodeItr();
    NodeItr(TreeItr const &node, bool inBounds);
    NodeItr Left() const;
    NodeItr Right() const;
    bool inBounds() const;
    DataItr<DataType> value() const;
    size_t index() const;
    size_t splitDim() const;

  private:
    TreeItr node_;
    bool inBounds_;
  };

  KDTree(DataContainer<DataType> const &points);

  KDTree(KDTree<Dim, Randomized, DataType> const &);

  KDTree(KDTree<Dim, Randomized, DataType> &&);

  KDTree<Dim, Randomized, DataType> &
  operator=(KDTree<Dim, Randomized, DataType> const &) = delete;

  size_t nLeaves() const;

  constexpr size_t nDims() const;

  int nVarianceSamples() const;

  void set_nVarianceSamples(int nVarianceSamples);

  NodeItr Root() const;

  static std::vector<KDTree<Dim, true, DataType>>
  BuildRandomizedTrees(DataContainer<DataType> const &points, int nTrees,
                       int nVarianceSamples);

private:
  TreeItr BuildTree(DataContainer<DataType> const &data,
                    std::vector<size_t>::iterator begin,
                    std::vector<size_t>::iterator const &end);

  size_t nLeaves_;
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
  size_t operator()(std::array<DataType, Dim> const &variances) {
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

  size_t operator()(std::array<DataType, Dim> const &variances) {
    std::array<size_t, Dim> indices;
    std::iota(indices.begin(), indices.end(), 0);
    auto begin = indices.begin();
    std::nth_element(begin, begin + nHighestVariances_, indices.end(),
                     [&variances](size_t a, size_t b) {
                       return variances[a] > variances[b];
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
    : nLeaves_(Dim > 0 ? points.size() / Dim : 0) {
  static_assert(Dim > 0, "KDTree data cannot be zero-dimensional.");
  if (nLeaves_ == 0) {
    throw std::invalid_argument("KDTree received empty data set.");
  }
  if (points.size() % Dim != 0) {
    throw std::invalid_argument("KDTree received unbalanced data.");
  }
  tree_.reserve(log2(nLeaves_)*nLeaves_);
  std::vector<size_t> indices(nLeaves_);
  std::iota(indices.begin(), indices.end(), 0);
  BuildTree(points, indices.begin(), indices.end());
}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::KDTree(
    KDTree<Dim, Randomized, DataType> const &other)
    : nLeaves_(other.nLeaves_), nVarianceSamples_(other.nVarianceSamples_),
      tree_(), getSplitDimImpl_() {
  tree_.reserve(other.tree_.size());
  const auto begin = tree_.cbegin();
  const auto beginOther = other.tree_.cbegin();
  for (auto &leaf : other.tree_) {
    tree_.emplace_back(leaf.value, leaf.index, leaf.splitDim,
                       begin + std::distance(beginOther, leaf.left),
                       begin + std::distance(beginOther, leaf.right));
  }
}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::KDTree(
    KDTree<Dim, Randomized, DataType> &&other)
    : nLeaves_(other.nLeaves_), nVarianceSamples_(other.nVarianceSamples_),
      tree_(std::move(other.tree_)), getSplitDimImpl_() {}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::Node::Node(DataItr<DataType> const &_value,
                                              size_t _index,
                                              const size_t _splitDim,
                                              TreeItr const &_left,
                                              TreeItr const &_right)
    : value(_value), index(_index), splitDim(_splitDim), left(_left),
      right(_right) {}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::NodeItr::NodeItr()
    : node_(), inBounds_(false) {}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::NodeItr::NodeItr(TreeItr const &node,
                                                    bool inBounds)
    : node_(node), inBounds_(inBounds) {}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::NodeItr
KDTree<Dim, Randomized, DataType>::NodeItr::Left() const {
  return {node_->left, inBounds_ && node_->left != node_};
}

template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::NodeItr
KDTree<Dim, Randomized, DataType>::NodeItr::Right() const {
  return {node_->right, inBounds_ && node_->right != node_};
}

template <size_t Dim, bool Randomized, typename DataType>
bool KDTree<Dim, Randomized, DataType>::NodeItr::inBounds() const {
  return inBounds_;
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
  return {tree_.cbegin(), true};
}

template <size_t Dim, bool Randomized, typename DataType>
size_t KDTree<Dim, Randomized, DataType>::nLeaves() const {
  return nLeaves_;
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
    std::vector<size_t>::iterator const &end) {

  const auto treeItr = tree_.end();
  if (std::distance(begin, end) > 1) {
    const auto variances =
        Variance<DataType, Dim>(points, nVarianceSamples_, begin, end);
    // Randomized/non-randomized machinery is hidden in getSplitDimImpl_
    const size_t splitDim = getSplitDimImpl_(variances);
    const size_t middle = std::distance(begin, end) / 2;
    std::nth_element(
        begin, begin + middle, end, [&points, splitDim](size_t a, size_t b) {
          return points[Dim * a + splitDim] < points[Dim * b + splitDim];
        });
    const size_t medianIndex = begin[middle];
    tree_.emplace_back(points.cbegin() + Dim * medianIndex, medianIndex,
                       splitDim, treeItr, treeItr);
    // Points are not consumed before a leaf is reached
    treeItr->left = BuildTree(points, begin, begin + middle);
    treeItr->right = BuildTree(points, begin + middle, end);
  } else {
    // Leaf node
    tree_.emplace_back(points.cbegin() + Dim * (*begin), *begin, 0, treeItr,
                       treeItr);
  }
  return treeItr;
}

template <size_t Dim, bool Randomized, typename DataType>
std::vector<KDTree<Dim, true, DataType>>
KDTree<Dim, Randomized, DataType>::BuildRandomizedTrees(
    DataContainer<DataType> const &points, const int nTrees,
    const int nVarianceSamples) {
  if (nTrees < 1) {
    throw std::invalid_argument(
        "BuildRandomizedTrees: number of trees must be >=0.");
  }
  std::vector<KDTree<Dim, true, DataType>> trees;
  trees.reserve(nTrees);
#pragma omp parallel for
  for (int i = 0; i < nTrees; ++i) {
    KDTree<Dim, true, DataType> tree(points);
    tree.set_nVarianceSamples(nVarianceSamples);
    trees.emplace_back(std::move(tree));
  }
  return trees;
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
    os << ") -> " << root.index() << "\n";
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
  os << "KDTree with " << tree.nLeaves() << " leaves and dimensionality "
     << tree.nDims() << ":\n";
  printRecursive(tree.Root(), "");
  return os;
}

} // End namespace knn
