#pragma once

#include <cassert>
#include <algorithm>  // std::nth_element
#include <functional> // std::function
#include <numeric>    // std::iota
#include <ostream>
#include <random>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>
#include <tbb/task_group.h>
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

  struct alignas(64) Node {
    DataItr<DataType> value;
    size_t index;
    size_t splitDim;
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

  enum class Pivot {
    median,
    mean
  };

  KDTree();

  /* KDTree(DataContainer<DataType> const &points, Pivot pivot = Pivot::median, */
  /*        int nVarianceSamples = -1, const bool parallel=false); */
  KDTree(DataContainer<DataType> const &points, Pivot pivot = Pivot::median,
         int nVarianceSamples = -1, const bool parallel=true);

  KDTree(KDTree<Dim, Randomized, DataType> const &);

  KDTree(KDTree<Dim, Randomized, DataType> &&);

  KDTree<Dim, Randomized, DataType> &
  operator=(KDTree<Dim, Randomized, DataType> const &);

  KDTree<Dim, Randomized, DataType> &
  operator=(KDTree<Dim, Randomized, DataType> &&);

  size_t nLeaves() const;

  constexpr size_t nDims() const;

  int nVarianceSamples() const;

  void set_nVarianceSamples(int nVarianceSamples);

  NodeItr Root() const;

  static std::vector<KDTree<Dim, true, DataType>>
  BuildRandomizedTrees(DataContainer<DataType> const &points, int nTrees,
                       Pivot pivot = Pivot::median, int nVarianceSamples = 100);

private:
  TreeItr BuildTree(DataContainer<DataType> const &data, Pivot pivot,
                    std::vector<size_t>::iterator begin,
                    const std::vector<size_t>::iterator end);

  TreeItr BuildTreeParallel(DataContainer<DataType> const &data, Pivot pivot,
                    std::vector<size_t>::iterator begin,
                    const std::vector<size_t>::iterator end,
                    const size_t mamaID=0);

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
KDTree<Dim, Randomized, DataType>::KDTree() : nLeaves_(0) {}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::KDTree(DataContainer<DataType> const &points,
                                          Pivot pivot, int nVarianceSamples,
                                          const bool parallel)
    : nLeaves_(Dim > 0 ? points.size() / Dim : 0),
      nVarianceSamples_(nVarianceSamples)
{
  static_assert(Dim > 0, "KDTree data cannot be zero-dimensional.");
  if (nLeaves_ == 0) {
    throw std::invalid_argument("KDTree received empty data set.");
  }
  if (points.size() % Dim != 0) {
    throw std::invalid_argument("KDTree received unbalanced data.");
  }
  /* TODO: (fabianw; Sat Nov 21 16:16:17 2015) these are too many */
  /* tree_.reserve(log2(nLeaves_)*nLeaves_); */
  tree_.reserve(2*nLeaves_-1); // exact number of nodes if points are not consumed until leaf is reached, is tree is evenly split, always odd
  std::vector<size_t> indices(nLeaves_);
  std::iota(indices.begin(), indices.end(), 0);
  if (!parallel)
      BuildTree(points, pivot, indices.begin(), indices.end());
  else
      BuildTreeParallel(points, pivot, indices.begin(), indices.end());
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
KDTree<Dim, Randomized, DataType> &KDTree<Dim, Randomized, DataType>::
operator=(KDTree<Dim, Randomized, DataType> const &rhs) {
  nLeaves_ = rhs.nLeaves_;
  nVarianceSamples_ = rhs.nVarianceSamples_;
  tree_.resize(rhs.tree_.size());
  const auto begin = tree_.cbegin();
  const auto beginOther = rhs.tree_.cbegin();
  for (int i = 0, iMax = nLeaves_; i < iMax; ++i) {
    tree_[i].value = rhs.tree_[i].value;
    tree_[i].index = rhs.tree_[i].index;
    tree_[i].splitDim = rhs.tree_[i].splitDim;
    tree_[i].left = begin + std::distance(beginOther, rhs.tree_[i].left);
    tree_[i].right = begin + std::distance(beginOther, rhs.tree_[i].right);
  }
  return *this;
}

template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType> &KDTree<Dim, Randomized, DataType>::
operator=(KDTree<Dim, Randomized, DataType> &&rhs) {
  nLeaves_ = rhs.nLeaves_;
  nVarianceSamples_ = rhs.nVarianceSamples_;
  tree_ = std::move(rhs.tree_);
  return *this;
}

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
    DataContainer<DataType> const &points, Pivot pivot,
    std::vector<size_t>::iterator begin,
    const std::vector<size_t>::iterator end) {

  const auto treeItr = tree_.end();
  if (std::distance(begin, end) > 1) {
    const auto meanAndVariance =
        MeanAndVariance<DataType, Dim>(points, nVarianceSamples_, begin, end);
    // Randomized/non-randomized machinery is hidden in getSplitDimImpl_
    const size_t splitDim = getSplitDimImpl_(meanAndVariance.second);
    size_t splitPivot;
    if (pivot == Pivot::median) {
      splitPivot = std::distance(begin, end) / 2;
      std::nth_element(begin, begin + splitPivot, end, [&points, &splitDim](
                                                           size_t a, size_t b) {
        return points[Dim * a + splitDim] < points[Dim * b + splitDim];
      });
    } else {
      const DataType splitMean = meanAndVariance.first[splitDim];
      splitPivot = std::distance(
          begin, std::partition(begin, end,
                                [&points, &splitDim, &splitMean](size_t i) {
                                  return points[Dim * i + splitDim] < splitMean;
                                }));
    }
    const size_t splitIndex = begin[splitPivot];
    tree_.emplace_back(points.cbegin() + Dim * splitIndex, splitIndex, splitDim,
                       treeItr, treeItr);
    // Points are not consumed before a leaf is reached
    treeItr->left  = BuildTree(points, pivot, begin, begin + splitPivot);
    treeItr->right = BuildTree(points, pivot, begin + splitPivot, end);
  } else {
    // Leaf node
    tree_.emplace_back(points.cbegin() + Dim * (*begin), *begin, 0, treeItr,
                       treeItr);
  }
  return treeItr;
}


template <size_t Dim, bool Randomized, typename DataType>
typename KDTree<Dim, Randomized, DataType>::TreeItr
KDTree<Dim, Randomized, DataType>::BuildTreeParallel(
    DataContainer<DataType> const &points, Pivot pivot,
    std::vector<size_t>::iterator begin,
    const std::vector<size_t>::iterator end,
    const size_t mamaID)
{
    const auto mySelf = tree_.cbegin() + mamaID;

    if (std::distance(begin, end) > 1) {
        const auto meanAndVariance =
        MeanAndVariance<DataType, Dim>(points, nVarianceSamples_, begin, end);
        // Randomized/non-randomized machinery is hidden in getSplitDimImpl_
        const size_t splitDim = getSplitDimImpl_(meanAndVariance.second);
        size_t splitPivot;
        if (pivot == Pivot::median) {
            splitPivot = std::distance(begin, end) / 2;
            std::nth_element(begin, begin + splitPivot, end, [&points, &splitDim](
                        size_t a, size_t b) {
                    return points[Dim * a + splitDim] < points[Dim * b + splitDim];
                    });
        } else {
            const DataType splitMean = meanAndVariance.first[splitDim];
            splitPivot = std::distance(
                    begin, std::partition(begin, end,
                        [&points, &splitDim, &splitMean](size_t i) {
                        return points[Dim * i + splitDim] < splitMean;
                        }));
        }
        const size_t splitIndex = begin[splitPivot];

        tree_[mamaID] = Node(points.cbegin() + Dim * splitIndex, splitIndex, splitDim, mySelf, mySelf);

        // Points are not consumed before a leaf is reached
        size_t offset = 2*std::distance(begin, begin+splitPivot); // (2*nSubleaves - 1) + 1

        tbb::task_group myGroup;
        myGroup.run([&]{tree_[mamaID].left  = BuildTreeParallel(points, pivot, begin, begin + splitPivot, mamaID + 1);});
        myGroup.run([&]{tree_[mamaID].right = BuildTreeParallel(points, pivot, begin + splitPivot, end, mamaID + offset);});
        myGroup.wait();

        /* tree_[mamaID].left  = BuildTreeParallel(points, pivot, begin, begin + splitPivot, mamaID + 1); */
        /* tree_[mamaID].right = BuildTreeParallel(points, pivot, begin + splitPivot, end, mamaID + offset); */
    }
    else
    {
        // Leaf node, no babies
        tree_[mamaID] = Node(points.cbegin() + Dim * (*begin), *begin, 0, mySelf, mySelf);
    }
    return mySelf;
}

template <size_t Dim, bool Randomized, typename DataType>
std::vector<KDTree<Dim, true, DataType>>
KDTree<Dim, Randomized, DataType>::BuildRandomizedTrees(
    DataContainer<DataType> const &points, const int nTrees,
    const Pivot pivot, const int nVarianceSamples) {
  if (nTrees < 1) {
    throw std::invalid_argument(
        "BuildRandomizedTrees: number of trees must be >=0.");
  }
  std::vector<KDTree<Dim, true, DataType>> trees(nTrees);
#pragma omp parallel for
  for (int i = 0; i < nTrees; ++i) {
    KDTree<Dim, true, DataType> tree(points, pivot, nVarianceSamples);
    trees[i] = std::move(tree);
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
