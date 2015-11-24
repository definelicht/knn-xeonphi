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
#ifdef _USE_TBB_
#include <tbb/task_group.h>
#endif /* _USE_TBB_ */

#include "knn/BoundedHeap.h"
#include "knn/Common.h"

namespace knn {

namespace {
template <bool Randomized, size_t Dim, typename T> class GetSplitDimImpl;
} // End anonymous namespace

template <size_t Dim, bool Randomized, typename T>
class KDTree {

private:
  struct Node;
  using TreeItr = Node const*;

  struct alignas(64) Node {
<<<<<<< HEAD
    DataItr<T> value;
=======
    DataItr<DataType> value;
>>>>>>> 8d5d28f574756e30deee952896648311c712bb2e
    size_t index;
    size_t splitDim;
    TreeItr left, right;
    Node(DataItr<T> _value, size_t _index, size_t _splitDim,
         TreeItr _left = nullptr, TreeItr _right = nullptr);
  };

public:
  class NodeItr {
  public:
    NodeItr();
    NodeItr(TreeItr node);
    NodeItr Left() const;
    NodeItr Right() const;
    bool inBounds() const;
    DataItr<T> value() const;
    size_t index() const;
    size_t splitDim() const;

  private:
    TreeItr node_;
  };

  enum class Pivot {
    median,
    mean
  };

  KDTree();

<<<<<<< HEAD
  KDTree(DataContainer<T> const &points, Pivot pivot = Pivot::median,
=======
  KDTree(DataContainer<DataType> const &points, Pivot pivot = Pivot::median,
>>>>>>> 8d5d28f574756e30deee952896648311c712bb2e
         int nVarianceSamples = -1, const bool parallel=false);

  KDTree(KDTree<Dim, Randomized, T> const &);

  KDTree(KDTree<Dim, Randomized, T> &&);

  KDTree<Dim, Randomized, T> &
  operator=(KDTree<Dim, Randomized, T> const &);

  KDTree<Dim, Randomized, T> &
  operator=(KDTree<Dim, Randomized, T> &&);

  size_t nLeaves() const;

  constexpr size_t nDims() const;

  int nVarianceSamples() const;

  void set_nVarianceSamples(int nVarianceSamples);

  NodeItr Root() const;

  static std::vector<KDTree<Dim, true, T>>
  BuildRandomizedTrees(DataContainer<T> const &points, int nTrees,
                       Pivot pivot = Pivot::median, int nVarianceSamples = 100);

private:
  TreeItr BuildTree(DataContainer<T> const &data, Pivot pivot,
                    std::vector<size_t>::iterator begin,
                    const std::vector<size_t>::iterator end);

<<<<<<< HEAD
  TreeItr BuildTreeParallel(DataContainer<T> const &data, Pivot pivot,
=======
  TreeItr BuildTreeParallel(DataContainer<DataType> const &data, Pivot pivot,
>>>>>>> 8d5d28f574756e30deee952896648311c712bb2e
                    std::vector<size_t>::iterator begin,
                    const std::vector<size_t>::iterator end,
                    const size_t mamaID=0);

  size_t nLeaves_;
  int nVarianceSamples_{0};
  std::vector<Node> tree_{};
  GetSplitDimImpl<Randomized, Dim, T> getSplitDimImpl_{};
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace {

template <size_t Dim, typename T>
class GetSplitDimImpl<false, Dim, T> {
public:
  size_t operator()(std::array<T, Dim> const &variances) {
    return std::distance(
        variances.cbegin(),
        std::max_element(variances.cbegin(), variances.cend()));
  }
};

template <size_t Dim, typename T>
class GetSplitDimImpl<true, Dim, T> {

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

  size_t operator()(std::array<T, Dim> const &variances) {
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

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree() : nLeaves_(0) {}

<<<<<<< HEAD
template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree(DataContainer<T> const &points,
=======
template <size_t Dim, bool Randomized, typename DataType>
KDTree<Dim, Randomized, DataType>::KDTree(DataContainer<DataType> const &points,
>>>>>>> 8d5d28f574756e30deee952896648311c712bb2e
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

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree(
    KDTree<Dim, Randomized, T> const &other)
    : nLeaves_(other.nLeaves_), nVarianceSamples_(other.nVarianceSamples_),
      tree_(), getSplitDimImpl_() {
  tree_.reserve(other.tree_.size());
  const auto begin = tree_.data();
  const auto beginOther = other.tree_.data();
  for (auto &leaf : other.tree_) {
    tree_.emplace_back(
        leaf.value, leaf.index, leaf.splitDim,
        leaf.left != nullptr ? begin + std::distance(beginOther, leaf.left)
                             : nullptr,
        leaf.right != nullptr ? begin + std::distance(beginOther, leaf.right)
                              : nullptr);
  }
}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree(
    KDTree<Dim, Randomized, T> &&other)
    : nLeaves_(other.nLeaves_), nVarianceSamples_(other.nVarianceSamples_),
      tree_(std::move(other.tree_)), getSplitDimImpl_() {}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T> &KDTree<Dim, Randomized, T>::
operator=(KDTree<Dim, Randomized, T> const &rhs) {
  nLeaves_ = rhs.nLeaves_;
  nVarianceSamples_ = rhs.nVarianceSamples_;
  tree_.resize(rhs.tree_.size());
  const auto begin = tree_.data();
  const auto beginOther = rhs.tree_.data();
  for (int i = 0, iMax = nLeaves_; i < iMax; ++i) {
    tree_[i].value = rhs.tree_[i].value;
    tree_[i].index = rhs.tree_[i].index;
    tree_[i].splitDim = rhs.tree_[i].splitDim;
    tree_[i].left = rhs.tree_[i].left != nullptr
                        ? begin + std::distance(beginOther, rhs.tree_[i].left)
                        : nullptr;
    tree_[i].right = rhs.tree_[i].right != nullptr
                         ? begin + std::distance(beginOther, rhs.tree_[i].right)
                         : nullptr;
  }
  return *this;
}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T> &KDTree<Dim, Randomized, T>::
operator=(KDTree<Dim, Randomized, T> &&rhs) {
  nLeaves_ = rhs.nLeaves_;
  nVarianceSamples_ = rhs.nVarianceSamples_;
  tree_ = std::move(rhs.tree_);
  return *this;
}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::Node::Node(const DataItr<T> _value,
                                       const size_t _index,
                                       const size_t _splitDim,
                                       const TreeItr _left,
                                       const TreeItr _right)
    : value(_value), index(_index), splitDim(_splitDim), left(_left),
      right(_right) {}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::NodeItr::NodeItr()
    : node_(nullptr) {}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::NodeItr::NodeItr(const TreeItr node)
    : node_(node) {}

template <size_t Dim, bool Randomized, typename T>
typename KDTree<Dim, Randomized, T>::NodeItr
KDTree<Dim, Randomized, T>::NodeItr::Left() const {
  return node_->left;
}

template <size_t Dim, bool Randomized, typename T>
typename KDTree<Dim, Randomized, T>::NodeItr
KDTree<Dim, Randomized, T>::NodeItr::Right() const {
  return node_->right;
}

template <size_t Dim, bool Randomized, typename T>
bool KDTree<Dim, Randomized, T>::NodeItr::inBounds() const {
  return node_ != nullptr;
}

template <size_t Dim, bool Randomized, typename T>
DataItr<T> KDTree<Dim, Randomized, T>::NodeItr::value() const {
  return node_->value;
}

template <size_t Dim, bool Randomized, typename T>
size_t KDTree<Dim, Randomized, T>::NodeItr::index() const {
  return node_->index;
}

template <size_t Dim, bool Randomized, typename T>
size_t KDTree<Dim, Randomized, T>::NodeItr::splitDim() const {
  return node_->splitDim;
}

template <size_t Dim, bool Randomized, typename T>
typename KDTree<Dim, Randomized, T>::NodeItr
KDTree<Dim, Randomized, T>::Root() const {
  return tree_.data();
}

template <size_t Dim, bool Randomized, typename T>
size_t KDTree<Dim, Randomized, T>::nLeaves() const {
  return nLeaves_;
}

template <size_t Dim, bool Randomized, typename T>
constexpr size_t KDTree<Dim, Randomized, T>::nDims() const {
  return Dim;
}

template <size_t Dim, bool Randomized, typename T>
int KDTree<Dim, Randomized, T>::nVarianceSamples() const {
  return nVarianceSamples_;
}

template <size_t Dim, bool Randomized, typename T>
void KDTree<Dim, Randomized, T>::set_nVarianceSamples(
    int nVarianceSamples) {
  nVarianceSamples_ = nVarianceSamples;
}

template <size_t Dim, bool Randomized, typename T>
typename KDTree<Dim, Randomized, T>::TreeItr
KDTree<Dim, Randomized, T>::BuildTree(DataContainer<T> const &points,
                                      Pivot pivot,
                                      std::vector<size_t>::iterator begin,
                                      const std::vector<size_t>::iterator end) {

  const auto treeItr = tree_.data() + tree_.size();
  if (std::distance(begin, end) > 1) {
    const auto meanAndVariance =
        MeanAndVariance<T, Dim>(points, nVarianceSamples_, begin, end);
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
      const T splitMean = meanAndVariance.first[splitDim];
      splitPivot = std::distance(
          begin, std::partition(begin, end,
                                [&points, &splitDim, &splitMean](size_t i) {
                                  return points[Dim * i + splitDim] < splitMean;
                                }));
    }
    const size_t splitIndex = begin[splitPivot];
    tree_.emplace_back(points.data() + Dim * splitIndex, splitIndex, splitDim);
    // Points are not consumed before a leaf is reached
    treeItr->left  = BuildTree(points, pivot, begin, begin + splitPivot);
    treeItr->right = BuildTree(points, pivot, begin + splitPivot, end);
  } else {
    // Leaf node
    tree_.emplace_back(points.data() + Dim * (*begin), *begin, 0);
  }
  return treeItr;
}


template <size_t Dim, bool Randomized, typename T>
typename KDTree<Dim, Randomized, T>::TreeItr
KDTree<Dim, Randomized, T>::BuildTreeParallel(
    DataContainer<T> const &points, Pivot pivot,
    std::vector<size_t>::iterator begin,
    const std::vector<size_t>::iterator end,
    const size_t mamaID)
{
    const auto mySelf = tree_.data() + mamaID;

    if (std::distance(begin, end) > 1) {
        const auto meanAndVariance =
        MeanAndVariance<T, Dim>(points, nVarianceSamples_, begin, end);
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
            const T splitMean = meanAndVariance.first[splitDim];
            splitPivot = std::distance(
                    begin, std::partition(begin, end,
                        [&points, &splitDim, &splitMean](size_t i) {
                        return points[Dim * i + splitDim] < splitMean;
                        }));
        }
        const size_t splitIndex = begin[splitPivot];

        *mySelf = Node(points.data() + Dim * splitIndex, splitIndex, splitDim, mySelf, mySelf);

        // Points are not consumed before a leaf is reached
        size_t offset = 2*std::distance(begin, begin+splitPivot); // (2*nSubleaves - 1) + 1

#ifdef _USE_TBB_
        tbb::task_group myGroup;
        myGroup.run([&]{mySelf->left  = BuildTreeParallel(points, pivot, begin, begin + splitPivot, mamaID + 1);});
        myGroup.run([&]{mySelf->right = BuildTreeParallel(points, pivot, begin + splitPivot, end, mamaID + offset);});
        myGroup.wait();
#else
        mySelf->left  = BuildTreeParallel(points, pivot, begin, begin + splitPivot, mamaID + 1);
        mySelf->right = BuildTreeParallel(points, pivot, begin + splitPivot, end, mamaID + offset);
#endif /* _USE_TBB_ */
    }
    else
    {
        // Leaf node, no babies
        *mySelf = Node(points.data() + Dim * (*begin), *begin, 0, mySelf, mySelf);
    }
    return mySelf;
}

template <size_t Dim, bool Randomized, typename T>
std::vector<KDTree<Dim, true, T>>
KDTree<Dim, Randomized, T>::BuildRandomizedTrees(
    DataContainer<T> const &points, const int nTrees,
    const Pivot pivot, const int nVarianceSamples) {
  if (nTrees < 1) {
    throw std::invalid_argument(
        "BuildRandomizedTrees: number of trees must be >=0.");
  }
  std::vector<KDTree<Dim, true, T>> trees(nTrees);
#pragma omp parallel for
  for (int i = 0; i < nTrees; ++i) {
    KDTree<Dim, true, T> tree(points, pivot, nVarianceSamples);
    trees[i] = std::move(tree);
  }
  return trees;
}

template <size_t Dim, bool Randomized, typename T>
std::ostream &
operator<<(std::ostream &os,
           KDTree<Dim, Randomized, T> const &tree) {
  std::function<void(
      typename KDTree<Dim, Randomized, T>::NodeItr,
      std::string indent)> printRecursive = [&os,
                                             &printRecursive](
      typename KDTree<Dim, Randomized, T>::NodeItr root,
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
