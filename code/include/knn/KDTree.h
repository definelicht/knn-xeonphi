#pragma once

#include <cassert>
#include <algorithm>  // std::nth_element
#include <functional> // std::function
#include <numeric>    // std::iota
#include <ostream>
#include <random>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <thread> // std::thread::hardware_concurrency
#include <vector>
#include <tbb/parallel_for_each.h>
#include <tbb/task_group.h>
#include <tbb/scalable_allocator.h>

#include "knn/BoundedHeap.h"
#include "knn/Common.h"
#include "knn/Random.h"

namespace knn {

template <size_t Dim, bool Randomized, typename T>
class KDTree {

private:
  struct Node;
  using TreeItr = Node const*;
  using IndexItr =
      typename std::vector<size_t, tbb::scalable_allocator<size_t>>::iterator;

  struct alignas(64) Node {
    DataItr<T> value;
    size_t index;
    size_t splitDim;
    TreeItr left, right;
    Node();
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

  KDTree(DataContainer<T> const &points, Pivot pivot = Pivot::median,
         int nVarianceSamples = -1, int nHighestVariances = -1,
         const bool parallel = true);

  KDTree(KDTree<Dim, Randomized, T> const &);

  KDTree(KDTree<Dim, Randomized, T> &&);

  KDTree<Dim, Randomized, T> &
  operator=(KDTree<Dim, Randomized, T> const &);

  KDTree<Dim, Randomized, T> &
  operator=(KDTree<Dim, Randomized, T> &&);

  NodeItr Root() const;

  size_t nLeaves() const;

  constexpr size_t nDims() const;

  static std::vector<KDTree<Dim, true, T>>
  BuildRandomizedTrees(DataContainer<T> const &points, int nTrees,
                       Pivot pivot = Pivot::median, int nVarianceSamples = 100,
                       int nHighestVariances = Dim > 10 ? 5 : Dim / 2);

private:
  TreeItr BuildTree(DataContainer<T> const &data, Pivot pivot, IndexItr begin,
                    const IndexItr end, const size_t mamaID,
                    int nVarianceSamples, int nHighestVariances, int treeWidth,
                    int splitWidth);

  std::vector<Node, tbb::scalable_allocator<Node>> tree_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace {

template <bool Randomized, size_t Dim, typename T>
class GetSplitDim;

template <size_t Dim, typename T>
class GetSplitDim<false, Dim, T> {
public:
  static size_t Get(const size_t, std::array<T, Dim> const &variances) {
    return std::distance(
        variances.cbegin(),
        std::max_element(variances.cbegin(), variances.cend()));
  }
};

template <size_t Dim, typename T>
class GetSplitDim<true, Dim, T> {

public:
  static size_t Get(const size_t nHighestVariances,
                    std::array<T, Dim> const &variances) {
    std::array<size_t, Dim> indices;
    std::iota(indices.begin(), indices.end(), 0);
    auto begin = indices.begin();
    std::nth_element(begin, begin + nHighestVariances, indices.end(),
                     [&variances](size_t a, size_t b) {
                       return variances[a] > variances[b];
                     });
    return indices[random::Uniform<size_t>() % nHighestVariances];
  }
};

} // End anonymous namespace

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree() {}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree(DataContainer<T> const &points, Pivot pivot,
                                   int nVarianceSamples, int nHighestVariances,
                                   const bool parallel) : tree_{} {
  const size_t nLeaves = Dim > 0 ? points.size() / Dim : 0;
  static_assert(Dim > 0, "KDTree data cannot be zero-dimensional.");
  if (nLeaves == 0) {
    throw std::invalid_argument("KDTree received empty data set.");
  }
  if (points.size() % Dim != 0) {
    throw std::invalid_argument("KDTree received unbalanced data.");
  }
  const size_t nNodes = 2 * nLeaves - 1;
  tree_.resize(nNodes); // exact number of nodes if points are not consumed
                        // until leaf is reached, is tree is evenly split,
                        // always odd
  std::vector<size_t, tbb::scalable_allocator<size_t>> indices(nLeaves);
  std::iota(indices.begin(), indices.end(), 0);
  if (!parallel) {
    BuildTree(points, pivot, indices.begin(), indices.end(), 0,
              nVarianceSamples, nHighestVariances, 1, 1);
  } else {
    BuildTree(points, pivot, indices.begin(), indices.end(), 0,
              nVarianceSamples, nHighestVariances, 1,
              std::thread::hardware_concurrency());
  }
}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree(KDTree<Dim, Randomized, T> const &other)
    : tree_(other.tree_.size()) {
  const auto begin = tree_.data();
  const auto beginOther = other.tree_.data();
  for (int i = 0, iMax = other.tree_.size(); i < iMax; ++i) {
    const auto &leaf = other.tree_[i];
    tree_[i] = Node(
        leaf.value, leaf.index, leaf.splitDim,
        leaf.left != nullptr ? begin + std::distance(beginOther, leaf.left)
                             : nullptr,
        leaf.right != nullptr ? begin + std::distance(beginOther, leaf.right)
                              : nullptr);
  }
}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::KDTree(KDTree<Dim, Randomized, T> &&other)
    : tree_(std::move(other.tree_)) {}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T> &KDTree<Dim, Randomized, T>::
operator=(KDTree<Dim, Randomized, T> const &rhs) {
  tree_.resize(rhs.tree_.size());
  const auto begin = tree_.data();
  const auto beginOther = rhs.tree_.data();
  for (int i = 0, iMax = rhs.tree_.size(); i < iMax; ++i) {
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
  tree_ = std::move(rhs.tree_);
  return *this;
}

template <size_t Dim, bool Randomized, typename T>
KDTree<Dim, Randomized, T>::Node::Node()
    : value(nullptr), index(0), splitDim(0), left(nullptr), right(nullptr) {}

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
  return tree_.size();
}

template <size_t Dim, bool Randomized, typename T>
constexpr size_t KDTree<Dim, Randomized, T>::nDims() const {
  return Dim;
}

template <size_t Dim, bool Randomized, typename T>
typename KDTree<Dim, Randomized, T>::TreeItr
KDTree<Dim, Randomized, T>::BuildTree(DataContainer<T> const &points,
                                      Pivot pivot, IndexItr begin,
                                      const IndexItr end, const size_t mamaID,
                                      int nVarianceSamples,
                                      int nHighestVariances, int treeWidth,
                                      int splitWidth) {

    const auto mySelf = tree_.data() + mamaID;

    if (std::distance(begin, end) > 1) {
        const auto meanAndVariance =
        MeanAndVariance<T, Dim>(points, nVarianceSamples, begin, end);
        const size_t splitDim = GetSplitDim<Randomized, Dim, T>::Get(
            nHighestVariances, meanAndVariance.second);
        size_t splitPivot;
        if (pivot == Pivot::median) {
            splitPivot = std::distance(begin, end) / 2;
            std::nth_element(begin, begin + splitPivot, end,
                             [&points, &splitDim](size_t a, size_t b) {
                               return points[Dim * a + splitDim] <
                                      points[Dim * b + splitDim];
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

        *mySelf = Node(points.data() + Dim * splitIndex, splitIndex, splitDim);

        // Points are not consumed before a leaf is reached
        size_t offset =
            2 *
            std::distance(begin, begin + splitPivot); // (2*nSubleaves - 1) + 1

        if (treeWidth < splitWidth) { 
          // Keep spawning more tasks
          tbb::task_group myGroup;
          myGroup.run([&] {
            mySelf->left = BuildTree(
                points, pivot, begin, begin + splitPivot, mamaID + 1,
                nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
          });
          myGroup.run([&] {
            mySelf->right = BuildTree(
                points, pivot, begin + splitPivot, end, mamaID + offset,
                nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
          });
          myGroup.wait();
        } else {
          // Enough tasks, recurse the subtree alone
          mySelf->left = BuildTree(
              points, pivot, begin, begin + splitPivot, mamaID + 1,
              nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
          mySelf->right = BuildTree(
              points, pivot, begin + splitPivot, end, mamaID + offset,
              nVarianceSamples, nHighestVariances, 2 * treeWidth, splitWidth);
        }
    } else {
      // Leaf node, no babies
      *mySelf = Node(points.data() + Dim * (*begin), *begin, 0);
    }
    return mySelf;
}

template <size_t Dim, bool Randomized, typename T>
std::vector<KDTree<Dim, true, T>>
KDTree<Dim, Randomized, T>::BuildRandomizedTrees(DataContainer<T> const &points,
                                                 const int nTrees,
                                                 const Pivot pivot,
                                                 const int nVarianceSamples,
                                                 const int nHighestVariances) {
  if (nTrees < 1) {
    throw std::invalid_argument(
        "BuildRandomizedTrees: number of trees must be >=0.");
  }
  std::vector<KDTree<Dim, true, T>> trees(nTrees);
  tbb::parallel_for_each(
      trees.begin(), trees.end(), [&](KDTree<Dim, true, T> &tree) {
        tree = KDTree<Dim, true, T>(points, pivot, nVarianceSamples,
                                    nHighestVariances, true);
      });
  return trees;
}

template <size_t Dim, bool Randomized, typename T>
std::ostream &
operator<<(std::ostream &os,
           KDTree<Dim, Randomized, T> const &tree) {
  std::function<void(typename KDTree<Dim, Randomized, T>::NodeItr,
                     std::string indent)> printRecursive = [&os,
                                                            &printRecursive](
      typename KDTree<Dim, Randomized, T>::NodeItr root, std::string indent) {
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
