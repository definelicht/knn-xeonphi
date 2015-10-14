#pragma once

#include <numeric>   // std::iota
#include <ostream>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>

template <typename DataType, typename LabelType> class KDTree {

private:
  struct Node {
    using TreeIterator_t = typename std::vector<Node>::const_iterator;
    std::vector<DataType> value;
    LabelType label;
    TreeIterator_t parent, left, right;
    Node(std::vector<DataType> &&_value, LabelType const &_label,
         TreeIterator_t const &_parent, TreeIterator_t const &_end);
  };

public:
  using Point_t = std::vector<DataType>;
  using Dim_t = std::vector<DataType>;

  class Ptr {
  public:
    Ptr(typename Node::TreeIterator_t const &node,
        typename Node::TreeIterator_t const &end);
    Ptr Left() const;
    Ptr Right() const;
    Ptr Parent() const;
    bool GoLeft();
    bool GoRight();
    bool GoParent();
    bool inBounds() const;
    Point_t const& value() const;
    LabelType const& label() const;
  private:
    typename Node::TreeIterator_t node_;
    const typename Node::TreeIterator_t end_;
  };

  KDTree(std::vector<Dim_t> const &points,
         std::vector<LabelType> const &labels);

  KDTree(KDTree<DataType, LabelType> const &other) = default;

  KDTree(KDTree<DataType, LabelType> &&other) = default;

  KDTree<DataType, LabelType> &
  operator=(KDTree<DataType, LabelType> const &rhs) = default;

  KDTree<DataType, LabelType> &
  operator=(KDTree<DataType, LabelType> &&rhs) = default;

  size_t size() const;

  size_t nDims() const;

  Ptr Root() const;

private:
  typename std::vector<Node>::const_iterator BuildTree(
      std::vector<Dim_t> const &points, std::vector<LabelType> const &labels,
      std::vector<size_t>::iterator begin, std::vector<size_t>::iterator end,
      size_t dim, typename Node::TreeIterator_t const &parent);

  const size_t nDims_, size_;
  std::vector<Node> tree_{};
};

template <typename DataType, typename LabelType>
KDTree<DataType, LabelType>::KDTree(std::vector<Dim_t> const &points,
                                    std::vector<LabelType> const &labels)
    : nDims_(points.size()), size_(nDims_ > 0 ? points[0].size() : 0) {
  if (nDims_ == 0) {
    throw std::invalid_argument("KDTree received zero-dimensional data.");
  }
  if (size_ == 0) {
    throw std::invalid_argument("KDTree received empty data set.");
  }
  if (labels.size() != size_) {
    throw std::invalid_argument(
        "KDTree received mismatched point and label size.");
  }
  for (auto &nDims_ : points) {
    if (nDims_.size() != size_) {
      throw std::invalid_argument("KDTree data has inconsistent length.");
    }
  }
  tree_.reserve(size_);
  std::vector<size_t> indices(size_);
  std::iota(indices.begin(), indices.end(), 0);
  BuildTree(points, labels, indices.begin(), indices.end(), 0,
            tree_.cbegin() + size_);
}

template <typename DataType, typename LabelType>
KDTree<DataType, LabelType>::Node::Node(std::vector<DataType> &&_value,
                                        LabelType const &_label,
                                        TreeIterator_t const &_parent,
                                        TreeIterator_t const &_end)
    : value(_value), label(_label), parent(_parent), left(_end), right(_end) {}

template <typename DataType, typename LabelType>
KDTree<DataType, LabelType>::Ptr::Ptr(typename Node::TreeIterator_t const &node,
                                      typename Node::TreeIterator_t const &end)
    : node_(node), end_(end) {}

template <typename DataType, typename LabelType>
typename KDTree<DataType, LabelType>::Ptr
KDTree<DataType, LabelType>::Ptr::Left() const {
  return {node_->left, end_};
}

template <typename DataType, typename LabelType>
typename KDTree<DataType, LabelType>::Ptr
KDTree<DataType, LabelType>::Ptr::Right() const {
  return {node_->right, end_};
}

template <typename DataType, typename LabelType>
typename KDTree<DataType, LabelType>::Ptr
KDTree<DataType, LabelType>::Ptr::Parent() const {
  return {node_->parent, end_};
}

template <typename DataType, typename LabelType>
bool KDTree<DataType, LabelType>::Ptr::GoLeft() {
  return (node_ = node_->left) != end_;
}

template <typename DataType, typename LabelType>
bool KDTree<DataType, LabelType>::Ptr::GoRight() {
  return (node_ = node_->right) != end_;
}

template <typename DataType, typename LabelType>
bool KDTree<DataType, LabelType>::Ptr::GoParent() {
  return (node_ = node_->parent) != end_;
}

template <typename DataType, typename LabelType>
bool KDTree<DataType, LabelType>::Ptr::inBounds() const {
  return node_ != end_;
}

template <typename DataType, typename LabelType>
typename KDTree<DataType, LabelType>::Point_t const &
KDTree<DataType, LabelType>::Ptr::value() const {
  return node_->value;
}

template <typename DataType, typename LabelType>
LabelType const &KDTree<DataType, LabelType>::Ptr::label() const {
  return node_->label;
}

template <typename DataType, typename LabelType>
typename KDTree<DataType, LabelType>::Ptr
KDTree<DataType, LabelType>::Root() const {
  return {tree_.cbegin(), tree_.cend()};
}

template <typename DataType, typename LabelType>
size_t KDTree<DataType, LabelType>::size() const {
  return size_;
}

template <typename DataType, typename LabelType>
size_t KDTree<DataType, LabelType>::nDims() const {
  return nDims_;
}

template <typename DataType, typename LabelType>
typename std::vector<typename KDTree<DataType, LabelType>::Node>::const_iterator
KDTree<DataType, LabelType>::BuildTree(
    std::vector<Dim_t> const &points, std::vector<LabelType> const &labels,
    std::vector<size_t>::iterator begin, std::vector<size_t>::iterator end,
    size_t dim, typename Node::TreeIterator_t const &parent) {
  if (begin >= end) {
    return tree_.cbegin() + size_;
  }
  std::sort(begin, end, [dim, &points](size_t a, size_t b) {
    return points[dim][a] < points[dim][b];
  });
  std::vector<size_t>::iterator median =
      begin + (std::distance(begin, end) >> 1);
  auto treeItr_ = tree_.end();
  {
    std::vector<DataType> pivot(nDims_);
    for (size_t d = 0; d < nDims_; ++d) {
      pivot[d] = points[d][*median];
    }
    tree_.emplace_back(std::move(pivot), labels[*median],
                       parent, tree_.cbegin() + size_);
  }
  if (++dim == points.size()) dim = 0;
  treeItr_->left = BuildTree(points, labels, begin, median, dim, treeItr_);
  treeItr_->right = BuildTree(points, labels, median + 1, end, dim, treeItr_);
  return treeItr_;
}

template <typename DataType, typename LabelType>
std::ostream &operator<<(std::ostream &os,
                         KDTree<DataType, LabelType> const &tree) {
  std::function<void(typename KDTree<DataType, LabelType>::Ptr,
                     std::string indent)> printRecursive = [&os,
                                                            &printRecursive](
      typename KDTree<DataType, LabelType>::Ptr root, std::string indent) {
    indent += "  ";
    auto &val = root.value();
    os << indent << "(" << val[0];
    for (int i = 1, iEnd = val.size(); i < iEnd; ++i) {
      os << ", " << val[i];
    }
    os << ") -> " << root.label() << "\n";
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
