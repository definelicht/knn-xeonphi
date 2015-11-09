#pragma once

#include <algorithm>
#include <functional>
#include <vector>

namespace knn {

/// Thread UNSAFE bounded heap.
template <typename T> class BoundedHeap {

public:
  BoundedHeap(size_t maxSize);

  BoundedHeap(size_t maxSize,
              std::function<bool(T const &, T const &b)> const &comp);

  size_t size() const;

  size_t maxSize() const;

  /// If called on an empty heap the returned value is undefined.
  T PeekFront() const;

  bool TryPush(T const &elem);

  /// Returns the data content of the heap, invalidating the instance of the
  /// class. Calling any function of the object after a call to Destroy() has
  /// undefined behaviour
  std::vector<T> Destroy();

private:
  size_t maxSize_;
  std::vector<T> content_{};
  std::function<bool(T const &, T const &b)> comp_;
};

template <typename T>
BoundedHeap<T>::BoundedHeap(const size_t maxSize)
    : BoundedHeap(maxSize, std::less<T>()) {}

template <typename T>
BoundedHeap<T>::BoundedHeap(
    const size_t maxSize, std::function<bool(T const &, T const &)> const &comp)
    : maxSize_(maxSize), comp_(comp) {
  content_.reserve(maxSize);
}

template <typename T> size_t BoundedHeap<T>::size() const {
  return content_.size();
}

template <typename T> size_t BoundedHeap<T>::maxSize() const {
  return maxSize_;
}

template <typename T> T BoundedHeap<T>::PeekFront() const {
  return content_[0];
}

template <typename T> bool BoundedHeap<T>::TryPush(T const &elem) {
  if (content_.size() == maxSize_) {
    if (elem < content_[0]) {
      content_[0] = elem;
      std::make_heap(content_.begin(), content_.end(), comp_);
      return true;
    }
  } else {
    content_.emplace_back(elem);
    std::push_heap(content_.begin(), content_.end(), comp_);
    return true;
  }
  return false;
}

template <typename T> std::vector<T> BoundedHeap<T>::Destroy() {
  return std::move(content_);
}

} // End namespace knn
