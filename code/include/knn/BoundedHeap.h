#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

namespace knn {

/// Thread UNSAFE bounded heap.
template <typename T, bool EvictWhenFull> class BoundedHeap {

public:
  BoundedHeap(size_t maxSize);

  BoundedHeap(size_t maxSize,
              std::function<bool(T const &, T const &b)> const &comp);

  size_t size() const;

  size_t maxSize() const;

  bool isFull() const;

  /// If called on an empty heap the returned value is undefined.
  T Max() const;

  /// If called on an empty heap the returned value is undefined.
  T Min() const;

  bool TryPopMax(T &elem);

  bool TryPush(T const &elem);

  /// Returns the data content of the heap, invalidating the instance of the
  /// class. Calling any function of the object after a call to Destroy() has
  /// undefined behaviour
  std::vector<T> Destroy();

private:
  size_t maxSize_;
  std::vector<T> content_{};
  T lowest_{};
  std::function<bool(T const &, T const &b)> comp_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace {

template <bool EvictWhenFull, typename T>
struct EvictImpl;

template <typename T>
struct EvictImpl<true, T> {
  EvictImpl() = delete;
  EvictImpl(EvictImpl<true, T> const &other) = delete;
  EvictImpl<true, T>& operator=(EvictImpl<true, T> const &other) = delete;
  static bool
  PushWhenFull(std::function<bool(T const &, T const &)> const &comp,
               T &back, std::vector<T> &content, T const &elem) {
    if (elem < content[0]) {
      if (elem < back) {
        back = elem;
      }
      content[0] = elem;
      std::make_heap(content.begin(), content.end(), comp);
      return true;
    }
    return false;
  }
};

template <typename T>
struct EvictImpl<false, T> {
  EvictImpl() = delete;
  EvictImpl(EvictImpl<false, T> const &other) = delete;
  EvictImpl<false, T>& operator=(EvictImpl<false, T> const &other) = delete;
  static bool PushWhenFull(std::function<bool(T const &, T const &)> const &,
                           T const &, std::vector<T> const &, T const &) {
    return false;
  }
};

} // End anonymous namespace

template <typename T, bool EvictWhenFull>
BoundedHeap<T, EvictWhenFull>::BoundedHeap(const size_t maxSize)
    : BoundedHeap(maxSize, std::less<T>()) {}

template <typename T, bool EvictWhenFull>
BoundedHeap<T, EvictWhenFull>::BoundedHeap(
    const size_t maxSize, std::function<bool(T const &, T const &)> const &comp)
    : maxSize_(maxSize), comp_(comp) {
  content_.reserve(maxSize);
}

template <typename T, bool EvictWhenFull>
size_t BoundedHeap<T, EvictWhenFull>::size() const {
  return content_.size();
}

template <typename T, bool EvictWhenFull>
size_t BoundedHeap<T, EvictWhenFull>::maxSize() const {
  return maxSize_;
}


template <typename T, bool EvictWhenFull>
bool BoundedHeap<T, EvictWhenFull>::isFull() const {
  return content_.size() == maxSize_;
}

template <typename T, bool EvictWhenFull>
T BoundedHeap<T, EvictWhenFull>::Max() const {
  assert(content_.size() > 0);
  return content_[0];
}

template <typename T, bool EvictWhenFull>
T BoundedHeap<T, EvictWhenFull>::Min() const {
  assert(content_.size() > 0);
  return lowest_;
}

template <typename T, bool EvictWhenFull>
bool BoundedHeap<T, EvictWhenFull>::TryPopMax(T &elem) {
  if (content_.size() > 0) {
    std::pop_heap(content_.begin(), content_.end(), comp_);
    elem = content_.back();
    content_.pop_back();
    return true;
  }
  return false;
}

template <typename T, bool EvictWhenFull>
bool BoundedHeap<T, EvictWhenFull>::TryPush(T const &elem) {
  if (content_.size() < maxSize_) {
    if (content_.size() == 0 || (content_.size() > 0 && comp_(elem, lowest_))) {
      lowest_ = elem;
    }
    content_.emplace_back(elem);
    std::push_heap(content_.begin(), content_.end(), comp_);
    return true;
  }
  return EvictImpl<EvictWhenFull, T>::PushWhenFull(comp_, lowest_, content_,
                                                   elem);
}

template <typename T, bool EvictWhenFull>
std::vector<T> BoundedHeap<T, EvictWhenFull>::Destroy() {
  return std::move(content_);
}

} // End namespace knn
