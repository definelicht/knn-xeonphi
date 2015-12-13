#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

namespace knn {

/// Thread UNSAFE bounded heap.
template <typename IteratorType, bool EvictWhenFull> class BoundedHeap {

public:
  using T = typename std::iterator_traits<IteratorType>::value_type;

  BoundedHeap(IteratorType begin, IteratorType end);

  BoundedHeap(IteratorType begin, IteratorType end,
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
  size_t size_, maxSize_;
  IteratorType begin_;
  T lowest_{};
  std::function<bool(T const &, T const &b)> comp_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace {

template <bool EvictWhenFull, typename IteratorType>
struct EvictImpl;

template <typename IteratorType>
struct EvictImpl<true, IteratorType> {
  using T = typename std::iterator_traits<IteratorType>::value_type;
  EvictImpl() = delete;
  EvictImpl(EvictImpl<true, IteratorType> const &other) = delete;
  EvictImpl<true, IteratorType> &
  operator=(EvictImpl<true, IteratorType> const &other) = delete;
  static bool
  PushWhenFull(std::function<bool(T const &, T const &)> const &comp, T &back,
               const IteratorType begin, const int maxSize, T const &elem) {
    if (elem < begin[0]) {
      if (elem < back) {
        back = elem;
      }
      begin[0] = elem;
      std::make_heap(begin, begin + maxSize, comp);
      return true;
    }
    return false;
  }
};

template <typename IteratorType>
struct EvictImpl<false, IteratorType> {
  using T = typename std::iterator_traits<IteratorType>::value_type;
  EvictImpl() = delete;
  EvictImpl(EvictImpl<false, IteratorType> const &other) = delete;
  EvictImpl<false, IteratorType> &
  operator=(EvictImpl<false, IteratorType> const &other) = delete;
  static bool PushWhenFull(std::function<bool(T const &, T const &)> const &,
                           T const &, const IteratorType, const int,
                           T const &) {
    return false;
  }
};

} // End anonymous namespace

template <typename IteratorType, bool EvictWhenFull>
BoundedHeap<IteratorType, EvictWhenFull>::BoundedHeap(const IteratorType begin,
                                                      const IteratorType end)
    : BoundedHeap(begin, end, std::less<T>()) {}

template <typename IteratorType, bool EvictWhenFull>
BoundedHeap<IteratorType, EvictWhenFull>::BoundedHeap(
    const IteratorType begin, const IteratorType end,
    std::function<bool(T const &, T const &)> const &comp)
    : size_(0), maxSize_(std::distance(begin, end)), begin_(begin),
      comp_(comp) {}

template <typename IteratorType, bool EvictWhenFull>
size_t BoundedHeap<IteratorType, EvictWhenFull>::size() const {
  return size_;
}

template <typename IteratorType, bool EvictWhenFull>
size_t BoundedHeap<IteratorType, EvictWhenFull>::maxSize() const {
  return maxSize_;
}

template <typename IteratorType, bool EvictWhenFull>
bool BoundedHeap<IteratorType, EvictWhenFull>::isFull() const {
  return size_ == maxSize_;
}

template <typename IteratorType, bool EvictWhenFull>
typename std::iterator_traits<IteratorType>::value_type
BoundedHeap<IteratorType, EvictWhenFull>::Max() const {
  return *begin_;
}

template <typename IteratorType, bool EvictWhenFull>
typename std::iterator_traits<IteratorType>::value_type
BoundedHeap<IteratorType, EvictWhenFull>::Min() const {
  return lowest_;
}

template <typename IteratorType, bool EvictWhenFull>
bool BoundedHeap<IteratorType, EvictWhenFull>::TryPopMax(
    typename std::iterator_traits<IteratorType>::value_type &elem) {
  if (size_ > 0) {
    std::pop_heap(begin_, begin_ + size_, comp_);
    --size_;
    elem = begin_[size_];
    return true;
  }
  return false;
}

template <typename IteratorType, bool EvictWhenFull>
bool BoundedHeap<IteratorType, EvictWhenFull>::TryPush(T const &elem) {
  if (size_ < maxSize_) {
    if (size_ == 0 || (size_ > 0 && comp_(elem, lowest_))) {
      lowest_ = elem;
    }
    begin_[size_] = elem;
    ++size_;
    std::push_heap(begin_, begin_ + size_, comp_);
    return true;
  }
  return EvictImpl<EvictWhenFull, IteratorType>::PushWhenFull(
      comp_, lowest_, begin_, maxSize_, elem);
}

} // End namespace knn
