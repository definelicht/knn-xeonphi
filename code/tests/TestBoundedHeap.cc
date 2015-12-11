#include <cassert>
#include <utility>
#include "knn/BoundedHeap.h"

using namespace knn;

int main() {
  auto comp = [](std::pair<float, int> const &a,
                 std::pair<float, int> const &b) { return a.first < b.first; };
  std::vector<std::pair<float, int>> heapEvictVec(3);
  std::vector<std::pair<float, int>> heapNoEvictVec(3);
  BoundedHeap<typename decltype(heapEvictVec)::iterator, true> heapEvict(
      heapEvictVec.begin(), heapEvictVec.end(), comp);
  BoundedHeap<typename decltype(heapNoEvictVec)::iterator, false> heapNoEvict(
      heapNoEvictVec.begin(), heapNoEvictVec.end(), comp);

  auto evictAssume = [&](float high, float low) {
    assert(heapEvict.Max().first == high);
    assert(heapEvict.Min().first == low);
  };
  auto noEvictAssume = [&](float high, float low) {
    assert(heapNoEvict.Max().first == high);
    assert(heapNoEvict.Min().first == low);
  };

  std::pair<float, int> pop;
  assert(heapEvict.TryPush(std::make_pair(3, 3)));
  evictAssume(3, 3);
  assert(heapEvict.TryPush(std::make_pair(2, 2)));
  evictAssume(3, 2);
  assert(heapEvict.TryPopMax(pop));
  assert(pop.second == 3);
  evictAssume(2, 2);
  assert(heapEvict.TryPush(std::make_pair(3, 3)));
  assert(heapEvict.TryPush(std::make_pair(4, 4)));
  evictAssume(4, 2);
  assert(heapEvict.TryPush(std::make_pair(5, 5)));
  evictAssume(4, 2);
  assert(heapEvict.TryPush(std::make_pair(1, 1)));
  evictAssume(3, 1);
  std::sort_heap(heapEvictVec.begin(), heapEvictVec.end(), comp);
  assert(heapEvictVec[0].second == 1);
  assert(heapEvictVec[1].second == 2);
  assert(heapEvictVec[2].second == 3);

  assert(heapNoEvict.TryPush(std::make_pair(3, 3)));
  noEvictAssume(3, 3);
  assert(heapNoEvict.TryPush(std::make_pair(2, 2)));
  noEvictAssume(3, 2);
  assert(heapNoEvict.TryPush(std::make_pair(4, 4)));
  noEvictAssume(4, 2);
  assert(!heapNoEvict.TryPush(std::make_pair(5, 5)));
  noEvictAssume(4, 2);
  assert(!heapNoEvict.TryPush(std::make_pair(1, 1)));
  noEvictAssume(4, 2);
  std::sort_heap(heapNoEvictVec.begin(), heapNoEvictVec.end(), comp);
  assert(heapNoEvictVec[0].second == 2);
  assert(heapNoEvictVec[1].second == 3);
  assert(heapNoEvictVec[2].second == 4);

  return 0;
}
