#include <cassert>
#include <utility>
#include "knn/BoundedHeap.h"

using namespace knn;

int main() {
  auto comp = [](std::pair<float, int> const &a,
                 std::pair<float, int> const &b) { return a.first < b.first; };
  BoundedHeap<std::pair<float, int>, true> heapEvict(3, comp);
  BoundedHeap<std::pair<float, int>, false> heapNoEvict(3, comp);

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
  assert(!heapEvict.TryPush(std::make_pair(5, 5)));
  evictAssume(4, 2);
  assert(heapEvict.TryPush(std::make_pair(1, 1)));
  evictAssume(3, 1);
  auto heapEvictContent = heapEvict.Destroy();
  std::sort_heap(heapEvictContent.begin(), heapEvictContent.end(), comp);
  assert(heapEvictContent[0].second == 1);
  assert(heapEvictContent[1].second == 2);
  assert(heapEvictContent[2].second == 3);

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
  auto heapNoEvictContent = heapNoEvict.Destroy(); 
  std::sort_heap(heapNoEvictContent.begin(), heapNoEvictContent.end(), comp);
  assert(heapNoEvictContent[0].second == 2);
  assert(heapNoEvictContent[1].second == 3);
  assert(heapNoEvictContent[2].second == 4);

  return 0;
}
