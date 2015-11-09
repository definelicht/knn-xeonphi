#include <cassert>
#include <utility>
#include "knn/BoundedHeap.h"

using namespace knn;

int main() {
  auto comp = [](std::pair<float, int> const &a,
                 std::pair<float, int> const &b) { return a.first < b.first; };
  BoundedHeap<std::pair<float, int>, true> heapEvict(3, comp);
  BoundedHeap<std::pair<float, int>, false> heapNoEvict(3, comp);

  assert(heapEvict.TryPush(std::make_pair(3, 3)));
  assert(heapEvict.PeekFront().second == 3);
  assert(heapEvict.TryPush(std::make_pair(2, 2)));
  assert(heapEvict.PeekFront().second == 3);
  assert(heapEvict.TryPush(std::make_pair(4, 4)));
  assert(heapEvict.PeekFront().second == 4);
  assert(!heapEvict.TryPush(std::make_pair(5, 5)));
  assert(heapEvict.PeekFront().second == 4);
  assert(heapEvict.TryPush(std::make_pair(1, 1)));
  assert(heapEvict.PeekFront().second == 3);
  auto heapEvictContent = heapEvict.Destroy();
  std::sort_heap(heapEvictContent.begin(), heapEvictContent.end(), comp);
  assert(heapEvictContent[0].second == 1);
  assert(heapEvictContent[1].second == 2);
  assert(heapEvictContent[2].second == 3);

  assert(heapNoEvict.TryPush(std::make_pair(3, 3)));
  assert(heapNoEvict.PeekFront().second == 3);
  assert(heapNoEvict.TryPush(std::make_pair(2, 2)));
  assert(heapNoEvict.PeekFront().second == 3);
  assert(heapNoEvict.TryPush(std::make_pair(4, 4)));
  assert(heapNoEvict.PeekFront().second == 4);
  assert(!heapNoEvict.TryPush(std::make_pair(5, 5)));
  assert(heapNoEvict.PeekFront().second == 4);
  assert(!heapNoEvict.TryPush(std::make_pair(1, 1)));
  assert(heapNoEvict.PeekFront().second == 4);
  auto heapNoEvictContent = heapNoEvict.Destroy(); 
  std::sort_heap(heapNoEvictContent.begin(), heapNoEvictContent.end(), comp);
  assert(heapNoEvictContent[0].second == 2);
  assert(heapNoEvictContent[1].second == 3);
  assert(heapNoEvictContent[2].second == 4);

  return 0;
}
