#include <cassert>
#include <utility>
#include "knn/BoundedHeap.h"

using namespace knn;

int main() {
  auto comp = [](std::pair<float, int> const &a,
                 std::pair<float, int> const &b) { return a.first < b.first; };
  BoundedHeap<std::pair<float, int>> heap(3, comp);
  assert(heap.TryPush(std::make_pair(3, 3)));
  assert(heap.PeekFront().second == 3);
  assert(heap.TryPush(std::make_pair(2, 2)));
  assert(heap.PeekFront().second == 3);
  assert(heap.TryPush(std::make_pair(4, 4)));
  assert(heap.PeekFront().second == 4);
  assert(!heap.TryPush(std::make_pair(5, 5)));
  assert(heap.PeekFront().second == 4);
  assert(heap.TryPush(std::make_pair(1, 1)));
  assert(heap.PeekFront().second == 3);
  auto heapContent = heap.Destroy();
  std::sort_heap(heapContent.begin(), heapContent.end(), comp);
  assert(heapContent[0].second == 1);
  assert(heapContent[1].second == 2);
  assert(heapContent[2].second == 3);
  return 0;
}
