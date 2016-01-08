#include <utility>
#include "knn/BoundedHeap.h"
#include "knn/Common.h"

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
    KNN_ASSERT(heapEvict.Max().first == high);
    KNN_ASSERT(heapEvict.Min().first == low);
  };
  auto noEvictAssume = [&](float high, float low) {
    KNN_ASSERT(heapNoEvict.Max().first == high);
    KNN_ASSERT(heapNoEvict.Min().first == low);
  };

  std::pair<float, int> pop;
  KNN_ASSERT(heapEvict.TryPush(std::make_pair(3, 3)));
  evictAssume(3, 3);
  KNN_ASSERT(heapEvict.TryPush(std::make_pair(2, 2)));
  evictAssume(3, 2);
  KNN_ASSERT(heapEvict.TryPopMax(pop));
  KNN_ASSERT(pop.second == 3);
  evictAssume(2, 2);
  KNN_ASSERT(heapEvict.TryPush(std::make_pair(3, 3)));
  KNN_ASSERT(heapEvict.TryPush(std::make_pair(4, 4)));
  evictAssume(4, 2);
  KNN_ASSERT(heapEvict.TryPush(std::make_pair(5, 5)));
  evictAssume(4, 2);
  KNN_ASSERT(heapEvict.TryPush(std::make_pair(1, 1)));
  evictAssume(3, 1);
  std::sort_heap(heapEvictVec.begin(), heapEvictVec.end(), comp);
  KNN_ASSERT(heapEvictVec[0].second == 1);
  KNN_ASSERT(heapEvictVec[1].second == 2);
  KNN_ASSERT(heapEvictVec[2].second == 3);

  KNN_ASSERT(heapNoEvict.TryPush(std::make_pair(3, 3)));
  noEvictAssume(3, 3);
  KNN_ASSERT(heapNoEvict.TryPush(std::make_pair(2, 2)));
  noEvictAssume(3, 2);
  KNN_ASSERT(heapNoEvict.TryPush(std::make_pair(4, 4)));
  noEvictAssume(4, 2);
  KNN_ASSERT(!heapNoEvict.TryPush(std::make_pair(5, 5)));
  noEvictAssume(4, 2);
  KNN_ASSERT(!heapNoEvict.TryPush(std::make_pair(1, 1)));
  noEvictAssume(4, 2);
  std::sort_heap(heapNoEvictVec.begin(), heapNoEvictVec.end(), comp);
  KNN_ASSERT(heapNoEvictVec[0].second == 2);
  KNN_ASSERT(heapNoEvictVec[1].second == 3);
  KNN_ASSERT(heapNoEvictVec[2].second == 4);

  return 0;
}
