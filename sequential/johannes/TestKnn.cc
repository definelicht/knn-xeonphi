#include "Knn.h"
#include "KnnNaive.h"
#include "KDTree.h"
#include <iostream>
#include <random>
#include <utility>

int main() {

  constexpr int nSamples = 32;

  std::vector<std::pair<float, float>> training{
      {1, 1},  {2, 1},  {1, 2},   {2, 2},   {-1, 1},  {-2, 1},
      {-1, 2}, {-2, 2}, {-1, -1}, {-2, -1}, {-1, -2}, {-2, -2},
      {1, -1}, {2, -1}, {1, -2},  {2, -2}};
  std::vector<int> labels{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  assert(training.size() == labels.size());
  KnnNaive<std::pair<float, float>, int> knnNaive(training, labels);
  Knn<float, int> knn(
      {{1, 2, 1, 2, -1, -2, -1, -2, -1, -2, -1, -2, 1, 2, 1, 2},
       {1, 1, 2, 2, 1, 1, 2, 2, -1, -1, -2, -2, -1, -1, -2, -2}},
      {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3});

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist(-10, 10);
  std::vector<std::pair<float, float>> test(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    test[i] = std::make_pair(dist(rng), dist(rng));
  }
  auto classesNaive =
      knnNaive.Classify<float>(5, test, [](std::pair<float, float> const &a,
                                           std::pair<float, float> const &b) {
        float dx = a.first - b.first;
        float dy = a.second - b.second;
        return dx * dx + dy * dy;
      });
  std::vector<int> classesKdTree;
  for (int i = 0; i < 32; ++i) {
    classesKdTree.emplace_back(knn.Classify<float>(
        5, std::vector<float>{test[i].first, test[i].second},
        [](std::vector<float> const &a, std::vector<float> const &b) {
          float dx = a[0] - b[0];
          float dy = a[1] - b[1];
          return dx * dx + dy * dy;
        }));
  }

  for (int i = 0, iEnd = classesNaive.size(); i < iEnd; ++i) {
    std::cout << "(" << test[i].first << ", " << test[i].second << ") -> "
              << classesNaive[i] << " / " << classesKdTree[i] << "\n";
  }

  return 0;
}
