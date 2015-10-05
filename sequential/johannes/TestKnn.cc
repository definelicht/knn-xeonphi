#include "knnnaive.h"
#include <iostream>
#include <random>

int main() {

  std::vector<std::pair<float, float>> training{
      {1, 1},  {2, 1},  {1, 2},   {2, 2},   {-1, 1},  {-2, 1},
      {-1, 2}, {-2, 2}, {-1, -1}, {-2, -1}, {-1, -2}, {-2, -2},
      {1, -1}, {2, -1}, {1, -2},  {2, -2}};
  std::vector<int> labels{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  KnnNaive<std::pair<float, float>, int> knn(training, labels);

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist(-10, 10);
  std::vector<std::pair<float, float>> test(32);
  for (auto &p : test) {
    p = std::make_pair(dist(rng), dist(rng));
  }
  auto classes =
      knn.Classify<float>(test, [](std::pair<float, float> const &a,
                                   std::pair<float, float> const &b) {
        float dx = a.first - b.first;
        float dy = a.second - b.second;
        return dx * dx + dy * dy;
      });

  for (int i = 0, iend = classes.size(); i < iend; ++i) {
    std::cout << "(" << test[i].first << ", " << test[i].second << ") -> "
              << classes[i] << "\n";
  }

  return 0;
}
