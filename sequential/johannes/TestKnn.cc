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
  std::vector<float> trainingKDTree;
  for (auto &p : training) {
    trainingKDTree.emplace_back(p.first);
    trainingKDTree.emplace_back(p.second);
  }
  KDTree<2, float, int> knn(trainingKDTree, labels);

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist(-10, 10);
  std::vector<std::pair<float, float>> testNaive(nSamples);
  std::vector<float> testKDTree(2*nSamples);
  for (int i = 0; i < nSamples; ++i) {
    testNaive[i] = std::make_pair(dist(rng), dist(rng));
    testKDTree[2*i] = testNaive[i].first;
    testKDTree[2*i+1] = testNaive[i].second;
  }
  auto classesNaive = knnNaive.Classify<float>(
      5, testNaive,
      [](std::pair<float, float> const &a, std::pair<float, float> const &b) {
        float dx = a.first - b.first;
        float dy = a.second - b.second;
        return dx * dx + dy * dy;
      });
  std::vector<int> classesKdTree;
  for (int i = 0; i < 32; ++i) {
    classesKdTree.emplace_back(knn.Knn<float>(
        5, testKDTree.cbegin() + 2 * i,
        [](typename std::vector<float>::const_iterator const &a,
           typename std::vector<float>::const_iterator const &b) {
          float dx = a[0] - b[0]; 
          float dy = a[1] - b[1];
          return dx * dx + dy * dy;
        }));
  }

  for (int i = 0, iEnd = classesNaive.size(); i < iEnd; ++i) {
    std::cout << "(" << testNaive[i].first << ", " << testNaive[i].second
              << ") -> " << classesNaive[i] << " / " << classesKdTree[i]
              << "\n";
  }

  return 0;
}
