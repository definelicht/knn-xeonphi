/* File:   myKNN.h */
/* Date:   Tue Oct  6 22:49:02 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Naive kNN algorithm */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef MYKNN_H_IKP2P367
#define MYKNN_H_IKP2P367

#include <cassert>
#include <vector>
#include <iterator>
#include <algorithm>
#include <utility>
#include <unordered_map>


template <typename TData, typename TagKernel, typename MetricKernel>
class myKNN
{
public:
    myKNN (const std::vector<TData>& test_data) : dataSet_(test_data) { }

    std::vector<typename TagKernel::TagType> classify(const size_t k, const std::vector<TData>& trainingSet) const;

private:
    const std::vector<TData> dataSet_;
    TagKernel tagKernel_;
    MetricKernel metricKernel_;
};


template <typename TData, typename TagKernel, typename MetricKernel>
std::vector<typename TagKernel::TagType> myKNN<TData, TagKernel, MetricKernel>::classify(const size_t k, const std::vector<TData>& trainingSet) const
{
    assert(k <= dataSet_.size());
    std::vector<typename TagKernel::TagType> bestGuess(trainingSet.size());

    // parallelize over training set
#pragma omp parallel for
    for (size_t i = 0; i < trainingSet.size(); ++i)
    {
        // 1.) compute k smallest distances
        auto minMetric = std::make_pair(std::vector<typename MetricKernel::MetricType>(k), std::vector<size_t>(k));
        for (size_t j = 0; j < k; ++j)
        {
            minMetric.first[j]  = metricKernel_(trainingSet[i], dataSet_[j]);
            minMetric.second[j] = j;
        }
        size_t jMax = std::distance(minMetric.first.cbegin(), std::max_element(minMetric.first.cbegin(), minMetric.first.cend()));

        for (size_t j = k; j < dataSet_.size(); ++j)
        {
            const typename MetricKernel::MetricType thisMetric = metricKernel_(trainingSet[i], dataSet_[j]);
            if (thisMetric > minMetric.first[jMax]) continue;
            minMetric.first[jMax]  = thisMetric;
            minMetric.second[jMax] = j;
            jMax = std::distance(minMetric.first.cbegin(), std::max_element(minMetric.first.cbegin(), minMetric.first.cend()));
        }

        // 2.) vote
        std::unordered_map<typename TagKernel::TagType, size_t> votes;
        for (size_t j = 0; j < k; ++j)
        {
            auto check = votes.emplace(tagKernel_(dataSet_[minMetric.second[j]]), 1);
            if (check.second == false) ++(check.first->second);
        }
        size_t maxVote = 0;
        typename TagKernel::TagType maxTag;
        for (auto& v : votes)
        {
            if (v.second > maxVote)
            {
                maxVote = v.second;
                maxTag = v.first;
            }
        }
        bestGuess[i] = maxTag;
    }
    return bestGuess;
}

#endif /* MYKNN_H_IKP2P367 */
