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
#include <unordered_map>
#include <utility>


template <typename TData, typename TTree>
class myKNN : public TTree
{
public:
    myKNN(std::vector<TData>&& test_data, const size_t data_dim) : TTree(std::move(test_data), data_dim) { }

    std::vector<typename TData::TagType> classify(const size_t k, const std::vector<TData>& trainingSet) const;
};


template <typename TData, typename TTree>
std::vector<typename TData::TagType> myKNN<TData, TTree>::classify(const size_t k, const std::vector<TData>& trainingSet) const
{
    assert(k <= TTree::_dataSet.size());
    std::vector<typename TData::TagType> bestGuess(trainingSet.size());

    // parallelize over training set
#pragma omp parallel for
    for (size_t i = 0; i < trainingSet.size(); ++i)
    {
        // 1.) compute k smallest distances
        auto minMetric = std::make_pair(std::vector<typename TData::MetricType>(k), std::vector<size_t>(k));
        for (size_t j = 0; j < k; ++j)
        {
            minMetric.first[j]  = TData::metricKernel(trainingSet[i], TTree::_dataSet[j]);
            minMetric.second[j] = j;
        }
        size_t jMax = std::distance(minMetric.first.cbegin(), std::max_element(minMetric.first.cbegin(), minMetric.first.cend()));

        for (size_t j = k; j < TTree::_dataSet.size(); ++j)
        {
            const typename TData::MetricType thisMetric = TData::metricKernel(trainingSet[i], TTree::_dataSet[j]);
            if (thisMetric > minMetric.first[jMax]) continue;
            minMetric.first[jMax]  = thisMetric;
            minMetric.second[jMax] = j;
            jMax = std::distance(minMetric.first.cbegin(), std::max_element(minMetric.first.cbegin(), minMetric.first.cend()));
        }

        // 2.) vote
        std::unordered_map<typename TData::TagType, size_t> votes;
        for (size_t j = 0; j < k; ++j)
        {
            auto check = votes.emplace(TData::tagKernel(TTree::_dataSet[minMetric.second[j]]), 1);
            if (check.second == false) ++(check.first->second);
        }
        size_t maxVote = 0;
        typename TData::TagType maxTag;
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
