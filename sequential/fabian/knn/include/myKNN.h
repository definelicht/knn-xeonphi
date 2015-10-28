/* File:   myKNN.h */
/* Date:   Tue Oct  6 22:49:02 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Naive kNN algorithm */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef MYKNN_H_IKP2P367
#define MYKNN_H_IKP2P367

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iterator>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <numeric>

#include "BPQ.h"
#include "myKDTree.h"

/* TODO: (fabianw; Tue Oct 27 22:39:56 2015) this is not general! Might fail
 * with exotic ArithmeticType's. Fix this! */
template <typename X> X myabs(X x) { abort(); return x; }
template <> int myabs(int x)   { return std::fabs(x); }
template <> float myabs(float x)   { return std::fabs(x); }
template <> double myabs(double x) { return std::abs(x); }

template <typename TData>
class myKNN : public myKDTree<TData>
{
public:
    myKNN(std::vector<TData>&& test_data, const size_t data_dim) : myKDTree<TData>(std::move(test_data), data_dim) { }

    std::vector<typename TData::TagType> classify(const size_t k, const std::vector<TData>& trainingSet) const;
    std::vector<typename TData::TagType> classifyUsingTree(const size_t k, const std::vector<TData>& trainingSet) const;
    std::vector<TData> findKNN(const size_t k, const std::vector<TData>& trainingSet) const;
    std::vector<TData> findKNNUsingTree(const size_t k, const std::vector<TData>& trainingSet) const;

private:
    typename TData::TagType _vote(const BPQ<TData>& bpq) const;
    void _recursiveTreeKNN(const TData& Q, const typename myKDTree<TData>::NodeType* N, BPQ<TData>& bpq) const;
};


template <typename TData>
std::vector<typename TData::TagType> myKNN<TData>::classify(const size_t k, const std::vector<TData>& trainingSet) const
{
    assert(k <= myKDTree<TData>::_dataSet.size());
    std::vector<typename TData::TagType> bestGuess(trainingSet.size());

    // parallelize over training set
#pragma omp parallel for
    for (size_t i = 0; i < trainingSet.size(); ++i)
    {
        BPQ<TData> bpq(k);

        // 1.) compute k smallest distances
        for (size_t j = 0; j < myKDTree<TData>::_dataSet.size(); ++j)
            bpq.enqueue(TData::metricKernel(trainingSet[i], myKDTree<TData>::_dataSet[j]), &myKDTree<TData>::_dataSet[j]);

        // 2.) vote
        bestGuess[i] = _vote(bpq);
    }
    return bestGuess;
}


template <typename TData>
std::vector<typename TData::TagType> myKNN<TData>::classifyUsingTree(const size_t k, const std::vector<TData>& trainingSet) const
{
    assert(k <= myKDTree<TData>::_dataSet.size());
    std::vector<typename TData::TagType> bestGuess(trainingSet.size());

    // parallelize over training set
#pragma omp parallel for
    for (size_t i = 0; i < trainingSet.size(); ++i)
    {
        BPQ<TData> bpq(k);

        // 1.) compute k smallest distances using tree
        _recursiveTreeKNN(trainingSet[i], myKDTree<TData>::_proot, bpq);

        // 2.) vote
        bestGuess[i] = _vote(bpq);
    }
    return bestGuess;
}


template <typename TData>
std::vector<TData> myKNN<TData>::findKNN(const size_t k, const std::vector<TData>& trainingSet) const
{
    assert(k <= myKDTree<TData>::_dataSet.size());
    std::vector<TData> nnList(k*trainingSet.size());

    // parallelize over training set
#pragma omp parallel for
    for (size_t i = 0; i < trainingSet.size(); ++i)
    {
        BPQ<TData> bpq(k);

        // 1.) compute k smallest distances
        for (size_t j = 0; j < myKDTree<TData>::_dataSet.size(); ++j)
            bpq.enqueue(TData::metricKernel(trainingSet[i], myKDTree<TData>::_dataSet[j]), &myKDTree<TData>::_dataSet[j]);

        // 2.) copy k-nearest neighbors in increasing distance order
        std::vector<TData> nn = bpq.getValues();
        for (size_t j = 0; j < k; ++j)
            nnList[i*k + j] = nn[j];
    }
    return nnList;
}


template <typename TData>
std::vector<TData> myKNN<TData>::findKNNUsingTree(const size_t k, const std::vector<TData>& trainingSet) const
{
    assert(k <= myKDTree<TData>::_dataSet.size());
    std::vector<TData> nnList(k*trainingSet.size());

    // parallelize over training set
#pragma omp parallel for
    for (size_t i = 0; i < trainingSet.size(); ++i)
    {
        BPQ<TData> bpq(k);

        // 1.) compute k smallest distances using tree
        _recursiveTreeKNN(trainingSet[i], myKDTree<TData>::_proot, bpq);

        // 2.) copy k-nearest neighbors in increasing distance order
        std::vector<TData> nn = bpq.getValues();
        for (size_t j = 0; j < k; ++j)
            nnList[i*k + j] = nn[j];
    }
    return nnList;
}


template <typename TData>
typename TData::TagType myKNN<TData>::_vote(const BPQ<TData>& bpq) const
{
    std::unordered_map<typename TData::TagType, size_t> votes;
    std::vector<const TData*> nearest = bpq.getValues();
    for (size_t j = 0; j < bpq.size(); ++j)
    {
        auto check = votes.emplace(TData::tagKernel(*nearest[j]), 1);
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
    return maxTag;
}


template <typename TData>
void myKNN<TData>::_recursiveTreeKNN(const TData& Q, const typename myKDTree<TData>::NodeType* node, BPQ<TData>& bpq) const
{
    if (node == nullptr) return;

    bpq.enqueue(TData::metricKernel(Q, *node->data), node->data);

    const size_t currDim = node->dim;
    typename myKDTree<TData>::NodeType* other;

    // search branch determined by currDim
    /* TODO: (fabianw; Tue Oct 27 22:41:27 2015) need to cast to an appropriate
     * ArithmeticType depending on the data.  Is this efficient? Probably not.
     * This caused problems when doing arithmetic with e.g. unsigned uint8_t
     * used for the image class.  Plus you need an appropriate myabs() function
     * below for the corresponding ArithmeticType. This is buggy. */
    const typename TData::ArithmeticType currDiff = static_cast<typename TData::ArithmeticType>(Q[currDim]) - static_cast<typename TData::ArithmeticType>((*node->data)[currDim]);
    if (currDiff < 0)
    {
        other = node->child_r;
        _recursiveTreeKNN(Q, node->child_l, bpq);
    }
    else
    {
        other = node->child_l;
        _recursiveTreeKNN(Q, node->child_r, bpq);
    }

    // must search other branch too if hyphersphere intersects with hyperplane
    // of currDim or if queue is not full yet, since we look for k neighbors
    if ((myabs(currDiff) < bpq.maxPriority()) || !bpq.full())
        _recursiveTreeKNN(Q, other, bpq);
}

#endif /* MYKNN_H_IKP2P367 */
