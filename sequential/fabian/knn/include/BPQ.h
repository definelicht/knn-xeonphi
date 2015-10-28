/* File:   BPQ.h */
/* Date:   Sat Oct 24 11:04:05 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Bounded Priority Queue */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef BPQ_H_IWWD4ZNV
#define BPQ_H_IWWD4ZNV

#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>

template <typename T>
class BPQ
{
public:
    BPQ(const size_t k) : _size(k) { _priority.reserve(k); _value.reserve(k); }

    inline size_t size() const { return _priority.size(); }
    inline bool full() const { return _priority.size() == _size; }
    inline typename T::MetricType maxPriority() const { return _priority[_iMax]; }
    void enqueue(typename T::MetricType m, const T* t)
    {
        if (!full())
        {
            _priority.push_back(m);
            _value.push_back(t);
        }
        else
        {
            if (m > _priority[_iMax]) return;
            _priority[_iMax] = m;
            _value[_iMax] = t;
        }
        _iMax = std::distance(_priority.cbegin(), std::max_element(_priority.cbegin(), _priority.cend()));
    }

    std::vector<const T*> getValues() const
    {
        std::vector<size_t> sortedIdx(_priority.size());
        std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                [this](const size_t lhs, const size_t rhs)
                { return this->_priority[lhs] < this->_priority[rhs]; });

        std::vector<const T*> copies;
        copies.reserve(_priority.size());
        for (size_t i = 0; i < _priority.size(); ++i)
            copies.push_back(_value[sortedIdx[i]]);
        return copies;
    }

    std::vector<const T> getValues() const
    {
        std::vector<size_t> sortedIdx(_priority.size());
        std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                [this](const size_t lhs, const size_t rhs)
                { return this->_priority[lhs] < this->_priority[rhs]; });

        std::vector<const T> copies;
        copies.reserve(_priority.size());
        for (size_t i = 0; i < _priority.size(); ++i)
            copies.push_back(*_value[sortedIdx[i]]);
        return copies;
    }

private:
    const size_t _size;
    std::vector<typename T::MetricType> _priority;
    std::vector<const T*> _value;
    size_t _iMax;
};

#endif /* BPQ_H_IWWD4ZNV */
