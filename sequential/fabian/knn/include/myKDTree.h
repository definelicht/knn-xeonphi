/* File:   myKDTree.h */
/* Date:   Sat Oct 17 15:44:36 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    KD-Tree */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef MYKDTREE_H_8IXIH5EW
#define MYKDTREE_H_8IXIH5EW

#include <cassert>
#include <cstddef>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <ostream>
#include <functional>
#include <string>
#include <utility>

template <typename T>
class KDNode
{
public:
    KDNode() : parent(nullptr), child_l(nullptr), child_r(nullptr), data(nullptr), dim(0) {}

    inline void set(KDNode<T>* p, KDNode<T>* cl, KDNode<T>* cr, T* d, const size_t D)
    {
        parent = p; child_l = cl, child_r = cr; data = d, dim = D;
    }

    KDNode<T> *parent, *child_l, *child_r;
    T* data;
    size_t dim;
};


template <typename TData>
class myKDTree
{
public:
    myKDTree(std::vector<TData>&& test_data, const size_t dim) :
        _dataSet(std::move(test_data)), _dataDim(dim), _treeSize(_dataSet.size()), _treeIdx(_dataSet.size()), _tree(), _proot(nullptr)
    {
        std::iota(_treeIdx.begin(), _treeIdx.end(), 0);
        _tree.reserve(_treeSize);
        _proot = _buildTree(_treeIdx.begin(), _treeIdx.end());
        assert(_proot == &_tree[0]);
    }

    using NodeType = KDNode<TData>;

    inline size_t dataDimension() const { return _dataDim; }
    inline size_t size() const { return _treeSize; }
    inline NodeType* root() { return _proot; }
    inline const NodeType* const root() const { return _proot; }

protected:
    std::vector<TData> _dataSet;
    const size_t _dataDim;
    const size_t _treeSize;
    std::vector<size_t> _treeIdx;
    std::vector<NodeType> _tree;
    NodeType* _proot;

private:
    using _indexIterator_t = std::vector<size_t>::iterator;
    NodeType* _buildTree(_indexIterator_t start, _indexIterator_t end, size_t curr_dim=0, NodeType* parent=nullptr);
};


template <typename TData>
typename myKDTree<TData>::NodeType* myKDTree<TData>::_buildTree(_indexIterator_t start, _indexIterator_t end, size_t curr_dim, NodeType* parent)
{
    const std::ptrdiff_t nElements = std::distance(start, end);
    if (0 >= nElements)
        return nullptr;
    else
    {
        std::sort(start, end,
                [curr_dim, this](size_t lhs, size_t rhs)
                { return this->_dataSet[lhs][curr_dim] < this->_dataSet[rhs][curr_dim]; });

        _indexIterator_t anchor = start + (nElements >> 1);
        NodeType* me = &(*_tree.emplace(_tree.end()));
        me->set(parent,nullptr,nullptr,&_dataSet[*anchor],curr_dim);

        curr_dim = (++curr_dim) % _dataDim;
        me->child_l = _buildTree(start, anchor, curr_dim, me);
        me->child_r = _buildTree(anchor+1, end, curr_dim, me);
        return me;
    }
}


template <typename T>
std::ostream& operator<<(std::ostream& lhs, myKDTree<T>& tree)
{
    std::function<void(typename myKDTree<T>::NodeType*, std::string)>
    printTree = [&lhs, &printTree](typename myKDTree<T>::NodeType* node, std::string indent)
    {
        T& nodeData = *(node->data);
        lhs << "(" << nodeData[0];
        for (size_t i = 1; i < T::Dim; ++i)
            lhs << ", " << nodeData[i];
        lhs << ")" << " Tag = " << nodeData.tag() << std::endl;
        indent += "----";
        if (node->child_l)
        {
            lhs << indent << "Left: ";
            printTree(node->child_l, indent);
        }
        if (node->child_r)
        {
            lhs << indent << "Right: ";
            printTree(node->child_r, indent);
        }
    };
    lhs << "Tree size = " << tree.size() << ", Data dimension = " << tree.dataDimension() << std::endl;
    lhs << "Root: ";
    printTree(tree.root(), "");
    return lhs;
}

#endif /* MYKDTREE_H_8IXIH5EW */
