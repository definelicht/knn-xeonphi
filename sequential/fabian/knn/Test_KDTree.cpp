/* File:   Test_KDTree.cpp */
/* Date:   Sun Oct 18 22:35:26 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test KD-Tree class */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#include <iostream>
#include <vector>
#include <utility>
#include "myKDTree.h"
#include "TestData.h"
using namespace std;

int main()
{
    vector<TestData> data({
            {-2,-2,"lower-left"},
            {-2,-1,"lower-left"},
            {-1,-2,"lower-left"},
            {-1,-1,"lower-left"},
            {-2, 2,"upper-left"},
            {-2, 1,"upper-left"},
            {-1, 2,"upper-left"},
            {-1, 1,"upper-left"},
            {2,-2,"lower-right"},
            {2,-1,"lower-right"},
            {1,-2,"lower-right"},
            {1,-1,"lower-right"},
            {2, 2,"upper-right"},
            {2, 1,"upper-right"},
            {1, 2,"upper-right"},
            {1, 1,"upper-right"}});

    myKDTree<TestData> tree(std::move(data), TestData::Dim);
    cout << tree;
    return 0;
}
