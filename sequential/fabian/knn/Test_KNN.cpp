/* File:   Test_KNN.cpp */
/* Date:   Wed Oct  7 20:37:13 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test myKNN */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#include "myKNN.h"
#include "myKDTree.h"
#include "TestData.h"
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <utility>

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("USAGE: %s <k>\n", argv[0]);
        exit(-1);
    }

    // KNN
    const size_t k = atoi(argv[1]);
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

    vector<TestData> test({
            {-0.4,-1.3,"lower-left"},
            {-3,2.4,"upper-left"},
            {1.2,-0.1,"lower-right"},
            {4.5,1.6,"upper-right"}});

    myKNN<TestData> knn(std::move(data), TestData::Dim);
    /* vector<typename TestData::TagType> c = knn.classify(k, test); */
    vector<typename TestData::TagType> c = knn.classifyUsingTree(k, test);

    // check quality of estimate
    size_t count = 0;
    for (size_t i = 0; i < c.size(); ++i)
    {
        printf("Predicted = %s, Actual = %s\n", c[i].c_str(), test[i].tag().c_str());
        if (c[i] == test[i].tag()) ++count;
    }
    const double accuracy = static_cast<double>(count)/static_cast<double>(c.size()) * 100.0;
    printf("Accuracy is %.2f%%\n", accuracy);

    return 0;
}
