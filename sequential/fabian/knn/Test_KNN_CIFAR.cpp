/* File:   Test_KNN_CIFAR.cpp */
/* Date:   Tue Oct 27 21:14:29 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test KNN using CIFAR 10 data */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <utility>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

#include "myKNN.h"
#include "myKDTree.h"
#include "CIFAR.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv)
{
    if (argc < 9)
    {
        printf("USAGE: %s <k> <CIFAR filenames>\n", argv[0]);
        exit(-1);
    }

    vector<CIFARData> data(50000);
    /* data.reserve(50000); */
    for (int i = 2; i < 7; ++i)
    {
        // load the test data
        vector<I8_32> batch = load_cifar_data(string(argv[i]));
        for (auto& v : batch)
            data.push_back(v);
    }

    vector<CIFARData> test(10000);
    {
        // load the training data
        vector<I8_32> batch = load_cifar_data(string(argv[7]));
        for (auto& v : batch)
            test.push_back(v);
    }

    vector<string> category(10);
    {
        ifstream catIn(argv[8]);
        string dummy;
        for (int i = 0; i < 10; ++i)
        {
            catIn >> dummy;
            category.push_back(dummy);
        }
    }

    // KNN
    const size_t k = atoi(argv[1]);
    myKNN<CIFARData> knn(std::move(data), CIFARData::Dim);
    /* vector<typename CIFARData::TagType> c = knn.classify(k, test); */
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    vector<typename CIFARData::TagType> c = knn.classifyUsingTree(k, test);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    cout << "Wallclock for classification = " << time_span.count() << " s" << endl;

    // print classification
    for (size_t i = 0; i < c.size(); ++i)
        printf("Classified: %s\n", category[c[i]].c_str());

    return 0;
}
