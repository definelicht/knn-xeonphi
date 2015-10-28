/* File:   Test_KNN_CIFAR.cpp */
/* Date:   Tue Oct 27 21:14:29 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test KNN using CIFAR 10 data */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */

#include <cassert>
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
#include "ArgumentParser.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv)
{
    ArgumentParser parser(argc, (const char**)argv);
    parser.print_args();

    parser.set_strict_mode();
    char** batches;
    int nbatches;
    bool bAlive = false;
    for (int i=1; i < argc; ++i)
    {
        const string argument(argv[i]);
        if (argument == "-batches")
        {
            batches  = (argv + (i+1));
            nbatches = argc - (i+1);
            bAlive = true;
            break;
        }
    }
    bAlive = bAlive && parser.check("-k");
    bAlive = bAlive && parser.check("-test");
    bAlive = bAlive && parser.check("-categories");
    bAlive = bAlive && nbatches==5;
    if (!bAlive)
    {
        fprintf(stderr, "USAGE: %s -k <k> -test <CIFAR test batch> -categories <CIFAR category file> -batches <list of 5 CIFAR data files>\n", argv[0]);
        abort();
    }
    parser.unset_strict_mode();

    vector<CIFARData> data;
    data.reserve(50000);
    for (int i = 0; i < nbatches; ++i)
    {
        // load the test data
        vector<I8_32> batch = load_cifar_data(string(batches[i]));
        for (auto& v : batch)
            data.push_back(CIFARData(v));
    }

    vector<CIFARData> test;
    test.reserve(10000);
    {
        // load the training data
        vector<I8_32> batch = load_cifar_data(parser("-test").asString());
        for (auto& v : batch)
            test.push_back(CIFARData(v));
    }

    vector<string> category(10);
    {
        ifstream catIn(parser("-categories").asString());
        for (int i = 0; i < 10; ++i)
            catIn >> category[i];
        catIn.close();
    }
    if (parser.check("-print"))
    {
        const size_t n = parser("-print").asInt();
        char buf[256];
        assert(n < 10000);
        for (size_t i = 0; i < n; ++i)
        {
            sprintf(buf, "data_%04d.jpg", i);
            data[i].dumpImage(string(buf));
            sprintf(buf, "test_%04d.jpg", i);
            test[i].dumpImage(string(buf));
        }
        for (size_t i = 0; i < 10; ++i)
            fprintf(stdout, "Category %d = %s\n", i, category[i].c_str());
    }

    // KNN
    myKNN<CIFARData> knn(std::move(data), CIFARData::Dim);
    vector<typename CIFARData::TagType> c;
    const size_t k = parser("-k").asInt();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    if (parser.check("-kdtree"))
    {
        cout << "Using KD-Tree..." << endl;
        c = knn.classifyUsingTree(k, test);
    }
    else
    {
        cout << "Naive classification..." << endl;
        c = knn.classify(k, test);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    cout << "Wallclock for classification = " << time_span.count() << " s" << endl;

    // print classification
    for (size_t i = 0; i < c.size(); ++i)
        fprintf(stdout, "Classified: %s\n", category[c[i]].c_str());

    return 0;
}
