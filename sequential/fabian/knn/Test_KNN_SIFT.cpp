/* File:   Test_KNN_SIFT.cpp */
/* Date:   Thu Oct 29 08:09:39 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test case using SIFT data set */
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
#include "TexMex.h"
#include "ArgumentParser.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv)
{
    using TestData = TexMexData<float, 128>;
    /* using TestData = TexMexData<float, 3>; */

    ArgumentParser parser(argc, (const char**)argv);
    parser.print_args();

    parser.set_strict_mode();
    bool bAlive = true;
    bAlive = bAlive && parser.check("-k");
    bAlive = bAlive && parser.check("-base");
    bAlive = bAlive && parser.check("-query");
    bAlive = bAlive && parser.check("-groundtruth");
    if (!bAlive)
    {
        fprintf(stderr, "USAGE: %s -k <k> -base <SIFT base vecs> -query <SIFT query vecs> -groundtruth <goundtruth vecs>\n", argv[0]);
        abort();
    }
    parser.unset_strict_mode();

    // load data
    vector<TestData> base   = load_texmex_data<float, 128>(parser("-base").asString());
    vector<TestData> query  = load_texmex_data<float, 128>(parser("-query").asString());
    /* vector<TestData> base   = load_texmex_data<float, 3>(parser("-base").asString()); */
    /* vector<TestData> query  = load_texmex_data<float, 3>(parser("-query").asString()); */
    vector<int> groundtruth = load_texmex_data_vec<int>(parser("-groundtruth").asString());

    // KNN
    myKNN<TestData> knn(std::move(base), TestData::Dim);
    vector<TestData> c;
    const size_t k = parser("-k").asInt();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    if (parser.check("-kdtree"))
    {
        cout << "Find kNN using KD-Tree..." << endl;
        c = knn.findKNNUsingTree(k, query);
    }
    else
    {
        cout << "Find kNN by naive algorithm..." << endl;
        c = knn.findKNN(k, query);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    cout << "Wallclock for kNN search = " << time_span.count() << " s" << endl;

    // check
    if (100==k)
    {
        vector<TestData> base = load_texmex_data<float, 128>(parser("-base").asString());
        size_t count = 0;
        for (size_t i = 0; i < query.size(); ++i)
            for (size_t j = 0; j < k; ++j)
            {
                TestData& b = c[i*k + j];
                const typename TestData::TagType nnTag = b.tag();
                const int trueTag = groundtruth[i*(k+1) + j+1];
                if (trueTag == nnTag) ++count;
                else
                {
                    TestData& a = base[trueTag];
                    const typename TestData::MetricType dist = TestData::metricKernel(a, b);
                    cout << "Tag missmatch:" << endl;
                    cout << "\tNearest-Neighbor tag = " << nnTag << endl;
                    cout << "\tGroundtruth tag      = " << trueTag << endl;
                    cout << "\tDistance             = " << dist << endl;
                }
            }
        const float accuracy = static_cast<float>(count)/static_cast<float>(k*query.size())*100.0f;
        cout << "Accuracy is " << accuracy << "%" << endl;
    }

    return 0;
}
