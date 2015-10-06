/* File:   Test_KNN.cpp */
/* Date:   Wed Oct  7 20:37:13 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test myKNN */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#include "myKNN.h"
#include "Flower.h"
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <random>

using namespace std;

int main(int argc, char** argv)
{
    if (3 != argc)
    {
        printf("USAGE: %s <k> <path to data file>\n", argv[0]);
        exit(-1);
    }
    vector<Flower> flowers = read_many_flowers(argv[2]);

    vector<Flower> dataSet, trainingSet;
    vector<size_t> imap;

    mt19937 generator;
    uniform_real_distribution<double> dist(0,1);
    for (size_t i = 0; i < flowers.size(); ++i)
    {
        if (dist(generator) < 0.33)
        {
            trainingSet.push_back(flowers[i]);
            imap.push_back(i);
        }
        else
            dataSet.push_back(flowers[i]);
    }

    // KNN
    const size_t k = atoi(argv[1]);
    myKNN<Flower, FlowerTag, FlowerMetric> knn(dataSet);
    vector<typename FlowerTag::TagType> c = knn.classify(k, trainingSet);

    // check quality of estimate
    size_t count = 0;
    for (size_t i = 0; i < c.size(); ++i)
    {
        printf("Predicted = %s, Actual = %s\n", c[i].c_str(), flowers[imap[i]].tag().c_str());
        if (c[i] == flowers[imap[i]].tag()) ++count;
    }
    const double accuracy = static_cast<double>(count)/static_cast<double>(c.size()) * 100.0;
    printf("Accuracy is %.2f%%\n", accuracy);

    return 0;
}
