/* File:   Test_KNN.cpp */
/* Date:   Wed Oct  7 20:37:13 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test myKNN */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#include "myKNN.h"
#include "Flower.h"
#include "CIFAR.h"
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <random>

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("USAGE: %s <k> <path to data file> [more files]\n", argv[0]);
        exit(-1);
    }
    /* vector<Flower> flowers = read_many_flowers(argv[2]); */

    // CIFAR images
    /* vector<I8_32> testData(50000); */
    /* { */
    /*     vector<I8_32> testData1 = load_cifar_data(argv[2]); */
    /*     vector<I8_32> testData2 = load_cifar_data(argv[3]); */
    /*     vector<I8_32> testData3 = load_cifar_data(argv[4]); */
    /*     vector<I8_32> testData4 = load_cifar_data(argv[5]); */
    /*     vector<I8_32> testData5 = load_cifar_data(argv[6]); */

    /*     // concat */
    /*     testData.insert(testData.end(), testData1.begin(), testData1.end()); */
    /*     testData.insert(testData.end(), testData2.begin(), testData2.end()); */
    /*     testData.insert(testData.end(), testData3.begin(), testData3.end()); */
    /*     testData.insert(testData.end(), testData4.begin(), testData4.end()); */
    /*     testData.insert(testData.end(), testData5.begin(), testData5.end()); */
    /* } */
    /* vector<I8_32> trainingData = load_cifar_data(argv[7]); */
    /* vector<string> category = load_cifar_category10(argv[8]); */

    vector<I8_32> testData = load_cifar_data(argv[2]);
    vector<I8_32> tr1 = load_cifar_data(argv[3]);
    vector<I8_32> trainingData;
    for (int i = 0; i < 10; ++i)
        trainingData.push_back(tr1[i]);
    vector<string> category = load_cifar_category10(argv[4]);

    /* vector<Flower> trainingData; */
    /* vector<Flower> testData; */
    /* vector<size_t> imap; */

    /* mt19937 generator; */
    /* uniform_real_distribution<double> dist(0,1); */
    /* for (size_t i = 0; i < flowers.size(); ++i) */
    /* { */
    /*     if (dist(generator) < 0.33) */
    /*     { */
    /*         trainingData.push_back(flowers[i]); */
    /*         imap.push_back(i); */
    /*     } */
    /*     else */
    /*         testData.push_back(flowers[i]); */
    /* } */

    // KNN
    const size_t k = atoi(argv[1]);
    /* myKNN<Flower, FlowerTag, FlowerMetric> knn(testData); */
    /* vector<typename FlowerTag::TagType> c = knn.classify(k, trainingData); */
    myKNN<I8_32, CIFARTag, CIFARMetric> knn(testData);
    vector<typename CIFARTag::TagType> c = knn.classify(k, trainingData);
    for (int i = 0; i < c.size(); ++i)
    {
        cout << category[c[i]] << endl;
        ostringstream name;
        name << "training_" << i << ".jpg";
        trainingData[i].print(name.str());
    }



    /* // check quality of estimate */
    /* size_t count = 0; */
    /* for (size_t i = 0; i < c.size(); ++i) */
    /* { */
    /*     printf("Predicted = %s, Actual = %s\n", c[i].c_str(), flowers[imap[i]].tag().c_str()); */
    /*     if (c[i] == flowers[imap[i]].tag()) ++count; */
    /* } */
    /* const double accuracy = static_cast<double>(count)/static_cast<double>(c.size()) * 100.0; */
    /* printf("Accuracy is %.2f%%\n", accuracy); */

    return 0;
}
