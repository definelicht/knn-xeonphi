/* File:   CIFAR.h */
/* Date:   Thu Oct 15 19:32:48 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Loader for CIFAR image set
 *         (http://www.cs.toronto.edu/~kriz/cifar.html) */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef CIFAR_H_VNCKES3Q
#define CIFAR_H_VNCKES3Q

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include "Image_8bit.h"

typedef Image_8bit<32> I8_32;

class CIFARTag
{
public:
    typedef uint8_t TagType;
    inline uint8_t operator()(const I8_32& img) const { return img.label(); }
};

class CIFARMetric
{
public:
    typedef double MetricType;
    double operator()(const I8_32& a, const I8_32& b) const
    {
        const uint8_t * const Ra = a.ptrR();
        const uint8_t * const Ga = a.ptrG();
        const uint8_t * const Ba = a.ptrB();
        const uint8_t * const Rb = b.ptrR();
        const uint8_t * const Gb = b.ptrG();
        const uint8_t * const Bb = b.ptrB();
        double d = 0.0;
        for (int i = 0; i < 1024; ++i)
        {
            d += (Ra[i] - Rb[i])*(Ra[i] - Rb[i]);
            d += (Ga[i] - Gb[i])*(Ga[i] - Gb[i]);
            d += (Ba[i] - Bb[i])*(Ba[i] - Bb[i]);
        }
        return d;
    }
};

std::vector<I8_32> load_cifar_data(const std::string& filename)
{
    // CIFAR contains 10000 items per batch
    std::vector<I8_32> ret(10000);
    const size_t chunk = 1 + 3*1024; // label + RGB for 32x32 8bit image
    uint8_t buffer[chunk];

    std::ifstream data(filename, std::ios::binary);
    for (int i = 0; i < 10000; ++i)
    {
        data.read((char*)&buffer[0], chunk);
        ret[i].put(buffer[0], &buffer[1], &buffer[1+1024], &buffer[1+2*1024]);
    }
    data.close();

    return ret;
}

std::vector<std::string> load_cifar_category10(const std::string& filename)
{
    // CIFAR cat 10
    std::vector<std::string> ret(10);
    std::ifstream cat(filename);
    for (int i = 0; i < 10; ++i)
        cat >> ret[i];
    cat.close();

    return ret;
}

std::vector<std::string> load_cifar_category100(const std::string& filename)
{
    // CIFAR cat 100
    std::vector<std::string> ret(100);
    std::ifstream cat(filename);
    for (int i = 0; i < 100; ++i)
        cat >> ret[i];
    cat.close();

    return ret;
}

#endif /* CIFAR_H_VNCKES3Q */
