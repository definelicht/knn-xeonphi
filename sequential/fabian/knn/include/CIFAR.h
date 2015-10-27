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

using I8_32 = Image_8bit<32>;

class CIFARData
{
public:
    typedef uint8_t DataType;
    typedef int ArithmeticType;
    typedef uint8_t TagType;
    typedef double MetricType;
    static const size_t Dim = I8_32::Dim;

    CIFARData() : _data() {}
    CIFARData(const I8_32& img) : _data(img) {}
    CIFARData(const CIFARData& c) : _data(c._data) {}
    CIFARData& operator=(const CIFARData& c)
    {
        if (this != &c)
            _data = c._data;
        return *this;
    }

    DataType& operator[](const size_t i) { return _data[i]; }
    DataType operator[](const size_t i) const { return _data[i]; }

    static inline TagType tagKernel(const CIFARData& c) { return c._data.label(); }
    static inline MetricType metricKernel(const CIFARData& a, const CIFARData& b)
    {
        const DataType * const pa = a._data.ptr();
        const DataType * const pb = b._data.ptr();
        MetricType d = 0.0;
        for (size_t i = 0; i < I8_32::Dim; ++i)
            d += (static_cast<MetricType>(pa[i]) - static_cast<MetricType>(pb[i]))*(static_cast<MetricType>(pa[i]) - static_cast<MetricType>(pb[i]));

        return d;
    }

    inline void dumpImage(const std::string& fname) const { _data.print(fname); }

private:
    I8_32 _data;
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
        ret[i].put(buffer);
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
