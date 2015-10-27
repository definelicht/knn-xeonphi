/* File:   TestData.h */
/* Date:   Sun Oct 18 22:23:58 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Test data class */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef TESTDATA_H_90MLSRGI
#define TESTDATA_H_90MLSRGI

#include <string>

class TestData
{
public:
    using DataType = double;
    using ArithmeticType = double;
    using MetricType = double;
    using TagType = std::string;
    static const size_t Dim = 2;

    TestData() : _a(0), _b(0), _tag("none") {}
    TestData(const DataType a, const DataType b, const TagType tag) : _a(a), _b(b), _tag(tag) {}

    DataType& operator[](const size_t i) { return (i==0) ? _a : _b; }
    DataType operator[](const size_t i) const { return (i==0) ? _a : _b; }

    static inline MetricType metricKernel(const TestData& a, const TestData& b) { return a.compare(b); }
    static inline TagType tagKernel(const TestData& c) { return c.tag(); }

    inline MetricType compare(const TestData& rhs) const { return (_a-rhs._a)*(_a-rhs._a) + (_b-rhs._b)*(_b-rhs._b); }
    inline TagType tag() const { return _tag; };

private:
    DataType _a, _b;
    TagType _tag;
};

#endif /* TESTDATA_H_90MLSRGI */
