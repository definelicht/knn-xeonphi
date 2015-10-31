/* File:   TexMex.h */
/* Date:   Wed Oct 28 22:19:26 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    TexMex data set */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef TEXMEX_H_ITJFVPGJ
#define TEXMEX_H_ITJFVPGJ

#include <cassert>
#include <cstddef>
#include <fstream>
#include <cstring>

template <typename T, size_t _dim>
class TexMexData
{
public:
    typedef T DataType;
    typedef T ArithmeticType;
    typedef size_t TagType;
    typedef T MetricType;
    static const size_t Dim = _dim;
    static size_t count;

    TexMexData() {}
    TexMexData(const T * const c) : _tag(count++) { _copy(c); }
    TexMexData(const TexMexData& c) { _tag = c._tag; _copy(c); }
    TexMexData& operator=(const TexMexData& c)
    {
        if (this != &c)
        {
            _tag = c._tag;
            _copy(c);
        }
        return *this;
    }

    DataType& operator[](const size_t i) { assert(i<_dim); return _data[i]; }
    DataType operator[](const size_t i) const { assert(i<_dim); return _data[i]; }

    inline size_t size() const { return _dim; }
    inline TagType tag() const { return _tag; }

    static inline TagType tagKernel(const TexMexData& c) { return c._tag; }
    static inline MetricType metricKernel(const TexMexData& a, const TexMexData& b)
    {
        MetricType d = 0.0;
        for (size_t i = 0; i < _dim; ++i)
            d += (a._data[i]-b._data[i])*(a._data[i]-b._data[i]);
        return d;
    }

private:
    TagType _tag;
    T _data[_dim];

    inline void _copy(const TexMexData& rhs)
    {
        memcpy(_data, rhs._data, _dim*sizeof(T));
    }
    inline void _copy(const T * const rhs)
    {
        memcpy(_data, rhs, _dim*sizeof(T));
    }
};

template <typename T, size_t _dim>
size_t TexMexData<T, _dim>::count = 0;


template <typename T, size_t d>
std::vector<TexMexData<T,d> > load_texmex_data(const std::string& filename)
{
    std::ifstream binfile(filename, std::ios::binary);
    binfile.seekg(0, std::ios::end);
    auto fileSize = binfile.tellg();
    binfile.seekg(0, std::ios::beg);

    const size_t N = fileSize / (sizeof(int) + d*sizeof(T));
    std::vector<TexMexData<T,d> > data;
    data.reserve(N);

    int dim;
    T buf[d];
    for (size_t i = 0; i < N; ++i)
    {
        binfile.read((char*)(&dim), sizeof(int));
        assert(dim == d);
        binfile.read((char*)(&buf[0]), d*sizeof(T));
        data.push_back(TexMexData<T,d>(&buf[0]));
    }
    return data;
}


template <typename T>
std::vector<T> load_texmex_data_vec(const std::string& filename)
{
    assert(sizeof(T) == 4); // T must be 4 bytes wide
    std::ifstream binfile(filename, std::ios::binary);
    binfile.seekg(0, std::ios::end);
    auto fileSize = binfile.tellg();
    binfile.seekg(0, std::ios::beg);

    int d;
    binfile.read((char*)(&d), sizeof(int));
    binfile.seekg(0, std::ios::beg);

    const size_t N = fileSize / (sizeof(int) + d*sizeof(T));
    std::vector<T> data(N*(1+d));

    int dim;
    T * const buf = new T[d];
    for (size_t i = 0; i < N; ++i)
    {
        binfile.read((char*)(&dim), sizeof(int));
        binfile.read((char*)(&buf[0]), d*sizeof(T));
        assert(dim == d);
        data[i*(1+d)] = dim;
        for (int j = 0; j < dim; ++j)
            data[i*(1+d) + 1 + j] = buf[j];
    }
    delete[] buf;
    return data;
}

#endif /* TEXMEX_H_ITJFVPGJ */
