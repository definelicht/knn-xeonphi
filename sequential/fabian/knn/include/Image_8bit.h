/* File:   Image_8bit.h */
/* Date:   Thu Oct 15 19:35:05 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Image class for 8bit data */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef IMAGE_8BIT_H_XC4WN5AJ
#define IMAGE_8BIT_H_XC4WN5AJ

#include <cstdint>
#include <string>
#include <cstring>
#include <vector>
#include "jpge.h"

struct pixel { uint8_t r, g, b; };

template <uint8_t N>
class Image_8bit
{
    void _copy(const Image_8bit& rhs)
    {
        _label = rhs._label;
        _ptrBase = &_pixelSet[0];
        memcpy(_ptrBase, rhs._ptrBase, 3*N*N);
    }

public:
    Image_8bit() : _label(0) { _ptrBase = &_pixelSet[0]; }
    Image_8bit(const Image_8bit& rhs)
    {
        _copy(rhs);
    }

    Image_8bit& operator=(const Image_8bit& rhs)
    {
        if (this != &rhs) _copy(rhs);
        return *this;
    }

    inline uint8_t& operator[](const size_t idx) { return _ptrBase[idx]; }
    inline const uint8_t& operator[](const size_t idx) const { return _ptrBase[idx]; }

    static const size_t Dim = 3*N*N;

    void print(const std::string filename) const
    {
        std::vector<pixel> img(N*N);
        for (int i = 0; i < N*N; ++i)
        {
            pixel px;
            px.r = _ptrBase[i];
            px.g = _ptrBase[i+1024];
            px.b = _ptrBase[i+2048];
            img[i] = px;
        }
        jpge::compress_image_to_jpeg_file(filename.c_str(), N, N, 3, (uint8_t*)img.data());
    }

    void put(const uint8_t * const ptrData)
    {
        _label = *ptrData;
        memcpy(_ptrBase, ptrData+1, 3*N*N);
    }

    inline uint8_t label() const { return _label; }
    inline uint8_t * ptr() { return _ptrBase; }
    inline uint8_t * ptrR() { return _ptrBase; }
    inline uint8_t * ptrG() { return _ptrBase+1024; }
    inline uint8_t * ptrB() { return _ptrBase+2048; }
    inline const uint8_t * ptr() const { return _ptrBase; }
    inline const uint8_t * ptrR() const { return _ptrBase; }
    inline const uint8_t * ptrG() const { return _ptrBase+1024; }
    inline const uint8_t * ptrB() const { return _ptrBase+2048; }

private:
    uint8_t _label;
    uint8_t _pixelSet[3*N*N];
    uint8_t* _ptrBase;
};

#endif /* IMAGE_8BIT_H_XC4WN5AJ */
