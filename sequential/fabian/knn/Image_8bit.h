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
public:
    Image_8bit() : _label(0) { }
    Image_8bit(const Image_8bit& rhs)
    {
        _label = rhs._label;
        memcpy(&_R[0][0], &(rhs._R[0][0]), N*N);
        memcpy(&_G[0][0], &(rhs._G[0][0]), N*N);
        memcpy(&_B[0][0], &(rhs._B[0][0]), N*N);
    }

    static const uint8_t size = N;

    void print(const std::string filename) const
    {
        std::vector<pixel> img(N*N);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
            {
                pixel px;
                px.r = _R[j][i];
                px.g = _G[j][i];
                px.b = _B[j][i];
                img[j*N+i] = px;
            }
        jpge::compress_image_to_jpeg_file(filename.c_str(), N, N, 3, (uint8_t*)img.data());
    }

    void put(const uint8_t l, const uint8_t * const r, const uint8_t * const g, const uint8_t * const b)
    {
        _label = l;
        memcpy(&_R[0][0], r, N*N);
        memcpy(&_G[0][0], g, N*N);
        memcpy(&_B[0][0], b, N*N);
    }

    inline uint8_t label() const { return _label; }
    inline uint8_t * const ptrR() { return &_R[0][0]; }
    inline uint8_t * const ptrG() { return &_G[0][0]; }
    inline uint8_t * const ptrB() { return &_B[0][0]; }
    inline const uint8_t * const ptrR() const { return &_R[0][0]; }
    inline const uint8_t * const ptrG() const { return &_G[0][0]; }
    inline const uint8_t * const ptrB() const { return &_B[0][0]; }

private:
    uint8_t _label;
    uint8_t _R[N][N];
    uint8_t _G[N][N];
    uint8_t _B[N][N];
};

#endif /* IMAGE_8BIT_H_XC4WN5AJ */
