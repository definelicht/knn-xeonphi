/* File:   Flower.h */
/* Date:   Thu Oct  8 07:37:08 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Flower data */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef FLOWER_H_FD2HYURO
#define FLOWER_H_FD2HYURO
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class Flower
{
    double a_, b_, c_, d_;
    std::string name_;

public:
    Flower() : a_(0.0), b_(0.0), c_(0.0), d_(0.0), name_("") { }
    Flower(const double a, const double b, const double c, const double d, const std::string name) :
        a_(a), b_(b), c_(c), d_(d), name_(name) { }
    Flower(const Flower& rhs) : a_(rhs.a_), b_(rhs.b_), c_(rhs.c_), d_(rhs.d_), name_(rhs.name_) { }
    ~Flower() { }
    Flower& operator=(const Flower& rhs)
    {
        if (this != &rhs)
        {
            a_ = rhs.a_;
            b_ = rhs.b_;
            c_ = rhs.c_;
            d_ = rhs.d_;
            name_ = rhs.name_;
        }
        return *this;
    }

    inline std::string tag() const { return name_; };
    inline double compare(const Flower& f) const { return (a_-f.a_)*(a_-f.a_) + (b_-f.b_)*(b_-f.b_) + (c_-f.c_)*(c_-f.c_) + (d_-f.d_)*(d_-f.d_); }
    inline void bloom() const { std::cout << a_ << " " << b_ << " " << c_ << " " << d_ << " " << name_ << std::endl; }
};

class FlowerTag
{
public:
    typedef std::string TagType;
    inline std::string operator()(const Flower& f) const { return f.tag(); }
};

class FlowerMetric
{
public:
    typedef double MetricType;
    inline double operator()(const Flower& f, const Flower& g) const { return f.compare(g); }
};

std::vector<Flower> read_many_flowers(const std::string file)
{
    std::ifstream in(file.c_str());
    double a, b, c, d;
    std::string name;
    std::vector<Flower> ret;
    while (in.peek() != EOF)
    {
        in >> a >> b >> c >> d >> name;
        ret.push_back(Flower(a, b, c, d, name));
        in >> std::ws;
    }
    return ret;
}

#endif /* FLOWER_H_FD2HYURO */
