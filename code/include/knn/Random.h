#pragma once

#include <algorithm>
#include <random>
#include <type_traits>

namespace knn {

#define CPPUTILS_RANDOM_THREADLOCAL thread_local

#ifdef __has_feature
  #if !__has_feature(cxx_thread_local)
    #undef CPPUTILS_RANDOM_THREADLOCAL
    #define CPPUTILS_RANDOM_THREADLOCAL
    #warning "Random.h: No support for thread_local: random number generation \
              is not thread safe."
  #endif
#endif

typedef std::mt19937 DefaultRng;

/// \brief Seed random number generator of the given type.
template <typename RngType = DefaultRng, typename SeedType>
void SetSeed(SeedType seed);

/// \brief Fills the provided range with uniform random numbers, using the
///        (optionally) specified random number generator.
template <typename Iterator, typename RngType = DefaultRng>
void Uniform(Iterator begin, Iterator end);

/// \return A uniform random number.
template <typename T, typename RngType = DefaultRng>
T Uniform();

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T, typename DistType = typename std::conditional<
                          std::is_floating_point<T>::value,
                          std::uniform_real_distribution<T>,
                          std::uniform_int_distribution<T>>::type>
DistType& UniformDist() {
  static CPPUTILS_RANDOM_THREADLOCAL DistType dist;
  return dist;
}

template <typename RngType>
struct EngineWrapper {
  RngType engine;
  EngineWrapper() : engine(0) {
    std::random_device rd;
    std::uniform_int_distribution<unsigned int> dist;
    engine.seed(dist(rd));
  }
};

template <typename RngType>
struct SharedRng {
  static EngineWrapper<RngType>& wrapper() {
    // Singleton pattern
    static CPPUTILS_RANDOM_THREADLOCAL EngineWrapper<RngType> wrapper;
    return wrapper;
  }
};

// Bitwise interpretation of input seed regardless of type
template <
    typename InputType,
    typename SeedType = typename std::conditional<
        sizeof(InputType) == sizeof(unsigned char), unsigned char,
        typename std::conditional<
            sizeof(InputType) == sizeof(unsigned short), unsigned short,
            typename std::conditional<
                sizeof(InputType) == sizeof(unsigned int), unsigned int,
                typename std::conditional<
                    sizeof(InputType) == sizeof(unsigned long), unsigned long,
                    typename std::conditional<
                        sizeof(InputType) == sizeof(unsigned long long),
                        unsigned long long, void>::type>::type>::type>::type>::
        type>
typename std::enable_if<!std::is_same<SeedType, void>::value, SeedType>::type
SeedToUnsigned(InputType seed) {
  return *reinterpret_cast<SeedType *>(&seed);
}

} // End anonymous namespace

template <typename RngType, typename SeedType>
void SetSeed(SeedType seed) {
  SharedRng<RngType>::wrapper().engine.seed(SeedToUnsigned<SeedType>(seed));
}

template <typename Iterator, typename RngType>
void Uniform(Iterator begin, Iterator end) {
  typedef typename std::iterator_traits<Iterator>::value_type T;
  std::for_each(begin, end, [](T &tgt) { tgt = Uniform<T, RngType>(); });
}

template <typename T, typename RngType>
T Uniform() {
  return UniformDist<T>()(SharedRng<RngType>::wrapper().engine);
}

} // End namespace knn 

#undef CPPUTILS_RANDOM_THREADLOCAL
