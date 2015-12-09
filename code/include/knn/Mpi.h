#pragma once

#include <array>
#include <iterator>
#include <stdexcept>
#include <mpi.h>
#include "knn/Common.h"

namespace knn {

namespace mpi {

enum class Op {
  max,
  min,
  sum,
  prod
};

namespace {

template <typename T> struct MpiType;

#define CPPUTILS_MPI_SENDBACKEND(TYPE, MPI_TYPE_NAME)                          \
  template <> struct MpiType<TYPE> {                                           \
    static MPI_Datatype value() { return MPI_TYPE_NAME; }                      \
  };
CPPUTILS_MPI_SENDBACKEND(int, MPI_INT);
CPPUTILS_MPI_SENDBACKEND(unsigned, MPI_UNSIGNED);
CPPUTILS_MPI_SENDBACKEND(long, MPI_LONG);
CPPUTILS_MPI_SENDBACKEND(char, MPI_CHAR);
CPPUTILS_MPI_SENDBACKEND(float, MPI_FLOAT);
CPPUTILS_MPI_SENDBACKEND(double, MPI_DOUBLE);
#undef CPPUTILS_MPI_SENDBACKEND

template <Op op> struct MpiOp;

#define CPPUTILS_MPI_OP(OP, MPI_OP_NAME)                                       \
  template <> struct MpiOp<OP> {                                               \
    static MPI_Op value() { return MPI_OP_NAME; }                              \
  };
CPPUTILS_MPI_OP(Op::max, MPI_MAX);
CPPUTILS_MPI_OP(Op::min, MPI_MIN);
CPPUTILS_MPI_OP(Op::sum, MPI_SUM);
CPPUTILS_MPI_OP(Op::prod, MPI_PROD);
#undef CPPUTILS_MPI_OP

} // End anonymous namespace

inline int rank(MPI_Comm const &comm) {
  int output;
  MPI_Comm_rank(comm, &output);
  return output;
}

inline int rank() { return rank(MPI_COMM_WORLD); }

inline int size(MPI_Comm const &comm) {
  int output;
  MPI_Comm_size(comm, &output);
  return output;
}

inline int size() { return size(MPI_COMM_WORLD); }

template <typename IteratorType, typename = CheckRandomAccess<IteratorType>>
void Send(IteratorType begin, const IteratorType end, const int destination,
          const int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  using T = typename std::iterator_traits<const IteratorType>::value_type;
  MPI_Send(&(*begin), std::distance(begin, end), MpiType<T>::value(),
           destination, tag, comm);
}

template <typename IteratorType, typename = CheckRandomAccess<IteratorType>>
MPI_Request SendAsync(IteratorType begin, const IteratorType end,
                      const int destination, const int tag = 0,
                      MPI_Comm comm = MPI_COMM_WORLD) {
  using T = typename std::iterator_traits<const IteratorType>::value_type;
  MPI_Request request;
  MPI_Isend(&(*begin), std::distance(begin, end), MpiType<T>::value(),
            destination, tag, comm, &request);
  return request;
}

template <typename IteratorType, typename = CheckRandomAccess<IteratorType>>
MPI_Status Receive(IteratorType begin, const IteratorType end, const int source,
                   const int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  using T = typename std::iterator_traits<IteratorType>::value_type;
  MPI_Status status;
  MPI_Recv(&(*begin), std::distance(begin, end), MpiType<T>::value(), source,
           tag, comm, &status);
  return status;
}

template <typename IteratorType, typename = CheckRandomAccess<IteratorType>>
MPI_Request ReceiveAsync(IteratorType begin, const IteratorType end,
                         const int source, const int tag = 0,
                         MPI_Comm comm = MPI_COMM_WORLD) {
  using T = typename std::iterator_traits<IteratorType>::value_type;
  MPI_Request request;
  MPI_Irecv(&(*begin), std::distance(begin, end), MpiType<T>::value(), source,
            tag, comm, &request);
  return request;
}

template <typename SendIterator, typename ReceiveIterator,
          typename = CheckRandomAccess<SendIterator>,
          typename = CheckRandomAccess<ReceiveIterator>>
void Gather(SendIterator sendBegin, const SendIterator sendEnd,
            ReceiveIterator receiveBegin, const int root,
            MPI_Comm comm = MPI_COMM_WORLD) {
  using TSend = typename std::iterator_traits<SendIterator>::value_type;
  using TReceive = typename std::iterator_traits<ReceiveIterator>::value_type;
  static_assert(sizeof(TSend) == sizeof(TReceive),
                "Send and receive types must be of equal size.");
  const int nElements = std::distance(sendBegin, sendEnd);
  MPI_Gather(&(*sendBegin), nElements, MpiType<TSend>::value(),
             &(*receiveBegin), nElements, MpiType<TReceive>::value(), root,
             comm);
}

template <typename SendIterator, typename ReceiveIterator,
          template <class, class> class ContainerType,
          typename = CheckRandomAccess<SendIterator>,
          typename = CheckRandomAccess<ReceiveIterator>>
void Gather(SendIterator sendBegin, const SendIterator sendEnd,
            ReceiveIterator receiveBegin,
            ContainerType<int, std::allocator<int>> &receiveSizes,
            ContainerType<int, std::allocator<int>> &receiveOffsets,
            const int root, MPI_Comm comm = MPI_COMM_WORLD) {
  using TSend = typename std::iterator_traits<SendIterator>::value_type;
  using TReceive = typename std::iterator_traits<ReceiveIterator>::value_type;
  static_assert(sizeof(TSend) == sizeof(TReceive),
                "Send and receive types must be of equal size.");
  MPI_Gatherv(&(*sendBegin), std::distance(sendBegin, sendEnd),
              MpiType<TSend>::value(), &(*receiveBegin), receiveSizes.data(),
              receiveOffsets.data(), MpiType<TReceive>::value(), root, comm);
}

template <typename SendIterator, typename ReceiveIterator,
          typename = CheckRandomAccess<SendIterator>,
          typename = CheckRandomAccess<ReceiveIterator>>
void GatherAll(SendIterator sendBegin, const SendIterator sendEnd,
               ReceiveIterator receiveBegin, MPI_Comm comm = MPI_COMM_WORLD) {
  using TSend = typename std::iterator_traits<SendIterator>::value_type;
  using TReceive = typename std::iterator_traits<ReceiveIterator>::value_type;
  static_assert(sizeof(TSend) == sizeof(TReceive),
                "Send and receive types must be of equal size.");
  const int nElements = std::distance(sendBegin, sendEnd);
  MPI_Allgather(&(*sendBegin), nElements, MpiType<TSend>::value(),
                &(*receiveBegin), nElements, MpiType<TReceive>::value(), comm);
}

template <typename SendIterator, typename ReceiveIterator,
          template <class, class> class ContainerType,
          typename = CheckRandomAccess<SendIterator>,
          typename = CheckRandomAccess<ReceiveIterator>>
void GatherAll(SendIterator sendBegin, const SendIterator sendEnd,
               ReceiveIterator receiveBegin,
               ContainerType<int, std::allocator<int>> &receiveSizes,
               ContainerType<int, std::allocator<int>> &receiveOffsets,
               MPI_Comm comm = MPI_COMM_WORLD) {
  using TSend = typename std::iterator_traits<SendIterator>::value_type;
  using TReceive = typename std::iterator_traits<ReceiveIterator>::value_type;
  static_assert(sizeof(TSend) == sizeof(TReceive),
                "Send and receive types must be of equal size.");
  MPI_Allgatherv(&(*sendBegin), std::distance(sendBegin, sendEnd),
                 MpiType<TSend>::value(), &(*receiveBegin), receiveSizes.data(),
                 receiveOffsets.data(), MpiType<TReceive>::value(), comm);
}

template <Op op, typename SendIterator, typename ReceiveIterator,
          typename = CheckRandomAccess<SendIterator>,
          typename = CheckRandomAccess<ReceiveIterator>>
void Reduce(SendIterator sendBegin, const SendIterator sendEnd,
            ReceiveIterator receiveBegin, const int root,
            MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Reduce(
      &(*sendBegin), &(*receiveBegin), std::distance(sendBegin, sendEnd),
      MpiType<typename std::iterator_traits<SendIterator>::value_type>::value(),
      MpiOp<op>::value(), root, comm);
}

    inline MPI_Status
    Wait(MPI_Request &request) {
  MPI_Status status;
  MPI_Wait(&request, &status);
  return status;
}

template <template <class, class> class ContainerType>
ContainerType<MPI_Status, std::allocator<MPI_Status>>
WaitAll(ContainerType<MPI_Request, std::allocator<MPI_Request>> &requests) {
  ContainerType<MPI_Status, std::allocator<MPI_Status>> statuses(
      requests.size());
  MPI_Waitall(requests.size(), requests.data(), statuses.data());
  return statuses;
}

class Context {

public:
  inline Context() : argc_(), argv_(nullptr) { MPI_Init(nullptr, nullptr); }
  inline Context(int argc, char **argv) : argc_(argc), argv_(argv) {
    MPI_Init(&argc_, &argv_);
  }
  inline ~Context() { MPI_Finalize(); }
  Context(Context const &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context const &) = delete;
  Context &operator=(Context &&) = delete;

private:
  int argc_;
  char **argv_;
};

template <size_t Dim> class CartesianGrid {

  static_assert(Dim > 0, "Cartesian grid must have a least one dimension.");

public:
  CartesianGrid(std::array<int, Dim> const &dimensions,
                const bool periodic = false, MPI_Comm comm = MPI_COMM_WORLD)
      : dimensions_(dimensions) {
    std::fill(periods_.begin(), periods_.end(), periodic);
    MPI_Cart_create(comm, Dim, dimensions_.data(), periods_.data(), true,
                    &cartComm_);
    MPI_Cart_get(cartComm_, Dim, dimensions_.data(), periods_.data(),
                 coords_.data());
  }

  template <size_t GetDim> int get() const { return coords_[GetDim]; }

  int get(const size_t dim) const { return coords_[dim]; }

  template <size_t GetDim> int getMax() const {
    static_assert(GetDim < Dim, "Requested dimension is out of bounds.");
    return dimensions_[GetDim];
  }

  int getMax(const size_t dim) const { return dimensions_[dim]; }

  typename std::enable_if<(Dim > 1), int>::type row() const {
    return coords_[Dim - 2];
  }

  typename std::enable_if<(Dim > 1), int>::type col() const {
    return coords_[Dim - 1];
  }

  typename std::enable_if<(Dim > 1), int>::type rowMax() const {
    return dimensions_[Dim - 2];
  }

  typename std::enable_if<(Dim > 1), int>::type colMax() const {
    return dimensions_[Dim - 1];
  }

  template <size_t ShiftDim>
  std::pair<int, bool> shift(const int amount) const {
    static_assert(ShiftDim < Dim, "Requested dimension is out of bounds.");
    std::pair<int, bool> output;
    int source;
    MPI_Cart_shift(cartComm_, ShiftDim, amount, &source, &output.first);
    output.second = output.first != MPI_PROC_NULL;
    return output;
  }

  std::pair<int, bool> shift(const size_t dim, const int amount) const {
    std::pair<int, bool> output;
    int source;
    MPI_Cart_shift(cartComm_, dim, amount, &source, &output.first);
    output.second = output.first != MPI_PROC_NULL;
    return output;
  }

  typename std::enable_if<(Dim > 1), std::pair<int, bool>>::type
  left(const int amount = 1) const {
    return shift<Dim - 1>(-amount);
  }

  typename std::enable_if<(Dim > 1), std::pair<int, bool>>::type
  right(const int amount = 1) const {
    return shift<Dim - 1>(amount);
  }

  typename std::enable_if<(Dim > 1), std::pair<int, bool>>::type
  up(const int amount = 1) const {
    return shift<Dim - 2>(-amount);
  }

  typename std::enable_if<(Dim > 1), std::pair<int, bool>>::type
  down(const int amount = 1) const {
    return shift<Dim - 2>(amount);
  }

  template <size_t PartitionDim> MPI_Comm Partition() {
    MPI_Comm comm;
    std::array<int, Dim> dimToSplit;
    for (size_t i = 0; i < Dim; ++i) {
      dimToSplit[i] = i == PartitionDim;
    }
    MPI_Cart_sub(cartComm_, dimToSplit.data(), &comm);
    return comm;
  }

private:
  std::array<int, Dim> dimensions_;
  std::array<int, Dim> periods_{};
  std::array<int, Dim> coords_{};
  MPI_Comm cartComm_{};
};

} // End namespace mpi

} // End namespace knn 
