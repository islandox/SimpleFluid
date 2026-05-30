/**
 * @file MPI_interface.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Lightweight C++ wrapper around raw MPI calls with type-deduced datatypes.
 *
 * Provides a thin, zero-overhead wrapper for the MPI C API.  All functions
 * use a static global communicator (`my_mpi::comm`) that defaults to
 * `MPI_COMM_WORLD` and can be overridden with `set_comm()`.
 *
 * The wrapper covers:
 * - Lifecycle (init / finalize / initialized / finalized)
 * - Point-to-point (send / recv / isend / irecv / wait / waitall / test)
 * - Collectives (broadcast / gather / gatherv / scatter / scatterv /
 *   allgather / allgatherv / alltoall / alltoallv / reduce / allreduce)
 * - Synchronization (barrier)
 * - Communicator management (dup / split / free / set_comm)
 * - Timing (wtime / wtick)
 *
 * @version 0.2
 * @date 2026-05-29
 *
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <mpi.h>

#include <cstdint>
#include <complex>

#include "utils/TMP_helpers.hh"

namespace my_mpi
{

// -- type aliases ---------------------------------------------------------

using Comm = MPI_Comm;
using Status = MPI_Status;
using Request = MPI_Request;
using ErrorCode = int;

// -- global communicator (defaults to MPI_COMM_WORLD) ---------------------

static inline Comm comm = MPI_COMM_WORLD;

// -- MPI datatype traits --------------------------------------------------

template<typename T>
MPI_Datatype type_trait();

// char / byte types
// Note: on most platforms std::int8_t ≡ signed char and std::uint8_t ≡ unsigned char,
// so we do NOT provide separate specializations for signed char / unsigned char.
template<> inline MPI_Datatype type_trait<char>()               { return MPI_CHAR; }
template<> inline MPI_Datatype type_trait<std::int8_t>()        { return MPI_INT8_T; }
template<> inline MPI_Datatype type_trait<std::int16_t>()       { return MPI_INT16_T; }
template<> inline MPI_Datatype type_trait<std::int32_t>()       { return MPI_INT32_T; }
template<> inline MPI_Datatype type_trait<std::int64_t>()       { return MPI_INT64_T; }
template<> inline MPI_Datatype type_trait<std::uint8_t>()       { return MPI_UINT8_T; }
template<> inline MPI_Datatype type_trait<std::uint16_t>()      { return MPI_UINT16_T; }
template<> inline MPI_Datatype type_trait<std::uint32_t>()      { return MPI_UINT32_T; }
template<> inline MPI_Datatype type_trait<std::uint64_t>()      { return MPI_UINT64_T; }

// floating point
template<> inline MPI_Datatype type_trait<float>()              { return MPI_FLOAT; }
template<> inline MPI_Datatype type_trait<double>()             { return MPI_DOUBLE; }
template<> inline MPI_Datatype type_trait<long double>()        { return MPI_LONG_DOUBLE; }

// complex
template<> inline MPI_Datatype type_trait<std::complex<float>>()  { return MPI_CXX_FLOAT_COMPLEX; }
template<> inline MPI_Datatype type_trait<std::complex<double>>() { return MPI_CXX_DOUBLE_COMPLEX; }

// boolean
template<> inline MPI_Datatype type_trait<bool>() { return MPI_CXX_BOOL; }

// unsupported types will cause a compile-time error due to missing specialization
template<typename T>
MPI_Datatype type_trait() {
    static_assert(utils::always_false_v<T>, "MPI datatype not defined for this type:");
}

// -- lifecycle ------------------------------------------------------------

/// Wraps MPI_Init.  Pass nullptr for both args to let MPI handle them.
inline ErrorCode init(int* argc, char*** argv) { return MPI_Init(argc, argv); }

/// Wraps MPI_Finalize.
inline ErrorCode finalize() { return MPI_Finalize(); }

/// Wraps MPI_Initialized.
inline bool initialized(ErrorCode& errorcode)
{
    int flag;
    errorcode = MPI_Initialized(&flag);
    return flag;
}

/// Wraps MPI_Finalized.
inline bool finalized(ErrorCode& errorcode)
{
    int flag;
    errorcode = MPI_Finalized(&flag);
    return flag;
}

/// Wraps MPI_Abort on the active communicator.
inline ErrorCode abort(int errorcode) { return MPI_Abort(comm, errorcode); }

// -- communicator queries -------------------------------------------------

/// Wraps MPI_Comm_size on the active communicator.
inline ErrorCode comm_size(int& size) { return MPI_Comm_size(comm, &size); }

/// Wraps MPI_Comm_rank on the active communicator.
inline ErrorCode comm_rank(int& rank) { return MPI_Comm_rank(comm, &rank); }

// -- communicator management ----------------------------------------------

/// Override the globally active communicator.
inline void set_comm(Comm c) { comm = c; }

/// Wraps MPI_Comm_dup.
inline ErrorCode comm_dup(Comm& newcomm) { return MPI_Comm_dup(comm, &newcomm); }

/// Wraps MPI_Comm_split.
inline ErrorCode comm_split(int color, int key, Comm& newcomm) {
    return MPI_Comm_split(comm, color, key, &newcomm);
}

/// Wraps MPI_Comm_free.
inline ErrorCode comm_free(Comm& comm_to_free) { return MPI_Comm_free(&comm_to_free); }

// -- point-to-point (blocking) --------------------------------------------

template<typename T>
inline ErrorCode send(const T* data, int count, int dest, int tag) {
    return MPI_Send(data, count, type_trait<T>(), dest, tag, comm);
}

template<typename T>
inline ErrorCode recv(T* data, int count, int source, int tag, Status& status) {
    return MPI_Recv(data, count, type_trait<T>(), source, tag, comm, &status);
}

// -- point-to-point (non-blocking) ----------------------------------------

template<typename T>
inline ErrorCode isend(const T* data, int count, int dest, int tag, Request& request) {
    return MPI_Isend(data, count, type_trait<T>(), dest, tag, comm, &request);
}

template<typename T>
inline ErrorCode irecv(T* data, int count, int source, int tag, Request& request) {
    return MPI_Irecv(data, count, type_trait<T>(), source, tag, comm, &request);
}

/// Wraps MPI_Wait for a single request.
inline ErrorCode wait(Request& request, Status& status) {
    return MPI_Wait(&request, &status);
}

/// Wraps MPI_Waitall.
inline ErrorCode waitall(int count, Request* requests, Status* statuses) {
    return MPI_Waitall(count, requests, statuses);
}

/// Wraps MPI_Test for a single request.
inline ErrorCode test(Request& request, int& flag, Status& status) {
    return MPI_Test(&request, &flag, &status);
}

// -- collectives ----------------------------------------------------------

/// Wraps MPI_Bcast.
template<typename T>
inline ErrorCode broadcast(T* data, int count, int root) {
    return MPI_Bcast(data, count, type_trait<T>(), root, comm);
}

/// Wraps MPI_Gather.
template<typename Ts, typename Tr>
inline ErrorCode gather(const Ts* sendbuf, int sendcount, Tr* recvbuf, int recvcount, int root) {
    return MPI_Gather(sendbuf, sendcount, type_trait<Ts>(),
                      recvbuf, recvcount, type_trait<Tr>(), root, comm);
}

/// Wraps MPI_Gatherv.
template<typename Ts, typename Tr>
inline ErrorCode gatherv(const Ts* sendbuf, int sendcount, Tr* recvbuf,
                         const int* recvcounts, const int* displs, int root) {
    return MPI_Gatherv(sendbuf, sendcount, type_trait<Ts>(),
                       recvbuf, recvcounts, displs, type_trait<Tr>(), root, comm);
}

/// Wraps MPI_Scatter.
template<typename Ts, typename Tr>
inline ErrorCode scatter(const Ts* sendbuf, int sendcount, Tr* recvbuf, int recvcount, int root) {
    return MPI_Scatter(sendbuf, sendcount, type_trait<Ts>(),
                       recvbuf, recvcount, type_trait<Tr>(), root, comm);
}

/// Wraps MPI_Scatterv.
template<typename Ts, typename Tr>
inline ErrorCode scatterv(const Ts* sendbuf, const int* sendcounts, const int* displs,
                   Tr* recvbuf, int recvcount, int root) {
    return MPI_Scatterv(sendbuf, sendcounts, displs, type_trait<Ts>(),
                        recvbuf, recvcount, type_trait<Tr>(), root, comm);
}

/// Wraps MPI_Allgather.
template<typename Ts, typename Tr>
inline ErrorCode allgather(const Ts* sendbuf, int sendcount, Tr* recvbuf, int recvcount) {
    return MPI_Allgather(sendbuf, sendcount, type_trait<Ts>(),
                         recvbuf, recvcount, type_trait<Tr>(), comm);
}

/// Wraps MPI_Allgatherv.
template<typename Ts, typename Tr>
inline ErrorCode allgatherv(const Ts* sendbuf, int sendcount, Tr* recvbuf,
                            const int* recvcounts, const int* displs) {
    return MPI_Allgatherv(sendbuf, sendcount, type_trait<Ts>(),
                          recvbuf, recvcounts, displs, type_trait<Tr>(), comm);
}

/// Wraps MPI_Alltoall.
template<typename Ts, typename Tr>
inline ErrorCode alltoall(const Ts* sendbuf, int sendcount, Tr* recvbuf, int recvcount) {
    return MPI_Alltoall(sendbuf, sendcount, type_trait<Ts>(),
                        recvbuf, recvcount, type_trait<Tr>(), comm);
}

/// Wraps MPI_Alltoallv.
template<typename Ts, typename Tr>
inline ErrorCode alltoallv(const Ts* sendbuf, const int* sendcounts, const int* sdispls,
                    Tr* recvbuf, const int* recvcounts, const int* rdispls) {
    return MPI_Alltoallv(sendbuf, sendcounts, sdispls, type_trait<Ts>(),
                         recvbuf, recvcounts, rdispls, type_trait<Tr>(), comm);
}

/// Wraps MPI_Reduce.
template<typename T>
inline ErrorCode reduce(const T* sendbuf, T* recvbuf, int count, MPI_Op op, int root) {
    return MPI_Reduce(sendbuf, recvbuf, count, type_trait<T>(), op, root, comm);
}

/// Wraps MPI_Allreduce.
template<typename T>
inline ErrorCode allreduce(const T* sendbuf, T* recvbuf, int count, MPI_Op op) {
    return MPI_Allreduce(sendbuf, recvbuf, count, type_trait<T>(), op, comm);
}

/// Global sum reduction wrappers.
template<typename T>
inline ErrorCode global_sum(const T* sendbuf, T* recvbuf, int count) {
    return allreduce(sendbuf, recvbuf, count, MPI_SUM);
}

template<typename T>
inline ErrorCode global_sum(const T& value, T& result) {
    return global_sum(&value, &result, 1);
}

/// Global max reduction wrappers.
template<typename T>
inline ErrorCode global_max(const T* sendbuf, T* recvbuf, int count) {
    return allreduce(sendbuf, recvbuf, count, MPI_MAX);
}

template<typename T>
inline ErrorCode global_max(const T& value, T& result) {
    return global_max(&value, &result, 1);
}

/// Global min reduction wrappers.
template<typename T>
inline ErrorCode global_min(const T* sendbuf, T* recvbuf, int count) {
    return allreduce(sendbuf, recvbuf, count, MPI_MIN);
}

template<typename T>
inline ErrorCode global_min(const T& value, T& result) {
    return global_min(&value, &result, 1);
}

// -- synchronization ------------------------------------------------------

/// Wraps MPI_Barrier on the active communicator.
inline ErrorCode barrier() { return MPI_Barrier(comm); }

// -- timing ---------------------------------------------------------------

/// Wraps MPI_Wtime.
inline double wtime() { return MPI_Wtime(); }

/// Wraps MPI_Wtick.
inline double wtick() { return MPI_Wtick(); }

} // namespace my_mpi