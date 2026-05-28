//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/Comm/Communicator.hpp>
#include <Einsums/Comm/Error.hpp>
#include <Einsums/Comm/Platform.hpp>

#include <complex>
#include <cstddef>
#include <span>

namespace einsums::comm {

/// Reduction operations for collective communication.
enum class ReduceOp : std::uint8_t { Sum, Max, Min, Prod };

/**
 * @brief Handle for a non-blocking collective operation.
 *
 * Call wait() to block until the operation completes, or test() to poll.
 * Integrates with DataflowExecutor's async_start/async_finish pattern.
 */
class EINSUMS_EXPORT Request {
  public:
    Request() = default;

    /// Block until the operation completes.
    void wait();

    /// Non-blocking poll. Returns true if complete.
    [[nodiscard]] bool test();

    /// Access the underlying native handle (MPI_Request* or nullptr).
    [[nodiscard]] void *native_handle() const;

    struct Impl;

    /// Construct from implementation (used by collective functions).
    explicit Request(std::shared_ptr<Impl> impl);

  private:
    std::shared_ptr<Impl> _impl;
};

/// Concept for types that can be communicated (arithmetic + complex).
template <typename T>
concept Communicable = std::is_arithmetic_v<T> || requires {
    typename T::value_type;
    requires std::is_arithmetic_v<typename T::value_type>;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Blocking collectives
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief All-reduce: combine values from all ranks using @p op.
 *
 * Every rank contributes @p send and receives the reduced result in @p recv.
 * Runtime dispatches to NCCL when pointers are device memory and NCCL is available.
 *
 * @param send  Source buffer (one element per count entry).
 * @param recv  Destination buffer (same size as send). May alias send for in-place.
 * @param op    Reduction operation.
 * @param comm  Communicator.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> allreduce(std::span<T const> send, std::span<T> recv, ReduceOp op,
                                                                 Communicator const &comm = Communicator::world());

/// In-place allreduce: result overwrites the input buffer.
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> allreduce_inplace(std::span<T> buf, ReduceOp op,
                                                                         Communicator const &comm = Communicator::world());

/**
 * @brief Broadcast: root sends data to all other ranks.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> broadcast(std::span<T> buf, int root = 0,
                                                                 Communicator const &comm = Communicator::world());

/**
 * @brief All-gather: each rank contributes a piece, all receive the whole.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> allgather(std::span<T const> send, std::span<T> recv,
                                                                 Communicator const &comm = Communicator::world());

/**
 * @brief Scatter: root distributes equal pieces to each rank.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> scatter(std::span<T const> send, std::span<T> recv, int root = 0,
                                                               Communicator const &comm = Communicator::world());

/**
 * @brief Point-to-point send.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> send(std::span<T const> buf, int dest, int tag = 0,
                                                            Communicator const &comm = Communicator::world());

/**
 * @brief Point-to-point receive.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> recv(std::span<T> buf, int src, int tag = 0,
                                                            Communicator const &comm = Communicator::world());

/// Barrier: synchronize all ranks.
[[nodiscard]] EINSUMS_EXPORT expected<void, CommError> barrier(Communicator const &comm = Communicator::world());

// ═══════════════════════════════════════════════════════════════════════════════
// Non-blocking collectives
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Non-blocking all-reduce. Returns a Request to wait on.
 *
 * Integrates with the ComputeGraph's async_start/async_finish pattern
 * for overlapping communication with computation.
 */
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<Request, CommError> iallreduce(std::span<T const> send, std::span<T> recv, ReduceOp op,
                                                                     Communicator const &comm = Communicator::world());

/// Non-blocking in-place allreduce.
template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<Request, CommError> iallreduce_inplace(std::span<T> buf, ReduceOp op,
                                                                             Communicator const &comm = Communicator::world());

template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<Request, CommError> ibroadcast(std::span<T> buf, int root = 0,
                                                                     Communicator const &comm = Communicator::world());

template <Communicable T>
[[nodiscard]] EINSUMS_EXPORT expected<Request, CommError> iallgather(std::span<T const> send, std::span<T> recv,
                                                                     Communicator const &comm = Communicator::world());

// ═══════════════════════════════════════════════════════════════════════════════
// Explicit instantiation declarations (export-control on every specialization).
//
// On Linux (both GCC and conda Clang) with EINSUMS_WITH_HIDDEN_VISIBILITY=ON,
// the EINSUMS_EXPORT on the primary templates above does NOT propagate to the
// specializations produced by the explicit instantiations in Collectives.cpp —
// the symbols come out hidden and disappear from libEinsums.so, breaking every
// downstream consumer (e.g. TensorFileBasic_test). Declaring each specialization
// here with ``extern template EINSUMS_EXPORT`` is the canonical libstdc++-style
// fix: the visibility attribute lives on a real declaration, and the matching
// ``template ... <T>(...)`` definitions in Collectives.cpp pick up the
// visibility from this declaration. Apple Clang propagates the primary template
// attribute on its own, so the macOS build doesn't need this — but the
// declarations are harmless there.
// ═══════════════════════════════════════════════════════════════════════════════
#define EINSUMS_COMM_EXTERN_INSTANTIATE(T)                                                                                                 \
    extern template EINSUMS_EXPORT expected<void, CommError> allreduce<T>(std::span<T const>, std::span<T>, ReduceOp,                      \
                                                                          Communicator const &);                                           \
    extern template EINSUMS_EXPORT expected<void, CommError> allreduce_inplace<T>(std::span<T>, ReduceOp, Communicator const &);           \
    extern template EINSUMS_EXPORT expected<void, CommError> broadcast<T>(std::span<T>, int, Communicator const &);                        \
    extern template EINSUMS_EXPORT expected<void, CommError> allgather<T>(std::span<T const>, std::span<T>, Communicator const &);         \
    extern template EINSUMS_EXPORT expected<void, CommError> scatter<T>(std::span<T const>, std::span<T>, int, Communicator const &);      \
    extern template EINSUMS_EXPORT expected<void, CommError> send<T>(std::span<T const>, int, int, Communicator const &);                  \
    extern template EINSUMS_EXPORT expected<void, CommError> recv<T>(std::span<T>, int, int, Communicator const &);                        \
    extern template EINSUMS_EXPORT expected<Request, CommError> iallreduce<T>(std::span<T const>, std::span<T>, ReduceOp,                  \
                                                                              Communicator const &);                                       \
    extern template EINSUMS_EXPORT expected<Request, CommError> iallreduce_inplace<T>(std::span<T>, ReduceOp, Communicator const &);       \
    extern template EINSUMS_EXPORT expected<Request, CommError> ibroadcast<T>(std::span<T>, int, Communicator const &);                    \
    extern template EINSUMS_EXPORT expected<Request, CommError> iallgather<T>(std::span<T const>, std::span<T>, Communicator const &);

EINSUMS_COMM_EXTERN_INSTANTIATE(float)
EINSUMS_COMM_EXTERN_INSTANTIATE(double)
EINSUMS_COMM_EXTERN_INSTANTIATE(int)
EINSUMS_COMM_EXTERN_INSTANTIATE(long)
EINSUMS_COMM_EXTERN_INSTANTIATE(long long)
EINSUMS_COMM_EXTERN_INSTANTIATE(std::complex<float>)
EINSUMS_COMM_EXTERN_INSTANTIATE(std::complex<double>)

#undef EINSUMS_COMM_EXTERN_INSTANTIATE

} // namespace einsums::comm
