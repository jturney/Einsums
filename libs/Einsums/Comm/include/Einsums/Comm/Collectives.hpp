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

} // namespace einsums::comm
