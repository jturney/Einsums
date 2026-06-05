//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Comm/Platform.hpp>

#include <memory>

namespace einsums::comm {

/**
 * @brief Communicator abstraction wrapping MPI_Comm.
 *
 * Provides rank/size queries and communicator management (split, duplicate).
 * On the mock backend, always reports rank=0, size=1.
 *
 * Value-semantic via shared_ptr to internal state — cheap to copy.
 *
 * @par Example
 * @code
 * auto &world = Communicator::world();
 * auto [my_rank, nprocs] = std::pair{world.rank(), world.size()};
 *
 * // Split into sub-communicators by parity
 * auto sub = world.split(my_rank % 2, my_rank);
 * @endcode
 */
class EINSUMS_EXPORT Communicator {
  public:
    /// Default constructor: creates an invalid communicator.
    Communicator() = default;

    /// Get the world communicator (wraps MPI_COMM_WORLD or mock singleton).
    [[nodiscard]] static Communicator &world();

    /// Rank of this process within this communicator.
    [[nodiscard]] int rank() const;

    /// Number of processes in this communicator.
    [[nodiscard]] int size() const;

    /// True if this process is rank 0 in this communicator.
    [[nodiscard]] bool is_root() const { return rank() == 0; }

    /// Split this communicator by color and key (MPI_Comm_split).
    [[nodiscard]] Communicator split(int color, int key) const;

    /// Duplicate this communicator (MPI_Comm_dup).
    [[nodiscard]] Communicator dup() const;

    /// Check validity.
    [[nodiscard]] explicit operator bool() const { return _impl != nullptr; }

    /// Access the underlying native handle (MPI_Comm* or nullptr for mock).
    [[nodiscard]] void *native_handle() const;

    /// Equality: same underlying communicator.
    [[nodiscard]] bool operator==(Communicator const &) const = default;

  private:
    struct Impl;
    std::shared_ptr<Impl> _impl;

    explicit Communicator(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}

    friend class Request; // Needs access for async operations
};

} // namespace einsums::comm
