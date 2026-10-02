//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file MemoryPool.hpp
 */

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(memory)

/**
 * @brief The library's one aligned-allocation primitive: 64-byte aligned, mimalloc-managed.
 *
 * Unmetered, unlike @ref einsums::BufferAllocator, which meters against
 * @c --einsums:buffer-size before landing here. Returns null on failure.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void *aligned_alloc(size_t bytes);

/**
 * @brief Release memory obtained from @ref aligned_alloc. Null is a no-op.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void aligned_free(void *ptr);

EINSUMS_NAMESPACE_END(memory)

EINSUMS_NAMESPACE_BEGIN()

namespace detail {
struct PoolState;
}

class MemoryPool;

/**
 * @brief Accounting snapshot for one @ref MemoryPool.
 *
 * @versionadded{2.0.0}
 */
struct MemoryPoolStats {
    size_t bytes_reserved{0}; ///< Total arena bytes registered with mimalloc.
    size_t bytes_used{0};     ///< Net bytes currently carved out, as mimalloc sizes them.
    size_t high_water{0};     ///< Largest value @ref bytes_used ever reached.
    size_t live_borrows{0};   ///< Outstanding keepalive tokens across every scope.
    size_t arenas{0};         ///< Arena count; more than one means the pool grew on overflow.
    size_t epoch_depth{0};    ///< Number of open epochs.
};

/**
 * @brief RAII scope whose carves are released together when it closes.
 *
 * Each epoch gets a heap of its own inside the pool's arena, so closing it
 * bulk-frees every carve made inside it, including ones nothing tracked.
 * Epochs nest and must close in reverse order of opening.
 *
 * If a keepalive token carved in the scope is still held, @ref close throws (Python's @c with takes
 * this path), while the destructor logs an error and moves the surviving blocks to the pool's base
 * heap, where they stay valid until their tokens die.
 *
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_NOCOPY MemoryPoolEpoch {
  public:
    /// A default-constructed epoch owns no scope and closing it is a no-op.
    MemoryPoolEpoch() = default;

    MemoryPoolEpoch(MemoryPoolEpoch &&other) noexcept;
    MemoryPoolEpoch &operator=(MemoryPoolEpoch &&other) noexcept;

    MemoryPoolEpoch(MemoryPoolEpoch const &)            = delete;
    MemoryPoolEpoch &operator=(MemoryPoolEpoch const &) = delete;

    ~MemoryPoolEpoch();

    /**
     * @brief Close the scope, releasing everything carved inside it.
     *
     * @throws std::runtime_error if a keepalive token from this scope is still
     *         held, or if an inner epoch is still open.
     *
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void close();

    /// True until @ref close or the destructor runs.
    APIARY_EXPOSE APIARY_GETTER("open") [[nodiscard]] bool is_open() const noexcept;

  private:
    friend class MemoryPool;

    MemoryPoolEpoch(std::shared_ptr<detail::PoolState> state, size_t depth);

    std::shared_ptr<detail::PoolState> _state{};
    size_t                             _depth{0};
};

/// The arena bytes a reservation of @p bytes actually claims, headroom and rounding included.
/// Planners budgeting against einsums:max-memory should charge this, not the raw sum.
APIARY_EXPOSE EINSUMS_EXPORT size_t pool_reserve_cost(size_t bytes);

/// Arena bytes currently charged against einsums:max-memory: every live
/// pool, plus dead PINNED pools, whose pages stay resident. A dead unpinned
/// pool's pages purge, so it stops counting when it dies.
[[nodiscard]] EINSUMS_EXPORT size_t pooled_reserved_bytes();

/**
 * @brief A pre-reserved region of memory that mimalloc manages exclusively.
 *
 * An exclusive mimalloc arena plus a heap bound to it. A carve takes about a microsecond for a
 * multi-megabyte block, against tens to hundreds for an mmap, and a free returns the bytes at once.
 * When the arena runs out the pool reserves another and warns, rather than throwing in a kernel.
 *
 * @ref allocate must run on the constructing thread (it throws otherwise); freeing may happen on any.
 * Arenas are capped per process and never returned to the OS, so keep pools few, long-lived and
 * reserved to their peak size up front.
 *
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_NOCOPY APIARY_NOMOVE MemoryPool {
  public:
    /**
     * @brief Reserve a pool.
     *
     * @param reserve_bytes Bytes the caller expects to carve; the arena adds headroom.
     * @param name          Label used in log messages.
     * @param warn_bytes    Usage that logs a one-shot warning; zero disables it.
     * @param pinned        Keep freed pages resident, so recarves are fault-free but the pool never
     *                      shrinks below its high water. For small, hot scratch pools only; unpinned
     *                      pages purge back to the OS.
     *
     * @throws std::runtime_error if the arena cannot be reserved.
     *
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE explicit MemoryPool(size_t reserve_bytes, std::string name = "pool", size_t warn_bytes = 0, bool pinned = false);

    ~MemoryPool();

    MemoryPool(MemoryPool const &)            = delete;
    MemoryPool &operator=(MemoryPool const &) = delete;
    MemoryPool(MemoryPool &&)                 = delete;
    MemoryPool &operator=(MemoryPool &&)      = delete;

    /// Convenience factory for the shared-ownership form the tensor factories expect.
    [[nodiscard]] static std::shared_ptr<MemoryPool> create(size_t reserve_bytes, std::string name = "pool", size_t warn_bytes = 0,
                                                            bool pinned = false);

    /**
     * @brief Carve @p bytes of 64-byte-aligned memory from the pool.
     *
     * A zero-byte request still returns a distinct address.
     *
     * @throws std::runtime_error if called off the owning thread, or if the
     *         pool could neither carve nor grow.
     *
     * @versionadded{2.0.0}
     */
    void *allocate(size_t bytes);

    /**
     * @brief Carve @p bytes and hand them back zeroed.
     *
     * Skips the memset only on pages mimalloc knows are OS-fresh.
     *
     * @versionadded{2.0.0}
     */
    void *allocate_zeroed(size_t bytes);

    /// Return a carve to the pool. Safe from any thread; null is a no-op.
    void deallocate(void *ptr) noexcept;

    /**
     * @brief Wrap a carve in a keepalive token that frees it when the last holder drops it.
     *
     * The token is a pooled tensor's storage owner, so the tensor's death frees the carve.
     *
     * @versionadded{2.0.0}
     */
    [[nodiscard]] std::shared_ptr<void const> borrow(void *ptr);

    /// Carve @p bytes and hand back the keepalive token for it in one step.
    [[nodiscard]] std::pair<void *, std::shared_ptr<void const>> allocate_borrowed(size_t bytes);

    /// @ref allocate_borrowed against @ref allocate_zeroed.
    [[nodiscard]] std::pair<void *, std::shared_ptr<void const>> allocate_borrowed_zeroed(size_t bytes);

    /**
     * @brief Grow the pool so at least @p bytes of capacity are reserved.
     *
     * A no-op when already that large. Reserve once rather than in steps: each growth costs an
     * arena, and arenas are capped and never reclaimed.
     *
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void reserve(size_t bytes);

    /**
     * @brief Bulk-free everything the pool holds and start over.
     *
     * @throws std::runtime_error if any keepalive token is outstanding, or if
     *         an epoch is open.
     *
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void reset();

    /// Open a nested scope; see @ref MemoryPoolEpoch.
    APIARY_EXPOSE [[nodiscard]] MemoryPoolEpoch epoch();

    /// Read the pool's accounting in one shot. The individual getters below are
    /// what the Python surface exposes; this is the C++ convenience form.
    [[nodiscard]] MemoryPoolStats stats() const noexcept;

    APIARY_EXPOSE APIARY_GETTER("bytes_reserved") [[nodiscard]] size_t   bytes_reserved() const noexcept;
    APIARY_EXPOSE APIARY_GETTER("bytes_used") [[nodiscard]] size_t       bytes_used() const noexcept;
    APIARY_EXPOSE APIARY_GETTER("high_water") [[nodiscard]] size_t       high_water() const noexcept;
    APIARY_EXPOSE APIARY_GETTER("live_borrows") [[nodiscard]] size_t     live_borrows() const noexcept;
    APIARY_EXPOSE APIARY_GETTER("arenas") [[nodiscard]] size_t           arenas() const noexcept;
    APIARY_EXPOSE APIARY_GETTER("epoch_depth") [[nodiscard]] size_t      epoch_depth() const noexcept;
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string const &name() const noexcept;

    /// Usage threshold that logs a one-shot warning when crossed; zero disables it.
    APIARY_EXPOSE APIARY_GETTER("warn_bytes") [[nodiscard]] size_t warn_bytes() const noexcept;
    APIARY_EXPOSE APIARY_SETTER("warn_bytes") void                 set_warn_bytes(size_t bytes) noexcept;

    /**
     * @brief Place a runtime-rank tensor of type @p TensorT on this pool.
     *
     * A template because this module sits below Tensor; see @c Einsums/Tensor/PooledTensor.hpp for
     * @c pool_empty, @c pool_zeros and @c pool_tensor. The result behaves as an owned tensor and frees
     * its bytes back to the pool when it dies.
     *
     * @versionadded{2.0.0}
     */
    template <typename TensorT, typename Dims>
    [[nodiscard]] TensorT empty_as(std::string tensor_name, Dims const &dims) {
        // Must be NRVO'd: the tensor types have no move constructor and their copy deep-copies, which
        // would un-pool the result. PooledTensor's test checks the carve survives the return.
        TensorT t(typename TensorT::DeferredAlloc{}, std::move(tensor_name), dims);
        place(t);
        return t;
    }

    /// @ref empty_as with the storage zeroed.
    template <typename TensorT, typename Dims>
    [[nodiscard]] TensorT zeros_as(std::string tensor_name, Dims const &dims) {
        TensorT t(typename TensorT::DeferredAlloc{}, std::move(tensor_name), dims);
        place(t, /*zeroed=*/true);
        return t;
    }

    /// Compile-time-rank form: the dimensions are the tensor constructor's own arguments.
    template <typename TensorT, std::integral... Dims>
    [[nodiscard]] TensorT empty_ranked(std::string tensor_name, Dims... dims) {
        TensorT t(typename TensorT::DeferredAlloc{}, std::move(tensor_name), dims...);
        place(t);
        return t;
    }

    /// @ref empty_ranked with the storage zeroed.
    template <typename TensorT, std::integral... Dims>
    [[nodiscard]] TensorT zeros_ranked(std::string tensor_name, Dims... dims) {
        TensorT t(typename TensorT::DeferredAlloc{}, std::move(tensor_name), dims...);
        place(t, /*zeroed=*/true);
        return t;
    }

    /**
     * @brief Carve storage for an already-shaped deferred tensor and attach it
     *        with a keepalive token.
     *
     * The copy-free form of the factories above: nothing is returned by value.
     *
     * @versionadded{2.0.0}
     */
    template <typename TensorT>
    void place(TensorT &t, bool zeroed = false) {
        using T           = typename TensorT::ValueType;
        size_t const size = t.size() * sizeof(T);
        auto [ptr, keep]  = zeroed ? allocate_borrowed_zeroed(size) : allocate_borrowed(size);
        t.materialize_into(static_cast<T *>(ptr), std::move(keep));
    }

  private:
    void *carve_bytes(size_t bytes, bool zeroed);

    std::shared_ptr<detail::PoolState> _state;
};

EINSUMS_NAMESPACE_END()
