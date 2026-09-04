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
 * @versionadded{2.1.0}
 */
EINSUMS_EXPORT void *aligned_alloc(size_t bytes);

/**
 * @brief Release memory obtained from @ref aligned_alloc. Null is a no-op.
 *
 * @versionadded{2.1.0}
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
 * @versionadded{2.1.0}
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
 * Two ways out, deliberately different:
 *
 * - @ref close throws if a keepalive token carved in the scope is still held,
 *   which is a use-after-free about to happen. That is the loud path, and the
 *   one Python's @c with statement takes.
 * - The destructor cannot throw, so it logs an error and demotes instead: the
 *   surviving blocks move to the pool's base heap and stay valid, to be freed
 *   individually when their tokens die.
 *
 * @versionadded{2.1.0}
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
     * @versionadded{2.1.0}
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

/// The arena bytes a MemoryPool reservation of @p bytes would actually claim:
/// the request plus the pool's utilization headroom and mimalloc's arena
/// rounding. Planners packing a working set against a memory budget
/// (einsums:max-memory, a chunked algorithm's in-core limit) should charge
/// this, not the raw sum, or the reserve that follows will overshoot the plan.
APIARY_EXPOSE EINSUMS_EXPORT size_t pool_reserve_cost(size_t bytes);

/// Arena bytes currently charged against einsums:max-memory: every live
/// pool, plus dead PINNED pools, whose pages stay resident. A dead unpinned
/// pool's pages purge, so it stops counting when it dies.
[[nodiscard]] EINSUMS_EXPORT size_t pooled_reserved_bytes();

/**
 * @brief A pre-reserved region of memory that mimalloc manages exclusively.
 *
 * A pool is an exclusive mimalloc arena plus a heap bound to it. Carving is a
 * real @c mi_heap_malloc_aligned against memory that is already reserved, which
 * is roughly a microsecond for a multi-megabyte block instead of the tens to
 * hundreds of microseconds an @c mmap-backed allocation costs; freeing returns
 * the bytes to the arena immediately, so a loop whose intermediates die each
 * iteration holds a flat footprint with no pool-specific discipline.
 *
 * Because the arena is exclusive, exhaustion is a defined event: mimalloc
 * returns null rather than falling back to OS memory. The pool answers by
 * reserving another arena and warning, so a mis-sized @c reserve degrades to a
 * log line rather than a throw inside a kernel.
 *
 * Thread affinity: a mimalloc heap allocates only from the thread that created
 * it, so every @ref allocate must run on the thread that constructed the pool
 * and the pool says so by throwing otherwise. Freeing has no such restriction:
 * a pooled tensor may be destroyed on any thread.
 *
 * Arena address space is never handed back to the OS: mimalloc declares
 * mi_arena_unload in its header but leaves it commented out through v3.5, so
 * there is no supported way to give one back. Arenas are also capped per
 * process. Pools are therefore meant to be FEW, LONG-LIVED, and reserved to
 * their peak size once rather than grown in steps.
 *
 * @versionadded{2.1.0}
 */
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_NOCOPY APIARY_NOMOVE MemoryPool {
  public:
    /**
     * @brief Reserve a pool.
     *
     * @param reserve_bytes Bytes the caller expects to carve. The arena is
     *                      sized with headroom on top, because multi-megabyte
     *                      blocks only reach about 87% of an arena's nominal
     *                      capacity, and is never smaller than mimalloc's
     *                      minimum arena size.
     * @param name          Label used in log messages.
     * @param warn_bytes    Usage threshold that logs a one-shot warning when
     *                      crossed; zero disables it.
     * @param pinned        Footprint policy. Unpinned (the default), freed
     *                      pages purge back to the OS after mimalloc's purge
     *                      delay, so the resident set tracks live bytes and a
     *                      recarve after a free re-pays first touch. Pinned,
     *                      freed pages stay resident and recarves are
     *                      fault-free, but the pool never shrinks below its
     *                      high water for its whole life - reserve pinning for
     *                      small, hot scratch pools that a machine can hold
     *                      permanently.
     *
     * @throws std::runtime_error if the arena cannot be reserved.
     *
     * @versionadded{2.1.0}
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
     * A zero-byte request still returns a distinct address, so no two tensors
     * ever share one.
     *
     * @throws std::runtime_error if called off the owning thread, or if the
     *         pool could neither carve nor grow.
     *
     * @versionadded{2.1.0}
     */
    void *allocate(size_t bytes);

    /**
     * @brief Carve @p bytes and hand them back zeroed.
     *
     * Cheaper than @ref allocate plus @c memset only where mimalloc can vouch
     * that the pages are still OS-fresh; on reused pages it is the same memory
     * write. Zeroing a multi-megabyte tensor costs full memory bandwidth
     * either way, which is why the pool's speedup is on uninitialized carves.
     *
     * @versionadded{2.1.0}
     */
    void *allocate_zeroed(size_t bytes);

    /// Return a carve to the pool. Safe from any thread; null is a no-op.
    void deallocate(void *ptr) noexcept;

    /**
     * @brief Wrap a carve in a keepalive token that frees it when the last holder drops it.
     *
     * This is what makes a pooled tensor's death its own reclamation: the token
     * rides along as the tensor's storage owner, on any thread.
     *
     * @versionadded{2.1.0}
     */
    [[nodiscard]] std::shared_ptr<void const> borrow(void *ptr);

    /// Carve @p bytes and hand back the keepalive token for it in one step.
    [[nodiscard]] std::pair<void *, std::shared_ptr<void const>> allocate_borrowed(size_t bytes);

    /// @ref allocate_borrowed against @ref allocate_zeroed.
    [[nodiscard]] std::pair<void *, std::shared_ptr<void const>> allocate_borrowed_zeroed(size_t bytes);

    /**
     * @brief Grow the pool so at least @p bytes of capacity are reserved.
     *
     * A no-op when the pool is already that large. Reserving up front is how a
     * caller keeps the overflow path from ever running - and it is worth doing
     * once rather than in steps: each growth takes another arena, mimalloc caps
     * how many arenas a process may hold, and it never reclaims one.
     *
     * @versionadded{2.1.0}
     */
    APIARY_EXPOSE void reserve(size_t bytes);

    /**
     * @brief Bulk-free everything the pool holds and start over.
     *
     * @throws std::runtime_error if any keepalive token is outstanding, or if
     *         an epoch is open. Destroying the heaps under live tensors is a
     *         use-after-free; this makes it a recoverable error instead.
     *
     * @versionadded{2.1.0}
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
     * The tensor type is a template parameter rather than a fixed
     * @c RuntimeTensor because this module sits below the tensor module; see
     * @c Einsums/Tensor/PooledTensor.hpp for the named @c pool_empty /
     * @c pool_zeros / @c pool_tensor wrappers.
     *
     * The tensor is built deferred, carved into, and handed the keepalive
     * token, so it is indistinguishable from an owned tensor to every consumer
     * and frees its bytes back to the pool when it dies.
     *
     * @versionadded{2.1.0}
     */
    template <typename TensorT, typename Dims>
    [[nodiscard]] TensorT empty_as(std::string tensor_name, Dims const &dims) {
        // The tensor types have no move constructor (a user-declared
        // destructor suppresses it) and their copy constructor deep-copies
        // external storage, so a non-elided return here would silently
        // un-pool the result. The shape is the one every compiler applies
        // NRVO to - one local, one return - and PooledTensor's unit test
        // asserts the carve survived the return, so a regression is loud.
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
     * The copy-free form: nothing is returned by value, so no tensor type's
     * deep-copying copy constructor can get between the carve and the caller.
     * The factories above are the ergonomic form.
     *
     * @versionadded{2.1.0}
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
