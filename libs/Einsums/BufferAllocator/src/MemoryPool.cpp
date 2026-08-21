//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/MemoryPool.hpp>
#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <mimalloc.h>
#if !defined(EINSUMS_WINDOWS)
#    include <sys/mman.h>
#endif
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(detail)

namespace {

/// Every carve is 64-byte aligned, matching einsums::memory::aligned_alloc and
/// the MemoryPlanning arena, so a pooled tensor's data pointer is as aligned as
/// any other tensor's.
constexpr size_t kPoolAlign = 64;

/// A zero-byte tensor still gets real bytes: two tensors sharing one address
/// would make the graph's span-identity checks ambiguous.
constexpr size_t kMinCarve = 64;

/// Multi-megabyte blocks only reach about 87.5% of an arena's nominal capacity
/// (size-class rounding plus mimalloc's own page metadata), so a request for
/// exactly N bytes of tensors needs an arena meaningfully larger than N. 25%
/// rather than the measured 12.5%: at the measured figure a reserve sized to
/// the workload lands exactly on the edge, and overshooting costs address
/// space while falling short costs an arena.
size_t with_headroom(size_t bytes) {
    return bytes + (bytes / 100) * 25 + kPoolAlign;
}

size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

/// One epoch's carve accounting. Outlives the epoch itself: a keepalive token
/// carved inside a scope has to be able to decrement the count from any thread,
/// at any time, including after the scope is gone.
struct ScopeCounter {
    std::atomic<size_t> live{0};
};

/// One epoch's heaps: at most one per arena, created on first use in that
/// scope. Allocation walks them newest-arena-first, since the newest arena is
/// the one with room when the pool has had to grow.
struct PoolScope {
    std::vector<mi_heap_t *>      heaps;
    std::shared_ptr<ScopeCounter> counter{std::make_shared<ScopeCounter>()};
};

/// The pool's interior, refcounted separately from @ref einsums::MemoryPool so
/// that a pooled tensor may outlive the pool handle: keepalive tokens hold a
/// reference, so the arena stays registered until the last carve is gone.
struct PoolState {
    std::string                name;
    std::thread::id            owner{std::this_thread::get_id()};
    std::vector<mi_arena_id_t> arenas;
    std::vector<PoolScope>     scopes;
    bool                       pinned{false};
    size_t                     reserved{0};
    std::atomic<size_t>        used{0};
    std::atomic<size_t>        high_water{0};
    std::atomic<size_t>        live{0};
    std::atomic<size_t>        warn_bytes{0};
    std::atomic<bool>          warned{false};

    ~PoolState();
};

namespace {

bool visit_area(mi_heap_t const * /*heap*/, mi_heap_area_t const *area, void * /*block*/, size_t /*block_size*/, void *arg) {
    auto *total = static_cast<size_t *>(arg);
    if (area != nullptr) {
        *total += area->used * area->block_size;
    }
    return true;
}

/// Bytes still allocated out of @p heap. Used to correct the pool's usage
/// counter when a scope is torn down with untracked carves still in it.
size_t heap_live_bytes(mi_heap_t *heap) {
    size_t total = 0;
    mi_heap_visit_blocks(heap, false, &visit_area, &total);
    return total;
}

void *reserve_region(size_t bytes, size_t alignment) {
#if defined(EINSUMS_WINDOWS)
    return _aligned_malloc(bytes, alignment);
#else
    // Raw mmap rather than posix_memalign: the reservation must be LAZY, so a
    // pool sized with headroom costs address space, not resident memory, until
    // the carves are actually written. Over-map by the alignment and trim.
    size_t const padded = bytes + alignment;
    void        *raw    = ::mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return nullptr;
    }
    auto const      base    = reinterpret_cast<uintptr_t>(raw);
    uintptr_t const aligned = (base + alignment - 1) & ~(alignment - 1);
    if (size_t const lead = aligned - base; lead != 0) {
        ::munmap(raw, lead);
    }
    if (size_t const tail = (base + padded) - (aligned + bytes); tail != 0) {
        ::munmap(reinterpret_cast<void *>(aligned + bytes), tail);
    }
    return reinterpret_cast<void *>(aligned);
#endif
}

/// Register another exclusive arena of at least @p bytes usable capacity.
/// Returns false when the OS refuses the reservation.
///
/// The region is obtained here and handed to mimalloc rather than reserved
/// through mi_reserve_os_memory_ex, so that the reservation is lazy (see
/// reserve_region). Whether it is PINNED is the pool's footprint policy:
/// pinned keeps freed pages resident so a recarve is fault-free (the 0.49 us
/// warm number on the monotonic fill benchmark), but the pages then never
/// shrink below the pool's high water for its whole life - correct for small
/// hot scratch pools, catastrophic for a multi-GB pool on a machine that also
/// has to hold the rest of the calculation (a 15 GB pinned T0 pool on a 32 GB
/// host thrashed the whole run). Unpinned, mimalloc purges freed pages after
/// its purge delay, so the resident set tracks LIVE bytes and only recarves
/// after a free re-pay first touch; a fill that carves and holds pays nothing.
///
/// mi_manage_os_memory_ex silently SHRINKS a region whose base is not
/// mi_arena_min_alignment()-aligned, hence the aligned reservation.
size_t arena_size_for(size_t bytes) {
    size_t const min_arena = mi_arena_min_size();
    size_t const alignment = mi_arena_min_alignment();
    size_t       size      = round_up(with_headroom(bytes), 4u << 20);
    if (size < min_arena) {
        size = min_arena;
    }
    return round_up(size, alignment);
}

/// Arena bytes counted against einsums:max-memory: every live pool, plus
/// dead PINNED pools (their pages stay resident to their high water; an
/// unpinned pool's purge on death returns its pages, so its share is
/// released in ~PoolState even though the arena registration itself is not).
std::atomic<size_t> pooled_reserved_total{0};

bool add_arena(PoolState &state, size_t bytes) {
    size_t const size = arena_size_for(bytes);

    // The planning ceiling. Checked here - the one choke point every
    // reservation passes - on the owning thread, before the OS is asked for
    // anything. Ordinary carves are never checked: a pool that is already
    // reserved can always be carved from, so nothing can throw mid-kernel.
    if (size_t const ceiling = string_util::memory_string(config::get(option::MaxMemory)); ceiling != 0) {
        size_t const already = pooled_reserved_total.load(std::memory_order_relaxed);
        if (already + size > ceiling) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "MemoryPool '{}': reserving {} more bytes would take the process's pooled total past "
                                    "--einsums:max-memory ({} already reserved, ceiling {}). Shrink the working set (for chunked "
                                    "algorithms, lower their memory budget) or raise the option.",
                                    state.name, size, already, ceiling);
        }
    }

    mi_arena_id_t id = nullptr;

    if (void *region = reserve_region(size, mi_arena_min_alignment()); region != nullptr) {
        // Exclusive: allocations that do not fit come back as null instead of
        // silently escaping into OS memory, which is what makes overflow a
        // policy decision rather than an invisible one.
        if (mi_manage_os_memory_ex(region, size, /*is_committed=*/true, /*is_pinned=*/state.pinned, /*is_zero=*/false,
                                   /*numa_node=*/-1, /*exclusive=*/true, &id) &&
            id != nullptr) {
            state.arenas.push_back(id);
            state.reserved += size;
            pooled_reserved_total.fetch_add(size, std::memory_order_relaxed);
            // The region is never released: mimalloc 3.3 has no arena unload,
            // so handing the pages back would leave the allocator pointing at
            // memory the OS could reissue.
            return true;
        }
        // mimalloc refused the region; it never took ownership, so the memory
        // is ours to release.
#if defined(EINSUMS_WINDOWS)
        _aligned_free(region);
#else
        ::munmap(region, size);
#endif
    }

    // Fall back to letting mimalloc do the reservation. Pages are then subject
    // to its purge policy, which costs first-touch faults on reuse but keeps
    // the pool working.
    if (mi_reserve_os_memory_ex(size, /*commit=*/true, /*allow_large=*/false, /*exclusive=*/true, &id) != 0 || id == nullptr) {
        return false;
    }

    state.arenas.push_back(id);
    state.reserved += size;
    pooled_reserved_total.fetch_add(size, std::memory_order_relaxed);
    return true;
}

/// Try every heap the current scope can reach, newest arena first.
///
/// @p zeroed routes through mimalloc's calloc path, which skips the memset for
/// pages it knows are still OS-fresh. Zeroing a multi-megabyte tensor is a full
/// memory write, so the pages mimalloc can vouch for are the only ones that
/// come free.
void *carve(PoolState &state, size_t bytes, bool zeroed) {
    PoolScope &scope = state.scopes.back();
    scope.heaps.resize(state.arenas.size(), nullptr);

    for (size_t i = state.arenas.size(); i-- > 0;) {
        if (scope.heaps[i] == nullptr) {
            scope.heaps[i] = mi_heap_new_in_arena(state.arenas[i]);
        }
        if (scope.heaps[i] == nullptr) {
            continue;
        }
        void *ptr =
            zeroed ? mi_heap_zalloc_aligned(scope.heaps[i], bytes, kPoolAlign) : mi_heap_malloc_aligned(scope.heaps[i], bytes, kPoolAlign);
        if (ptr != nullptr) {
            return ptr;
        }
    }
    return nullptr;
}

/// Free every carve in @p scope at once, correcting the usage counter for
/// anything that was never handed a keepalive token.
void destroy_scope(PoolState &state, PoolScope &scope) {
    for (mi_heap_t *heap : scope.heaps) {
        if (heap == nullptr) {
            continue;
        }
        size_t const stale = heap_live_bytes(heap);
        if (stale != 0) {
            state.used.fetch_sub(stale, std::memory_order_relaxed);
        }
        mi_heap_destroy(heap);
    }
    scope.heaps.clear();
}

/// Retire @p scope's heaps without freeing their blocks: the blocks move to the
/// backing heap and stay valid, so outstanding tokens can still free them.
void demote_scope(PoolScope &scope) {
    for (mi_heap_t *heap : scope.heaps) {
        if (heap != nullptr) {
            mi_heap_delete(heap);
        }
    }
    scope.heaps.clear();
}

/// Deleter carried by every keepalive token. Runs on whichever thread drops the
/// last reference, which is why it only ever calls mi_free (thread-safe) and
/// touches atomics.
struct CarveDeleter {
    std::shared_ptr<PoolState>    state;
    std::shared_ptr<ScopeCounter> counter;

    void operator()(void *ptr) const noexcept {
        if (ptr != nullptr) {
            state->used.fetch_sub(mi_usable_size(ptr), std::memory_order_relaxed);
            mi_free(ptr);
        }
        counter->live.fetch_sub(1, std::memory_order_relaxed);
        state->live.fetch_sub(1, std::memory_order_relaxed);
    }
};

} // namespace

PoolState::~PoolState() {
    // The einsums:max-memory ceiling tracks what can be RESIDENT at once. A
    // dead unpinned pool's blocks are freed and its pages purge, so its
    // reservation stops counting - even though the arena registration itself
    // is unreclaimable (address space, not memory). A pinned pool's pages
    // stay resident to its high water forever, so it keeps counting.
    // Serial pool lifecycles (a (T0) pool, then the iterative (T)'s own pool
    // of the same size) depend on this: charging dead pools made the second
    // reservation exceed a ceiling the machine could honor.
    if (!pinned) {
        pooled_reserved_total.fetch_sub(reserved, std::memory_order_relaxed);
    }

    // mimalloc heaps belong to the thread that created them, so a state that
    // outlived its pool and is dying on a worker thread cannot tear them down.
    // Leaking heap descriptors is the only safe answer; the blocks themselves
    // are already gone, since a live token would still be holding this state.
    if (std::this_thread::get_id() != owner) {
        EINSUMS_LOG_DEBUG("MemoryPool '{}': destroyed off the owning thread; heaps left to mimalloc.", name);
        return;
    }

    for (size_t i = scopes.size(); i-- > 0;) {
        for (mi_heap_t *heap : scopes[i].heaps) {
            if (heap != nullptr) {
                mi_heap_destroy(heap);
            }
        }
    }
    // The arena outlives the pool. mimalloc HAS written mi_arena_unload, and
    // its preconditions are exactly what add_arena() builds - an exclusive
    // arena over externally owned memory - but the whole section is commented
    // out in src/arena.c and its declaration is commented out in mimalloc.h,
    // in every 3.x release through v3.5.0 (checked 2026-08-20; the conda 3.3.2
    // dylib exports no such symbol). If a release ever enables it, releasing
    // an arena here is a few lines: unload, then free the region add_arena
    // reserved.
}

EINSUMS_NAMESPACE_END(detail)

EINSUMS_NAMESPACE_BEGIN()

// ── MemoryPoolEpoch ─────────────────────────────────────────────────────────

MemoryPoolEpoch::MemoryPoolEpoch(std::shared_ptr<detail::PoolState> state, size_t depth) : _state{std::move(state)}, _depth{depth} {
}

MemoryPoolEpoch::MemoryPoolEpoch(MemoryPoolEpoch &&other) noexcept : _state{std::move(other._state)}, _depth{other._depth} {
    other._state.reset();
    other._depth = 0;
}

MemoryPoolEpoch &MemoryPoolEpoch::operator=(MemoryPoolEpoch &&other) noexcept {
    if (this != &other) {
        _state = std::move(other._state);
        _depth = other._depth;
        other._state.reset();
        other._depth = 0;
    }
    return *this;
}

MemoryPoolEpoch::~MemoryPoolEpoch() {
    if (!_state) {
        return;
    }

    if (_state->scopes.size() != _depth + 1) {
        EINSUMS_LOG_ERROR("MemoryPool '{}': epoch at depth {} outlived an inner epoch; its memory stays with the pool.", _state->name,
                          _depth);
        _state.reset();
        return;
    }

    auto &scope = _state->scopes.back();
    if (size_t const live = scope.counter->live.load(std::memory_order_relaxed); live != 0) {
        // A destructor cannot throw, so the loud failure lives in close().
        // Demoting keeps the surviving blocks valid instead of turning a leak
        // into a use-after-free.
        EINSUMS_LOG_ERROR("MemoryPool '{}': epoch closed with {} carve(s) still borrowed; their memory stays live until released.",
                          _state->name, live);
        detail::demote_scope(scope);
    } else {
        detail::destroy_scope(*_state, scope);
    }
    _state->scopes.pop_back();
    _state.reset();
}

void MemoryPoolEpoch::close() {
    if (!_state) {
        return;
    }

    if (_state->scopes.size() != _depth + 1) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': epoch at depth {} cannot close while {} inner epoch(s) are open.",
                                _state->name, _depth, _state->scopes.size() - _depth - 1);
    }

    auto &scope = _state->scopes.back();
    if (size_t const live = scope.counter->live.load(std::memory_order_relaxed); live != 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "MemoryPool '{}': epoch cannot close while {} carve(s) are still borrowed. Something still holds a tensor "
                                "created inside this epoch; releasing it before the epoch ends is required.",
                                _state->name, live);
    }

    detail::destroy_scope(*_state, scope);
    _state->scopes.pop_back();
    _state.reset();
}

bool MemoryPoolEpoch::is_open() const noexcept {
    return static_cast<bool>(_state);
}

// ── MemoryPool ──────────────────────────────────────────────────────────────

size_t pool_reserve_cost(size_t bytes) {
    // The MARGINAL planning cost: headroom plus rounding, without the 32 MiB
    // arena minimum. The minimum is a fixed address-space floor the first
    // reservation pays once, not a per-chunk resident cost - charging it per
    // chunk made every budget under 32 MiB unsatisfiable, and small budgets
    // are how the chunking tests exercise multi-chunk paths.
    return detail::round_up(detail::with_headroom(bytes), 4u << 20);
}

size_t pooled_reserved_bytes() {
    return detail::pooled_reserved_total.load(std::memory_order_relaxed);
}

MemoryPool::MemoryPool(size_t reserve_bytes, std::string name, size_t warn_bytes, bool pinned)
    : _state{std::make_shared<detail::PoolState>()} {
    _state->name   = std::move(name);
    _state->pinned = pinned;
    _state->warn_bytes.store(warn_bytes, std::memory_order_relaxed);
    _state->scopes.emplace_back();

    if (!detail::add_arena(*_state, reserve_bytes)) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': could not reserve an arena for {} bytes.", _state->name,
                                reserve_bytes);
    }
}

MemoryPool::~MemoryPool() = default;

std::shared_ptr<MemoryPool> MemoryPool::create(size_t reserve_bytes, std::string name, size_t warn_bytes, bool pinned) {
    return std::make_shared<MemoryPool>(reserve_bytes, std::move(name), warn_bytes, pinned);
}

void *MemoryPool::allocate(size_t bytes) {
    return carve_bytes(bytes, /*zeroed=*/false);
}

void *MemoryPool::allocate_zeroed(size_t bytes) {
    return carve_bytes(bytes, /*zeroed=*/true);
}

void *MemoryPool::carve_bytes(size_t bytes, bool zeroed) {
    if (std::this_thread::get_id() != _state->owner) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "MemoryPool '{}': allocate() ran off the owning thread. A mimalloc heap serves only the thread that "
                                "created it; carve on the pool's own thread, or give each thread a pool.",
                                _state->name);
    }

    size_t const request = bytes == 0 ? detail::kMinCarve : bytes;

    void *ptr = detail::carve(*_state, request, zeroed);
    if (ptr == nullptr) {
        // The arena is exclusive, so this is exhaustion, not OS pressure. Grow
        // rather than throw: a throw here could fire inside a kernel's OpenMP
        // region, which is the failure mode the buffer-size ceiling taught us
        // to avoid.
        EINSUMS_LOG_WARN("MemoryPool '{}': {} bytes did not fit in {} reserved byte(s); reserving another arena. Raise the pool's "
                         "reserve to avoid this.",
                         _state->name, request, _state->reserved);
        // Grow geometrically. A growth arena sized to just this request would
        // hold one or two more blocks and then overflow again, and each
        // overflow is a fresh OS reservation.
        size_t const growth = std::max(request, _state->reserved / 2);
        if (!detail::add_arena(*_state, growth)) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "MemoryPool '{}': could not carve {} bytes and could not add a {}th arena. mimalloc caps how many "
                                    "arenas one process may hold and never reclaims one, so a pool that grows repeatedly eventually runs "
                                    "out of them; reserve the pool once, to its full size, instead.",
                                    _state->name, request, _state->arenas.size() + 1);
        }
        ptr = detail::carve(*_state, request, zeroed);
        if (ptr == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': could not carve {} bytes even from a fresh arena.", _state->name,
                                    request);
        }
    }

    size_t const usable = mi_usable_size(ptr);
    size_t const used   = _state->used.fetch_add(usable, std::memory_order_relaxed) + usable;

    size_t high = _state->high_water.load(std::memory_order_relaxed);
    while (used > high && !_state->high_water.compare_exchange_weak(high, used, std::memory_order_relaxed)) {
    }

    if (size_t const warn = _state->warn_bytes.load(std::memory_order_relaxed);
        warn != 0 && used > warn && !_state->warned.exchange(true, std::memory_order_relaxed)) {
        EINSUMS_LOG_WARN("MemoryPool '{}': usage passed the expected {} bytes (now {} bytes).", _state->name, warn, used);
    }

    return ptr;
}

void MemoryPool::deallocate(void *ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    _state->used.fetch_sub(mi_usable_size(ptr), std::memory_order_relaxed);
    mi_free(ptr);
}

std::shared_ptr<void const> MemoryPool::borrow(void *ptr) {
    auto counter = _state->scopes.back().counter;
    counter->live.fetch_add(1, std::memory_order_relaxed);
    _state->live.fetch_add(1, std::memory_order_relaxed);
    return {ptr, detail::CarveDeleter{.state = _state, .counter = std::move(counter)}};
}

std::pair<void *, std::shared_ptr<void const>> MemoryPool::allocate_borrowed(size_t bytes) {
    void *ptr = allocate(bytes);
    return {ptr, borrow(ptr)};
}

std::pair<void *, std::shared_ptr<void const>> MemoryPool::allocate_borrowed_zeroed(size_t bytes) {
    void *ptr = allocate_zeroed(bytes);
    return {ptr, borrow(ptr)};
}

void MemoryPool::reserve(size_t bytes) {
    if (std::this_thread::get_id() != _state->owner) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': reserve() ran off the owning thread.", _state->name);
    }
    if (bytes <= _state->reserved) {
        return;
    }
    size_t const shortfall = bytes - _state->reserved;
    if (!detail::add_arena(*_state, shortfall)) {
        EINSUMS_THROW_EXCEPTION(
            std::runtime_error,
            "MemoryPool '{}': could not reserve {} more bytes as a {}th arena - the OS refused the reservation (not enough "
            "memory or swap), or mimalloc's per-process arena cap is exhausted. mimalloc caps how many arenas one "
            "process may hold and never reclaims one, so many small growths exhaust them where one large reserve "
            "would not; size the pool to its peak up front.",
            _state->name, shortfall, _state->arenas.size() + 1);
    }
}

void MemoryPool::reset() {
    if (std::this_thread::get_id() != _state->owner) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': reset() ran off the owning thread.", _state->name);
    }
    if (_state->scopes.size() != 1) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': reset() cannot run while {} epoch(s) are open.", _state->name,
                                _state->scopes.size() - 1);
    }
    if (size_t const live = _state->live.load(std::memory_order_relaxed); live != 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "MemoryPool '{}': reset() cannot run while {} carve(s) are still borrowed. Destroying the heaps under a "
                                "live tensor is a use-after-free; release the tensors first.",
                                _state->name, live);
    }

    detail::destroy_scope(*_state, _state->scopes.back());
    _state->used.store(0, std::memory_order_relaxed);
    _state->warned.store(false, std::memory_order_relaxed);
}

MemoryPoolEpoch MemoryPool::epoch() {
    if (std::this_thread::get_id() != _state->owner) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "MemoryPool '{}': epoch() ran off the owning thread.", _state->name);
    }
    _state->scopes.emplace_back();
    return {_state, _state->scopes.size() - 1};
}

MemoryPoolStats MemoryPool::stats() const noexcept {
    return MemoryPoolStats{.bytes_reserved = _state->reserved,
                           .bytes_used     = _state->used.load(std::memory_order_relaxed),
                           .high_water     = _state->high_water.load(std::memory_order_relaxed),
                           .live_borrows   = _state->live.load(std::memory_order_relaxed),
                           .arenas         = _state->arenas.size(),
                           .epoch_depth    = _state->scopes.size() - 1};
}

size_t MemoryPool::bytes_reserved() const noexcept {
    return _state->reserved;
}

size_t MemoryPool::bytes_used() const noexcept {
    return _state->used.load(std::memory_order_relaxed);
}

size_t MemoryPool::high_water() const noexcept {
    return _state->high_water.load(std::memory_order_relaxed);
}

size_t MemoryPool::live_borrows() const noexcept {
    return _state->live.load(std::memory_order_relaxed);
}

size_t MemoryPool::arenas() const noexcept {
    return _state->arenas.size();
}

size_t MemoryPool::epoch_depth() const noexcept {
    return _state->scopes.size() - 1;
}

std::string const &MemoryPool::name() const noexcept {
    return _state->name;
}

size_t MemoryPool::warn_bytes() const noexcept {
    return _state->warn_bytes.load(std::memory_order_relaxed);
}

void MemoryPool::set_warn_bytes(size_t bytes) noexcept {
    _state->warn_bytes.store(bytes, std::memory_order_relaxed);
    _state->warned.store(false, std::memory_order_relaxed);
}

EINSUMS_NAMESPACE_END()
