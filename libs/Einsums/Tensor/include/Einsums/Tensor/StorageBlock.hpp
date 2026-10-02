//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(detail)

/**
 * @brief Ask the kernel to back @p bytes at @p begin with transparent huge pages.
 *
 * Linux only; a no-op elsewhere, below @ref huge_page_advice_threshold, or
 * unless `--einsums:tensor:huge-pages` is set. It must run BEFORE the buffer is
 * first touched: THP is decided at fault time, and memory already faulted in
 * 4 KB pages only becomes huge if khugepaged gets round to collapsing it.
 * Failures are ignored, because the advice changes only how the buffer is
 * paged, never whether it works.
 */
EINSUMS_EXPORT void advise_huge_pages(void *begin, size_t bytes) noexcept;

/// The smallest buffer @ref advise_huge_pages advises, in bytes.
EINSUMS_EXPORT size_t huge_page_advice_threshold() noexcept;

/// True for @c std::vector with any allocator: the one owning container whose
/// @c reserve hands back untouched memory that the advice can still shape.
template <typename V>
struct IsStdVector : std::false_type {};
template <typename U, typename A>
struct IsStdVector<std::vector<U, A>> : std::true_type {};

/**
 * @brief Type-erased handle to a tensor's current backing buffer.
 *
 * Hold this, not a raw pointer, to follow the buffer across materialize, materialize_into,
 * release and resize. @c base is the current pointer; @c generation counts relocations. The
 * destructor need not be virtual: blocks come from @c make_shared, which captures the right deleter.
 */
struct StorageBase {
    void  *base{nullptr}; ///< Current start of the buffer, or null when unallocated.
    size_t generation{0}; ///< Incremented on every relocation of @ref base.
};

/**
 * @brief Refcounted backing storage for one tensor.
 *
 * Owns whichever of the two storage modes a tensor is in: @c owned holds
 * memory the tensor allocated for itself, @c external points at memory
 * somebody else owns (a MemoryPlanning arena slice, attached through
 * @c materialize_into). They are mutually exclusive, and @ref refresh
 * republishes whichever is live as @ref StorageBase::base.
 *
 * Blocks are shared, not copied. A tensor copy constructor still deep-copies
 * (into a fresh block), so value semantics are unchanged; sharing is opt-in,
 * through @c shallow_alias() and through graph capture, whose whole purpose is
 * to keep an operand's buffer alive past the wrapper that created it.
 *
 * @tparam T      Element type.
 * @tparam Vector Owning container, @c std::vector or @c gpu::DeviceVector.
 */
template <typename T, typename Vector>
struct StorageBlock final : StorageBase {
    Vector owned{};           ///< Self-allocated storage. Empty when external or unallocated.
    T     *external{nullptr}; ///< Caller-provided storage; never freed here.

    /// Keepalive for @ref external (a pool carve, freed when dropped); null for a raw attach.
    std::shared_ptr<void const> external_owner{};

    StorageBlock()                                = default;
    StorageBlock(StorageBlock const &)            = delete;
    StorageBlock &operator=(StorageBlock const &) = delete;
    StorageBlock(StorageBlock &&)                 = delete;
    StorageBlock &operator=(StorageBlock &&)      = delete;

    ~StorageBlock() { WAGGLE_MEM_FREE(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T))); }

    /// Republish the live pointer and count the relocation. Call after any
    /// change to @ref owned or @ref external.
    void refresh() noexcept {
        base = external != nullptr ? static_cast<void *>(external) : static_cast<void *>(owned.data());
        ++generation;
    }

    /// Reserve @p elems and, before first touch, advise huge pages (host vectors only).
    void reserve_owned(size_t elems) {
        if constexpr (IsStdVector<Vector>::value) {
            if (elems > owned.capacity()) {
                owned.reserve(elems);
                advise_huge_pages(static_cast<void *>(owned.data()), owned.capacity() * sizeof(T));
            }
        } else {
            (void)elems;
        }
    }

    /// Resize owned storage. Not with @ref external set; callers detach first.
    void resize_owned(size_t elems) {
        WAGGLE_MEM_FREE(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        reserve_owned(elems);
        owned.resize(elems);
        WAGGLE_MEM_ALLOC(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        refresh();
    }

    /// Replace self-allocated storage with a copy of @p src. Used by the
    /// deep-copying tensor copy constructor, which gets a block of its own.
    void copy_owned_from(Vector const &src) {
        WAGGLE_MEM_FREE(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        if constexpr (IsStdVector<Vector>::value) {
            // Land the copy in advised memory: assign() reuses the capacity
            // reserve_owned just shaped, where `owned = src` may allocate anew.
            owned.clear();
            reserve_owned(src.size());
            owned.assign(src.begin(), src.end());
        } else {
            owned = src;
        }
        external = nullptr;
        external_owner.reset();
        WAGGLE_MEM_ALLOC(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        refresh();
    }

    /// Take ownership of @p src's buffer, leaving @p src empty.
    void adopt_owned(Vector &&src) {
        WAGGLE_MEM_FREE(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        owned    = std::move(src);
        external = nullptr;
        external_owner.reset();
        refresh();
    }

    /// Stop pointing at external storage, without touching the owned buffer.
    /// Dropping the keepalive here is what returns a pooled carve to its pool.
    void detach_external() {
        external = nullptr;
        external_owner.reset();
    }

    /// Attach caller-owned storage, dropping any self-allocated buffer.
    void attach_external(T *ptr) { attach_external(ptr, nullptr); }

    /// Attach caller-owned storage along with a keepalive token for it.
    void attach_external(T *ptr, std::shared_ptr<void const> owner) {
        if (!owned.empty()) {
            WAGGLE_MEM_FREE(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
            owned.clear();
            owned.shrink_to_fit();
        }
        external       = ptr;
        external_owner = std::move(owner);
        refresh();
    }

    /// Drop both storage modes and return to the unallocated state.
    void clear() {
        if (!owned.empty()) {
            WAGGLE_MEM_FREE(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
            owned.clear();
            owned.shrink_to_fit();
        }
        external = nullptr;
        // Dropping the keepalive last: for pooled storage this is the free, so
        // the pointer must already be unpublished when it runs.
        external_owner.reset();
        refresh();
    }

    /// True when either storage mode is live.
    [[nodiscard]] bool allocated() const noexcept { return !owned.empty() || external != nullptr; }
};

/**
 * @brief Tag selecting a tensor constructor that SHARES the source's storage.
 *
 * For @c make_shared: these types have no move constructor, so passing ``shallow_alias()`` would
 * deep-copy and share nothing.
 */
struct SharedStorageTag {};

/// Allocate an empty block. Every tensor holds one from construction, so no
/// code path has to null-check the block itself, only its @ref StorageBase::base.
template <typename T, typename Vector>
[[nodiscard]] std::shared_ptr<StorageBlock<T, Vector>> make_storage_block() {
    return std::make_shared<StorageBlock<T, Vector>>();
}

EINSUMS_NAMESPACE_END(detail)
