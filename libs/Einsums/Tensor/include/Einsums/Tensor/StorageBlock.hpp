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
#include <utility>

EINSUMS_NAMESPACE_BEGIN(detail)

/**
 * @brief Type-erased handle to a tensor's current backing buffer.
 *
 * Anything that must keep reading a buffer across a relocation holds one of
 * these rather than a copy of the pointer. A tensor's storage moves in more
 * situations than it looks: @c materialize() allocates a deferred tensor,
 * @c materialize_into() re-seats it on an arena slice, @c release() drops it,
 * and @c resize() reallocates. A raw pointer taken before any of those is
 * silently stale afterwards, which is what made views of deferred tensors
 * dangle once a graph ran.
 *
 * @c base is the one authoritative copy of that pointer. @c generation counts
 * relocations, so an observer that cached a derived pointer can tell whether
 * it needs to recompute without comparing pointers it may no longer read.
 *
 * The destructor is deliberately non-virtual: every block is created through
 * @c std::make_shared of the concrete @ref StorageBlock, and @c shared_ptr
 * captures the concrete deleter at construction, so upcasting to
 * @c shared_ptr<StorageBase> still destroys the derived object.
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

    StorageBlock()                                = default;
    StorageBlock(StorageBlock const &)            = delete;
    StorageBlock &operator=(StorageBlock const &) = delete;
    StorageBlock(StorageBlock &&)                 = delete;
    StorageBlock &operator=(StorageBlock &&)      = delete;

    ~StorageBlock() { ProfileMemFree(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T))); }

    /// Republish the live pointer and count the relocation. Call after any
    /// change to @ref owned or @ref external.
    void refresh() noexcept {
        base = external != nullptr ? static_cast<void *>(external) : static_cast<void *>(owned.data());
        ++generation;
    }

    /// Grow or shrink self-allocated storage. Detaches nothing: calling this
    /// while @ref external is set would leave two live storage modes, which the
    /// tensor types prevent by releasing first.
    void resize_owned(size_t elems) {
        ProfileMemFree(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        owned.resize(elems);
        ProfileMemAlloc(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        refresh();
    }

    /// Replace self-allocated storage with a copy of @p src. Used by the
    /// deep-copying tensor copy constructor, which gets a block of its own.
    void copy_owned_from(Vector const &src) {
        ProfileMemFree(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        owned    = src;
        external = nullptr;
        ProfileMemAlloc(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        refresh();
    }

    /// Take ownership of @p src's buffer, leaving @p src empty.
    void adopt_owned(Vector &&src) {
        ProfileMemFree(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
        owned    = std::move(src);
        external = nullptr;
        refresh();
    }

    /// Attach caller-owned storage, dropping any self-allocated buffer.
    void attach_external(T *ptr) {
        if (!owned.empty()) {
            ProfileMemFree(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
            owned.clear();
            owned.shrink_to_fit();
        }
        external = ptr;
        refresh();
    }

    /// Drop both storage modes and return to the unallocated state.
    void clear() {
        if (!owned.empty()) {
            ProfileMemFree(static_cast<int64_t>(owned.size()) * static_cast<int64_t>(sizeof(T)));
            owned.clear();
            owned.shrink_to_fit();
        }
        external = nullptr;
        refresh();
    }

    /// True when either storage mode is live.
    [[nodiscard]] bool allocated() const noexcept { return !owned.empty() || external != nullptr; }
};

/**
 * @brief Tag selecting a tensor constructor that SHARES the source's storage.
 *
 * The tag exists so the alias can be built in place. ``shallow_alias()``
 * returns by value, and these tensor types have no move constructor (a
 * user-declared destructor suppresses it), so handing that prvalue to
 * @c std::make_shared binds it to the *copy* constructor and silently
 * deep-copies -- the alias then shares nothing and every write through it is
 * lost. @c new T(t.shallow_alias()) avoids that only through guaranteed copy
 * elision, at the cost of a second allocation for the control block.
 * Constructing with this tag gets both: one allocation, and real sharing.
 */
struct SharedStorageTag {};

/// Allocate an empty block. Every tensor holds one from construction, so no
/// code path has to null-check the block itself, only its @ref StorageBase::base.
template <typename T, typename Vector>
[[nodiscard]] std::shared_ptr<StorageBlock<T, Vector>> make_storage_block() {
    return std::make_shared<StorageBlock<T, Vector>>();
}

EINSUMS_NAMESPACE_END(detail)
