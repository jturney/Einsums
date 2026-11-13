//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Which of a tensor's two copies, host and device, hold its current value during one
 *        replay, on a build whose device memory is separate from the host's.
 *
 * A tensor is in one of three states. @c HostOnly: the device copy is absent or stale.
 * @c Both: the copies agree. @c DeviceOnly: a device kernel wrote it and the host copy is stale.
 * Every tracked tensor starts a replay @c HostOnly.
 *
 * A node that reads or writes a tensor on one side first asks for that side to be current
 * (@ref need_on_device, @ref need_on_host), which copies from the other side only when the
 * requested side is stale; a node that wrote it then says on which side (@ref wrote_on_device,
 * @ref wrote_on_host). Reading never makes a copy stale, so two readers never copy toward a side
 * the other is reading.
 *
 * @par Concurrency
 * Two nodes that both access a tensor and one of which writes it are ordered by the graph's
 * dependency edges, which every executor honors, so a write never races another access to the
 * same tensor. Readers are not ordered among themselves: a reader that has to copy does it under
 * the tensor's lock, and one that finds the requested side current copies nothing. Any number of
 * threads may therefore call these for different nodes at once. @ref reset and @ref flush are
 * for the one thread that owns the replay, before and after the nodes run.
 *
 * The copies are the caller's: each takes a callable that performs the copy and returns whether
 * it happened, and the state changes only when it did.
 */
class DeviceCoherence {
  public:
    /// Where a tensor's current value is.
    enum class State : std::uint8_t {
        HostOnly,   ///< The device copy is absent or stale.
        Both,       ///< The two copies agree.
        DeviceOnly, ///< A device kernel wrote it; the host copy is stale.
    };

    DeviceCoherence()                                   = default;
    DeviceCoherence(DeviceCoherence const &)            = delete;
    DeviceCoherence &operator=(DeviceCoherence const &) = delete;
    DeviceCoherence(DeviceCoherence &&)                 = delete;
    DeviceCoherence &operator=(DeviceCoherence &&)      = delete;
    ~DeviceCoherence()                                  = default;

    /// @brief Track @p tensors, each starting @c HostOnly, and forget any tracked before.
    /// @param[in] tensors The tensors a device node or a transfer can touch; repeats are harmless.
    void reset(std::span<TensorId const> tensors) {
        _index.clear();
        for (TensorId const id : tensors) {
            _index.try_emplace(id, _index.size());
        }
        // Built in place: an entry holds a mutex and cannot move, but assigning the vector moves
        // only its buffer.
        _entries = std::vector<Entry>(_index.size());
    }

    /// @brief Whether @p id is tracked. An untracked tensor never leaves the host.
    [[nodiscard]] bool tracked(TensorId id) const noexcept { return _index.contains(id); }

    /// @brief Where @p id's current value is; @c HostOnly for an untracked tensor.
    [[nodiscard]] State state(TensorId id) const {
        Entry *entry = find(id);
        if (entry == nullptr) {
            return State::HostOnly;
        }
        std::scoped_lock const lock(entry->mutex);
        return entry->state;
    }

    /**
     * @brief Make the device copy of @p id current, copying the host copy up when it is stale.
     * @param[in] id      The tensor; must be tracked.
     * @param[in] copy_up Performs the host-to-device copy, returning whether it did.
     * @return Whether the device copy is current afterwards.
     */
    template <typename CopyUp>
    bool need_on_device(TensorId id, CopyUp &&copy_up) {
        Entry *entry = find(id);
        if (entry == nullptr) {
            return false;
        }
        std::scoped_lock const lock(entry->mutex);
        if (entry->state == State::HostOnly) {
            if (!copy_up()) {
                return false;
            }
            entry->state = State::Both;
        }
        return true;
    }

    /**
     * @brief Make the host copy of @p id current, copying the device copy down when it is stale.
     * @param[in] id        The tensor; an untracked one is always current on the host.
     * @param[in] copy_down Performs the device-to-host copy, returning whether it did.
     * @return Whether the host copy is current afterwards.
     */
    template <typename CopyDown>
    bool need_on_host(TensorId id, CopyDown &&copy_down) {
        Entry *entry = find(id);
        if (entry == nullptr) {
            return true;
        }
        std::scoped_lock const lock(entry->mutex);
        if (entry->state == State::DeviceOnly) {
            if (!copy_down()) {
                return false;
            }
            entry->state = State::Both;
        }
        return true;
    }

    /// @brief A device kernel wrote @p id: the host copy is stale. No effect when untracked.
    void wrote_on_device(TensorId id) { set(id, State::DeviceOnly); }

    /// @brief A host kernel wrote @p id: the device copy is stale. No effect when untracked.
    void wrote_on_host(TensorId id) { set(id, State::HostOnly); }

    /**
     * @brief Copy down every tensor whose current value is on the device alone.
     * @param[in] copy_down Called with each such tensor's id; performs its device-to-host copy and
     *                      returns whether it did.
     */
    template <typename CopyDown>
    void flush(CopyDown &&copy_down) {
        for (auto const &[id, slot] : _index) {
            Entry &entry = _entries[slot];
            if (entry.state == State::DeviceOnly && copy_down(id)) {
                entry.state = State::Both;
            }
        }
    }

  private:
    struct Entry {
        std::mutex mutex;
        State      state{State::HostOnly};
    };

    [[nodiscard]] Entry *find(TensorId id) const {
        auto const it = _index.find(id);
        return it == _index.end() ? nullptr : &_entries[it->second];
    }

    void set(TensorId id, State state) {
        if (Entry *entry = find(id); entry != nullptr) {
            std::scoped_lock const lock(entry->mutex);
            entry->state = state;
        }
    }

    /// Tensor to its entry. Written only by @ref reset, so concurrent lookups are reads.
    std::unordered_map<TensorId, std::size_t> _index;
    /// One per tracked tensor. Mutable because each entry's state is guarded by its own lock,
    /// which a const query takes too.
    mutable std::vector<Entry> _entries;
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
