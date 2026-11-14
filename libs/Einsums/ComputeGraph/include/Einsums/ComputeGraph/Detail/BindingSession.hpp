//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file BindingSession.hpp
/// @brief What binding a graph's interface to a caller's storage has recorded: the slots bound
///        and the span each was given, the interface names they keep, the ragged extent tables,
///        the setup key, and the slots of a bind still being assembled.

#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

namespace detail {

/**
 * @brief A graph's bind-time state: facts about the CALLER's problem rather than the graph's.
 *
 * None of it is saved, since a loaded graph is bound afresh. @ref Graph runs the bind
 * transaction and decides what a bind refuses; this records what the binds that succeeded
 * established.
 */
class EINSUMS_EXPORT BindingSession {
  public:
    /// Half-open byte span of a bound operand's storage; a null @ref lo means the
    /// operand offered none.
    ///
    /// A span rather than a base address, because two DISJOINT slices of one buffer
    /// have distinct base addresses and must still bind without complaint, while two
    /// OVERLAPPING ones share no address at all when neither starts where the other
    /// does. Comparing base pointers answers the second case wrong, and answering it
    /// wrong is how a bind loses the hazard edges between two slots.
    struct BoundSpan {
        char const *lo{nullptr}; ///< First byte, or null when there is no span.
        char const *hi{nullptr}; ///< One past the last byte.

        /// Whether the two spans share a byte. False whenever either is absent:
        /// an operand with no address cannot be shown to alias anything.
        [[nodiscard]] bool overlaps(BoundSpan const &other) const noexcept;

        /// Number of bytes the two spans share, for the diagnostic.
        [[nodiscard]] std::size_t overlap_bytes(BoundSpan const &other) const noexcept;
    };

    /**
     * @brief What one bind's first walk worked out about the symbols.
     *
     * Values only, no storage identities: the second walk repoints, this one decides
     * whether it may.
     */
    struct DimSolution {
        /// Symbol name to the extent this bind solved it at.
        std::unordered_map<std::string, std::size_t> values;
        /// Symbol name to the manifest slot that first supplied its extent, so a
        /// disagreement can name both sides rather than only the loser.
        std::unordered_map<std::string, std::string> witness;
        /// True when at least one bound entry declares a symbolic or ragged axis.
        bool any_symbolic{false};
        /// True when some bound extent differs from the one the graph holds. Everything
        /// expensive below this is gated on it, so the ordinary same-shape bind - a replay
        /// loop's - pays for one comparison per axis and nothing else.
        bool extents_changed{false};
    };

    /// One slot handed to a multi-slot bind, held until the commit runs the transaction.
    /// The two steps are stored type-erased because the pairs are heterogeneous and the
    /// commit has to walk them twice, once to solve and once to repoint.
    ///
    /// The steps take the graph as an argument rather than capturing it, so a pending bind
    /// stays valid when the graph it was added to is moved before the commit.
    struct PendingBind {
        std::string                                                                  name;
        std::function<void(Graph &, InterfaceManifest const &, DimSolution &)>       collect;
        std::function<void(Graph &, InterfaceManifest const &, DimSolution const &)> apply;
    };

    // ── A bind being assembled ────────────────────────────────────────────

    /// Drop any slots added since the last commit.
    void begin_pending() noexcept { _pending.clear(); }

    /// Add one slot to the bind being assembled.
    void add_pending(PendingBind slot) { _pending.push_back(std::move(slot)); }

    /// Take the assembled slots, leaving none: a refused transaction must not leak into the
    /// next one.
    [[nodiscard]] std::vector<PendingBind> take_pending() noexcept { return std::exchange(_pending, {}); }

    // ── Slots bound ───────────────────────────────────────────────────────

    /// Whether a bind has supplied storage for @p id.
    [[nodiscard]] bool is_bound(TensorId id) const noexcept { return _bound.contains(id); }

    /// Every bound slot and the span it was given.
    [[nodiscard]] std::unordered_map<TensorId, BoundSpan> const &bound_spans() const noexcept { return _bound; }

    /// Record @p id as bound to @p span; an empty span still counts as bound.
    void mark_bound(TensorId id, BoundSpan span) { _bound.insert_or_assign(id, span); }

    // ── Interface names ───────────────────────────────────────────────────

    /// The interface name pinned for @p id, or null when none was.
    [[nodiscard]] std::string const *interface_name(TensorId id) const noexcept;

    /// Pin @p name as the interface name of @p id, replacing any before.
    void pin_interface_name(TensorId id, std::string const &name) { _interface_names.insert_or_assign(id, name); }

    /// Pin @p name for @p id unless a name is already pinned.
    void pin_interface_name_once(TensorId id, std::string const &name) { _interface_names.emplace(id, name); }

    // ── Ragged extents ────────────────────────────────────────────────────

    /// Every ragged extent table accepted, in supply order.
    [[nodiscard]] std::vector<RaggedExtentTable> const &ragged_tables() const noexcept { return _ragged; }

    /// The table held for @p name axis @p axis, or null.
    [[nodiscard]] RaggedExtentTable const *find_ragged(std::string_view name, std::size_t axis) const noexcept;

    /// Hold @p table, replacing the one for the same slot and axis.
    void hold_ragged(RaggedExtentTable table);

    // ── Setup key ─────────────────────────────────────────────────────────

    [[nodiscard]] std::string const &setup_key() const noexcept { return _setup_key; }
    void                             set_setup_key(std::string key) noexcept { _setup_key = std::move(key); }

    /// Forget which slots are bound and the ragged tables they came with.
    void clear() noexcept {
        _bound.clear();
        _ragged.clear();
    }

  private:
    std::vector<PendingBind> _pending;

    /// Manifest slots a bind has supplied storage for, and the byte span each was
    /// given. Keyed by TensorId, not by name: a rebind renames the handle after
    /// the tensor it now points at, so a name is not stable across the very
    /// operation this records. The span is empty when the operand offered no address
    /// (a deferred shell, a tile-wise sparse tensor), which still counts as bound and
    /// simply cannot participate in the aliasing check.
    std::unordered_map<TensorId, BoundSpan> _bound;

    /// Interface names pinned by a bind, by TensorId.
    ///
    /// A manifest name is the contract, and a contract that renamed itself every
    /// time a caller bound different storage to it would be unusable: the second
    /// ``bind("t2", ...)`` of a replay loop would report "no such interface tensor"
    /// because the first one renamed the slot after the tensor it was handed.
    /// A rebind's rename is right for the handle (which names a tensor) and wrong
    /// for the interface (which names a slot), so the interface keeps its own record.
    std::unordered_map<TensorId, std::string> _interface_names;

    /// Per-instance extent tables accepted, in supply order.
    std::vector<RaggedExtentTable> _ragged;

    /// The problem identity a caller declared, or empty when none was.
    std::string _setup_key;
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
