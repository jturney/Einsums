//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Which way data flows across one manifest slot.
 *
 * Derived from the graph's own reader/writer sets rather than declared: a slot every
 * node only reads is an @ref Input, a slot only written is an @ref Output, and one both
 * read and written is @ref InOut. That last case is not rare and not a diagnostic - a
 * ``C`` accumulated into by a chain of contractions is read and written by design.
 *
 * @see Graph::manifest
 * @versionadded{2.0.0}
 */
enum class ManifestDirection : std::uint8_t {
    Input,  ///< Read by at least one node, written by none.
    Output, ///< Written by at least one node, read by none.
    InOut,  ///< Both read and written.
};

/**
 * @brief The name of a direction, for diagnostics and for the saved form.
 * @param[in] direction The direction to name.
 * @return A stable spelling. Written by NAME into any saved artifact, never by
 *         numeric value, per the compatibility policy.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view manifest_direction_name(ManifestDirection direction) noexcept;

/**
 * @brief The name of an ownership scope, for diagnostics and for the saved form.
 * @param[in] scope The scope to name.
 * @return A stable spelling ("graph", "pipeline", "workspace").
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view tensor_ownership_name(TensorOwnership scope) noexcept;

/**
 * @brief The name of an element type, for diagnostics and for the saved form.
 * @param[in] dtype The type to name.
 * @return A stable spelling ("float32", "float64", "complex64", "complex128", "unknown").
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view scalar_type_name(packed_gemm::ScalarType dtype) noexcept;

/**
 * @brief The @ref ManifestDirection spelled @p name, if there is one.
 * @param[in] name A spelling @ref manifest_direction_name produces.
 * @return The direction, or an empty optional when nothing is spelled that way.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<ManifestDirection> manifest_direction_from_name(std::string_view name) noexcept;

/**
 * @brief The @ref TensorOwnership spelled @p name, if there is one.
 * @param[in] name A spelling @ref tensor_ownership_name produces.
 * @return The scope, or an empty optional when nothing is spelled that way.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<TensorOwnership> tensor_ownership_from_name(std::string_view name) noexcept;

/**
 * @brief The element type spelled @p name, if there is one.
 * @param[in] name A spelling @ref scalar_type_name produces.
 * @return The type, or an empty optional when nothing is spelled that way.
 *
 * @c "unknown" resolves to @ref packed_gemm::ScalarType::Unknown, which is a
 * legitimate thing for a handle to carry (an integral scalar has no BLAS type);
 * a name that is not one of the five is the empty optional a loader reports.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<packed_gemm::ScalarType> scalar_type_from_name(std::string_view name) noexcept;

/**
 * @brief The prefix that marks a dim symbol as ragged over an index space.
 *
 * A ragged axis has no single extent, so it cannot be a plain symbol; it is spelled
 * ``"ragged:<space>"`` inside the same per-axis string vector a plain symbol lives in.
 * One vector rather than a vector plus a parallel flag array, because the pair would be
 * two things a serializer has to keep in step and a loader has to validate against each
 * other, and the bug history of this module is two derivations of one fact disagreeing.
 *
 * @see TensorHandle::dim_symbols
 * @versionadded{2.0.0}
 */
inline constexpr std::string_view ragged_symbol_prefix = "ragged:";

/**
 * @brief Whether @p symbol declares a ragged axis rather than a plain one.
 * @param[in] symbol One entry of @ref TensorHandle::dim_symbols or @ref ManifestEntry::dim_symbols.
 * @return True when it carries @ref ragged_symbol_prefix.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline bool is_ragged_symbol(std::string_view symbol) noexcept {
    return symbol.size() > ragged_symbol_prefix.size() && symbol.starts_with(ragged_symbol_prefix);
}

/**
 * @brief The index-space name a ragged symbol is ragged over.
 * @param[in] symbol A symbol @ref is_ragged_symbol accepts.
 * @return The space name, or an empty view when @p symbol is not ragged.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::string_view ragged_symbol_space(std::string_view symbol) noexcept {
    if (!is_ragged_symbol(symbol)) {
        return {};
    }
    // Pointer arithmetic rather than substr, which is only conditionally noexcept and so
    // would make this function a throwing one for a bound it has already checked.
    return std::string_view{symbol.data() + ragged_symbol_prefix.size(), symbol.size() - ragged_symbol_prefix.size()};
}

/**
 * @brief Spell the ragged symbol for @p space.
 * @param[in] space The index space the axis is ragged over.
 * @return ``"ragged:<space>"``.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::string make_ragged_symbol(std::string_view space) {
    return std::string(ragged_symbol_prefix) + std::string(space);
}

/**
 * @brief Per-instance extents a caller supplied for one ragged axis.
 *
 * A ragged axis is the design's answer to a family whose members genuinely differ in
 * size - PNO domains, block and tile dims - where declaring one symbol per space would be
 * a lie. ``Graph::bind_ragged_extents`` accepts the table, validates it, and stores it
 * here; the grouped-batched-GEMM path already consumes exactly this shape, which is why
 * the representation is not new work.
 *
 * @note STORED, not yet consumed. Re-forming the batched nodes that read a table is the
 *       resource/tuning phase's job on the load path, and that phase is a later task; see
 *       @ref Graph::bind_ragged_extents for the deferral and what a bind refuses in the
 *       meantime.
 *
 * @see Graph::bind_ragged_extents
 * @see Graph::ragged_extent_tables
 * @versionadded{2.0.0}
 */
struct RaggedExtentTable {
    TensorId    id{0};   ///< This graph's id for the slot the table describes.
    std::string name;    ///< The manifest name the table was supplied under.
    std::size_t axis{0}; ///< Which axis of that slot is ragged.
    std::string space;   ///< The index space the axis is ragged over.

    /// One extent per instance of @ref space, in instance order. Its length is the
    /// instance count, which every table over one space in one graph must agree on.
    std::vector<std::size_t> extents;
};

/**
 * @brief One named slot of a graph's interface: an operand the graph does not own.
 *
 * The formalization of the complement of @ref TensorHandle::is_intermediate. A handle
 * flagged intermediate is storage the graph created for itself and is never an entry
 * here; everything else a node touches is part of the contract a caller has to satisfy,
 * and this says what satisfying it means.
 *
 * @note @ref dims holds the extents of the instance the graph was captured at.
 *       @ref dim_symbols says which of them are part of the contract and which a bind may
 *       move; an entry with no symbols is fixed at exactly @ref dims, as @ref Graph::rebind
 *       always required.
 *
 * @see Graph::manifest
 * @see Graph::bind
 * @versionadded{2.0.0}
 */
struct ManifestEntry {
    /// The binding key. Copied from @ref TensorHandle::name, which capture in turn copied
    /// from the tensor. Names are the interface: a saved graph is bound by name, never by
    /// id. A tensor registered without a meaningful name of its own (the generated
    /// "tensor_<n>" spellings) is a legal entry and a fragile key - it is stable only as
    /// long as nothing upstream renumbers, so a graph meant to be saved should name its
    /// operands.
    std::string name;

    /// This graph's id for the slot. Graph-LOCAL and not stable across processes: ids are
    /// minted in registration order, so the same tensor in a re-captured graph can hold a
    /// different one. Useful to reach @ref Graph::tensor and to express @ref aliases_input
    /// within one manifest; never a binding key and never written to a file.
    TensorId id{0};

    ManifestDirection        direction{ManifestDirection::Input};     ///< Read, written, or both.
    packed_gemm::ScalarType  dtype{packed_gemm::ScalarType::Unknown}; ///< Element type a bound tensor must match.
    std::size_t              rank{0};                                 ///< Number of dimensions.
    std::vector<std::size_t> dims;                                    ///< Extents at capture, one per axis. See the note above.

    /// Per-axis symbolic extent declaration, parallel to @ref dims, or empty when every
    /// axis is literal. An empty string in a populated vector is one literal axis, an
    /// ordinary string is a symbol every slot naming it must agree on, and a
    /// ``"ragged:<space>"`` spelling is an axis whose extent differs per instance.
    ///
    /// Names, never ids, for the same reason @ref spaces holds names: this is what a
    /// serializer writes and a loader reads back, and a symbol is only meaningful by name.
    /// @see TensorHandle::dim_symbols
    /// @see Graph::annotate_dims
    std::vector<std::string> dim_symbols;

    /// Per-axis index-space NAMES, parallel to @ref dims, or empty when the slot is
    /// unannotated. Names, not @ref SpaceId values: an id is a handle into the registry
    /// that issued it and means nothing in another process or another registry.
    std::vector<std::string> spaces;

    /// Whether @ref spaces was inferred rather than declared by a user.
    /// @see TensorHandle::spaces_inferred
    bool spaces_inferred{false};

    /// The scope that owns the slot's storage: @c Graph for an operand supplied to this
    /// graph alone, @c Pipeline for one carried between a pipeline's stages, @c Workspace
    /// for one that outlives every pipeline over it.
    TensorOwnership scope{TensorOwnership::Graph};

    /// Manifest-level alias declaration: the @ref id of the entry whose storage this entry
    /// is part of, or 0 when the slot aliases nothing another entry names.
    ///
    /// Populated from @ref TensorHandle::aliases, but only when BOTH ends are manifest
    /// entries - an alias of an intermediate is the graph's own business. A loaded graph
    /// has no addresses, so this declaration is how an aliasing relation between two
    /// caller-supplied tensors survives a save at all; @ref Graph::bind validates a bind
    /// against it and rejects an aliasing bind that no entry declares.
    TensorId aliases_input{0};
};

/**
 * @brief A graph's named, typed, space-annotated, ownership-scoped interface.
 *
 * Entries are partitioned by direction: @ref ManifestDirection::InOut slots appear in
 * BOTH @ref inputs and @ref outputs, because a caller has to supply such a slot and reads
 * a result out of it. Within each vector the order is by @ref ManifestEntry::name, then by
 * @ref ManifestEntry::id, and that order is part of the contract - a serializer writes
 * entries in it and a test asserts against it.
 *
 * @note TOP-LEVEL only. A ``Loop`` body and a ``Conditional`` branch are separate graphs
 *       whose handles are deliberately fresh and default (see the MetadataBoundary
 *       contract), so a body has no meaningful interface of its own and this never
 *       descends into one. A buffer the parent hands to a body still appears here: the
 *       parent's control-flow node reports it through effective-IO expansion, which is a
 *       use of the PARENT's handle.
 *
 * @see Graph::manifest
 * @versionadded{2.0.0}
 */
struct InterfaceManifest {
    std::vector<ManifestEntry> inputs;  ///< Slots the graph reads. Sorted by name, then id.
    std::vector<ManifestEntry> outputs; ///< Slots the graph writes. Sorted by name, then id.

    /**
     * @brief Find an entry by binding key.
     * @param[in] name The name to look for.
     * @return The entry, or nullptr when no slot carries that name. An @c InOut slot is
     *         held in both vectors; the input copy is returned, and the two are equal.
     */
    [[nodiscard]] EINSUMS_EXPORT ManifestEntry const *find(std::string_view name) const noexcept;

    /**
     * @brief Find an entry by this graph's id for it.
     * @param[in] id The @ref ManifestEntry::id to look for.
     * @return The entry, or nullptr when no slot carries that id.
     */
    [[nodiscard]] EINSUMS_EXPORT ManifestEntry const *find_by_id(TensorId id) const noexcept;

    /**
     * @brief Every entry once, @c InOut slots included exactly once.
     * @return The union of @ref inputs and @ref outputs, in the same name-then-id order.
     */
    [[nodiscard]] EINSUMS_EXPORT std::vector<ManifestEntry> entries() const;

    /**
     * @brief Every binding key, in the manifest's own order.
     * @return The names, for a diagnostic that has to list what a caller could have meant.
     */
    [[nodiscard]] EINSUMS_EXPORT std::vector<std::string> names() const;

    /// Number of distinct slots (an @c InOut slot counts once).
    [[nodiscard]] EINSUMS_EXPORT std::size_t size() const noexcept;
};

/**
 * @brief Ownership scopes of tensors a scope declared, keyed by tensor object address.
 *
 * A @ref Workspace and a @ref Pipeline declare tensors that many graphs then capture, and
 * each capturing graph builds its OWN handle for such a tensor rather than sharing the
 * declaring scope's (see @ref make_handle). Scope would be lost at that boundary, so the
 * declaring scope publishes it here and the graph attaches the table - the same shape of
 * plumbing @ref ParamTable already uses, and shared by @c shared_ptr for the same reason:
 * a declaration made after a stage was added still has to reach that stage.
 *
 * Keyed on the tensor object's address, which is the identity that survives the
 * per-graph re-registration (a @ref TensorId does not).
 *
 * @see Graph::add_scope_map
 * @versionadded{2.0.0}
 */
using TensorScopeMap = std::unordered_map<void const *, TensorOwnership>;

/// A scope table shared between the scope that declares tensors and the graphs that use them.
using TensorScopeMapPtr = std::shared_ptr<TensorScopeMap>;

EINSUMS_NAMESPACE_END(compute_graph)
