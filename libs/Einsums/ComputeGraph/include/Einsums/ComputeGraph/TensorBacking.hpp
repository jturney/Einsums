//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file TensorBacking.hpp
/// @brief Where a tensor's data lives when it is not in memory, and how a graph reads it.
///
/// A tensor annotated with a @ref TensorBacking is an input whose elements stay in storage the
/// graph reads at each replay, so the caller never holds it. The backing names its storage by a
/// KIND and a LOCATION, which is what a saved graph keeps, and carries the reader, which is not
/// saved: a module that knows the storage registers a @ref BackingKind under the kind's name, and
/// loading a graph rebuilds the reader through it. This module knows no storage format itself.

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {
class RunContext;
} // namespace detail

/**
 * @brief Read one slab of a backed tensor.
 *
 * @p ranges gives, per axis, the half-open range ``[first, second)`` of the stored tensor to read.
 * The elements land in @p destination densely and in column-major order over the slab's own
 * extents, which is the order a freshly constructed tensor of the slab's shape holds them in.
 * Throws on any failure to read; a short read is a failure, never a partial slab.
 */
using SlabReader = std::function<void(void *destination, std::vector<std::pair<std::size_t, std::size_t>> const &ranges)>;

/**
 * @brief Write one slab of a backed tensor.
 *
 * The mirror of @ref SlabReader: @p source holds the slab densely and in column-major order over
 * its own extents, and the writer stores it at @p ranges of the stored tensor. Throws on any
 * failure to write.
 */
using SlabWriter = std::function<void(void const *source, std::vector<std::pair<std::size_t, std::size_t>> const &ranges)>;

/**
 * @brief What a backing's storage has moved, counted as it happens.
 *
 * Relaxed atomics: slabs are read and written on whichever thread runs the node, and the counts
 * are read only between replays. @ref Graph::annotate_backing attaches a fresh set to each backing
 * it accepts, and @ref RunMetrics reports one replay's share as the difference of two readings.
 * @versionadded{2.0.0}
 */
struct BackingCounters {
    /// The run context of the graph that accepted the backing, which tells an attached observer
    /// (@ref ExecutionObserver) what the storage moves.
    std::shared_ptr<detail::RunContext> context;
    /// The backed tensor's name, for those events.
    std::string tensor;

    std::atomic<std::uint64_t> bytes_read{0};    ///< Bytes read from the storage.
    std::atomic<std::uint64_t> bytes_written{0}; ///< Bytes written to it.
    std::atomic<std::uint64_t> reads{0};         ///< Slab reads, a whole load counting as one.
    std::atomic<std::uint64_t> writes{0};        ///< Slab writes, a whole store counting as one.
    std::atomic<std::uint64_t> publishes{0};     ///< Publishes of an update backing.
};

/**
 * @brief The storage behind a tensor the graph reads, or writes, but never holds whole.
 *
 * @ref kind and @ref location are what a saved graph records, with @ref write. @ref dims and
 * @ref dtype describe what the storage holds and are checked against the tensor when the backing
 * is annotated, so a mismatch is reported where it is made rather than at the first replay.
 * @ref read_slab and @ref write_slab are rebuilt at load by the registered @ref BackingKind and
 * never written to a file.
 *
 * A READ backing names storage that already holds the tensor: an input the graph only reads. A
 * WRITE backing names storage the graph fills: the tensor is an output the graph produces, and
 * may read back after producing it. A write backing is completed by the tensor it is annotated
 * on, through @ref prepare, since the storage may not exist yet to describe it. An UPDATE backing
 * is a write backing whose storage also holds the value each replay starts from: the graph reads
 * the tensor, writes its next value, and the storage is switched to that value at the end of the
 * replay (@ref update, @ref publish), together with every other update backing of the graph that
 * shares its @ref publish_scope.
 *
 * Built by the module that knows the storage, as ``tensor_io::etn_backing`` builds one for a
 * ``.etn`` entry, and handed to @ref Graph::annotate_backing.
 *
 * @see Graph::annotate_backing
 * @see register_backing_kind
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") TensorBacking {
    /// The registered kind's name, qualified as in ``"tensor_io.etn"``.
    APIARY_EXPOSE APIARY_READONLY std::string kind;

    /// Key/value pairs that name the storage within its kind, sorted by key, as in
    /// ``{{"entry", "B_vv"}, {"path", "ints.etn"}}``.
    APIARY_EXPOSE APIARY_READONLY std::vector<std::pair<std::string, std::string>> location;

    /// The stored tensor's extents.
    APIARY_EXPOSE APIARY_READONLY std::vector<std::size_t> dims;

    /// The stored tensor's element type.
    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown};

    /// Whether the graph writes the storage (an output it produces) rather than only reading it.
    APIARY_EXPOSE APIARY_READONLY bool write{false};

    /// Whether the graph also reads the value the storage holds when a replay starts: an UPDATE
    /// backing, set with @ref write. @ref read_slab reads that value, @ref write_slab writes the
    /// next one beside it, and @ref publish makes the next one current once the replay has made
    /// every access. A crash before the publish leaves the storage holding the previous value.
    APIARY_EXPOSE APIARY_READONLY bool update{false};

    /// Makes what @ref write_slab wrote the value @ref read_slab reads, in one step that survives a
    /// crash, for every backing in @p group at once: the update backings of one graph that share
    /// this one's @ref publish_scope, this one among them, or this one alone when it has none. Run
    /// by the graph once per replay, after every access of every storage the graph names. Set on an
    /// update backing only.
    std::function<void(std::vector<TensorBacking const *> const &group)> publish;

    /// Which update backings publish together: those of one graph with the same non-empty scope
    /// are published by one call, so a crash leaves all of them at the values the replay started
    /// from or all at the values it wrote. A kind sets it to what one atomic step can cover, as
    /// ``tensor_io.etn`` sets the file's path. Empty: published alone.
    APIARY_EXPOSE APIARY_READONLY std::string publish_scope;

    /// What the storage has moved, attached by @ref Graph::annotate_backing; null on a backing
    /// no graph has accepted. See @ref RunMetrics.
    std::shared_ptr<BackingCounters> counters;

    /// Reads a slab of the stored tensor; see @ref SlabReader.
    SlabReader read_slab;

    /// Writes a slab of the stored tensor; see @ref SlabWriter. Set on a write backing only.
    SlabWriter write_slab;

    /// The order the storage holds the tensor's axes in: the declared axis at each stored
    /// position, ``0, 1, ..., rank - 1`` for the declared order. Unset for a kind that stores only
    /// the declared order. The slabs @ref read_slab and @ref write_slab move are in the declared
    /// order whatever this is; it decides only which slabs are long contiguous runs of the storage.
    std::function<std::vector<std::size_t>()> stored_order;

    /// Store the tensor's axes in @p order from now on. Returns false, changing nothing, when the
    /// storage's order is no longer the graph's to choose: storage this graph did not create, an
    /// order its creator fixed, or storage a slab has already been written to. Set on a write
    /// backing whose kind can store another order.
    std::function<bool(std::vector<std::size_t> const &order)> reorder;

    /// Whether @ref reorder would change the order now: what a pass choosing how to move the tensor
    /// asks, to know whether the order it has to work with is the one stored or the one it wants.
    /// Unset reads as false.
    std::function<bool()> order_open;

    /// Completes a write backing for the tensor it is annotated on: given the tensor's extents
    /// and element type, makes the storage hold a tensor of that shape and returns the backing
    /// with @ref dims, @ref dtype, @ref read_slab and @ref write_slab filled in. Throws when the
    /// storage already holds something of another shape or type. Set on a write backing only.
    std::function<TensorBacking(std::vector<std::size_t> const &dims, packed_gemm::ScalarType dtype)> prepare;

    /// @brief The stored tensor's element type by name, as a ``dtype=`` argument spells it.
    /// @return ``"float32"``, ``"float64"``, ``"complex64"`` or ``"complex128"``; nothing for a write
    ///         backing not yet completed by its tensor, whose storage has no type to give.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("dtype") [[nodiscard]] EINSUMS_EXPORT std::optional<std::string> dtype_name() const;

    /// @brief The kind and location in one phrase, for diagnostics.
    /// @return Text such as ``tensor_io.etn (entry=B_vv, path=ints.etn)``.
    APIARY_EXPOSE [[nodiscard]] EINSUMS_EXPORT std::string describe() const;
};

/**
 * @brief How to open the storage of one backing kind from its saved location.
 *
 * @ref name must be QUALIFIED - contain a ``.``, as in ``"tensor_io.etn"`` - for the same reason a
 * registered descriptor's must: no two modules can then claim one name by accident.
 */
struct BackingKind {
    /// The kind's name, as @ref TensorBacking::kind spells it.
    std::string name;

    /// The backing for a saved location. For a read backing, with its dims, dtype and reader
    /// filled in; for a write backing (@p write true), unprepared, as the kind's own constructor
    /// returns one, to be completed by @ref TensorBacking::prepare when it is annotated. Throw
    /// @c std::invalid_argument when the storage cannot be opened or does not hold the location;
    /// the loader reports the message against the tensor.
    std::function<TensorBacking(std::vector<std::pair<std::string, std::string>> const &location, bool write)> open;
};

/**
 * @brief Register @p kind so a saved graph naming it can be loaded.
 *
 * Do it from a namespace-scope initializer in the module that knows the storage. Registration is
 * thread-safe.
 *
 * @throws std::invalid_argument when the name is not qualified, is already registered, or when
 *         @ref BackingKind::open is empty.
 */
EINSUMS_EXPORT void register_backing_kind(BackingKind kind);

/**
 * @brief Open a saved location through its registered kind.
 * @param[in] kind The kind's name.
 * @param[in] location The saved location.
 * @param[in] write Whether the backing is a write backing.
 * @return The backing, its reader rebuilt (a write backing unprepared).
 * @throws std::invalid_argument when no kind is registered under @p kind, or when the kind's
 *         @ref BackingKind::open throws.
 */
[[nodiscard]] EINSUMS_EXPORT TensorBacking open_backing(std::string const                                      &kind,
                                                        std::vector<std::pair<std::string, std::string>> const &location, bool write);

EINSUMS_NAMESPACE_END(compute_graph)
