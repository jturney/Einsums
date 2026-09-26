//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Common.hpp
/// @brief What the three halves of the ``einsums_graph_ir`` implementation share.
///
/// Private to the `GraphIR` sources. The schema itself is documented once, in
/// `GraphIR.hpp`; the member walk, which of `Graph`'s members are structure and
/// travel in a file, is documented once in `GraphIR.cpp`.
///
/// The implementation is three passes that meet in exactly three places, and
/// this header is those three places:
///
///  1. `Write.cpp` turns a graph into a document. It owns every `write_*`
///     helper and exports @ref write_structure and @ref write_document.
///  2. `Read.cpp` turns a document into the plain intermediate form below,
///     collecting EVERY problem rather than stopping at the first. It owns every
///     `read_*` helper and exports @ref read_document.
///  3. `Build.cpp` turns a clean intermediate form into a graph. It exports
///     @ref build_graph.
///
/// That the reader and the builder are separate passes is what lets
/// `validate_graph_ir` report three seeded errors as three messages while
/// `load_graph` stops at the first. The separation is enforced here by what this
/// header does NOT carry: the reader's `Problems` plumbing and the writer's
/// `Frame` numbering each stay inside their own translation unit, so neither
/// pass can reach into the other's machinery by accident.

#include <Einsums/ComputeGraph/Approximation.hpp>
#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::graph_ir)

using json::Array;
using json::Object;
using json::Value;

// ── Small shared helpers ───────────────────────────────────────────────────

/// The section keys, in the order the schema fixes them.
constexpr std::string_view key_version    = "einsums_graph_ir";
constexpr std::string_view key_provenance = "provenance";

/// Render a double, or the tagged string that stands for a value JSON cannot
/// spell. The three tags are the whole special-value policy: JSON has no NaN and
/// no infinity, and a schema that wrote `null` for them would lose the
/// distinction between "not a number" and "no value here".
inline Value number_or_tag(double value) {
    if (std::isnan(value)) {
        return Value{"nan"};
    }
    if (std::isinf(value)) {
        return Value{value > 0 ? "inf" : "-inf"};
    }
    return Value{value};
}

/// The inverse of @ref number_or_tag.
inline std::optional<double> tagged_number(Value const &value) {
    if (value.is_number()) {
        return value.as_double();
    }
    if (!value.is_string()) {
        return std::nullopt;
    }
    if (value.as_string() == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (value.as_string() == "inf") {
        return std::numeric_limits<double>::infinity();
    }
    if (value.as_string() == "-inf") {
        return -std::numeric_limits<double>::infinity();
    }
    return std::nullopt;
}

// ── The intermediate form ──────────────────────────────────────────────────
//
// A document is read into these first. Building a Graph is a separate step that
// only runs on a clean read, which is what lets the reader keep going after a
// problem and report every one of them.

struct IrTensor {
    std::size_t              id{0};
    std::string              name;
    packed_gemm::ScalarType  dtype{packed_gemm::ScalarType::Unknown};
    std::size_t              rank{0};
    std::vector<std::size_t> dims;
    std::vector<std::string> dim_symbols;
    std::vector<std::string> spaces;
    bool                     spaces_inferred{false};

    /// What the tensor is declared to BE. Optional, because the vast majority of tensors are
    /// untagged and writing an empty object for each of them would cost every file bytes for
    /// nothing; absent reads as untagged.
    ProvenanceTag tag;

    bool            intermediate{false};
    TensorOwnership scope{TensorOwnership::Graph};
    InitKind        init{InitKind::None};

    /// Whether the tensor still needs storage. Absent in files written before this was
    /// recorded, and Materialized is the right reading of those: a graph that never said
    /// otherwise had its intermediates allocated at capture.
    AllocState alloc{AllocState::Materialized};

    /// A rank-0 record's storage: a tensor object rather than a bare element. Written as
    /// ``"rank0": "tensor"`` or ``"scalar"`` on every rank-0 record since 1.9.0; absent reads as
    /// a bare element, which is what every earlier reader built.
    bool rank0_tensor{false};

    /// Set on a FRAGMENT tensor that denotes a buffer the enclosing frame
    /// already defines; the value is that frame's id for it.
    std::optional<std::size_t> outer;

    /// Manifest-only: the entry whose storage this one is part of, by name.
    std::string aliases_input;

    /// Manifest-only: what the file claims the direction is, checked against
    /// what the loaded nodes actually do.
    ManifestDirection direction{ManifestDirection::Input};
};

struct IrFragment;

struct IrNode {
    std::size_t              id{0};
    OpKind                   kind{OpKind::Custom};
    std::string              label;
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    packed_gemm::ScalarType  dtype{packed_gemm::ScalarType::Unknown};
    std::size_t              rank{0};
    OpData                   descriptor;

    /// Dense operand ids a descriptor names, remapped at build time. Only a GEMM
    /// hint has any, and they are kept out of the descriptor until then because a
    /// descriptor holds real @ref TensorId values.
    std::vector<std::size_t> hint_ids;

    std::shared_ptr<IrFragment> then_branch;
    std::shared_ptr<IrFragment> else_branch;
    std::shared_ptr<IrFragment> body;
};

struct IrFragment {
    std::string           name;
    std::vector<IrTensor> tensors;
    std::vector<IrNode>   nodes;
};

/// Gate-flag buffers, by name. Made while READING, because the predicates read
/// out of the document already hold them; the graph then adopts exactly these
/// rather than allocating a second set nothing points at.
using GateFlagTable = std::unordered_map<std::string, std::shared_ptr<std::vector<std::uint8_t>>>;

struct IrDocument {
    std::string version;
    std::string name;
    /// The provenance block's pass list. CARRIED, never acted on: a load does not re-run one
    /// of these, and does not refuse a graph because of what is in the list. It is put back on
    /// the loaded graph so a load, further optimization and a re-save do not silently drop the
    /// history of everything that shaped the file in the first place.
    std::vector<std::string>                          structural_passes;
    GateFlagTable                                     gate_buffers;
    std::vector<IrTensor>                             manifest;
    std::vector<std::string>                          space_names;
    std::vector<std::pair<std::string, std::string>>  symbol_ties;
    std::vector<ApproximationRecord>                  approximations;
    std::vector<std::pair<std::string, std::int64_t>> params;
    std::vector<std::pair<std::string, std::size_t>>  gate_flags;
    std::vector<IrTensor>                             tensors;
    std::vector<std::pair<std::size_t, std::size_t>>  slot_redirects;
    std::vector<IrNode>                               nodes;
};

// ── What each pass throws ──────────────────────────────────────────────────

/// Thrown by the writer when a graph holds something a file cannot carry.
/// Caught at the API boundary and turned into a @ref GraphError.
struct SaveRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Thrown when a structurally valid document still cannot be turned into a
/// graph. Caught at the API boundary.
struct BuildFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ── What each pass exports ─────────────────────────────────────────────────

/// Problems found while reading, in document order.
using Problems = std::vector<std::string>;

/// Render @p graph's structure, the sections a content hash is taken over.
/// Throws @ref SaveRefusal when the graph holds something a file cannot carry.
[[nodiscard]] json::Object write_structure(Graph const &graph);

/// Render @p graph whole, structure plus the provenance block.
/// Throws @ref SaveRefusal for the same reasons @ref write_structure does.
[[nodiscard]] json::Value write_document(Graph const &graph, SaveOptions const &options);

/// Read @p root into the intermediate form, appending every problem to
/// @p problems rather than throwing. A document with problems still yields an
/// @ref IrDocument; it is the caller that decides whether to go on.
[[nodiscard]] IrDocument read_document(json::Value const &root, Problems &problems, SpaceRegistry const &registry);

/// Turn a clean @p document into a graph. Throws @ref BuildFailure when a
/// document that READS clean still cannot be built.
[[nodiscard]] Graph build_graph(IrDocument const &document, SpaceRegistry &registry);

EINSUMS_NAMESPACE_END(compute_graph::graph_ir)
