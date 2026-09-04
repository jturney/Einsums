//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file GraphIR.hpp
 * @brief Save a graph's structure to a file, and read it back as a runnable graph.
 *
 * @par What this is, and what `to_json` is not
 * ``Graph::to_json`` writes ids, kinds, labels, edges and timings for the
 * profile viewer, and nothing reads it back. This is the round-trip IR of the
 * an exact, versioned, by-name encoding of everything a
 * graph needs to be rebuilt and executed in another process, and the interchange
 * format an offline optimizer transforms.
 *
 * Both stay. The debug view is allowed to be lossy; this one is not allowed to
 * be, which is why it refuses a graph it cannot write instead of writing a
 * partial one.
 *
 * @par The schema: ``einsums_graph_ir``, semver, currently 1.3.0
 * One JSON object with a FIXED top-level key order, so a tool can validate the
 * interface before it parses a single node:
 *
 * @code{.json}
 * {
 *   "einsums_graph_ir": "1.3.0",
 *   "provenance":       { ... },
 *   "name":             "ccsd_doubles",
 *   "manifest":         [ ... ],
 *   "spaces":           { "names": [ ... ], "symbol_ties": [ ... ] },
 *   "approximations":   [ ... ],
 *   "params":           [ ... ],
 *   "gate_flags":       [ ... ],
 *   "tensors":          [ ... ],
 *   "slot_redirects":   [ ... ],
 *   "nodes":            [ ... ]
 * }
 * @endcode
 *
 * Every section is written even when empty, so the layout is fixed rather than
 * merely conventional and a diff of two files lines up section for section.
 *
 * ``approximations`` arrived at 1.1.0 and is the one section a READER treats as
 * optional, because the golden corpus is written by older builds and a file
 * without it describes a graph nothing approximated. Its ``origin`` key arrived
 * at 1.2.0 and is optional within it for the same reason, defaulting to
 * ``asserted``: a number whose provenance nobody recorded is not evidence. It is written
 * unconditionally, empty list included, so the layout rule above still holds
 * for everything this build produces.
 *
 * An @ref OpKind::ElementTransform node's ``param`` key arrived at 1.3.0, is
 * optional, and is written only where the capture site chose a number. Its
 * default is not a constant in this file: an absent key means the default the
 * NAMED op's registration documents, so an older file keeps computing what it
 * computed and a reader has one place to look up what it will run at.
 *
 * @par The hash domain
 * @ref Graph::content_hash digests the canonical bytes of that object with
 * ``provenance`` REMOVED and nothing else changed. Provenance is therefore
 * outside the hash by construction: a field can only enter the hash domain by
 * being placed in a structure section. Two captures of the same program under
 * different builds, in different working directories, after different pass
 * pipelines, hash equal; a changed prefactor, a reordered node or a renamed slot
 * does not.
 *
 * @par Provenance is data, never instructions
 * The block records the library version, the ABI @c config_fingerprint, and the
 * names of the structural passes the caller says produced this form. A loader
 * READS it and acts on none of it: it does not re-run a named pass, does not
 * refuse a foreign fingerprint, and does not trust a version string over the
 * schema version. Deferred: nothing in the pass manager records which passes ran
 * on a given graph, so the pass list is supplied by the caller
 * (@ref SaveOptions::structural_passes) rather than recovered.
 *
 * @par Everything by NAME
 * Op kinds, dtypes, directions, ownership scopes, comparison operators,
 * parameter source types, element-op kernels, index spaces and dim symbols are
 * all written as strings. @ref OpKind is a @c std::uint8_t enum whose members
 * are added in the middle by whoever adds an operation, so a numeric value in a
 * file silently means a different operation the next time someone inserts one.
 * Anything a loading process cannot resolve fails AT LOAD with the name in the
 * message, and for a registry-backed name (an element op, an index space) the
 * message lists what is registered.
 *
 * @par The renumbering, and why files are byte-identical
 * Ids in a file are dense and positional, never the graph's own: a @c TensorId
 * is minted in registration order, so the same program captured twice can carry
 * different ones. On save both id spaces are renumbered from 0, deterministically:
 *
 * - **Nodes** take their program order, which is the order @ref Graph::nodes
 *   holds them in.
 * - **Tensors** take FIRST-MENTION order in one fixed walk: every manifest entry
 *   in manifest order (by name, then by the graph's id), then, for each node in
 *   program order, its inputs in order, its outputs in order, and finally the
 *   operands its descriptor names (a GEMM hint's three). A tensor no walk
 *   reaches is not written, because nothing could refer to it.
 *
 * Two captures of one program therefore produce byte-identical files, which is
 * asserted rather than assumed.
 *
 * @par Fragments: one encoding for a graph, a subgraph and a region
 * A ``Loop`` body and each ``Conditional`` branch is a separate @ref Graph with
 * its own tensor table, so it is written as a FRAGMENT: an object holding a
 * name, a local tensor list and a local node list, with its own ids renumbered
 * from 0 by the same rule. A fragment tensor that denotes a buffer the enclosing
 * frame already names carries ``"outer"`` with that frame's id for it, which is
 * the boundary reference: without it a loaded body would allocate its own copy
 * of the parent's buffer and write to the wrong storage. Boundary identity is
 * resolved at SAVE time by tensor-object address, which a live captured graph
 * has and a file does not - which is exactly why it has to be recorded.
 *
 * Fragments nest, so a loop inside a conditional inside a loop is three levels of
 * the same shape. A region dump reuses this shape rather than inventing a second
 * one: a region is a node list plus the tensors it touches plus references to
 * what crosses its boundary, which is what a fragment already is.
 *
 * @par What is saved, and what is deliberately not
 * Structure only. Saved: nodes,
 * descriptors, dataflow edges, slot redirects, the manifest, index-space and
 * symbolic-extent annotations, parameters, named gate-flag arrays, and the
 * schema version. NOT saved, at all: @ref Node::thread_width,
 * @ref Node::admission_priority, @ref Node::stream_id, per-node timings,
 * @ref Node::estimated_flops and @ref Node::estimated_bytes, and the planned
 * thread count. Those are tuning artifacts of one machine and one BLAS vendor,
 * and reusing one across machines is not a performance risk to weigh against
 * convenience - it is a correctness bug waiting for a different conda
 * environment.
 *
 * ``_slot_redirects`` IS saved, and the reason is worth stating because it looks
 * like a derived cache. CSE merges a duplicate by redirecting the loser's SLOT
 * and deliberately leaving the node dataflow alone, so the redirect is the only
 * record that a node naming the loser must read the winner's buffer. Dropping it
 * would leave a loaded graph reading a buffer nothing writes.
 *
 * @par What refuses, and when
 * At SAVE: any node @ref Graph::serializability_report names (the error carries
 * the whole report); a @ref BoundExpr or @ref PredExpr callback arm, naming the
 * node and the field; an anonymous @ref OpKind::ElementTransform kernel; and a
 * @ref PredExpr::FlagTest whose array has no name, pointing at
 * @ref Graph::name_gate_flags. At LOAD: an unknown schema version, an
 * unresolvable name of any kind, a structural inconsistency (an id no tensor
 * defines, a manifest direction that disagrees with what the nodes actually do),
 * and ANY unconsumed key.
 *
 * @par What a loaded graph still needs
 * ``load_graph`` gives back structure plus placeholder storage, with the manifest
 * intact so @ref Graph::bind works by name. The RESOURCE and TUNING phases are
 * the caller's next step and are deliberately not run here:
 *
 * @code
 * auto graph = cg::load_graph("ccsd_doubles.eig.json").value();
 * graph.bind("t2", T2, "fock", F);
 * graph.apply(cg::PassManager::resource_pass_manager());
 * graph.apply(cg::PassManager::tuning_pass_manager());
 * graph.execute();
 * @endcode
 *
 * Auto-applying them would be wrong twice over: the caller may want to bind
 * first (and an extent change invalidates a batching decision), and a loader
 * that silently rewrote the node set would make the loaded graph something other
 * than what the file says.
 *
 * @see Graph::serializability_report for what a graph must satisfy to be saved
 * @see Einsums/ComputeGraph/Detail/Json.hpp for the strict document model
 */

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief The schema version this build writes.
 *
 * Semver, and the compatibility policy is: a newer build loads an
 * older IR, the reverse is refused with both versions named. A descriptor field
 * may be ADDED (an absent field takes its documented default) but never
 * repurposed; a semantic change is a new field name and a minor bump.
 * @versionadded{2.0.0}
 */
inline constexpr std::string_view graph_ir_schema_version = "1.3.0";

/// @brief Knobs for @ref save_graph.
/// @versionadded{2.0.0}
struct SaveOptions {
    /// Lay the file out for diffing (one node per line) rather than canonically.
    /// The content is identical either way; only the whitespace differs, and a
    /// pretty file reads back to the same canonical bytes and the same hash.
    bool pretty{true};

    /// Names of the structural-algebraic passes that produced this form, for the
    /// provenance block. Supplied by the caller because nothing records what ran
    /// on a graph; see the file note on provenance.
    std::vector<std::string> structural_passes;
};

/**
 * @brief Write @p graph to @p path as ``einsums_graph_ir``.
 *
 * @param[in] graph   The graph to write. Must be fully reconstructible; see
 *                    @ref Graph::serializability_report.
 * @param[in] path    Destination file.
 * @param[in] options Layout and provenance; see @ref SaveOptions.
 * @return Nothing on success, or the reason it refused.
 *
 * Refuses rather than writing something partial. A graph with blockers comes
 * back as a @ref GraphError::Kind::Validation error whose message is the whole
 * serializability report, node by node, so the answer to "why can this not be
 * saved" is in the failure itself.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<void, GraphError> save_graph(Graph const &graph, std::string const &path,
                                                                   SaveOptions const &options = {});

/**
 * @brief Render @p graph as ``einsums_graph_ir`` text.
 * @param[in] graph   The graph to write.
 * @param[in] options Layout and provenance.
 * @return The document, or the reason it refused.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<std::string, GraphError> save_graph_string(Graph const &graph, SaveOptions const &options = {});

/**
 * @brief Read an ``einsums_graph_ir`` file back as a runnable graph.
 *
 * @param[in] path The file to read.
 * @return The graph, or the first reason it could not be built.
 *
 * The graph comes back with every node's executor rebuilt through
 * @ref build_executor, its alias relation derived structurally
 * (@ref Graph::link_alias_structural, reached through
 * @ref Graph::link_alias_storage since a loaded graph has no addresses), and its
 * manifest intact, so @ref Graph::bind works by name immediately. Placeholder
 * storage is allocated for every slot so an unbound graph still executes rather
 * than dereferencing nothing.
 *
 * The resource and tuning pass phases are NOT run; see the file note.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<Graph, GraphError> load_graph(std::string const &path);

/**
 * @brief Read @p path back, resolving its index spaces against @p registry.
 * @param[in] path The file to read.
 * @param[in] registry The registry the file's space NAMES are looked up in. Must outlive the graph.
 * @return The graph, or the failure.
 *
 * A saved graph carries space NAMES rather than ids, because a @ref SpaceId is meaningless
 * outside the registry that issued it. Resolving them needs a registry, and the no-registry
 * overload uses the process-global one, which is right for a program with one set of spaces and
 * wrong for a caller who keeps their own: such a caller got "index space 'occ' is not registered
 * in this process" with an empty list of what IS registered, having registered everything.
 *
 * The loaded graph is also told to USE @p registry, so its annotations, a later save, and any
 * pass that reads a space all agree with the ids it was built from.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<Graph, GraphError> load_graph(std::string const &path, SpaceRegistry &registry);

/**
 * @brief Read ``einsums_graph_ir`` text back as a runnable graph.
 * @param[in] text The document.
 * @return The graph, or the first reason it could not be built.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<Graph, GraphError> load_graph_string(std::string_view text);

/// @brief As @ref load_graph_string, resolving index spaces against @p registry.
/// @param[in] text The document.
/// @param[in] registry The registry to resolve space names in. Must outlive the graph.
/// @return The graph, or the failure.
/// @versionadded{2.0.0}
[[nodiscard]] EINSUMS_EXPORT expected<Graph, GraphError> load_graph_string(std::string_view text, SpaceRegistry &registry);

/**
 * @brief Check an ``einsums_graph_ir`` file without building a graph, reporting
 *        EVERY problem rather than the first.
 *
 * @param[in] path The file to check.
 * @return Nothing when the file is valid, or an error listing every problem
 *         found, one per line.
 *
 * @ref load_graph stops at the first error, which is right for a loader and
 * wrong for a tool: an offline optimizer that emitted three malformed nodes
 * wants to see three messages, not to rediscover them one build at a time. This
 * runs the same checks and collects instead of returning.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<void, GraphError> validate_graph_ir(std::string const &path);

/**
 * @brief Check ``einsums_graph_ir`` text, reporting every problem.
 * @param[in] text The document.
 * @return Nothing when valid, or an error listing every problem found.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<void, GraphError> validate_graph_ir_string(std::string_view text);

/// @brief As @ref validate_graph_ir_string, resolving index spaces against @p registry.
/// @param[in] text The document.
/// @param[in] registry The registry to resolve space names in.
/// @return Nothing, or every problem found.
/// @versionadded{2.0.0}
[[nodiscard]] EINSUMS_EXPORT expected<void, GraphError> validate_graph_ir_string(std::string_view text, SpaceRegistry &registry);

// ── The Python spelling ────────────────────────────────────────────────────
//
// Three thin wrappers, and the only thing they change is the failure channel:
// an ``expected`` is the right shape for a C++ caller and unbindable, so these
// throw instead. They carry the same names on the Python side (APIARY_RENAME),
// so ``einsums.graph.save_graph`` and the C++ ``save_graph`` are one API with
// two error conventions rather than two APIs.

/**
 * @brief Write @p graph to @p path, throwing on refusal.
 * @param[in] graph The graph to write.
 * @param[in] path  Destination file.
 * @throws std::runtime_error With the refusal's message.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("save_graph") EINSUMS_EXPORT void save_graph_file(Graph const       &graph,
                                                                                                     std::string const &path);

/**
 * @brief Read @p path back as a runnable graph, throwing on failure.
 * @param[in] path The file to read.
 * @return The graph. The caller owns it; Python's reference counting takes it
 *         from here.
 * @throws std::runtime_error With the failure's message.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("load_graph") APIARY_RVP(take_ownership) [[nodiscard]] EINSUMS_EXPORT Graph *
load_graph_file(std::string const &path);

/**
 * @brief Read @p path back against @p registry, throwing on failure. The Python spelling.
 * @param[in] path The file to read.
 * @param[in] registry The registry the file's space NAMES resolve in. Must outlive the graph.
 * @return The graph, owned by the caller.
 * @throws std::runtime_error With every problem the file has.
 *
 * The counterpart of the @ref load_graph overload taking a @ref SpaceRegistry, for a caller
 * who keeps their own registry rather than using the process-global one.
 *
 * The overload is named in prose rather than written out as a signature: a cross-reference
 * followed immediately by a parameter list renders as an inline literal with an unterminated
 * start-string, and the docs build treats that warning as an error.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("load_graph_into") APIARY_RVP(take_ownership) APIARY_KEEP_ALIVE(0, 2) [[nodiscard]]
EINSUMS_EXPORT Graph *load_graph_file_into(std::string const &path, SpaceRegistry &registry);

/**
 * @brief Check @p path, throwing with every problem when it is not valid.
 * @param[in] path The file to check.
 * @throws std::runtime_error With one line per problem found.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("validate_graph_ir") EINSUMS_EXPORT void validate_graph_ir_file(std::string const &path);

EINSUMS_NAMESPACE_END(compute_graph)
