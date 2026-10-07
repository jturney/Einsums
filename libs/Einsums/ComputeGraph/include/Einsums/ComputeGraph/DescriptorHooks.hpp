//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file DescriptorHooks.hpp
/// @brief What a descriptor registered outside this library can tell the passes about its node.
///
/// A pass asks one question of a node at a time, through one function per question:
/// @ref named_reads and @ref named_writes for the parameters and named resources it touches,
/// @ref effects_of for what it does beyond its tensors, @ref reads_destination and its neighbours
/// for whether it reads what it writes, and @ref kernel_moldability for whether it can use more
/// than one thread. Each of those answers a library operation from its kind and descriptor, and
/// answers an @ref OpKind::Custom node whose descriptor has a registered codec (see
/// DescriptorRegistry.hpp) from that codec's hook for the question, when it has one. A hook left
/// empty keeps the answer the function gives any other Custom node.
///
/// Hooks are pure functions of their arguments: no side effects and no global state, safe to call
/// from any thread any number of times, and never a mutation of the node or its graph. They are
/// code, not data, so a saved graph never contains them; a loader reaches them by having the
/// codec registered, as it reaches the codec's @c build.

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;
class ParamTable;

/**
 * @brief The parameters and named resources a node touches outside its tensor lists.
 *
 * A parameter is an entry of the graph's @ref ParamTable, by name; a node whose executor reads one
 * through @c Graph::params_ptr must list it here, or nothing orders it after the @c WriteParam
 * that sets it. A named resource is any other state two nodes can share without sharing a tensor,
 * as an opaque key: a dataset (``"disk:<file>#<dataset>"``), a cache, a communicator. Keys from
 * different libraries should carry the library's prefix (``"nectar:plan-cache"``).
 */
struct NamedAccesses {
    std::vector<std::string> param_reads;  ///< @ref ParamTable entries read at run time.
    std::vector<std::string> param_writes; ///< @ref ParamTable entries written at run time.
    std::vector<std::string> named_reads;  ///< Other named resources read.
    std::vector<std::string> named_writes; ///< Other named resources written.
};

/**
 * @brief Whether a node reads the tensor it writes, and how its input list says so.
 *
 * The destination rule (DestinationRead.hpp) lists a node's operands first and its destination
 * once, trailing, exactly when the node reads it. A node with one output that gives
 * @ref operand_count is held to that rule: its input list is brought into line at capture and
 * after every pass, and @c Graph::verify reports a node that breaks it.
 */
struct DestinationUse {
    /// Reads the prior contents of its destination (accumulates into it, or updates it in place).
    bool reads{false};
    /// Writes every element of its destination, so any earlier write to it is dead.
    bool overwrites_all{false};
    /// Operand inputs listed before the destination, for a node with one output. Empty leaves the
    /// node outside the rule, listing its inputs as it chooses.
    std::optional<std::size_t> operand_count;
};

/// @brief How a node's executor uses threads, which decides whether ThreadPlanning may widen it.
enum class Threading : std::uint8_t {
    /// Forks its parallelism from the OpenMP thread count the executor sets, as a plain
    /// ``omp parallel`` region or ``omp_get_max_threads()`` does, and starts no threads of its own.
    Moldable,
    /// Runs on the calling thread.
    Serial,
    /// Runs its own thread pool, which a planned width would oversubscribe.
    OwnPool,
};

/// @brief The extents of a node's outputs, one list per output in the node's output order.
using ExtentList = std::vector<std::vector<std::size_t>>;

/**
 * @brief What an @c output_extents hook is given.
 *
 * @ref input_extents holds the current extents of every input, in the node's input order (an
 * input whose extents are not known yet has an empty list). @ref params is the graph's
 * @ref ParamTable, for a node whose output size is a parameter's value; it may be null.
 */
struct ExtentQuery {
    ExtentList        input_extents;
    ParamTable const *params{nullptr};
};

/**
 * @brief What a node costs to run, in the terms the planners price every node in.
 *
 * The planner turns this into a serial time: @ref serial_us when given, otherwise @ref flops
 * priced as a GEMM of that many flops and @ref bytes as memory traffic, whichever takes longer.
 * Widening divides the time by @ref speedup when given, otherwise by the curve of @ref family.
 * A measured run of the node always takes precedence over the estimate.
 */
struct CostEstimate {
    /// Floating-point operations, independent of the device.
    double flops{0.0};
    /// Bytes moved, independent of the device. Zero leaves it to the planner, which counts the
    /// node's tensors.
    std::size_t bytes{0};
    /// The speedup curve the node scales along. Empty picks a GEMM curve when @ref flops is
    /// given and the element-wise curve otherwise.
    std::optional<KernelFamily> family;
    /// Serial time on the queried device, when the descriptor knows its rate there.
    std::optional<double> serial_us;
    /// Speedup at a width, overriding @ref family's curve for an engine none of them fits. Called
    /// with widths of 2 and above; returns how many times faster than serial.
    std::function<double(unsigned width)> speedup;
};

/// @brief What a @c cost hook is asked about: the node, its graph, and the device it is priced on.
struct CostQuery {
    Graph const         &graph;
    Node const          &node;
    DeviceProfile const &profile;
    Target               target{Target::CPU};
};

/**
 * @brief The optional answers a registered descriptor gives about its node.
 *
 * Every member may be empty. Members are only ever appended, since a @ref DescriptorCodec is
 * passed by value across the library boundary.
 */
struct DescriptorHooks {
    /// The parameters and named resources the node touches; see @ref NamedAccesses. Without it the
    /// node touches none.
    std::function<NamedAccesses(OpData const &descriptor, Node const &node)> accesses;

    /// What the node does beyond its tensors. Without it a registered descriptor counts as pure,
    /// since its codec's @c build is a function of the descriptor and operands.
    std::function<NodeEffects(OpData const &descriptor)> effects;

    /// Whether the node reads its destination; see @ref DestinationUse. Without it the node does
    /// not, and stays outside the destination rule.
    std::function<DestinationUse(OpData const &descriptor, Node const &node)> destination;

    /// How the executor uses threads. Without it the node is @ref Threading::Moldable.
    std::function<Threading(OpData const &descriptor)> threading;

    /// The extents the node's outputs take for the inputs in @p query, one list per output, or
    /// empty when the descriptor cannot say. A rebind re-derives a graph-owned output from it, and
    /// validation reports an output whose declared extents differ. Without it a rebind leaves the
    /// outputs as they were captured.
    std::function<std::optional<ExtentList>(OpData const &descriptor, Node const &node, ExtentQuery const &query)> output_extents;

    /// What the node costs; see @ref CostEstimate. Without it, or when it returns empty, the node
    /// is priced as the memory traffic of its tensors, as any Custom node is.
    std::function<std::optional<CostEstimate>(OpData const &descriptor, CostQuery const &query)> cost;
};

/**
 * @brief The hooks registered for @p node's descriptor, or null.
 *
 * Null unless @p node is an @ref OpKind::Custom node whose descriptor has a registered codec. A
 * library descriptor's name is never qualified, so only a registered one is looked up. The
 * returned table lives for the rest of the process.
 */
[[nodiscard]] EINSUMS_EXPORT DescriptorHooks const *descriptor_hooks(Node const &node) noexcept;

/// @brief What @p node does beyond its tensors: its registered @c effects hook's answer, or
///        @ref Node::effects.
[[nodiscard]] EINSUMS_EXPORT NodeEffects effects_of(Node const &node);

/// @brief The extents @p node's registered @c output_extents hook gives for its inputs' current
///        extents in @p graph, or empty when it has no such hook or the hook cannot say.
[[nodiscard]] EINSUMS_EXPORT std::optional<ExtentList> hooked_output_extents(Graph const &graph, Node const &node);

/// @brief Why @p node's outputs in @p graph disagree with its registered @c output_extents hook:
///        a different number of outputs, or an output whose known extents differ. Empty when they
///        agree, or when there is no hook or it cannot say.
[[nodiscard]] EINSUMS_EXPORT std::optional<std::string> hooked_extent_mismatch(Graph const &graph, Node const &node);

EINSUMS_NAMESPACE_END(compute_graph)
