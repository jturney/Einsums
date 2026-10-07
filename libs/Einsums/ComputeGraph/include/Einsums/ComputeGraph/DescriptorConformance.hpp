//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file DescriptorConformance.hpp
/// @brief Check that a registered descriptor's codec and hooks tell the truth.
///
/// A codec's hooks are claims the passes act on without checking: a node said to be
/// deterministic is hoisted and merged, one said to overwrite its destination makes the previous
/// write dead, one that declares the parameters it reads is ordered by them. A claim that is false
/// produces a wrong answer far from the codec. @ref check_descriptor runs each claim against the
/// codec's own executor on samples the caller supplies, so the library that registers a
/// descriptor finds a false claim in its own tests.
///
/// It reports rather than asserts, so it fits any test framework:
///
/// @code
/// auto const problems = cg::check_descriptor("nectar.Integral", samples);
/// INFO(fmt::format("{}", fmt::join(problems, "\n")));
/// CHECK(problems.empty());
/// @endcode

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/// @brief The tensors one sample's node reads and writes, in the node's order.
struct ConformanceOperands {
    std::vector<TensorId> inputs;  ///< The operands the node reads, without its destination.
    std::vector<TensorId> outputs; ///< The tensors the node writes.
};

/**
 * @brief One case @ref check_descriptor runs a registered descriptor's node on.
 *
 * The kit builds several one-node graphs from each sample and compares what their outputs hold
 * byte for byte, so @ref operands must produce the same values every time it is called.
 */
struct ConformanceSample {
    /// Names the sample in a reported problem.
    std::string name;

    /// The descriptor the node carries.
    OpData descriptor;

    /// The element type and rank the node's executor is built for, as for @c record_built.
    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Float64};
    std::size_t             rank{0};

    /// Creates the node's tensors in @p graph, which is capturing, fills its inputs and its
    /// outputs' prior contents, and returns their ids. Make dense tensors the graph owns
    /// (``graph.create_runtime_tensor<T>(name, dims, false)``) so nothing outlives the graph.
    std::function<ConformanceOperands(Graph &graph)> operands;

    /// The parameters the node's executor may read, set on each graph before capture.
    std::map<std::string, std::int64_t> params;

    /// The width a moldable node is run at to compare against width 1; 0 means the machine's
    /// thread count.
    unsigned width{0};

    /// The relative difference allowed between the two, for a kernel whose reduction order
    /// changes with the thread count. Zero demands the same bytes.
    double width_tolerance{0.0};
};

/**
 * @brief Run the claims of the codec registered as @p name against @p samples.
 *
 * For each sample, the node is captured as @c record_built would capture it, run once for a
 * reference, and then held to:
 * - its saved form: @c write then @c read gives the descriptor back (the same JSON, every key
 *   read), and the node rebuilt from it computes the reference;
 * - its effects: a node said to be deterministic computes the same bytes when run again;
 * - its destination: a node said to overwrite its destination without reading it computes the
 *   reference over a poisoned destination, and its input list keeps the destination rule;
 * - its accesses: a node run with only the parameters it declares computes the reference, and
 *   declares none the sample does not set;
 * - its output extents: the hook's answer matches the outputs;
 * - its threading: a moldable node run at @ref ConformanceSample::width under the
 *   DataflowExecutor computes the reference, within @ref ConformanceSample::width_tolerance.
 *
 * Across the samples, an @c equal hook must call each descriptor equal to itself, give the same
 * answer both ways round, and call two descriptors equal only when the second, run on the first
 * sample's inputs, computes the first's reference.
 *
 * @return One line per problem found, each naming the sample; empty when every claim held.
 *         A codec not registered under @p name is one problem.
 */
[[nodiscard]] EINSUMS_EXPORT std::vector<std::string> check_descriptor(std::string_view name, std::span<ConformanceSample const> samples);

EINSUMS_NAMESPACE_END(compute_graph)
