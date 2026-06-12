//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ChainParenthesization.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Check if an EinsumDescriptor represents a simple matrix multiplication
/// pattern: C[i,j] = A[i,k] * B[k,j] (or transposed variants).
/// Returns true and sets M, K, N dimensions if so.
bool is_gemm_pattern(EinsumDescriptor const &desc, Graph const &graph, Node const &node, size_t &M, size_t &K, size_t &N) {
    auto const &spec = desc.spec;

    // Need exactly 2 indices on each operand and C
    if (spec.c_indices.size() != 2 || spec.a_indices.size() != 2 || spec.b_indices.size() != 2) {
        return false;
    }

    // Need exactly 1 link index
    if (spec.link_indices.size() != 1) {
        return false;
    }

    // Need exactly 2 target indices
    if (spec.target_indices.size() != 2) {
        return false;
    }

    // c_prefactor must be 0 (pure multiplication, not accumulation)
    if (!is_zero(desc.c_prefactor)) {
        return false;
    }

    // Get tensor dimensions from the graph
    if (node.outputs.empty() || node.inputs.size() < 2) {
        return false;
    }

    auto const &tensors = graph.tensors_map();
    auto        c_it    = tensors.find(node.outputs[0]);
    auto        a_it    = tensors.find(node.inputs[0]);
    auto        b_it    = tensors.find(node.inputs[1]);

    if (c_it == tensors.end() || a_it == tensors.end() || b_it == tensors.end()) {
        return false;
    }

    auto const &c_handle = c_it->second;
    auto const &a_handle = a_it->second;
    auto const &b_handle = b_it->second;

    if (c_handle.dims.size() != 2 || a_handle.dims.size() != 2 || b_handle.dims.size() != 2) {
        return false;
    }

    // Determine M, K, N from the contraction pattern
    // A[i,k]: M = A.dim(0), K = A.dim(1) (if not transposed)
    // B[k,j]: K = B.dim(0), N = B.dim(1) (if not transposed)
    // C[i,j]: M = C.dim(0), N = C.dim(1)
    M = c_handle.dims[0];
    N = c_handle.dims[1];

    // K is the link dimension
    std::string const &link = spec.link_indices[0];
    // Find K in A's indices
    if (spec.a_indices[0] == link) {
        K = a_handle.dims[0];
    } else if (spec.a_indices[1] == link) {
        K = a_handle.dims[1];
    } else {
        return false;
    }

    return true;
}

// Analyze the GEMM chains in a single graph, adding their original and
// optimal FLOP counts into the accumulators. Pulled out of run() so the
// pass can aggregate over loop bodies / conditional branches.
void analyze_chain_flops(Graph &graph, size_t &orig_acc, size_t &opt_acc) {
    graph.topological_sort();

    auto const &nodes = graph.nodes();

    // Find chains of GEMM einsums: sequences where node[i]'s output
    // is node[i+1]'s input (one of them).
    struct GemmInfo {
        size_t node_idx;
        size_t M, K, N;
    };

    std::vector<std::vector<GemmInfo>> chains;
    std::vector<bool>                  in_chain(nodes.size(), false);

    for (size_t i = 0; i < nodes.size(); i++) {
        if (in_chain[i])
            continue;
        if (nodes[i].kind != OpKind::Einsum)
            continue;

        auto *desc = std::get_if<EinsumDescriptor>(&nodes[i].op_data);
        if (!desc)
            continue;

        size_t M, K, N;
        if (!is_gemm_pattern(*desc, graph, nodes[i], M, K, N))
            continue;

        // Start building a chain
        std::vector<GemmInfo> chain;
        chain.push_back({.node_idx = i, .M = M, .K = K, .N = N});
        in_chain[i] = true;

        // Extend: look for next node that reads this node's output
        TensorId output_tid = nodes[i].outputs[0];
        for (size_t j = i + 1; j < nodes.size(); j++) {
            if (in_chain[j])
                break;
            if (nodes[j].kind != OpKind::Einsum)
                break;

            auto *next_desc = std::get_if<EinsumDescriptor>(&nodes[j].op_data);
            if (!next_desc)
                break;

            // Check if this node reads the previous output
            bool const reads_output = std::find(nodes[j].inputs.begin(), nodes[j].inputs.end(), output_tid) != nodes[j].inputs.end();
            if (!reads_output)
                break;

            size_t M2, K2, N2;
            if (!is_gemm_pattern(*next_desc, graph, nodes[j], M2, K2, N2))
                break;

            chain.push_back({.node_idx = j, .M = M2, .K = K2, .N = N2});
            in_chain[j] = true;
            output_tid  = nodes[j].outputs[0];
        }

        if (chain.size() >= 2) {
            chains.push_back(std::move(chain));
        }
    }

    if (chains.empty()) {
        return;
    }

    // For each chain, compute original and optimal FLOP counts
    for (auto const &chain : chains) {
        size_t n = chain.size();

        // Dimensions array: p[0] = M of first, p[1] = N of first = M of second, ...
        // For chain of n multiplications, we have n+1 dimensions.
        std::vector<size_t> p(n + 1);
        p[0] = chain[0].M;
        for (size_t idx = 0; idx < n; idx++) {
            p[idx + 1] = chain[idx].N;
        }

        // Original FLOPs: left-to-right evaluation
        // Each GEMM of (p[i] x p[i+1]) * (p[i+1] x p[i+2]) costs 2*p[i]*p[i+1]*p[i+2]
        size_t orig = 0;
        for (size_t idx = 0; idx < n; idx++) {
            orig += 2 * chain[idx].M * chain[idx].K * chain[idx].N;
        }
        orig_acc += orig;

        // DP for optimal parenthesization
        // m[i][j] = minimum cost to multiply matrices i through j
        std::vector<std::vector<size_t>> m(n, std::vector<size_t>(n, 0));
        std::vector<std::vector<size_t>> s(n, std::vector<size_t>(n, 0));

        for (size_t len = 2; len <= n; len++) {
            for (size_t idx = 0; idx <= n - len; idx++) {
                size_t const j_idx = idx + len - 1;
                m[idx][j_idx]      = std::numeric_limits<size_t>::max();
                for (size_t k_idx = idx; k_idx < j_idx; k_idx++) {
                    size_t const cost = m[idx][k_idx] + m[k_idx + 1][j_idx] + 2 * p[idx] * p[k_idx + 1] * p[j_idx + 1];
                    if (cost < m[idx][j_idx]) {
                        m[idx][j_idx] = cost;
                        s[idx][j_idx] = k_idx;
                    }
                }
            }
        }

        opt_acc += m[0][n - 1];

        if (m[0][n - 1] < orig) {
            EINSUMS_LOG_INFO("ChainParenthesization: chain of {} GEMMs, dimensions: [{}]", n, fmt::join(p, " x "));
            EINSUMS_LOG_INFO("  Original FLOPs: {}, Optimal FLOPs: {}, Savings: {:.1f}%", orig, m[0][n - 1],
                             100.0 * (1.0 - static_cast<double>(m[0][n - 1]) / static_cast<double>(orig)));
        }
    }
}

// Accumulate chain-flop analysis over a graph and all its descendants.
void accumulate(Graph &graph, size_t &orig_acc, size_t &opt_acc) {
    analyze_chain_flops(graph, orig_acc, opt_acc);
    graph.for_each_subgraph([&](Graph &sub) { accumulate(sub, orig_acc, opt_acc); });
}

} // namespace

bool ChainParenthesization::run(Graph &graph) {
    // Aggregate over the whole graph tree: a GEMM chain that lives entirely
    // inside a loop body is counted too. recurse_into_subgraphs() stays
    // false: we aggregate here rather than being re-run per sub-graph.
    _original_flops = 0;
    _optimal_flops  = 0;
    accumulate(graph, _original_flops, _optimal_flops);

    if (_optimal_flops > 0 && _optimal_flops < _original_flops) {
        report(1, fmt::format("GEMM-chain FLOPs {} -> {} achievable ({:.1f}% saving) by reparenthesizing", _original_flops, _optimal_flops,
                              100.0 * (1.0 - static_cast<double>(_optimal_flops) / static_cast<double>(_original_flops))));
    }

    // This pass is analysis/reporting only, restructuring the graph would
    // require creating new intermediate tensors with type information we
    // don't have in type-erased form. The user can use the recommendations
    // to restructure their code.
    return false;
}

} // namespace einsums::compute_graph::passes
