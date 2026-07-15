//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ContractionPlanning.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/**
 * Analyze an einsum contraction of arbitrary rank and compute the effective
 * GEMM dimensions (M, K, N) that the contraction maps to when flattened.
 *
 * - M = product of dimensions for target indices that come from A
 * - N = product of dimensions for target indices that come from B
 * - K = product of dimensions for link indices
 *
 * Returns false if the node isn't a pure contraction (c_prefactor != 0,
 * no link indices, etc.).
 */
bool analyze_contraction(EinsumDescriptor const &desc, Graph const &graph, Node const &node, size_t &M, size_t &K, size_t &N) {
    auto const &spec = desc.spec;

    // Must be a pure multiplication (c_prefactor == 0, i.e., C = ab_pf * A * B)
    if (!is_zero(desc.c_prefactor))
        return false;

    // Must have at least one link index (something to contract over)
    if (spec.link_indices.empty())
        return false;

    // Must have exactly one output and two inputs
    if (node.outputs.empty() || node.inputs.size() < 2)
        return false;

    auto const &tensors = graph.tensors_map();
    auto        a_it    = tensors.find(node.inputs[0]);
    auto        b_it    = tensors.find(node.inputs[1]);
    auto        c_it    = tensors.find(node.outputs[0]);

    if (a_it == tensors.end() || b_it == tensors.end() || c_it == tensors.end())
        return false;

    auto const &a_h = a_it->second;
    auto const &b_h = b_it->second;

    // Build index sets
    std::set<std::string> const link_set(spec.link_indices.begin(), spec.link_indices.end());
    std::set<std::string> const a_set(spec.a_indices.begin(), spec.a_indices.end());
    std::set<std::string> const b_set(spec.b_indices.begin(), spec.b_indices.end());

    // K = product of link dimensions (from A's perspective)
    K = 1;
    for (size_t idx = 0; idx < spec.a_indices.size(); idx++) {
        if (link_set.count(spec.a_indices[idx])) {
            if (idx < a_h.dims.size())
                K *= a_h.dims[idx];
        }
    }

    // M = product of A's target dimensions (indices in A that are NOT link)
    M = 1;
    for (size_t idx = 0; idx < spec.a_indices.size(); idx++) {
        if (!link_set.count(spec.a_indices[idx])) {
            if (idx < a_h.dims.size())
                M *= a_h.dims[idx];
        }
    }

    // N = product of B's target dimensions (indices in B that are NOT link)
    N = 1;
    for (size_t idx = 0; idx < spec.b_indices.size(); idx++) {
        if (!link_set.count(spec.b_indices[idx])) {
            if (idx < b_h.dims.size())
                N *= b_h.dims[idx];
        }
    }

    // Sanity: M, K, N must all be > 0
    return M > 0 && K > 0 && N > 0;
}

/// Information about one contraction in a chain.
struct ContractionInfo {
    size_t    node_idx;
    size_t    M, K, N;      ///< Effective GEMM dimensions after index flattening
    size_t    output_elems; ///< Total elements in the output tensor
    TensorId  output_tid;
    TensorId  input_a_tid;
    TensorId  input_b_tid;
    Target    target;
    Residency a_residency;
    Residency b_residency;
};

/// Find chains of contractions where each feeds the next.
std::vector<std::vector<ContractionInfo>> find_contraction_chains(Graph const &graph) {
    auto const                               &nodes = graph.nodes();
    std::vector<bool>                         in_chain(nodes.size(), false);
    std::vector<std::vector<ContractionInfo>> chains;
    auto const                               &tensors = graph.tensors_map();

    auto make_info = [&](size_t idx, size_t M, size_t K, size_t N) -> ContractionInfo {
        ContractionInfo ci;
        ci.node_idx    = idx;
        ci.M           = M;
        ci.K           = K;
        ci.N           = N;
        ci.output_tid  = nodes[idx].outputs[0];
        ci.input_a_tid = nodes[idx].inputs[0];
        ci.input_b_tid = nodes[idx].inputs[1];
        ci.target      = nodes[idx].target;

        ci.output_elems = M * N;

        auto a_it      = tensors.find(ci.input_a_tid);
        auto b_it      = tensors.find(ci.input_b_tid);
        ci.a_residency = (a_it != tensors.end()) ? a_it->second.residency : Residency::Host;
        ci.b_residency = (b_it != tensors.end()) ? b_it->second.residency : Residency::Host;
        return ci;
    };

    for (size_t i = 0; i < nodes.size(); i++) {
        if (in_chain[i] || nodes[i].kind != OpKind::Einsum)
            continue;

        auto *desc = std::get_if<EinsumDescriptor>(&nodes[i].op_data);
        if (!desc)
            continue;

        size_t M, K, N;
        if (!analyze_contraction(*desc, graph, nodes[i], M, K, N))
            continue;

        std::vector<ContractionInfo> chain;
        chain.push_back(make_info(i, M, K, N));
        in_chain[i] = true;

        // Extend: look for next node that reads this output
        TensorId output_tid = nodes[i].outputs[0];
        for (size_t j = i + 1; j < nodes.size(); j++) {
            if (in_chain[j] || nodes[j].kind != OpKind::Einsum)
                break;

            auto *next_desc = std::get_if<EinsumDescriptor>(&nodes[j].op_data);
            if (!next_desc)
                break;

            bool const reads_output = std::find(nodes[j].inputs.begin(), nodes[j].inputs.end(), output_tid) != nodes[j].inputs.end();
            if (!reads_output)
                break;

            size_t M2, K2, N2;
            if (!analyze_contraction(*next_desc, graph, nodes[j], M2, K2, N2))
                break;

            chain.push_back(make_info(j, M2, K2, N2));
            in_chain[j] = true;
            output_tid  = nodes[j].outputs[0];
        }

        if (chain.size() >= 2)
            chains.push_back(std::move(chain));
    }

    return chains;
}

/// Extract the n+1 leaf tensor IDs from a chain of n contractions.
/// For chain[0]: both inputs are leaves.
/// For chain[i>0]: one input is the chain link (previous output), the other is a leaf.
std::vector<TensorId> extract_leaves(std::vector<ContractionInfo> const &chain) {
    std::vector<TensorId> leaves;
    leaves.reserve(chain.size() + 1);

    // First contraction: both inputs are leaves
    leaves.push_back(chain[0].input_a_tid);
    leaves.push_back(chain[0].input_b_tid);

    // Subsequent: the "other" input (not the chain link) is a leaf
    for (size_t i = 1; i < chain.size(); i++) {
        TensorId const prev_output = chain[i - 1].output_tid;
        if (chain[i].input_a_tid == prev_output)
            leaves.push_back(chain[i].input_b_tid);
        else
            leaves.push_back(chain[i].input_a_tid);
    }
    return leaves;
}

/// Declare a DEFERRED graph-owned intermediate for a restructured chain.
/// Deferred (not create_tensor_dynamic, which allocates eagerly and used to
/// leak these for the graph's lifetime): the Materialization pass allocates
/// it at the right position, and FreeInsertion / InplaceOptimization /
/// MemoryPlanning's arena manage its buffer like any other intermediate.
/// Returns 0 if the dtype is unsupported.
TensorId declare_chain_intermediate(Graph &graph, std::string name, packed_gemm::ScalarType dtype, size_t rows, size_t cols) {
    // Typed Tensor<T,2>, not a runtime tensor: make_einsum_executor's rank-2
    // fast path casts tensor_ptr to Tensor<T,2>*.
    void const *ptr = nullptr;
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        ptr = &graph.declare_tensor<float, 2>(std::move(name), rows, cols);
        break;
    case packed_gemm::ScalarType::Float64:
        ptr = &graph.declare_tensor<double, 2>(std::move(name), rows, cols);
        break;
    case packed_gemm::ScalarType::Complex64:
        ptr = &graph.declare_tensor<std::complex<float>, 2>(std::move(name), rows, cols);
        break;
    case packed_gemm::ScalarType::Complex128:
        ptr = &graph.declare_tensor<std::complex<double>, 2>(std::move(name), rows, cols);
        break;
    default:
        return 0;
    }
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.tensor_ptr == ptr) {
            // declare_tensor defaults to user-visible; these are pass-created
            // scratch the memory passes should manage.
            graph.tensor(tid).is_intermediate = true;
            return tid;
        }
    }
    return 0;
}

/// Recursively build the optimal tree from the DP split table.
/// Returns the TensorId of the sub-result for leaves[i..j+1].
/// For a single leaf (i == j), returns leaves[i].
/// For a pair (j == i+1), creates a single GEMM of leaves[i] and leaves[i+1].
/// For larger ranges, splits at s[i][j] and recurses.
TensorId reconstruct_tree(size_t i, size_t j, std::vector<std::vector<size_t>> const &split, std::vector<TensorId> const &leaves, // NOLINT
                          std::vector<size_t> const &p, Graph &graph, packed_gemm::ScalarType dtype, size_t element_size,
                          TensorId final_output_tid, std::vector<Node> &new_nodes, size_t &intermediates_created) {
    // Base case: single leaf
    if (i == j)
        return leaves[i];

    size_t const k = split[i][j];

    // Recursively build left and right sub-results
    TensorId const left_id  = reconstruct_tree(i, k, split, leaves, p, graph, dtype, element_size, 0, new_nodes, intermediates_created);
    TensorId const right_id = reconstruct_tree(k + 1, j, split, leaves, p, graph, dtype, element_size, 0, new_nodes, intermediates_created);

    // Determine output tensor: full chain → use original final output; subchain → create intermediate
    TensorId out_id;
    size_t   out_M = p[i];
    size_t   out_N = p[j + 1];
    size_t   K_dim = p[k + 1];

    if (i == 0 && j == leaves.size() - 1 && final_output_tid != 0) {
        // This is the top-level result → use the original output tensor
        out_id = final_output_tid;
    } else {
        // Declare a deferred intermediate; the memory passes downstream
        // (FreeInsertion, MemoryPlanning's arena) own its lifecycle - this
        // pass now runs BEFORE them.
        std::string name = fmt::format("_cp_{}x{}_{}", out_M, out_N, intermediates_created);
        out_id           = declare_chain_intermediate(graph, std::move(name), dtype, out_M, out_N);
        if (out_id == 0) {
            // Unsupported dtype: leave the chain unrestructured.
            return left_id;
        }
        intermediates_created++;

        // Emit the Materialize for our own intermediate rather than relying
        // on a later Materialization pass: the restructured graph must be
        // executable even when this pass is applied standalone. The
        // allocation still happens at execute time (deferred until then),
        // and materialize_fn is idempotent if anything downstream adds
        // bookkeeping of its own.
        {
            auto const &handle = graph.tensor(out_id);
            Node        mat_node;
            mat_node.kind            = OpKind::Materialize;
            mat_node.label           = fmt::format("materialize({})", handle.name);
            mat_node.outputs         = {out_id};
            mat_node.execute         = [mat_fn = handle.materialize_fn]() { mat_fn(); };
            mat_node.estimated_bytes = out_M * out_N * element_size;
            new_nodes.push_back(std::move(mat_node));
        }
    }

    // Build GEMM/einsum node.
    // Construct a spec for the folded GEMM: C[m,n] = A[m,k] * B[k,n]
    ParsedEinsumSpec spec;
    spec.raw       = fmt::format("m,n <- m,k ; k,n");
    spec.c_indices = {"m", "n"};
    spec.a_indices = {"m", "k"};
    spec.b_indices = {"k", "n"};

    Node node;
    node.kind            = OpKind::Gemm;
    node.label           = fmt::format("cp_gemm({}x{}x{})", out_M, K_dim, out_N);
    node.execute         = graph.make_einsum_executor(left_id, right_id, out_id, spec, 1.0, 0.0);
    node.inputs          = {left_id, right_id};
    node.outputs         = {out_id};
    node.estimated_flops = 2 * out_M * K_dim * out_N;
    node.estimated_bytes = (out_M * K_dim + K_dim * out_N + out_M * out_N) * element_size;

    new_nodes.push_back(std::move(node));
    return out_id;
}

Target determine_target(HardwareProfile const &profile, size_t M, size_t N, size_t K, size_t elem_size) {
    if (!profile.has_gpu())
        return Target::CPU;
    double const cpu_time = profile.estimate_total_gemm_time_us(M, N, K, elem_size, Target::CPU);
    double const gpu_time = profile.estimate_total_gemm_time_us(M, N, K, elem_size, Target::GPU) + profile.gpu.gpu_launch_latency_us;
    return (gpu_time < cpu_time * 0.8) ? Target::GPU : Target::CPU;
}

double transfer_cost_us(HardwareProfile const &profile, size_t bytes, Residency current, Target needed) {
    if (needed == Target::GPU) {
        if (current == Residency::Device || current == Residency::Both)
            return 0.0;
    } else {
        if (current == Residency::Host || current == Residency::Both)
            return 0.0;
    }
    return profile.estimate_transfer_time_us(bytes);
}

} // namespace

ContractionPlanning::ContractionPlanning() : _profile(HardwareProfile::detect_default()) {
}

ContractionPlanning::ContractionPlanning(HardwareProfile profile) : _profile(std::move(profile)) {
}

bool ContractionPlanning::run(Graph &graph) {
    graph.topological_sort();

    _reports.clear();
    _chains_restructured   = 0;
    _intermediates_created = 0;

    auto chains = find_contraction_chains(graph);
    if (chains.empty())
        return false;

    // Determine element size and dtype from first chain
    size_t                  element_size = 8;
    packed_gemm::ScalarType dtype        = packed_gemm::ScalarType::Float64;
    bool                    modified     = false;
    {
        auto const &tensors = graph.tensors_map();
        auto        it      = tensors.find(chains[0][0].output_tid);
        if (it != tensors.end()) {
            element_size = it->second.element_size;
            dtype        = it->second.dtype;
        }
    }

    // Device memory budget
    size_t device_budget = 0;
    if (_profile.has_gpu()) {
        device_budget = gpu::available_device_memory();
        for (auto const &node : graph.nodes()) {
            if (node.target == Target::GPU)
                device_budget -= std::min(device_budget, node.estimated_bytes);
        }
    }

    for (auto const &chain : chains) {
        size_t n = chain.size(); // Number of GEMMs

        // Extract leaf matrices: n+1 leaves from n GEMMs.
        auto         leaves     = extract_leaves(chain);
        size_t const num_leaves = leaves.size(); // n + 1

        // Build dimension array p[0..num_leaves] (num_leaves+1 entries).
        // Leaf matrix i has effective dimensions p[i] × p[i+1].
        // p[0] = M of first GEMM (= rows of leaf 0)
        // p[1] = K of first GEMM (= cols of leaf 0 = rows of leaf 1)
        // p[i+1] for i>=1: N of GEMM i-1 (= cols of leaf i = rows of leaf i+1)
        //
        // Derivation:
        //   leaf 0 = first input of GEMM 0 → dims: chain[0].M × chain[0].K
        //   leaf 1 = second input of GEMM 0 → dims: chain[0].K × chain[0].N
        //   leaf 2 = non-link input of GEMM 1 → dims: chain[1].K × chain[1].N
        //   ...but chain[1].K == chain[0].N (chain link), so p flows correctly.
        std::vector<size_t> p(num_leaves + 1);
        p[0] = chain[0].M;
        p[1] = chain[0].K;
        for (size_t idx = 0; idx < n; idx++)
            p[idx + 2] = chain[idx].N;

        // ── Standard matrix chain DP on num_leaves matrices ────────────────
        // m[i][j] = min cost to multiply leaf matrices i through j
        // s[i][j] = split position k: (leaves i..k) * (leaves k+1..j)
        std::vector<std::vector<double>> m(num_leaves, std::vector<double>(num_leaves, 0.0));
        std::vector<std::vector<size_t>> s(num_leaves, std::vector<size_t>(num_leaves, 0));

        for (size_t len = 2; len <= num_leaves; len++) {
            for (size_t i_idx = 0; i_idx <= num_leaves - len; i_idx++) {
                size_t const j_idx = i_idx + len - 1;
                m[i_idx][j_idx]    = std::numeric_limits<double>::max();

                for (size_t k_idx = i_idx; k_idx < j_idx; k_idx++) {
                    // Multiply result of (i..k) [shape p[i] × p[k+1]]
                    //       with result of (k+1..j) [shape p[k+1] × p[j+1]]
                    // GEMM dimensions: M=p[i], K=p[k+1], N=p[j+1]
                    Target const split_target = determine_target(_profile, p[i_idx], p[j_idx + 1], p[k_idx + 1], element_size);

                    double const gemm_time =
                        _profile.estimate_total_gemm_time_us(p[i_idx], p[j_idx + 1], p[k_idx + 1], element_size, split_target);

                    double const cost = m[i_idx][k_idx] + m[k_idx + 1][j_idx] + gemm_time;

                    if (cost < m[i_idx][j_idx]) {
                        m[i_idx][j_idx] = cost;
                        s[i_idx][j_idx] = k_idx;
                    }
                }
            }
        }

        // ── Compute original (left-to-right) cost ──────────────────────────
        double original_time = 0.0;
        for (size_t idx = 0; idx < n; idx++) {
            original_time +=
                _profile.estimate_total_gemm_time_us(chain[idx].M, chain[idx].N, chain[idx].K, element_size, chain[idx].target);
        }

        double optimal_time = m[0][num_leaves - 1];

        // Check if any tensor in the chain is distributed (non-replicated).
        // If so, add allreduce communication cost to the total.
        double      comm_cost = 0.0;
        bool        has_dist  = false;
        int const   num_ranks = comm::world_size();
        auto const &tensors   = graph.tensors_map();

        for (auto const &ci : chain) {
            for (auto tid : {ci.input_a_tid, ci.input_b_tid, ci.output_tid}) {
                auto it = tensors.find(tid);
                if (it != tensors.end() && it->second.is_distributed && !it->second.is_replicated) {
                    has_dist = true;
                    // Each non-replicated contraction needs an allreduce of the result
                    size_t const result_bytes = ci.M * ci.N * element_size;
                    comm_cost += _profile.estimate_allreduce_time_us(result_bytes, num_ranks);
                    break;
                }
            }
        }

        // Add communication cost to both original and optimal estimates
        original_time += comm_cost;
        optimal_time += comm_cost;

        double speedup = (optimal_time > 0) ? original_time / optimal_time : 1.0;

        ChainReport report;
        report.chain_length          = n;
        report.dimensions            = p;
        report.original_time_us      = original_time;
        report.optimal_time_us       = optimal_time;
        report.speedup               = speedup;
        report.intermediates_created = 0;
        report.comm_cost_us          = comm_cost;
        report.has_distributed       = has_dist;

        if (speedup < 1.05) {
            EINSUMS_LOG_DEBUG("ContractionPlanning: chain of {} contractions [eff. dims {}], speedup {:.2f}x — below threshold", n,
                              fmt::join(p, "x"), speedup);
            _reports.push_back(std::move(report));
            continue;
        }

        // Check if all tensors in the chain are rank-2 (safe for direct GEMM restructuring).
        // Higher-rank chains require folding which needs more careful validation.
        // For now, only restructure rank-2 chains. Higher-rank gets analysis only.
        bool all_rank2 = true;
        for (auto const &ci : chain) {
            auto a_it = graph.tensors_map().find(ci.input_a_tid);
            auto b_it = graph.tensors_map().find(ci.input_b_tid);
            if ((a_it != graph.tensors_map().end() && a_it->second.rank != 2) ||
                (b_it != graph.tensors_map().end() && b_it->second.rank != 2)) {
                all_rank2 = false;
                break;
            }
        }

        if (!all_rank2) {
            EINSUMS_LOG_INFO("ContractionPlanning: chain of {} contractions [eff. dims {}], {:.1f}us → {:.1f}us ({:.2f}x speedup) — "
                             "analysis only (higher-rank, needs folding)",
                             n, fmt::join(p, "x"), original_time, optimal_time, speedup);
            _reports.push_back(std::move(report));
            continue;
        }

        // Restructured intermediates are declared deferred with the chain's
        // dtype; gate on the dtypes we can declare rather than discovering
        // failure mid-rebuild.
        if (dtype == packed_gemm::ScalarType::Unknown) {
            EINSUMS_LOG_INFO("ContractionPlanning: chain of {} contractions — analysis only (unknown dtype)", n);
            _reports.push_back(std::move(report));
            continue;
        }

        EINSUMS_LOG_INFO(
            "ContractionPlanning: chain of {} contractions [eff. dims {}], {:.1f}us → {:.1f}us ({:.2f}x speedup) — restructuring", n,
            fmt::join(p, "x"), original_time, optimal_time, speedup);

        // ── Graph restructuring ────────────────────────────────────────────
        // leaves and p are already computed above. Build optimal tree from DP.
        {
            size_t            inter_count = 0;
            std::vector<Node> new_nodes;
            TensorId const    final_tid = chain.back().output_tid;

            // reconstruct_tree indices are 0..num_leaves-1 (leaf indices)
            reconstruct_tree(0, num_leaves - 1, s, leaves, p, graph, dtype, element_size, final_tid, new_nodes, inter_count);

            if (!new_nodes.empty()) {
                // Mark original chain nodes for removal
                auto                      &nodes = graph.nodes();
                std::unordered_set<size_t> remove_indices;
                for (auto const &ci : chain)
                    remove_indices.insert(ci.node_idx);

                // Find insertion point (where first chain node was)
                size_t const insert_pos = chain[0].node_idx;

                // Build new node list
                std::vector<Node> result;
                result.reserve(nodes.size() - remove_indices.size() + new_nodes.size());

                for (size_t idx = 0; idx < nodes.size(); idx++) {
                    if (idx == insert_pos) {
                        for (auto &nn : new_nodes)
                            result.push_back(std::move(nn));
                    }
                    if (remove_indices.count(idx) == 0) {
                        result.push_back(std::move(nodes[idx]));
                    }
                }

                nodes = std::move(result);
                graph.mark_sorted();

                report.intermediates_created = inter_count;
                _intermediates_created += inter_count;
                _chains_restructured++;
                modified = true;

                EINSUMS_LOG_INFO("ContractionPlanning: restructured chain — {} new GEMM nodes, {} intermediates created", new_nodes.size(),
                                 inter_count);
                this->report(2, fmt::format("restructure GEMM chain into {} node(s), {} intermediate(s)", new_nodes.size(), inter_count));
            }
        }

        _reports.push_back(std::move(report));
    }

    if (!_reports.empty()) {
        EINSUMS_LOG_INFO("ContractionPlanning: analyzed {} chains using CPU='{}' GPU='{}', restructured {}", _reports.size(),
                         _profile.cpu.name, _profile.gpu.name, _chains_restructured);
        this->report(1, fmt::format("analyzed {} GEMM chain(s), restructured {}", _reports.size(), _chains_restructured));
    }

    return modified;
}

} // namespace einsums::compute_graph::passes
