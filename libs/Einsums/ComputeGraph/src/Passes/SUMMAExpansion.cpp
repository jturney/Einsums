//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/ProcessGrid.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <cstring>
#include <variant>
#include <vector>

using namespace einsums::index;

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// One process's SUMMA panel loop for element type @p T: prescale C by its
/// prefactor, then for each panel broadcast the A block along the process row
/// and the B block along the process column and accumulate the local GEMM via
/// einsum (which routes through PackedGemm). Templating over T collapses what
/// were two byte-identical double/float copies into a single body.
template <typename T>
void run_summa_panels(comm::ProcessGrid const &grid, int panels, void *a_ptr, void *b_ptr, void *c_ptr, PrefactorScalar c_pf) {
    auto *A_local = static_cast<Tensor<T, 2> *>(a_ptr);
    auto *B_local = static_cast<Tensor<T, 2> *>(b_ptr);
    auto *C_local = static_cast<Tensor<T, 2> *>(c_ptr);

    size_t const local_m   = C_local->dim(0);
    size_t const local_n   = C_local->dim(1);
    size_t const local_k_a = A_local->dim(1); // K/Pc
    size_t const local_k_b = B_local->dim(0); // K/Pr (== K/Pc for a square grid)

    // Apply the C prefactor (typically 0 on the first call).
    auto const c_pf_v = as<T>(c_pf);
    if (c_pf_v == T{0}) {
        C_local->zero();
    } else if (c_pf_v != T{1}) {
        linear_algebra::scale(c_pf_v, C_local);
    }

    int const my_col = grid.my_col();
    int const my_row = grid.my_row();

    // Temporary panel buffers (same size as the local blocks).
    Tensor<T, 2> A_panel("A_panel", local_m, local_k_a);
    Tensor<T, 2> B_panel("B_panel", local_k_b, local_n);

    LabeledSection("SUMMA({}x{}x{}, {} panels)", local_m, local_k_a, local_n, panels);

    for (int p = 0; p < panels; p++) {
        // Step 1: broadcast the A panel along the process row.
        {
            LabeledSection("broadcast_A");
            if (my_col == p) {
                std::memcpy(A_panel.data(), A_local->data(), local_m * local_k_a * sizeof(T));
            }
            auto const placeholder = comm::broadcast<T>(std::span<T>(A_panel.data(), A_panel.size()), p, grid.row_comm());
            (void)placeholder;
        }

        // Step 2: broadcast the B panel along the process column.
        {
            LabeledSection("broadcast_B");
            if (my_row == p) {
                std::memcpy(B_panel.data(), B_local->data(), local_k_b * local_n * sizeof(T));
            }
            auto const placeholder = comm::broadcast<T>(std::span<T>(B_panel.data(), B_panel.size()), p, grid.col_comm());
            (void)placeholder;
        }

        // Step 3: local GEMM accumulate via einsum dispatch (enables PackedGemm).
        {
            LabeledSection("local_gemm");
            tensor_algebra::einsum(T{1}, Indices{i, j}, C_local, T{1}, Indices{i, k}, A_panel, Indices{k, j}, B_panel);
        }
    }
}

} // namespace

void SUMMAExpansion::reset_stats() {
    _num_expanded = 0;
}

bool SUMMAExpansion::run(Graph &graph) {
    PassCounter const expanded{_num_expanded};
    if (comm::world_size() <= 1)
        return false;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();
    auto       &grid    = comm::ProcessGrid::default_grid();

    if (grid.rows() <= 1 || grid.cols() <= 1)
        return false; // Need true 2D grid for SUMMA

    for (auto &node : nodes) {
        // Only Einsum nodes. BatchedGemm is intentionally ignored,
        // distributed batched contractions aren't supported today (see
        // libs/Einsums/ComputeGraph/docs/gemm_batching.rst).
        if (node.kind != OpKind::Einsum)
            continue;

        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (!desc)
            continue;

        // Check if output is SUMMA-distributed
        if (node.outputs.empty())
            continue;
        auto out_it = tensors.find(node.outputs[0]);
        if (out_it == tensors.end())
            continue;
        auto const &out_handle = out_it->second;
        if (!out_handle.distribution_info)
            continue;
        auto out_desc = std::static_pointer_cast<comm::DistributionDescriptor>(out_handle.distribution_info);
        if (!out_desc->summa)
            continue;

        // Check both inputs are SUMMA-distributed
        if (node.inputs.size() < 2)
            continue;
        auto a_it = tensors.find(node.inputs[0]);
        auto b_it = tensors.find(node.inputs[1]);
        if (a_it == tensors.end() || b_it == tensors.end())
            continue;

        auto const &a_handle = a_it->second;
        auto const &b_handle = b_it->second;
        if (!a_handle.distribution_info || !b_handle.distribution_info)
            continue;

        auto a_desc = std::static_pointer_cast<comm::DistributionDescriptor>(a_handle.distribution_info);
        auto b_desc = std::static_pointer_cast<comm::DistributionDescriptor>(b_handle.distribution_info);
        if (!a_desc->summa || !b_desc->summa)
            continue;

        // Only handle rank-2 GEMM for now
        if (out_handle.rank != 2 || a_handle.rank != 2 || b_handle.rank != 2)
            continue;

        // Extract dimensions:
        // A_local = (M/Pr, K/Pc), B_local = (K/Pr, N/Pc), C_local = (M/Pr, N/Pc)
        // SUMMA iterates over Pc panels (for A broadcast) or Pr panels (for B broadcast).
        // Since A: k→Col and B: k→Row, the panels iterate over the same K dimension.
        // Number of panels = max(Pc, Pr), but for a consistent SUMMA, we iterate over
        // the grid dimension that splits K in A (which is Pc) for A-broadcasts,
        // and the grid dimension that splits K in B (which is Pr) for B-broadcasts.
        //
        // Standard SUMMA with Pc == Pr: iterate over Pc panels.
        // For non-square grids, we need Pc panels for A (broadcast along rows of size Pc)
        // and Pr panels for B (broadcast along cols of size Pr).
        // But K must be consistently split: K/Pc for A's k-dim, K/Pr for B's k-dim.
        // These are only equal when Pc == Pr. For non-square, the inner dimension sizes differ.
        //
        // Simplification: for the initial implementation, require Pc == Pr.
        // For non-square grids, fall back to outer-product (skip SUMMA).
        if (grid.rows() != grid.cols()) {
            EINSUMS_LOG_INFO("SUMMAExpansion: skipping non-square grid {}x{} (not yet supported)", grid.rows(), grid.cols());
            continue;
        }

        int panels = grid.cols(); // == grid.rows() for square grid

        // Replace the einsum's execute lambda with a SUMMA loop.
        // Capture the original execute lambda as a fallback (for the local GEMM step).
        auto original_execute = node.execute;
        auto c_pf             = desc->c_prefactor; // PrefactorScalar; unwrapped per-dtype below

        // Tensor IDS, resolved at EXECUTE time through Graph::live_tensor_ptr,
        // rather than pointers baked here. Two reasons, and the second is a
        // correctness bug rather than a style point. Materialization may
        // relocate a deferred tensor after this pass runs, so a pointer taken
        // now can be stale by replay. And `tensor_ptr` names the CALLER's
        // wrapper, which capture allows to be destroyed before execute()
        // because it adopted the storage into a stand-in; only
        // `live_tensor_ptr` returns the object the graph actually keeps alive.
        auto          *graph_ptr = &graph;
        TensorId const a_id      = node.inputs[0];
        TensorId const b_id      = node.inputs[1];
        TensorId const c_id      = node.outputs[0];
        auto           dtype     = out_handle.dtype;

        // We need type-specific SUMMA. Use dtype to dispatch.
        // For now, support double only.
        if (dtype != packed_gemm::ScalarType::Float64 && dtype != packed_gemm::ScalarType::Float32) {
            EINSUMS_LOG_INFO("SUMMAExpansion: skipping unsupported dtype for '{}'", out_handle.name);
            continue;
        }

        // Build the SUMMA executor lambda. The panels are broadcast through
        // comm::broadcast on the row/col communicators, not through the handle's
        // allreduce hook.
        node.execute = [&grid, panels, graph_ptr, a_id, b_id, c_id, dtype, c_pf, original_execute]() {
            void *a_ptr = graph_ptr->live_tensor_ptr(a_id);
            void *b_ptr = graph_ptr->live_tensor_ptr(b_id);
            void *c_ptr = graph_ptr->live_tensor_ptr(c_id);
            if (dtype == packed_gemm::ScalarType::Float64) {
                run_summa_panels<double>(grid, panels, a_ptr, b_ptr, c_ptr, c_pf);
            } else if (dtype == packed_gemm::ScalarType::Float32) {
                run_summa_panels<float>(grid, panels, a_ptr, b_ptr, c_ptr, c_pf);
            }
        };

        _num_expanded++;
        EINSUMS_LOG_INFO("SUMMAExpansion: replaced einsum '{}' with SUMMA loop ({} panels on {}x{} grid)", node.label, panels, grid.rows(),
                         grid.cols());
        report(2, fmt::format("expand einsum '{}' into a SUMMA broadcast+GEMM loop ({} panels, {}x{} grid)", node.label, panels,
                              grid.rows(), grid.cols()));
    }

    return expanded.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
