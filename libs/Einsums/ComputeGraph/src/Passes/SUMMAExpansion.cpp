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
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <cstring>
#include <variant>
#include <vector>

using namespace einsums::index;

namespace einsums::compute_graph::passes {

bool SUMMAExpansion::run(Graph &graph) {
    _num_expanded = 0;

    if (comm::world_size() <= 1)
        return false;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();
    auto       &grid    = comm::ProcessGrid::default_grid();

    if (grid.rows() <= 1 || grid.cols() <= 1)
        return false; // Need true 2D grid for SUMMA

    for (auto &node : nodes) {
        // Only Einsum nodes. BatchedGemm is intentionally ignored —
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
        // Number of panels = max(Pc, Pr) — but for a consistent SUMMA, we iterate over
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

        // Capture tensor pointers (type-erased) for broadcast
        // A_local, B_local, C_local are the tensors after materialization
        auto *a_ptr = a_handle.tensor_ptr; // void* to Tensor<T,2>
        auto *b_ptr = b_handle.tensor_ptr;
        auto *c_ptr = out_handle.tensor_ptr;
        auto  dtype = out_handle.dtype;

        // We need type-specific SUMMA. Use dtype to dispatch.
        // For now, support double only.
        if (dtype != packed_gemm::ScalarType::Float64 && dtype != packed_gemm::ScalarType::Float32) {
            EINSUMS_LOG_INFO("SUMMAExpansion: skipping unsupported dtype for '{}'", out_handle.name);
            continue;
        }

        // Build the SUMMA executor lambda
        auto a_alloc_fn = a_handle.allreduce_sum_fn; // not used, but we need broadcast
        // We'll use comm::broadcast directly with the row/col communicators

        node.execute = [&grid, panels, a_ptr, b_ptr, c_ptr, dtype, c_pf, original_execute]() {
            if (dtype == packed_gemm::ScalarType::Float64) {
                auto *A_local = static_cast<Tensor<double, 2> *>(a_ptr);
                auto *B_local = static_cast<Tensor<double, 2> *>(b_ptr);
                auto *C_local = static_cast<Tensor<double, 2> *>(c_ptr);

                size_t       local_m   = C_local->dim(0);
                size_t       local_n   = C_local->dim(1);
                size_t       local_k_a = A_local->dim(1); // K/Pc
                size_t const local_k_b = B_local->dim(0); // K/Pr (== K/Pc for square grid)

                // Apply C prefactor (typically 0.0 for first call)
                auto c_pf_d = as<double>(c_pf);
                if (c_pf_d == 0.0) {
                    C_local->zero();
                } else if (c_pf_d != 1.0) {
                    linear_algebra::scale(c_pf_d, C_local);
                }

                int const my_col = grid.my_col();
                int const my_row = grid.my_row();

                // Allocate temporary panel buffers (same size as local blocks)
                Tensor<double, 2> A_panel("A_panel", local_m, local_k_a);
                Tensor<double, 2> B_panel("B_panel", local_k_b, local_n);

                profile::Profiler::instance().push(fmt::format("SUMMA({}x{}x{}, {} panels)", local_m, local_k_a, local_n, panels));

                for (int p = 0; p < panels; p++) {
                    // Step 1: Broadcast A panel along rows.
                    profile::Profiler::instance().push("broadcast_A");
                    if (my_col == p) {
                        std::memcpy(A_panel.data(), A_local->data(), local_m * local_k_a * sizeof(double));
                    }
                    auto placeholder = comm::broadcast<double>(std::span<double>(A_panel.data(), A_panel.size()), p, grid.row_comm());
                    (void)placeholder;
                    profile::Profiler::instance().pop();

                    // Step 2: Broadcast B panel along cols.
                    profile::Profiler::instance().push("broadcast_B");
                    if (my_row == p) {
                        std::memcpy(B_panel.data(), B_local->data(), local_k_b * local_n * sizeof(double));
                    }
                    placeholder = comm::broadcast<double>(std::span<double>(B_panel.data(), B_panel.size()), p, grid.col_comm());
                    (void)placeholder;
                    profile::Profiler::instance().pop();

                    // Step 3: Local GEMM accumulate via einsum dispatch (enables PackedGemm)
                    profile::Profiler::instance().push("local_gemm");
                    tensor_algebra::einsum(1.0, Indices{i, j}, C_local, 1.0, Indices{i, k}, A_panel, Indices{k, j}, B_panel);
                    profile::Profiler::instance().pop();
                }

                profile::Profiler::instance().pop(); // SUMMA
            } else if (dtype == packed_gemm::ScalarType::Float32) {
                auto *A_local = static_cast<Tensor<float, 2> *>(a_ptr);
                auto *B_local = static_cast<Tensor<float, 2> *>(b_ptr);
                auto *C_local = static_cast<Tensor<float, 2> *>(c_ptr);

                size_t       local_m   = C_local->dim(0);
                size_t       local_n   = C_local->dim(1);
                size_t       local_k_a = A_local->dim(1);
                size_t const local_k_b = B_local->dim(0);

                auto c_pf_f = as<float>(c_pf);
                if (c_pf_f == 0.0F) {
                    C_local->zero();
                } else if (c_pf_f != 1.0F) {
                    linear_algebra::scale(c_pf_f, C_local);
                }

                Tensor<float, 2> A_panel("A_panel", local_m, local_k_a);
                Tensor<float, 2> B_panel("B_panel", local_k_b, local_n);

                int const my_col = grid.my_col();
                int const my_row = grid.my_row();

                profile::Profiler::instance().push(fmt::format("SUMMA({}x{}x{}, {} panels, float)", local_m, local_k_a, local_n, panels));

                for (int p = 0; p < panels; p++) {
                    profile::Profiler::instance().push("broadcast_A");
                    if (my_col == p) {
                        std::memcpy(A_panel.data(), A_local->data(), local_m * local_k_a * sizeof(float));
                    }
                    auto placeholder = comm::broadcast<float>(std::span<float>(A_panel.data(), A_panel.size()), p, grid.row_comm());
                    (void)placeholder;
                    profile::Profiler::instance().pop();

                    profile::Profiler::instance().push("broadcast_B");
                    if (my_row == p) {
                        std::memcpy(B_panel.data(), B_local->data(), local_k_b * local_n * sizeof(float));
                    }
                    placeholder = comm::broadcast<float>(std::span<float>(B_panel.data(), B_panel.size()), p, grid.col_comm());
                    (void)placeholder;
                    profile::Profiler::instance().pop();

                    profile::Profiler::instance().push("local_gemm");
                    tensor_algebra::einsum(1.0f, Indices{i, j}, C_local, 1.0f, Indices{i, k}, A_panel, Indices{k, j}, B_panel);
                    profile::Profiler::instance().pop();
                }

                profile::Profiler::instance().pop(); // SUMMA
            }
        };

        _num_expanded++;
        EINSUMS_LOG_INFO("SUMMAExpansion: replaced einsum '{}' with SUMMA loop ({} panels on {}x{} grid)", node.label, panels, grid.rows(),
                         grid.cols());
        report(2, fmt::format("expand einsum '{}' into a SUMMA broadcast+GEMM loop ({} panels, {}x{} grid)", node.label, panels,
                              grid.rows(), grid.cols()));
    }

    return _num_expanded > 0;
}

} // namespace einsums::compute_graph::passes
