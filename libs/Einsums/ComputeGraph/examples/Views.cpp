//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Views.cpp
/// @brief Demonstrates ``cg::view`` (graph-aware non-owning slices) and
///        ``cg::write_param`` (graph-driven parameter updates).
///
/// Shows three patterns, in order:
///   1. Constant-bound view, slice with literal integer bounds.
///   2. Pipeline-parameter view, slice bounds resolved each iteration
///      from a named ParamTable entry. Update the param between
///      ``Pipeline::execute()`` calls; the next execute picks up the new
///      slice without rebuilding the graph.
///   3. Mid-loop parameter update, ``cg::write_param`` from a callback
///      changes the param *within* a single execute, so subsequent
///      iterations of the same loop body see the new bounds.
///
/// All three patterns use the same underlying machinery; pick the form
/// that matches when the value is known.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cstdint>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Constant-bound view, slice the first 2 rows of a 4×4 matrix
    // ═══════════════════════════════════════════════════════════════════════
    println("=== Constant-bound view ===\n");
    {
        Tensor<double, 2> A("A", 4, 4);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                A(i, j) = static_cast<double>(10 * i + j);

        Tensor<double, 2> top_two("top_two", 2, 4);

        cg::Pipeline pipe("const_view");
        {
            auto                  &stage = pipe.add_stage("copy_top_two");
            cg::CaptureGuard const g(stage);

            // slice = A[0:2, :], rank-preserving, zero-copy alias.
            auto &slice = cg::view<double, 2>(A, cg::ViewAxis::range(0, 2), // rows 0..2
                                              cg::ViewAxis::full());        // all cols

            cg::permute("ij <- ij", 0.0, &top_two, 1.0, slice);
        }
        pipe.execute();

        println("  A[0,:]      = [{}, {}, {}, {}]", A(0, 0), A(0, 1), A(0, 2), A(0, 3));
        println("  A[1,:]      = [{}, {}, {}, {}]", A(1, 0), A(1, 1), A(1, 2), A(1, 3));
        println("  top_two[0]  = [{}, {}, {}, {}]", top_two(0, 0), top_two(0, 1), top_two(0, 2), top_two(0, 3));
        println("  top_two[1]  = [{}, {}, {}, {}]\n", top_two(1, 0), top_two(1, 1), top_two(1, 2), top_two(1, 3));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Pipeline-parameter view, slice extent set per execute
    // ═══════════════════════════════════════════════════════════════════════
    //
    // The HF case: ``C_occ = C[:, 0:n_occ]``. ``n_occ`` is decided once per
    // molecule (here once per outer loop) and never changes during the
    // execute, but should not be baked into the graph at capture time.
    println("=== Pipeline-parameter view ===\n");
    {
        Tensor<double, 2> A("A", 4, 6);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 6; ++j)
                A(i, j) = 1.0;

        cg::Pipeline pipe("param_view");
        pipe.set_param("n_occ", 0); // initialized below per execute

        // Body: zero the first n_occ columns of A.
        {
            auto                  &stage = pipe.add_stage("zero_head");
            cg::CaptureGuard const g(stage);
            auto                  &head = cg::view<double, 2>(A,
                                             cg::ViewAxis::full(),             // all rows
                                             cg::ViewAxis::range(0, "n_occ")); // cols 0..n_occ
            cg::scale(0.0, &head);
        }

        // Reset, then run with n_occ = 2 and n_occ = 4 in turn.
        for (int const trial : {2, 4}) {
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 6; ++j)
                    A(i, j) = 1.0;

            pipe.set_param("n_occ", trial);
            pipe.execute();

            // Count zeros in row 0 to confirm.
            int zeros = 0;
            for (size_t j = 0; j < 6; ++j)
                if (A(0, j) == 0.0)
                    ++zeros;
            println("  n_occ = {}: row 0 has {} leading zeros (expected {})", trial, zeros, trial);
        }
        println("");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Mid-loop parameter update, cg::write_param from a callback
    // ═══════════════════════════════════════════════════════════════════════
    //
    // SCF-style loop where the slice grows by one each iteration. ``n_occ``
    // is updated *within* the loop body via ``cg::write_param``; the next
    // iteration's View picks up the new value.
    //
    // We zero the first ``n_occ`` columns of A on each pass; by iteration 4
    // all 4 columns of a 1×4 row should be zero.
    println("=== Mid-loop write_param ===\n");
    {
        Tensor<double, 2> A("A", 1, 4);
        for (size_t j = 0; j < 4; ++j)
            A(0, j) = 1.0;

        cg::Pipeline pipe("write_param_view");
        pipe.set_param("n_occ", 1);

        size_t iter_count = 0;
        auto   cond       = [&](size_t /*iter*/) {
            ++iter_count;
            return iter_count < 4; // 4 iterations
        };

        {
            auto                  &body = pipe.add_loop("grow", 10, cond);
            cg::CaptureGuard const g(body);

            auto &head = cg::view<double, 2>(A, cg::ViewAxis::full(), cg::ViewAxis::range(0, "n_occ"));
            cg::scale(0.0, &head);

            // Increase n_occ by 1 for the next iteration.
            cg::write_param("n_occ", std::function<std::int64_t()>([&pipe] { return pipe.get_param("n_occ") + 1; }));
        }

        pipe.execute();

        println("  After 4 iterations, A = [{}, {}, {}, {}]", A(0, 0), A(0, 1), A(0, 2), A(0, 3));
        println("  Final n_occ = {}\n", pipe.get_param("n_occ"));
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
