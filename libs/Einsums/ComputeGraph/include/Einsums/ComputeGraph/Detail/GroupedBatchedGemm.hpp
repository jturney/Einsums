//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/Detail/BatchedGemm.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>

#include <complex>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Shared executor for @ref OpKind::GroupedBatchedGemm nodes.
 *
 * The same contract as @ref make_batched_gemm_executor and for the same
 * reason: more than one producer builds these nodes (`cg::grouped_batched_gemm`
 * today, the GEMMBatching pass later), and two copies of the dispatch would
 * drift. Pointers are re-extracted on every call rather than baked in, because
 * `Graph::rebind` and the MemoryPlanning arena can both move a tensor's storage
 * between executions.
 *
 * What it adds over the uniform node is that the shape parameters are per
 * group, so they are hoisted into typed arrays once at capture and only the
 * pointers are rebuilt per execution.
 */

/// Whether to time each group separately instead of issuing one grouped call.
///
/// Reading it costs a config lookup, so the executor asks once at capture.
EINSUMS_EXPORT bool grouped_gemm_group_profiling();

/// The per-group BLAS parameters, hoisted out of the descriptor into the
/// argument arrays `blas::gemm_batch_grouped` wants, and typed.
///
/// Built once when the node is captured. Nothing here depends on where the
/// operands live, which is exactly why it can outlive a `rebind`.
template <typename T>
struct GroupedGemmPlan {
    std::vector<char>        transa, transb;
    std::vector<blas::int_t> m, n, k, lda, ldb, ldc, size;
    std::vector<T>           alpha, beta;
    std::vector<std::string> labels;
    /// Where each group's members start in the flattened operand arrays.
    std::vector<int> first;
    int              total{0};
};

template <typename T>
GroupedGemmPlan<T> make_grouped_gemm_plan(GroupedBatchedGemmDescriptor const &d) {
    GroupedGemmPlan<T> p;
    auto const         g_count = d.groups.size();
    p.transa.reserve(g_count);
    p.transb.reserve(g_count);
    p.m.reserve(g_count);
    p.n.reserve(g_count);
    p.k.reserve(g_count);
    p.lda.reserve(g_count);
    p.ldb.reserve(g_count);
    p.ldc.reserve(g_count);
    p.size.reserve(g_count);
    p.alpha.reserve(g_count);
    p.beta.reserve(g_count);
    p.first.reserve(g_count);

    for (auto const &g : d.groups) {
        p.transa.push_back(g.trans_a);
        p.transb.push_back(g.trans_b);
        p.m.push_back(static_cast<blas::int_t>(g.m));
        p.n.push_back(static_cast<blas::int_t>(g.n));
        p.k.push_back(static_cast<blas::int_t>(g.k));
        p.lda.push_back(static_cast<blas::int_t>(g.lda));
        p.ldb.push_back(static_cast<blas::int_t>(g.ldb));
        p.ldc.push_back(static_cast<blas::int_t>(g.ldc));
        p.size.push_back(static_cast<blas::int_t>(g.count));
        p.first.push_back(g.first);
        // The descriptor carries the full complex prefactor; preserve both
        // parts, a phase factor must not be truncated to its real part.
        if constexpr (IsComplexV<T>) {
            using R = typename T::value_type;
            p.alpha.push_back(T{static_cast<R>(g.alpha.real()), static_cast<R>(g.alpha.imag())});
            p.beta.push_back(T{static_cast<R>(g.beta.real()), static_cast<R>(g.beta.imag())});
        } else {
            p.alpha.push_back(static_cast<T>(g.alpha.real()));
            p.beta.push_back(static_cast<T>(g.beta.real()));
        }
    }
    p.labels = d.labels;
    p.total  = d.total;
    return p;
}

/**
 * @brief Extract every member's pointer and issue the call, in one pass.
 *
 * The extraction walk is threaded for the same reason the uniform node's is:
 * each pointer is three dependent loads over separate allocations, so a few
 * thousand members walk a few thousand cache misses with nothing to prefetch.
 * Guarded on @c omp_in_parallel because the OpenMP executor already replays
 * whole graphs as a team, and nesting a second one inside a node is how the
 * OpenMP-built OpenBLAS gets miscomputed.
 */
template <typename T>
void extract_and_run_grouped(GroupedGemmPlan<T> const &p, BatchedGemmExtractors const &a_exs, BatchedGemmExtractors const &b_exs,
                             BatchedGemmOutputExtractors const &c_exs, bool profile_groups) {
    std::vector<T const *> a_arr(p.total);
    std::vector<T const *> b_arr(p.total);
    std::vector<T *>       c_arr(p.total);

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (!omp_in_parallel())
#endif
    for (int i = 0; i < p.total; ++i) {
        a_arr[i] = static_cast<T const *>(a_exs[i]().first);
        b_arr[i] = static_cast<T const *>(b_exs[i]().first);
        c_arr[i] = static_cast<T *>(c_exs[i]().first);
    }

    if (!profile_groups) {
        blas::gemm_batch_grouped<T>(p.transa.data(), p.transb.data(), p.m.data(), p.n.data(), p.k.data(), p.alpha.data(), a_arr.data(),
                                    p.lda.data(), b_arr.data(), p.ldb.data(), p.beta.data(), c_arr.data(), p.ldc.data(),
                                    static_cast<blas::int_t>(p.size.size()), p.size.data());
        return;
    }

    // Group profiling deliberately does NOT time the grouped call: there is no
    // way to attribute one parallel loop's wall time to the groups inside it
    // without a clock read per member. It runs the batch the way this node's
    // ancestors did instead, one uniform call per group under its own zone, so
    // the per-shape breakdown that made these investigations possible is one
    // flag away.
    //
    // The consequence is that a profiled run measures a DIFFERENT execution:
    // the whole point of the grouped node is that the looped form is slower.
    // Read the breakdown for where the arithmetic is, never for what the node
    // costs.
    for (size_t g = 0; g < p.size.size(); ++g) {
        LabeledSectionRuntime(g < p.labels.size() ? p.labels[g] : fmt::format("group {}", g));
        int const off = p.first[g];
        blas::gemm_batch<T>(p.transa[g], p.transb[g], p.m[g], p.n[g], p.k[g], p.alpha[g], a_arr.data() + off, p.lda[g], b_arr.data() + off,
                            p.ldb[g], p.beta[g], c_arr.data() + off, p.ldc[g], p.size[g]);
    }
}

/**
 * @brief Build the node executor for a @ref GroupedBatchedGemmDescriptor.
 *
 * @param d Per-group BLAS parameters and the element type.
 * @param a_exs,b_exs,c_exs Per-member extractors, flattened in group order so
 *          that @ref GemmGroup::first indexes them.
 */
inline std::function<void()> make_grouped_batched_gemm_executor(GroupedBatchedGemmDescriptor const &d, BatchedGemmExtractors a_exs,
                                                                BatchedGemmExtractors b_exs, BatchedGemmOutputExtractors c_exs) {
    bool const profile = grouped_gemm_group_profiling();

    // The plan is typed, so it is built here, once, under the same switch the
    // uniform node does its dispatch with. Only the pointers move afterwards.
    switch (d.scalar) {
    case BlasScalar::Float:
        return [p = make_grouped_gemm_plan<float>(d), a_exs = std::move(a_exs), b_exs = std::move(b_exs), c_exs = std::move(c_exs),
                profile]() { extract_and_run_grouped<float>(p, a_exs, b_exs, c_exs, profile); };
    case BlasScalar::Double:
        return [p = make_grouped_gemm_plan<double>(d), a_exs = std::move(a_exs), b_exs = std::move(b_exs), c_exs = std::move(c_exs),
                profile]() { extract_and_run_grouped<double>(p, a_exs, b_exs, c_exs, profile); };
    case BlasScalar::ComplexFloat:
        return [p = make_grouped_gemm_plan<std::complex<float>>(d), a_exs = std::move(a_exs), b_exs = std::move(b_exs),
                c_exs = std::move(c_exs), profile]() { extract_and_run_grouped<std::complex<float>>(p, a_exs, b_exs, c_exs, profile); };
    case BlasScalar::ComplexDouble:
        break;
    }
    return [p = make_grouped_gemm_plan<std::complex<double>>(d), a_exs = std::move(a_exs), b_exs = std::move(b_exs),
            c_exs = std::move(c_exs), profile]() { extract_and_run_grouped<std::complex<double>>(p, a_exs, b_exs, c_exs, profile); };
}

/// Run a grouped batch straight from raw pointers, for the eager path.
template <typename T>
void run_grouped_batched_gemm(GroupedBatchedGemmDescriptor const &d, std::vector<void const *> const &a_vs,
                              std::vector<void const *> const &b_vs, std::vector<void *> const &c_vs) {
    auto const             p = make_grouped_gemm_plan<T>(d);
    std::vector<T const *> a_arr(p.total);
    std::vector<T const *> b_arr(p.total);
    std::vector<T *>       c_arr(p.total);
    for (int i = 0; i < p.total; ++i) {
        a_arr[i] = static_cast<T const *>(a_vs[i]);
        b_arr[i] = static_cast<T const *>(b_vs[i]);
        c_arr[i] = static_cast<T *>(c_vs[i]);
    }
    blas::gemm_batch_grouped<T>(p.transa.data(), p.transb.data(), p.m.data(), p.n.data(), p.k.data(), p.alpha.data(), a_arr.data(),
                                p.lda.data(), b_arr.data(), p.ldb.data(), p.beta.data(), c_arr.data(), p.ldc.data(),
                                static_cast<blas::int_t>(p.size.size()), p.size.data());
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
