//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>

#if defined(I)
#    undef I
#endif

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(tensor_permute::detail)

// Per-thread plan cache for hptt::Transpose<T>. Plans are expensive to build but depend only on
// shape, so they are keyed on the structural parameters and rebound to new pointers and scalars.

template <typename T>
struct HpttPlanKey {
    int dim{0};
    /// The team size the plan was built for. It shapes the work decomposition, so plans for different
    /// counts are different plans (and the count differs inside and outside an OpenMP region).
    int                 num_threads{1};
    bool                row_major{false};
    size_t              innerStrideA{1};
    size_t              innerStrideB{1};
    std::vector<int>    perm;
    std::vector<size_t> sizeA;
    std::vector<size_t> outerSizeA;
    std::vector<size_t> outerSizeB;
    std::vector<size_t> offsetA;
    std::vector<size_t> offsetB;

    bool operator==(HpttPlanKey const &) const = default;
};

template <typename T>
struct HpttPlanKeyHash {
    size_t operator()(HpttPlanKey<T> const &k) const noexcept {
        // FNV-1a over the structural fields. The vectors are short
        // (rank ≤ ~6 in practice) so the per-key cost is small.
        size_t h   = 0xcbf29ce484222325ULL;
        auto   mix = [&h](size_t x) {
            h ^= x;
            h *= 0x100000001b3ULL;
        };
        mix(static_cast<size_t>(k.dim));
        mix(static_cast<size_t>(k.num_threads));
        mix(k.innerStrideA);
        mix(k.innerStrideB);
        mix(k.row_major ? 1ULL : 0ULL);
        for (auto v : k.perm)
            mix(static_cast<size_t>(v));
        for (auto v : k.sizeA)
            mix(v);
        for (auto v : k.outerSizeA)
            mix(v);
        for (auto v : k.outerSizeB)
            mix(v);
        for (auto v : k.offsetA)
            mix(v);
        for (auto v : k.offsetB)
            mix(v);
        return h;
    }
};

template <typename T>
inline std::shared_ptr<hptt::Transpose<T>>
get_or_create_hptt_plan(int const *perm, int dim, T alpha, T const *A, size_t const *sizeA, size_t const *outerSizeA, size_t const *offsetA,
                        size_t innerStrideA, T beta, T *B, size_t const *outerSizeB, size_t const *offsetB, size_t innerStrideB,
                        bool row_major, hptt::SelectionMethod method = hptt::ESTIMATE) {
    // Small transposes run on one thread: a thread-team fork cost 24 us against 0.6 us serial for a
    // 64-element permute.
    size_t elements = 1;
    for (int d = 0; d < dim; ++d) {
        elements *= sizeA[d];
    }

    int const numThreads = [elements]() {
#ifdef _OPENMP
        return elements >= ::einsums::hardware::omp_min_parallel_elements() ? omp_get_max_threads() : 1;
#else
        return 1;
#endif
    }();

    // The autotuning methods are not cached: the caller is paying for the search deliberately.
    if (method != hptt::ESTIMATE) {
        return hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                                 method, numThreads, nullptr, row_major);
    }

    HpttPlanKey<T> key;
    key.dim          = dim;
    key.num_threads  = numThreads;
    key.row_major    = row_major;
    key.innerStrideA = innerStrideA;
    key.innerStrideB = innerStrideB;
    key.perm.assign(perm, perm + dim);
    key.sizeA.assign(sizeA, sizeA + dim);
    if (outerSizeA != nullptr)
        key.outerSizeA.assign(outerSizeA, outerSizeA + dim);
    else
        key.outerSizeA = key.sizeA;
    if (outerSizeB != nullptr)
        key.outerSizeB.assign(outerSizeB, outerSizeB + dim);
    else {
        // HPTT default: outerSizeB defaults to sizeB which equals sizeA permuted by perm.
        key.outerSizeB.resize(dim);
        for (int i = 0; i < dim; ++i)
            key.outerSizeB[i] = key.sizeA[perm[i]];
    }
    if (offsetA != nullptr)
        key.offsetA.assign(offsetA, offsetA + dim);
    else
        key.offsetA.assign(dim, 0);
    if (offsetB != nullptr)
        key.offsetB.assign(offsetB, offsetB + dim);
    else
        key.offsetB.assign(dim, 0);

    // Holds plan templates. Each caller gets a copy with its own pointers and scalars, sharing the
    // expensive plan tree, since callers keep plans across calls and would otherwise overwrite each
    // other's state.
    thread_local std::unordered_map<HpttPlanKey<T>, std::shared_ptr<hptt::Transpose<T>>, HpttPlanKeyHash<T>> cache;

    auto fresh_copy = [&](std::shared_ptr<hptt::Transpose<T>> const &templ) {
        auto p = templ->clone();
        p->set_input_ptr(A);
        p->set_output_ptr(B);
        p->set_alpha(alpha);
        p->set_beta(beta);
        return p;
    };

    if (auto it = cache.find(key); it != cache.end()) {
        return fresh_copy(it->second);
    }

    auto templ = hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB,
                                   innerStrideB, hptt::ESTIMATE, numThreads, nullptr, row_major);
    cache.emplace(std::move(key), templ);
    return fresh_copy(templ);
}

// Convenience overload for the basic signature (no offsets, no innerStrides).
template <typename T>
inline std::shared_ptr<hptt::Transpose<T>> get_or_create_hptt_plan(int const *perm, int dim, T alpha, T const *A, size_t const *sizeA,
                                                                   T beta, T *B, bool row_major) {
    return get_or_create_hptt_plan<T>(perm, dim, alpha, A, sizeA, nullptr, nullptr, 1, beta, B, nullptr, nullptr, 1, row_major);
}

// Convenience overload with offsets and outer sizes but unit inner strides.
template <typename T>
inline std::shared_ptr<hptt::Transpose<T>> get_or_create_hptt_plan(int const *perm, int dim, T alpha, T const *A, size_t const *sizeA,
                                                                   size_t const *outerSizeA, size_t const *offsetA, T beta, T *B,
                                                                   size_t const *outerSizeB, size_t const *offsetB, bool row_major) {
    return get_or_create_hptt_plan<T>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, 1, beta, B, outerSizeB, offsetB, 1, row_major);
}

EINSUMS_NAMESPACE_END(tensor_permute::detail)
