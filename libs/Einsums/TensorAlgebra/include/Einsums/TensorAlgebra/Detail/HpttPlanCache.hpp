//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/HPTT/HPTT.hpp>

#if defined(I)
#    undef I
#endif

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::tensor_algebra::detail {

// Plan cache for hptt::Transpose<T>.
//
// HPTT plans are expensive to build (per call: structural analysis,
// loop ordering, parallelism strategy selection, and even in ESTIMATE
// mode it walks the candidate list. The plan tree depends only on
// shape, not on the runtime A/B/alpha/beta, so we hash on the
// structural parameters and reuse the plan across calls. On hit, the
// pointers and scalars are reset via the existing setters.
//
// Cache is thread_local: no locks, no contention. Each thread builds
// its own plans on first use; for typical Einsums workloads the
// shape catalog is small and the per-thread footprint is negligible.

template <typename T>
struct HpttPlanKey {
    int                 dim{0};
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
    int const numThreads = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();

    // MEASURE/PATIENT/CRAZY do their own autotuning; the caller is
    // paying for it deliberately, so don't replace it with a cached
    // ESTIMATE plan.
    if (method != hptt::ESTIMATE) {
        return hptt::create_plan(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB, innerStrideB,
                                 method, numThreads, nullptr, row_major);
    }

    HpttPlanKey<T> key;
    key.dim          = dim;
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

    // The cache stores plan templates keyed on shape. Each caller gets a
    // fresh ``Transpose<T>`` copy with its own ``_A``/``_B``/``_alpha``/
    // ``_beta``/``_conjA`` slots, while the plan tree (``_masterPlan``) is the
    // expensive thing to build and is shared via ``shared_ptr`` inside the
    // copy. We can't return the template directly because higher-level
    // callers (e.g. ``cached_permute`` in Dispatch.hpp) hang onto the plan
    // across calls, and two of those callers sharing a shape would trample
    // each other's pointer/scalar state on the shared object.
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

} // namespace einsums::tensor_algebra::detail
