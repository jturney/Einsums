//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Logging.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/Packing.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#elif defined(__linux__)
#    include <fstream>
#    include <string>
#endif

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

namespace {

template <typename T>
void hash_combine(size_t &seed, T const &v) {
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

size_t hash_vec_str(std::vector<std::string> const &v) {
    size_t h = v.size();
    for (auto const &s : v) {
        h = h * 31 + static_cast<size_t>(static_cast<unsigned char>(s[0]));
    }
    return h;
}

size_t hash_vec_i64(std::vector<int64_t> const &v) {
    size_t h = 0;
    for (auto x : v) {
        hash_combine(h, x);
    }
    return h;
}

} // anonymous namespace

CpuConfig const &cpu_config() {
    static CpuConfig const cfg = []() -> CpuConfig {
        CpuConfig c{};
        c.VL = einsums::hardware::cpu_info().simd_width_f64;
        c.MR = 2 * c.VL; // Two full vector registers as C accumulators per j-column.
        c.NR = 6;        // Fixed: LLVM fully unrolls trip counts <= ~16.

        // Facts come from the one detector (Einsums_Hardware); MR/NR and the BLIS
        // blocking below stay here, being kernel policy rather than hardware.
        auto const &cache = einsums::hardware::cpu_info().cache;
        c.l1_cache_size   = cache.l1;
        c.l2_cache_size   = cache.l2;
        c.l3_cache_size   = cache.l3;

        // Single measured source, shared with the elementwise kernels (Config).
        c.omp_region_cost_ns = einsums::hardware::omp_region_cost_ns();
        // Same derivation everyone else uses, from the one detector.
        c.min_parallel_flops = einsums::hardware::omp_min_parallel_flops();

        EINSUMS_LOG_INFO("cpu_config: VL={}, MR={}, NR={}, L1={}K, L2={}K, L3={}K, omp_region={:.2f}us, min_parallel_flops={}", c.VL, c.MR,
                         c.NR, c.l1_cache_size / 1024, c.l2_cache_size / 1024, c.l3_cache_size / 1024, c.omp_region_cost_ns / 1000.0,
                         c.min_parallel_flops);
        return c;
    }();
    return cfg;
}

BlockingParams compute_blocking(int64_t elem_size) {
    auto const &cfg = cpu_config();
    int const   MR  = cfg.MR;
    int const   NR  = cfg.NR;

    // Use half the L2 for the A panel (MC * KC * elem_size ≤ L2/2).
    // Use half the L3 for the B panel (KC * NC * elem_size ≤ L3/2).
    int64_t const l2_budget = cfg.l2_cache_size / 2;
    int64_t const l3_budget = cfg.l3_cache_size / 2;

    // Start with KC sized so that one column of packed A (MR * KC) fits in ~L1.
    // Then adjust MC so the full A panel (MC * KC) fits in L2.
    int64_t KC = std::max(int64_t{64}, cfg.l1_cache_size / (MR * elem_size));
    // Round KC down to a multiple of 8 for alignment.
    KC = (KC / 8) * 8;
    if (KC < 64) {
        KC = 64;
    }

    // MC: fit A panel in L2 budget.  MC must be a multiple of MR.
    int64_t MC = l2_budget / (KC * elem_size);
    MC         = (MC / MR) * MR;
    if (MC < MR) {
        MC = MR;
    }

    // NC: fit B panel in L3 budget.  NC must be a multiple of NR.
    int64_t NC = l3_budget / (KC * elem_size);
    NC         = (NC / NR) * NR;
    if (NC < NR) {
        NC = NR;
    }

    return {.KC = KC, .MC = MC, .NC = NC, .NR = NR};
}

// ---------------------------------------------------------------------------
// PackingPlanCache implementation
// ---------------------------------------------------------------------------

struct PackingPlanCache::Impl {
    std::unordered_map<ContractionKey, PackingPlan> cache;
    mutable std::shared_mutex                       mutex;
};

PackingPlanCache::PackingPlanCache() : _impl(std::make_unique<Impl>()) {
}

PackingPlanCache::~PackingPlanCache() = default;

PackingPlanCache &PackingPlanCache::instance() {
    static PackingPlanCache cache;
    return cache;
}

PackingPlan const *PackingPlanCache::lookup(ContractionKey const &key) const {
    std::shared_lock<std::shared_mutex> const rlock(_impl->mutex);
    auto                                      it = _impl->cache.find(key);
    if (it == _impl->cache.end()) {
        return nullptr;
    }
    return &it->second;
}

void PackingPlanCache::insert(ContractionKey const &key, PackingPlan plan) {
    std::unique_lock<std::shared_mutex> const wlock(_impl->mutex);
    _impl->cache.emplace(key, std::move(plan));
}

EINSUMS_NAMESPACE_END(packed_gemm)

// Must be in namespace std:
size_t std::hash<einsums::packed_gemm::ContractionKey>::operator()(einsums::packed_gemm::ContractionKey const &key) const noexcept {
    using namespace einsums::packed_gemm;
    size_t h = 0;
    hash_combine(h, hash_vec_str(key.spec.c_indices));
    hash_combine(h, hash_vec_str(key.spec.a_indices));
    hash_combine(h, hash_vec_str(key.spec.b_indices));
    hash_combine(h, hash_vec_str(key.spec.all_indices));
    hash_combine(h, hash_vec_str(key.spec.link_indices));
    hash_combine(h, static_cast<int>(key.spec.scalar_type));
    hash_combine(h, static_cast<int>(key.spec.conj_a));
    hash_combine(h, static_cast<int>(key.spec.conj_b));
    hash_combine(h, static_cast<int>(key.spec.scalar_output));
    hash_combine(h, key.a_desc.rank);
    hash_combine(h, key.b_desc.rank);
    hash_combine(h, key.c_desc.rank);
    hash_combine(h, static_cast<int>(key.a_desc.dtype));
    hash_combine(h, hash_vec_i64(key.a_desc.strides));
    hash_combine(h, hash_vec_i64(key.b_desc.strides));
    hash_combine(h, hash_vec_i64(key.c_desc.strides));
    hash_combine(h, hash_vec_i64(key.target_dims));
    hash_combine(h, hash_vec_i64(key.link_dims));
    return h;
}
