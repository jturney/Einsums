//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/Packing.hpp>

#include <algorithm>
#include <cstdint>
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

namespace einsums::packed_gemm {

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

// ---------------------------------------------------------------------------
// cpu_config(): detect native SIMD width and compute MR, NR.
// ---------------------------------------------------------------------------

namespace {

/// Detect CPU cache sizes. Returns {L1, L2, L3} in bytes.
/// Falls back to conservative defaults if detection fails.
struct CacheSizes {
    int64_t l1 = 32 * 1024;       // 32 KB
    int64_t l2 = 256 * 1024;      // 256 KB
    int64_t l3 = 8 * 1024 * 1024; // 8 MB
};

CacheSizes detect_cache_sizes() {
    CacheSizes cs;

#if defined(__APPLE__)
    auto sysctl_i64 = [](char const *name, int64_t fallback) -> int64_t {
        int64_t val = 0;
        size_t  len = sizeof(val);
        if (sysctlbyname(name, &val, &len, nullptr, 0) == 0 && val > 0) {
            return val;
        }
        return fallback;
    };
    cs.l1 = sysctl_i64("hw.l1dcachesize", cs.l1);
    cs.l2 = sysctl_i64("hw.l2cachesize", cs.l2);
    cs.l3 = sysctl_i64("hw.l3cachesize", cs.l3);
    // Apple Silicon may report L3 as 0; fall back to a reasonable default.
    if (cs.l3 <= 0) {
        cs.l3 = 8 * 1024 * 1024;
    }
#elif defined(__linux__)
    // Read from sysfs: /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/
    auto read_cache = [](int index) -> int64_t {
        std::string   base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
        std::ifstream size_file(base + "size");
        if (!size_file.is_open()) {
            return -1;
        }
        std::string size_str;
        std::getline(size_file, size_str);
        if (size_str.empty()) {
            return -1;
        }
        // Parse "32K", "256K", "8192K" etc.
        int64_t val  = std::stoll(size_str);
        char    unit = size_str.back();
        if (unit == 'K' || unit == 'k') {
            val *= 1024;
        } else if (unit == 'M' || unit == 'm') {
            val *= 1024 * 1024;
        }
        return val;
    };
    // index0 = L1d (usually), index2 = L2, index3 = L3
    // Verify via the "level" file to be safe.
    for (int idx = 0; idx <= 4; ++idx) {
        std::string   level_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/level";
        std::ifstream level_file(level_path);
        if (!level_file.is_open()) {
            continue;
        }
        int level = 0;
        level_file >> level;
        // Also check type: we want "Data" or "Unified", not "Instruction"
        std::string   type_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/type";
        std::ifstream type_file(type_path);
        std::string   type_str;
        if (type_file.is_open()) {
            std::getline(type_file, type_str);
        }
        if (type_str == "Instruction") {
            continue;
        }
        int64_t val = read_cache(idx);
        if (val <= 0) {
            continue;
        }
        if (level == 1) {
            cs.l1 = val;
        } else if (level == 2) {
            cs.l2 = val;
        } else if (level == 3) {
            cs.l3 = val;
        }
    }
#endif

    return cs;
}

} // anonymous namespace

CpuConfig const &cpu_config() {
    static CpuConfig const cfg = []() -> CpuConfig {
        CpuConfig c{};
#if defined(__AVX512F__)
        c.VL = 8;
#elif defined(__AVX__) || defined(__AVX2__)
        c.VL = 4;
#else
        // SSE2 (x86-64 baseline), NEON (ARM), or unknown: use 128-bit / 2 doubles.
        c.VL = 2;
#endif
        c.MR = 2 * c.VL; // Two full vector registers as C accumulators per j-column.
        c.NR = 6;        // Fixed: LLVM fully unrolls trip counts <= ~16.

        auto cs         = detect_cache_sizes();
        c.l1_cache_size = cs.l1;
        c.l2_cache_size = cs.l2;
        c.l3_cache_size = cs.l3;

        EINSUMS_LOG_INFO("cpu_config: VL={}, MR={}, NR={}, L1={}K, L2={}K, L3={}K", c.VL, c.MR, c.NR, cs.l1 / 1024, cs.l2 / 1024,
                         cs.l3 / 1024);
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

} // namespace einsums::packed_gemm

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
    hash_combine(h, hash_vec_i64(key.target_dims));
    hash_combine(h, hash_vec_i64(key.link_dims));
    return h;
}
