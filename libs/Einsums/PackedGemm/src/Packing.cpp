//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Profile/Profile.hpp>

#if !defined(EINSUMS_WINDOWS)
#    include <Einsums/HPTT/HPTT.hpp>
// HPTT includes <complex> which on some platforms defines I as a macro.
// Undefine it to avoid conflicts with einsums' MAKE_INDEX(I).
#    if defined(I)
#        undef I
#    endif
#endif

#include <unordered_set>

namespace einsums::packed_gemm {

PackingPlan compute_packing_topology(ContractionKey const &key) {
    LabeledSection0();
    PackingPlan plan;
    auto const &spec = key.spec;

    // --- Pre-flight validity checks ---

    // Reject scalar output (rank-0 C cannot be tiled).
    if (spec.scalar_output) {
        EINSUMS_LOG_TRACE("compute_packing_topology: rejected (scalar output)");
        return plan;
    }

    // Reject Hadamard: repeated indices in A or B.
    {
        std::unordered_set<std::string> seen_a, seen_b;
        for (auto const &idx : spec.a_indices) {
            if (!seen_a.insert(idx).second) {
                EINSUMS_LOG_TRACE("compute_packing_topology: rejected (repeated index '{}' in A)", idx);
                return plan;
            }
        }
        for (auto const &idx : spec.b_indices) {
            if (!seen_b.insert(idx).second) {
                EINSUMS_LOG_TRACE("compute_packing_topology: rejected (repeated index '{}' in B)", idx);
                return plan;
            }
        }
    }

    // Build index sets for membership tests.
    std::unordered_set<std::string> const a_set(spec.a_indices.begin(), spec.a_indices.end());
    std::unordered_set<std::string> const b_set(spec.b_indices.begin(), spec.b_indices.end());
    std::unordered_set<std::string> const link_set(spec.link_indices.begin(), spec.link_indices.end());
    std::unordered_set<std::string> const target_set(spec.target_indices.begin(), spec.target_indices.end());

    // --- Helpers to look up sizes from the key ---

    auto find_target_size = [&](std::string const &idx) -> int64_t {
        for (size_t i = 0; i < spec.target_indices.size(); ++i) {
            if (spec.target_indices[i] == idx) {
                return key.target_dims[i];
            }
        }
        return 0;
    };

    // --- Extract batch dims: target indices in both A and B ---
    // Batch dims are NOT rejected — they are looped over in blis_contraction.
    std::unordered_set<std::string> batch_set;
    for (auto const &idx : spec.target_indices) {
        if (a_set.count(idx) && b_set.count(idx)) {
            batch_set.insert(idx);
            BatchDimSpec bds;
            bds.size = find_target_size(idx);
            // Find positions in A, B, C for fill_strides.
            for (size_t ai = 0; ai < spec.a_indices.size(); ++ai) {
                if (spec.a_indices[ai] == idx) {
                    bds.a_pos = ai;
                    break;
                }
            }
            for (size_t bi = 0; bi < spec.b_indices.size(); ++bi) {
                if (spec.b_indices[bi] == idx) {
                    bds.b_pos = bi;
                    break;
                }
            }
            for (size_t ci = 0; ci < spec.c_indices.size(); ++ci) {
                if (spec.c_indices[ci] == idx) {
                    bds.c_pos = ci;
                    break;
                }
            }
            plan.batch_dims.push_back(bds);
        }
    }

    // --- m_dims: target indices in A but not B, in a_indices order ---
    for (size_t ai = 0; ai < spec.a_indices.size(); ++ai) {
        auto const &idx = spec.a_indices[ai];
        if (target_set.count(idx) && !b_set.count(idx)) {
            plan.m_dims.push_back({.tensor_pos = ai, .size = find_target_size(idx), .tensor_stride = 0});
        }
    }

    // --- n_dims: target indices in B but not A, in b_indices order ---
    for (size_t bi = 0; bi < spec.b_indices.size(); ++bi) {
        auto const &idx = spec.b_indices[bi];
        if (target_set.count(idx) && !a_set.count(idx)) {
            plan.n_dims.push_back({.tensor_pos = bi, .size = find_target_size(idx), .tensor_stride = 0});
        }
    }

    // --- k_dims_in_a / k_dims_in_b: link dims in link_indices order ---
    for (size_t li = 0; li < spec.link_indices.size(); ++li) {
        auto const   &idx       = spec.link_indices[li];
        int64_t const link_size = key.link_dims[li];

        for (size_t ai = 0; ai < spec.a_indices.size(); ++ai) {
            if (spec.a_indices[ai] == idx) {
                plan.k_dims_in_a.push_back({.tensor_pos = ai, .size = link_size, .tensor_stride = 0});
                break;
            }
        }
        for (size_t bi = 0; bi < spec.b_indices.size(); ++bi) {
            if (spec.b_indices[bi] == idx) {
                plan.k_dims_in_b.push_back({.tensor_pos = bi, .size = link_size, .tensor_stride = 0});
                break;
            }
        }
    }

    // --- c_m_dims / c_n_dims: m/n dims as seen in C ---
    for (auto const &ds : plan.m_dims) {
        auto const &idx = spec.a_indices[ds.tensor_pos];
        for (size_t ci = 0; ci < spec.c_indices.size(); ++ci) {
            if (spec.c_indices[ci] == idx) {
                plan.c_m_dims.push_back({.tensor_pos = ci, .size = ds.size, .tensor_stride = 0});
                break;
            }
        }
    }

    for (auto const &ds : plan.n_dims) {
        auto const &idx = spec.b_indices[ds.tensor_pos];
        for (size_t ci = 0; ci < spec.c_indices.size(); ++ci) {
            if (spec.c_indices[ci] == idx) {
                plan.c_n_dims.push_back({.tensor_pos = ci, .size = ds.size, .tensor_stride = 0});
                break;
            }
        }
    }

    // --- Validity: need at least one dim in each group ---
    if (plan.m_dims.empty() || plan.n_dims.empty() || plan.k_dims_in_a.empty()) {
        EINSUMS_LOG_TRACE("compute_packing_topology: rejected (empty m/n/k group: m={}, n={}, k={})", plan.m_dims.size(),
                          plan.n_dims.size(), plan.k_dims_in_a.size());
        return plan;
    }

    // Multi-M and multi-N are now supported via flat-index-to-offset conversion
    // in pack_A/pack_B (same approach as multi-K).

    // --- Compute flat totals ---
    plan.M_total = 1;
    for (auto const &ds : plan.m_dims) {
        plan.M_total *= ds.size;
    }
    plan.N_total = 1;
    for (auto const &ds : plan.n_dims) {
        plan.N_total *= ds.size;
    }
    plan.K_total = 1;
    for (auto const &ds : plan.k_dims_in_a) {
        plan.K_total *= ds.size;
    }

    // --- Compute batch total ---
    plan.batch_total = 1;
    for (auto const &bds : plan.batch_dims) {
        plan.batch_total *= bds.size;
    }

    plan.valid = true;
    EINSUMS_LOG_TRACE("compute_packing_topology: valid plan M={}, N={}, K={} ({} K dims, {} batch dims, batch_total={})", plan.M_total,
                      plan.N_total, plan.K_total, plan.k_dims_in_a.size(), plan.batch_dims.size(), plan.batch_total);
    return plan;
}

// ---------------------------------------------------------------------------
// HPTT transpose wrappers with plan caching
// ---------------------------------------------------------------------------

#if !defined(EINSUMS_WINDOWS)

namespace {

struct HpttPlanKey {
    std::vector<int>    perm;
    std::vector<size_t> sizes;
    int                 num_threads;

    bool operator==(HpttPlanKey const &o) const { return perm == o.perm && sizes == o.sizes && num_threads == o.num_threads; }
};

template <typename T>
struct HpttPlanCache {
    static constexpr int                                    kSlots = 2;
    std::array<HpttPlanKey, kSlots>                         keys{};
    std::array<std::shared_ptr<hptt::Transpose<T>>, kSlots> plans{};
    int                                                     lru{0};

    std::shared_ptr<hptt::Transpose<T>> const &get(int const *perm, int rank, T const *src, size_t const *sizes, T *dst, int num_threads) {
        HpttPlanKey candidate;
        candidate.perm.assign(perm, perm + rank);
        candidate.sizes.assign(sizes, sizes + rank);
        candidate.num_threads = num_threads;

        for (int s = 0; s < kSlots; ++s) {
            if (plans[s] && keys[s] == candidate) {
                plans[s]->set_input_ptr(src);
                plans[s]->set_output_ptr(dst);
                lru = 1 - s;
                return plans[s];
            }
        }

        int const evict = lru;
        plans[evict] =
            hptt::create_plan(perm, rank, T{1}, src, sizes, nullptr, T{0}, dst, nullptr, hptt::ESTIMATE, num_threads, nullptr, false);
        keys[evict] = std::move(candidate);
        lru         = 1 - evict;
        return plans[evict];
    }
};

} // anonymous namespace

void hptt_transpose(int const *perm, int rank, float const *src, size_t const *sizes, float *dst, int num_threads, bool /*conj*/) {
    static thread_local HpttPlanCache<float> cache;
    cache.get(perm, rank, src, sizes, dst, num_threads)->execute();
}

void hptt_transpose(int const *perm, int rank, double const *src, size_t const *sizes, double *dst, int num_threads, bool /*conj*/) {
    static thread_local HpttPlanCache<double> cache;
    cache.get(perm, rank, src, sizes, dst, num_threads)->execute();
}

void hptt_transpose(int const *perm, int rank, std::complex<float> const *src, size_t const *sizes, std::complex<float> *dst,
                    int num_threads, bool conj) {
    static thread_local HpttPlanCache<std::complex<float>> cache;
    auto const                                            &p = cache.get(perm, rank, src, sizes, dst, num_threads);
    p->set_conj_a(conj);
    p->execute();
}

void hptt_transpose(int const *perm, int rank, std::complex<double> const *src, size_t const *sizes, std::complex<double> *dst,
                    int num_threads, bool conj) {
    static thread_local HpttPlanCache<std::complex<double>> cache;
    auto const                                             &p = cache.get(perm, rank, src, sizes, dst, num_threads);
    p->set_conj_a(conj);
    p->execute();
}

#endif // !EINSUMS_WINDOWS

} // namespace einsums::packed_gemm
