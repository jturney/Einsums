//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// What would a batched PackedGemm buy?
//
// Tiled expansion turns one tiled contraction into many small dense ones of the
// SAME topology and shape. Each currently goes through try_packed_gemm on its
// own, and profiling a CCSD tile call showed ~85% of the time inside
// try_packed_gemm is setup rather than blis_contraction: building the
// ContractionSpec, hashing a ContractionKey, the plan-cache lookup, then
// fill_strides + sort_k_dims_for_packing + coalesce_plan, which all run again on
// every call even when the plan cache hits.
//
// None of that depends on the operand POINTERS -- blis_contraction takes the plan
// and the tensors separately, and the plan holds no data pointers. So a group of
// same-shape contractions could share one plan. This measures the ceiling of that
// idea by hoisting the setup by hand: prepare once, then loop blis_contraction
// over N operand triples.
//
// It is a measurement, not the feature. If the gap is small, a
// BatchedPackedGemm node is not worth building.

#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace pg = einsums::packed_gemm;

namespace {

RuntimeTensor<double> make(std::string name, std::vector<size_t> const &dims, double salt) {
    RuntimeTensor<double> t(std::move(name), dims);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = salt + 0.001 * static_cast<double>(i % 89);
    }
    return t;
}

pg::ContractionSpec make_spec(std::vector<std::string> c, std::vector<std::string> a, std::vector<std::string> b) {
    pg::ContractionSpec s;
    s.c_indices   = std::move(c);
    s.a_indices   = std::move(a);
    s.b_indices   = std::move(b);
    s.scalar_type = pg::ScalarType::Float64;
    return s;
}

/// Best of several trials: the mean is dominated by scheduler noise whenever
/// anything else is running on the machine.
template <typename F>
double best_of(int trials, size_t reps, F &&fn) {
    for (size_t i = 0; i < reps; ++i) {
        fn();
    }
    double best = 1e300;
    for (int t = 0; t < trials; ++t) {
        auto const t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < reps; ++i) {
            fn();
        }
        auto const t1 = std::chrono::steady_clock::now();
        best          = std::min(best, std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(reps));
    }
    return best;
}

} // namespace

TEST_CASE("BatchedPackedGemm ceiling - CCSD ladder tile", "[performance][packed_gemm][batched]") {
    // The particle-particle ladder, per tile: C[i,j,a,b] += A[i,j,e,f] B[e,f,a,b]
    // with an occupied block of 2 and a virtual block of 4 -- the shape tiled
    // expansion emits for a symmetry-blocked CCSD.
    size_t const NT   = 64; // tiles in one group
    auto const   spec = make_spec({"i", "j", "a", "b"}, {"i", "j", "e", "f"}, {"e", "f", "a", "b"});

    std::vector<RuntimeTensor<double>> As, Bs, Cs, Cs2;
    As.reserve(NT);
    Bs.reserve(NT);
    Cs.reserve(NT);
    Cs2.reserve(NT);
    for (size_t n = 0; n < NT; ++n) {
        As.push_back(make("A" + std::to_string(n), {2, 2, 4, 4}, 1.0 + 0.01 * static_cast<double>(n)));
        Bs.push_back(make("B" + std::to_string(n), {4, 4, 4, 4}, 2.0 + 0.01 * static_cast<double>(n)));
        Cs.push_back(make("C" + std::to_string(n), {2, 2, 4, 4}, 0.0));
        Cs2.push_back(make("D" + std::to_string(n), {2, 2, 4, 4}, 0.0));
    }

    // ── Baseline: what expansion produces today, one full call per tile ──────
    auto const per_call = [&] {
        for (size_t n = 0; n < NT; ++n) {
            pg::try_packed_gemm(spec, 1.0, &Cs[n], 1.0, As[n], Bs[n]);
        }
    };

    // ── Prototype: prepare once, then only blis_contraction per tile ─────────
    // Every tile in the group has identical dims and strides, so one plan serves
    // all of them. That is the property a real BatchedPackedGemm would key on.
    pg::ContractionSpec prepared = spec;
    prepared.target_indices      = pg::unique_ordered(prepared.c_indices);
    prepared.link_indices        = pg::compute_link_indices(prepared.a_indices, prepared.b_indices, prepared.target_indices);

    pg::ContractionKey key;
    key.spec   = prepared;
    key.a_desc = pg::tensor_descriptor(As[0]);
    key.b_desc = pg::tensor_descriptor(Bs[0]);
    key.c_desc = pg::tensor_descriptor(Cs2[0]);
    key.target_dims.resize(prepared.target_indices.size());
    for (size_t ti = 0; ti < prepared.target_indices.size(); ++ti) {
        auto const &nm = prepared.target_indices[ti];
        for (size_t ci = 0; ci < prepared.c_indices.size(); ++ci) {
            if (prepared.c_indices[ci] == nm) {
                key.target_dims[ti] = static_cast<int64_t>(Cs2[0].dim(ci));
                break;
            }
        }
    }
    key.link_dims.resize(prepared.link_indices.size());
    for (size_t li = 0; li < prepared.link_indices.size(); ++li) {
        auto const &nm = prepared.link_indices[li];
        for (size_t ai = 0; ai < prepared.a_indices.size(); ++ai) {
            if (prepared.a_indices[ai] == nm) {
                key.link_dims[li] = static_cast<int64_t>(As[0].dim(ai));
                break;
            }
        }
    }

    pg::PackingPlan plan = pg::compute_packing_topology(key);
    REQUIRE(plan.valid);
    pg::fill_strides(plan, As[0], Bs[0], Cs2[0]);
    pg::sort_k_dims_for_packing(plan);
    pg::coalesce_plan(plan);

    auto const batched = [&] {
        for (size_t n = 0; n < NT; ++n) {
            pg::blis_contraction<double>(plan, Cs2[n], As[n], Bs[n], 1.0, 1.0);
        }
    };

    // Same answer both ways, or the comparison is meaningless.
    per_call();
    batched();
    for (size_t n = 0; n < NT; ++n) {
        REQUIRE(Cs[n].size() == Cs2[n].size());
        for (size_t i = 0; i < Cs[n].size(); ++i) {
            REQUIRE_THAT(Cs2[n].data()[i], Catch::Matchers::WithinRel(Cs[n].data()[i], 1e-12));
        }
    }

    double const t_per_call = best_of(7, 20, per_call);
    double const t_batched  = best_of(7, 20, batched);

    // 2*M*N*K per tile with M=(i,j)=4, N=(a,b)=16, K=(e,f)=16.
    double const flops = static_cast<double>(NT) * 2.0 * 4.0 * 16.0 * 16.0;

    WARN(fmt::format("\n  {} tiles, one full try_packed_gemm each : {:8.1f} us  ({:6.2f} GFLOP/s, {:5.2f} us/tile)"
                     "\n  {} tiles, one shared plan               : {:8.1f} us  ({:6.2f} GFLOP/s, {:5.2f} us/tile)"
                     "\n  setup amortized away                    : {:5.1f}x",
                     NT, t_per_call * 1e6, flops / t_per_call / 1e9, t_per_call * 1e6 / static_cast<double>(NT), NT, t_batched * 1e6,
                     flops / t_batched / 1e9, t_batched * 1e6 / static_cast<double>(NT), t_per_call / t_batched));
}

// What an OpenMP parallel region costs, against what a small contraction is
// worth. blis_contraction parallelizes the NC loop for any contraction without a
// batch dimension (EinsumPackedGemm.hpp:753), shrinking NC_blk so every thread
// gets a block -- correct for tall N, ruinous for a tile whose whole contraction
// is a couple of KFLOP. This measures the region cost so a work threshold can be
// derived from it rather than guessed, and so the crossover can be re-measured on
// other hardware, where both the barrier cost and the achievable rate differ.
TEST_CASE("OpenMP region cost vs small-contraction work", "[performance][packed_gemm][threading]") {
    int const maxt = omp_get_max_threads();

    auto bench = [](int trials, size_t reps, auto &&fn) {
        for (size_t i = 0; i < reps; ++i)
            fn();
        double best = 1e300;
        for (int t = 0; t < trials; ++t) {
            auto const t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < reps; ++i)
                fn();
            auto const t1 = std::chrono::steady_clock::now();
            best          = std::min(best, std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(reps));
        }
        return best;
    };

    std::string out = fmt::format("\n  omp_get_max_threads = {}\n", maxt);

    // Cost of entering/leaving a parallel region that does nothing, warm team.
    int volatile sink = 0;
    for (int nt : {1, 2, 4, 8}) {
        if (nt > maxt)
            continue;
        double const t = bench(7, 20000, [&] {
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int i = 0; i < nt; ++i) {
                sink = i;
            }
        });
        out += fmt::format("  empty parallel region, {} threads : {:8.3f} us\n", nt, t * 1e6);
    }

    // Same, but the team sleeps in between (a serial gap), which is what an
    // interleaved graph replay does.
    for (int nt : {8}) {
        if (nt > maxt)
            continue;
        double const t = bench(7, 2000, [&] {
            for (int spin = 0; spin < 20000; ++spin) {
                sink = spin;
            }
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int i = 0; i < nt; ++i) {
                sink = i;
            }
        });
        double const gap = bench(7, 2000, [&] {
            for (int spin = 0; spin < 20000; ++spin) {
                sink = spin;
            }
        });
        out +=
            fmt::format("  region after a serial gap, {} threads : {:8.3f} us  (gap alone {:8.3f} us)\n", nt, (t - gap) * 1e6, gap * 1e6);
    }
    // Break-even: a region only pays for itself once the work it distributes takes
    // longer than entering it. At ~20 GFLOP/s achievable on one core, a 20 us
    // region needs ~400 KFLOP of work to break even -- three orders of magnitude
    // more than a CCSD tile contraction (~2 KFLOP).
    WARN(out);
}
