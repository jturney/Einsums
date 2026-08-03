//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TriplesBlocking.cpp
/// @brief The CCSD(T) energy two ways: the materialized rank-6 spelling, and the
///        production-shaped blocked loop that a blocking pass would have to emit.
///
/// This is the reference form for a future "block a contraction region over a
/// subset of its indices" optimization pass. The pass would consume the first
/// graph below and produce the second, so the second is written by hand here to
/// pin down what it must emit and to show that the machinery already supports it.
///
/// The two spellings compute
///
///     Xc[ijkabc] = sum_e t2[jkae] <ei||bc> - sum_m t2[imbc] <ma||jk>
///     Xd[ijkabc] = t1[ia] <jk||bc>
///     W = P(i/jk) P(a/bc) Xc          V = P(i/jk) P(a/bc) Xd
///     E(T) = 1/36 sum_{ijkabc} W (W + V) / D
///
/// and must agree to round-off. See examples/toy/ccsd_t_spinorbital_toy.py for
/// the same comparison in Python (validated there against a numpy oracle, and
/// the equations against psi4's CCSD(T)).
///
/// ── what the blocked form does differently ──────────────────────────────────
///
/// It never builds a rank-6 tensor. It loops over occupied triples i < j < k and
/// keeps one v^3 block, which is what psi4's cc/cctriples/ET_RHF.cc does and the
/// only thing that fits: at o=50, v=500 the rank-6 tensor is 1.25e16 bytes.
///
/// Two consequences show up in the body below. P(i/jk) stops being a permutation
/// and becomes three evaluations of the producer at permuted loop parameters -
/// that costs 3x the contraction flops, and is repaid many times over by dropping
/// to i < j < k, which is legal because W and V are fully antisymmetric in ijk.
/// Only P(a/bc) survives as an actual permute, on a block that fits in cache.
///
/// ── the machinery it leans on ───────────────────────────────────────────────
///
/// ViewAxis bounds are BoundExpr values that may name a parameter, and the View
/// executor re-resolves them and rebuilds the TensorView on every iteration. So
/// the loop body is written ONCE, with views like ``t2[j, k, :, :]`` spelled as
/// ``ViewAxis::drop("j"), ViewAxis::drop("k"), full(), full()``, and three
/// write_param nodes at the top of the body advance i, j, k per iteration. The
/// body is a fixed 43 nodes no matter how many triples run; unrolling would be
/// about 20 nodes per triple, or 400k nodes at o=50.
///
/// The dependency between a write_param and the views that resolve against it
/// lives in the ParamTable and touches no tensor, which every TensorId-keyed
/// scheduler and pass in the pipeline used to miss. See
/// tests/unit/ParametricLoop.cpp for the regression tests, and param_writes /
/// param_reads / has_runtime_view_bounds in Node.hpp for the analysis hooks.
///
/// ── measured ────────────────────────────────────────────────────────────────
///
///     o, v      whole-tensor         blocked loop
///     6, 10      3.0 ms   10 MB      1.1 ms   0.05 MB
///     10, 16    33.1 ms  197 MB     12.2 ms   0.20 MB
///     12, 20   108.8 ms  664 MB     45.3 ms   0.38 MB
///
/// Consistently ~2.5x faster in ~1/1700th the scratch, on top of being the only
/// spelling that scales at all. Optimizing the blocked graph takes 0.2 ms
/// against 14 ms for the equivalent unrolled one, since the pipeline sees one
/// body rather than one copy per triple.
///
/// Set EINSUMS_TRIPLES_OCC / EINSUMS_TRIPLES_VIRT (spatial orbital counts) to run
/// a larger system than the small default this registers with ctest as, and
/// EINSUMS_TRIPLES_NOPASS / EINSUMS_TRIPLES_VERBOSE to skip or narrate the passes.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace cg = einsums::compute_graph;

using einsums::println;
using einsums::RuntimeTensor;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

size_t env_size(char const *name, size_t fallback) {
    char const *raw = std::getenv(name);
    if (raw == nullptr) {
        return fallback;
    }
    auto const value = std::strtoul(raw, nullptr, 10);
    return value == 0 ? fallback : static_cast<size_t>(value);
}

/// The toy system: random integrals carrying the physical symmetries, so the
/// antisymmetry of W in ijk (which the i < j < k restriction rests on) holds.
struct System {
    size_t no{0}, nv{0};

    RuntimeTensor<double> t1{"t1", {1UL, 1UL}};
    RuntimeTensor<double> t2{"t2", {1UL, 1UL, 1UL, 1UL}};
    RuntimeTensor<double> oovv{"oovv", {1UL, 1UL, 1UL, 1UL}};
    RuntimeTensor<double> vovv{"vovv", {1UL, 1UL, 1UL, 1UL}};
    RuntimeTensor<double> ovoo{"ovoo", {1UL, 1UL, 1UL, 1UL}};

    std::vector<double> eo, ev;
};

System build_system(size_t nspatial_occ, size_t nspatial_vir) {
    System       sys;
    size_t const nsp = nspatial_occ + nspatial_vir;
    sys.no           = 2 * nspatial_occ;
    sys.nv           = 2 * nspatial_vir;
    size_t const nso = sys.no + sys.nv;

    std::mt19937                           rng(2026);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double>       normal(0.0, 1.0);

    // Chemist-like spatial integrals with the 8-fold symmetry, then physicist
    // <pq|rs> = (pr|qs), spin-blocked and antisymmetrized into <PQ||RS>.
    std::vector<double> ao(nsp * nsp * nsp * nsp);
    auto const          at = [&](size_t p, size_t q, size_t r, size_t s) { return ((p * nsp + q) * nsp + r) * nsp + s; };
    for (auto &x : ao) {
        x = normal(rng);
    }
    std::vector<double> sym(ao.size());
    for (size_t p = 0; p < nsp; ++p) {
        for (size_t q = 0; q < nsp; ++q) {
            for (size_t r = 0; r < nsp; ++r) {
                for (size_t s = 0; s < nsp; ++s) {
                    sym[at(p, q, r, s)] = 0.02 * (ao[at(p, q, r, s)] + ao[at(q, p, r, s)] + ao[at(p, q, s, r)] + ao[at(q, p, s, r)] +
                                                  ao[at(r, s, p, q)] + ao[at(s, r, p, q)] + ao[at(r, s, q, p)] + ao[at(s, r, q, p)]);
                }
            }
        }
    }
    // <PQ||RS> over spin orbitals, interleaved alpha/beta.
    std::vector<double> g(nso * nso * nso * nso, 0.0);
    auto const          gt = [&](size_t p, size_t q, size_t r, size_t s) { return ((p * nso + q) * nso + r) * nso + s; };
    for (size_t P = 0; P < nso; ++P) {
        for (size_t Q = 0; Q < nso; ++Q) {
            for (size_t R = 0; R < nso; ++R) {
                for (size_t S = 0; S < nso; ++S) {
                    double const dir  = (P % 2 == R % 2 && Q % 2 == S % 2) ? sym[at(P / 2, R / 2, Q / 2, S / 2)] : 0.0;
                    double const exc  = (P % 2 == S % 2 && Q % 2 == R % 2) ? sym[at(P / 2, S / 2, Q / 2, R / 2)] : 0.0;
                    g[gt(P, Q, R, S)] = dir - exc;
                }
            }
        }
    }

    std::vector<double> eps_sp(nsp);
    for (size_t p = 0; p < nspatial_occ; ++p) {
        eps_sp[p] = -(4.0 + unit(rng));
    }
    for (size_t p = nspatial_occ; p < nsp; ++p) {
        eps_sp[p] = 4.0 + unit(rng);
    }
    sys.eo.resize(sys.no);
    sys.ev.resize(sys.nv);
    for (size_t p = 0; p < sys.no; ++p) {
        sys.eo[p] = eps_sp[p / 2];
    }
    for (size_t p = 0; p < sys.nv; ++p) {
        sys.ev[p] = eps_sp[(sys.no + p) / 2];
    }

    size_t const o = sys.no, v = sys.nv;
    sys.t1   = RuntimeTensor<double>("t1", {o, v});
    sys.t2   = RuntimeTensor<double>("t2", {o, o, v, v});
    sys.oovv = RuntimeTensor<double>("oovv", {o, o, v, v});
    sys.vovv = RuntimeTensor<double>("vovv", {v, o, v, v});
    sys.ovoo = RuntimeTensor<double>("ovoo", {o, v, o, o});

    for (size_t i = 0; i < o; ++i) {
        for (size_t a = 0; a < v; ++a) {
            sys.t1(i, a) = 0.01 * normal(rng);
        }
    }
    // t2 from the MP1 amplitudes: antisymmetric in ij and in ab by construction,
    // which is what the blocked form's i < j < k restriction needs.
    for (size_t i = 0; i < o; ++i) {
        for (size_t j = 0; j < o; ++j) {
            for (size_t a = 0; a < v; ++a) {
                for (size_t b = 0; b < v; ++b) {
                    double const num     = g[gt(i, j, o + a, o + b)];
                    sys.oovv(i, j, a, b) = num;
                    sys.t2(i, j, a, b)   = num / (sys.eo[i] + sys.eo[j] - sys.ev[a] - sys.ev[b]);
                }
            }
        }
    }
    for (size_t e = 0; e < v; ++e) {
        for (size_t i = 0; i < o; ++i) {
            for (size_t b = 0; b < v; ++b) {
                for (size_t c = 0; c < v; ++c) {
                    sys.vovv(e, i, b, c) = g[gt(o + e, i, o + b, o + c)];
                }
            }
        }
    }
    for (size_t m = 0; m < o; ++m) {
        for (size_t a = 0; a < v; ++a) {
            for (size_t j = 0; j < o; ++j) {
                for (size_t k = 0; k < o; ++k) {
                    sys.ovoo(m, a, j, k) = g[gt(m, o + a, j, k)];
                }
            }
        }
    }
    return sys;
}

/// Brute-force E(T) with explicit loops - slow, obviously correct, and the
/// arbiter when the two graph spellings disagree.
double reference_energy(System const &sys) {
    size_t const o = sys.no, v = sys.nv;
    auto const   x6 = [&](size_t i, size_t j, size_t k, size_t a, size_t b, size_t c) {
        return ((((i * o + j) * o + k) * v + a) * v + b) * v + c;
    };
    std::vector<double> Xc(o * o * o * v * v * v, 0.0), Xd(o * o * o * v * v * v, 0.0);
    for (size_t i = 0; i < o; ++i) {
        for (size_t j = 0; j < o; ++j) {
            for (size_t k = 0; k < o; ++k) {
                for (size_t a = 0; a < v; ++a) {
                    for (size_t b = 0; b < v; ++b) {
                        for (size_t c = 0; c < v; ++c) {
                            double acc = 0.0;
                            for (size_t e = 0; e < v; ++e) {
                                acc += sys.t2(j, k, a, e) * sys.vovv(e, i, b, c);
                            }
                            for (size_t m = 0; m < o; ++m) {
                                acc -= sys.t2(i, m, b, c) * sys.ovoo(m, a, j, k);
                            }
                            Xc[x6(i, j, k, a, b, c)] = acc;
                            Xd[x6(i, j, k, a, b, c)] = sys.t1(i, a) * sys.oovv(j, k, b, c);
                        }
                    }
                }
            }
        }
    }
    // P(i/jk) P(a/bc), applied by summing the nine permuted reads.
    static constexpr std::array<std::array<size_t, 6>, 9> perm = {{{0, 1, 2, 3, 4, 5},
                                                                   {1, 0, 2, 3, 4, 5},
                                                                   {2, 1, 0, 3, 4, 5},
                                                                   {0, 1, 2, 4, 3, 5},
                                                                   {0, 1, 2, 5, 4, 3},
                                                                   {1, 0, 2, 4, 3, 5},
                                                                   {1, 0, 2, 5, 4, 3},
                                                                   {2, 1, 0, 4, 3, 5},
                                                                   {2, 1, 0, 5, 4, 3}}};
    static constexpr std::array<double, 9>                sgn  = {1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0};

    double energy = 0.0;
    for (size_t i = 0; i < o; ++i) {
        for (size_t j = 0; j < o; ++j) {
            for (size_t k = 0; k < o; ++k) {
                for (size_t a = 0; a < v; ++a) {
                    for (size_t b = 0; b < v; ++b) {
                        for (size_t c = 0; c < v; ++c) {
                            std::array<size_t, 6> const idx = {i, j, k, a, b, c};
                            double                      W = 0.0, V = 0.0;
                            for (size_t n = 0; n < 9; ++n) {
                                auto const  &p = perm[n];
                                size_t const s = x6(idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]], idx[p[4]], idx[p[5]]);
                                W += sgn[n] * Xc[s];
                                V += sgn[n] * Xd[s];
                            }
                            double const D = sys.eo[i] + sys.eo[j] + sys.eo[k] - sys.ev[a] - sys.ev[b] - sys.ev[c];
                            energy += W * (W + V) / D;
                        }
                    }
                }
            }
        }
    }
    return energy / 36.0;
}

struct Result {
    double energy{0.0};
    size_t nodes{0};
    double optimize_ms{0.0};
    double execute_ms{0.0};
    double peak_mb{0.0};
};

/// The rank-6 spelling: build W and V whole, contract once at the end.
Result whole_tensor(System const &sys) {
    size_t const o = sys.no, v = sys.nv;
    // P(i/jk) P(a/bc): nine (permutation, sign) pairs, each an involution.
    static constexpr std::array<std::string_view, 9> specs = {
        "i,j,k,a,b,c <- i,j,k,a,b,c", "j,i,k,a,b,c <- i,j,k,a,b,c", "k,j,i,a,b,c <- i,j,k,a,b,c",
        "i,j,k,b,a,c <- i,j,k,a,b,c", "i,j,k,c,b,a <- i,j,k,a,b,c", "j,i,k,b,a,c <- i,j,k,a,b,c",
        "j,i,k,c,b,a <- i,j,k,a,b,c", "k,j,i,b,a,c <- i,j,k,a,b,c", "k,j,i,c,b,a <- i,j,k,a,b,c"};
    static constexpr std::array<double, 9> signs = {1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0};

    RuntimeTensor<double> Xc("Xc", {o, o, o, v, v, v}), Xd("Xd", {o, o, o, v, v, v});
    RuntimeTensor<double> W("W", {o, o, o, v, v, v}), V("V", {o, o, o, v, v, v}), Wd("Wd", {o, o, o, v, v, v});
    RuntimeTensor<double> D("D", {o, o, o, v, v, v});
    RuntimeTensor<double> Et("Et", {1UL});

    for (size_t i = 0; i < o; ++i) {
        for (size_t j = 0; j < o; ++j) {
            for (size_t k = 0; k < o; ++k) {
                for (size_t a = 0; a < v; ++a) {
                    for (size_t b = 0; b < v; ++b) {
                        for (size_t c = 0; c < v; ++c) {
                            D(i, j, k, a, b, c) = sys.eo[i] + sys.eo[j] + sys.eo[k] - sys.ev[a] - sys.ev[b] - sys.ev[c];
                        }
                    }
                }
            }
        }
    }

    auto const antisymmetrize = [](RuntimeTensor<double> &dst, RuntimeTensor<double> const &src) {
        cg::axpby(signs[0], src, 0.0, &dst);
        for (size_t n = 1; n < 9; ++n) {
            cg::permute(cg::PermuteFormatString(specs[n]), 1.0, &dst, signs[n], src);
        }
    };

    cg::Graph graph("triples_whole_tensor");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("i,j,k,a,b,c <- j,k,a,e ; e,i,b,c", 0.0, &Xc, 1.0, sys.t2, sys.vovv);
        cg::einsum("i,j,k,a,b,c <- i,m,b,c ; m,a,j,k", 1.0, &Xc, -1.0, sys.t2, sys.ovoo);
        cg::einsum("i,j,k,a,b,c <- i,a ; j,k,b,c", 0.0, &Xd, 1.0, sys.t1, sys.oovv);

        antisymmetrize(W, Xc);
        antisymmetrize(V, Xd);
        cg::direct_division(1.0, W, D, 0.0, &Wd);
        cg::axpby(1.0, W, 1.0, &V); // V := W + V
        // dot_python, not dot: the raw-pointer dot registers the result through
        // get_or_register_scalar, which keys on &data()[0] and so lands on a
        // DIFFERENT tensor id than the scale below reads. The dependency goes
        // invisible and the scale is free to run first.
        cg::dot_python(&Et, Wd, V);
        cg::scale(1.0 / 36.0, &Et);
    }

    Result r;
    auto   pm = cg::PassManager::create_default();
    auto   t0 = Clock::now();
    graph.apply(pm);
    r.optimize_ms = ms_since(t0);
    r.nodes       = graph.num_nodes();

    t0 = Clock::now();
    graph.execute();
    r.execute_ms = ms_since(t0);
    r.energy     = Et(0);
    r.peak_mb    = static_cast<double>(6 * o * o * o * v * v * v * sizeof(double)) / 1e6;
    return r;
}

/// The production spelling: ONE loop body, v^3 blocks, parametric views.
Result blocked_loop(System const &sys) {
    size_t const o = sys.no, v = sys.nv;

    // The i < j < k triples, walked by the loop. The body reads triples[trip]
    // through its three write_param callbacks.
    std::vector<std::array<std::int64_t, 3>> triples;
    for (size_t i = 0; i < o; ++i) {
        for (size_t j = i + 1; j < o; ++j) {
            for (size_t k = j + 1; k < o; ++k) {
                triples.push_back({static_cast<std::int64_t>(i), static_cast<std::int64_t>(j), static_cast<std::int64_t>(k)});
            }
        }
    }

    // Constants. eo3 carries e_i + e_j + e_k with a trailing extent-1 axis so a
    // rank-reducing view of it is a rank-1 tensor the einsum below can sum over
    // (the loop parameters index tensors, they cannot scale a prefactor).
    RuntimeTensor<double> eo3("eo3", {o, o, o, 1UL});
    RuntimeTensor<double> Dv("Dv", {v, v, v});
    RuntimeTensor<double> ones("ones", {v, v, v});
    for (size_t i = 0; i < o; ++i) {
        for (size_t j = 0; j < o; ++j) {
            for (size_t k = 0; k < o; ++k) {
                eo3(i, j, k, 0) = sys.eo[i] + sys.eo[j] + sys.eo[k];
            }
        }
    }
    for (size_t a = 0; a < v; ++a) {
        for (size_t b = 0; b < v; ++b) {
            for (size_t c = 0; c < v; ++c) {
                Dv(a, b, c)   = -(sys.ev[a] + sys.ev[b] + sys.ev[c]);
                ones(a, b, c) = 1.0;
            }
        }
    }

    // The whole working set: six v^3 blocks and two scalars.
    RuntimeTensor<double> Xtot("Xtot", {v, v, v}), Vtot("Vtot", {v, v, v});
    RuntimeTensor<double> W("W", {v, v, v}), V("V", {v, v, v}), Wd("Wd", {v, v, v}), D("D", {v, v, v});
    RuntimeTensor<double> E("E", {1UL}), e_trip("e_trip", {1UL});
    E(0) = 0.0;

    size_t     trip = 0;
    auto const cond = [&](size_t /*iter*/) {
        ++trip;
        return trip < triples.size();
    };

    cg::Graph graph("triples_blocked_loop");
    auto     &body = graph.add_loop("triple", triples.size(), cond);
    {
        cg::CaptureGuard const capture(body);

        // Advance the loop parameters. These run first: the scheduler orders a
        // write_param before every View that names the parameter it writes.
        cg::write_param("i", std::function<std::int64_t()>([&] { return triples[trip][0]; }));
        cg::write_param("j", std::function<std::int64_t()>([&] { return triples[trip][1]; }));
        cg::write_param("k", std::function<std::int64_t()>([&] { return triples[trip][2]; }));

        using cg::ViewAxis;
        auto const full = [] { return ViewAxis::full(); };
        auto const drop = [](char const *p) { return ViewAxis::drop(p); };

        // D[a,b,c] = (e_i + e_j + e_k) - e_a - e_b - e_c
        auto &esum = cg::view_runtime(eo3, {drop("i"), drop("j"), drop("k"), full()});
        cg::axpby(1.0, Dv, 0.0, &D);
        cg::einsum("a,b,c <- x ; a,b,c", 1.0, &D, 1.0, esum, ones);

        // P(i/jk) as three producer evaluations at permuted parameters.
        static constexpr std::array<std::array<char const *, 3>, 3> sigma = {{{{"i", "j", "k"}}, {{"j", "i", "k"}}, {{"k", "j", "i"}}}};
        static constexpr std::array<double, 3>                      sign  = {1.0, -1.0, -1.0};
        for (size_t n = 0; n < 3; ++n) {
            char const *ii = sigma[n][0], *jj = sigma[n][1], *kk = sigma[n][2];
            auto       &t2jk = cg::view_runtime(sys.t2, {drop(jj), drop(kk), full(), full()});
            auto       &gvi  = cg::view_runtime(sys.vovv, {full(), drop(ii), full(), full()});
            auto       &t2i  = cg::view_runtime(sys.t2, {drop(ii), full(), full(), full()});
            auto       &gojk = cg::view_runtime(sys.ovoo, {full(), full(), drop(jj), drop(kk)});
            cg::einsum("a,b,c <- a,e ; e,b,c", n == 0 ? 0.0 : 1.0, &Xtot, sign[n], t2jk, gvi);
            cg::einsum("a,b,c <- m,b,c ; m,a", 1.0, &Xtot, -sign[n], t2i, gojk);

            auto &t1i   = cg::view_runtime(sys.t1, {drop(ii), full()});
            auto &goovv = cg::view_runtime(sys.oovv, {drop(jj), drop(kk), full(), full()});
            cg::einsum("a,b,c <- a ; b,c", n == 0 ? 0.0 : 1.0, &Vtot, sign[n], t1i, goovv);
        }

        // P(a/bc), the only permutation left, on a v^3 block.
        cg::axpby(1.0, Xtot, 0.0, &W);
        cg::permute("b,a,c <- a,b,c", 1.0, &W, -1.0, Xtot);
        cg::permute("c,b,a <- a,b,c", 1.0, &W, -1.0, Xtot);
        cg::axpby(1.0, Vtot, 0.0, &V);
        cg::permute("b,a,c <- a,b,c", 1.0, &V, -1.0, Vtot);
        cg::permute("c,b,a <- a,b,c", 1.0, &V, -1.0, Vtot);

        cg::direct_division(1.0, W, D, 0.0, &Wd);
        cg::axpby(1.0, W, 1.0, &V);            // V := W + V
        cg::dot_python(&e_trip, Wd, V);        // see the note in whole_tensor()
        cg::axpby(1.0 / 6.0, e_trip, 1.0, &E); // 1/36 * 6, the i < j < k orbit
    }

    Result r;
    auto   pm = cg::PassManager::create_default();
    auto   t0 = Clock::now();
    if (std::getenv("EINSUMS_TRIPLES_VERBOSE") != nullptr) {
        pm.set_verbosity(2);
    }
    if (std::getenv("EINSUMS_TRIPLES_NOPASS") == nullptr) {
        graph.apply(pm);
    }
    r.optimize_ms = ms_since(t0);
    r.nodes       = body.num_nodes();

    trip = 0;
    E(0) = 0.0;
    t0   = Clock::now();
    graph.execute();
    r.execute_ms = ms_since(t0);
    r.energy     = E(0);
    r.peak_mb    = static_cast<double>(6 * v * v * v * sizeof(double)) / 1e6;
    return r;
}

} // namespace

int einsums_main() {
    size_t const nocc = env_size("EINSUMS_TRIPLES_OCC", 3);
    size_t const nvir = env_size("EINSUMS_TRIPLES_VIRT", 5);

    System const sys = build_system(nocc, nvir);
    println("spin orbitals: {} occupied, {} virtual   ({} occupied triples with i < j < k)", sys.no, sys.nv,
            sys.no * (sys.no - 1) * (sys.no - 2) / 6);
    println("");

    Result const whole   = whole_tensor(sys);
    Result const blocked = blocked_loop(sys);

    println("{:<16} {:>8} {:>13} {:>13} {:>12}", "spelling", "nodes", "optimize ms", "execute ms", "scratch MB");
    println("{:<16} {:>8} {:>13.2f} {:>13.2f} {:>12.2f}", "whole-tensor", whole.nodes, whole.optimize_ms, whole.execute_ms, whole.peak_mb);
    println("{:<16} {:>8} {:>13.2f} {:>13.2f} {:>12.4f}", "blocked loop", blocked.nodes, blocked.optimize_ms, blocked.execute_ms,
            blocked.peak_mb);
    println("");
    double const reference = reference_energy(sys);
    println("E(T) reference    = {:.14f}", reference);
    println("E(T) whole-tensor = {:.14f}   (error {:.2e})", whole.energy, std::abs(whole.energy - reference));
    println("E(T) blocked loop = {:.14f}   (error {:.2e})", blocked.energy, std::abs(blocked.energy - reference));

    double const tol = 1e-10 * std::max(1.0, std::abs(reference));
    if (std::abs(whole.energy - reference) > tol || std::abs(blocked.energy - reference) > tol) {
        println("MISMATCH against the brute-force reference");
        return 1;
    }
    println("");
    println("The blocked body is {} nodes regardless of system size; unrolling the same", blocked.nodes);
    println("work would be about 20 nodes per triple, so 400k nodes at o = 50.");

    einsums::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
