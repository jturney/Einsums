//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// J and K builds timed separately, one row per implementation strategy:
//
//   J(mu,nu) =  2 sum_{lam,sig} (mu nu|lam sig) D(lam,sig)
//   K(mu,nu) = -1 sum_{lam,sig} (mu lam|nu sig) D(lam,sig)
//
//   c_loops          four nested loops per matrix, serial
//   c_omp_loops      the same nests, OpenMP over the (mu,nu) output elements
//   blas_loop        J as one GEMV over the TEI, K as the serial loop nest
//   blas_permute     J as one GEMV, K as a hand-written sort of the TEI into
//                    (mu nu|lam sig) order followed by one GEMV
//   einsum           two templated einsum calls, automatic dispatch
//   einsum_permute   templated einsum for J, an HPTT permute of the TEI plus
//                    templated einsum for K
//   string_eager     two string-spec einsum calls outside capture (the engine
//                    the graph, RuntimeTensor and Python use)
//   graph            J and K each captured into its own graph, optimized
//   graph_fused      J and K captured into ONE graph and optimized, so the
//                    default pipeline may fuse them; only the total is
//                    separable, reported as the J column with K = 0
//
// Every strategy fills the same symmetric operands (8-fold permutational
// symmetry for the TEI, a symmetric density), so its layout cannot change the
// answer, and every J and K is checked against the c_loops result before a
// time is printed. Times are wall clock, the median of the trials after one
// untimed warm-up call.
//
// Usage: profile_jk_breakdown -n <norbs> -t <trials> [-c]
//   -c prints a header line and one CSV line per strategy:
//   strategy,j_ms,k_ms

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

// Deterministic operand values, reproduced exactly by profile_fort_loop.f03
// so the Fortran row is checked against the same answer. The TEI value
// depends only on the canonical pair-of-pairs index, which gives it the
// 8-fold symmetry of real two-electron integrals.
std::int64_t pair_index(std::int64_t a, std::int64_t b) {
    std::int64_t const hi = std::max(a, b), lo = std::min(a, b);
    return hi * (hi + 1) / 2 + lo;
}

double tei_value(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    std::int64_t const idx = pair_index(pair_index(a, b), pair_index(c, d));
    return static_cast<double>((idx * 2654435761LL) % 1000003LL) / 500001.5 - 1.0;
}

double density_value(std::int64_t a, std::int64_t b) {
    return static_cast<double>((pair_index(a, b) * 40503LL) % 10007LL) / 5003.5 - 1.0;
}

// Column-major, matching Tensor: element (a,b,c,d) at a + n b + n^2 c + n^3 d.
void fill(Tensor<double, 4> &TEI, Tensor<double, 2> &D) {
    auto const n = static_cast<std::int64_t>(D.dim(0));
    double    *t = TEI.data();
#pragma omp parallel for
    for (std::int64_t d = 0; d < n; d++) {
        for (std::int64_t c = 0; c < n; c++) {
            for (std::int64_t b = 0; b < n; b++) {
                for (std::int64_t a = 0; a < n; a++) {
                    t[a + n * (b + n * (c + n * d))] = tei_value(a, b, c, d);
                }
            }
        }
    }
    for (std::int64_t b = 0; b < n; b++) {
        for (std::int64_t a = 0; a < n; a++) {
            D.data()[a + n * b] = density_value(a, b);
        }
    }
}

// The original figure's loop nests, written C-style: the buffer read as
// T[i][j][k][l] with l fastest. The operand symmetry makes that the same
// tensor as the column-major one the other strategies read.
void        loop_j(double *J, double const *D, double const *T, std::int64_t n, bool parallel) {
#pragma omp parallel for collapse(2) if (parallel)
    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (std::int64_t k = 0; k < n; k++) {
                for (std::int64_t l = 0; l < n; l++) {
                    sum += D[k * n + l] * T[n * (n * (n * i + j) + k) + l];
                }
            }
            J[i * n + j] = 2.0 * sum;
        }
    }
}

void        loop_k(double *K, double const *D, double const *T, std::int64_t n, bool parallel) {
#pragma omp parallel for collapse(2) if (parallel)
    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (std::int64_t k = 0; k < n; k++) {
                for (std::int64_t l = 0; l < n; l++) {
                    sum += D[k * n + l] * T[n * (n * (n * i + k) + j) + l];
                }
            }
            K[i * n + j] = -sum;
        }
    }
}

void blas_j(double *J, double const *D, double const *T, std::int64_t n) {
    auto const m = static_cast<blas::int_t>(n * n);
    blas::gemv('N', m, m, 2.0, T, m, D, 1, 0.0, J, 1);
}

// The sort the original BLAS+permute row timed: a plain serial loop that
// writes (mu nu|lam sig) from (mu lam|nu sig), then one GEMV.
void blas_permute_k(double *K, double const *D, double const *T, double *sorted, std::int64_t n) {
    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < n; j++) {
            for (std::int64_t k = 0; k < n; k++) {
                for (std::int64_t l = 0; l < n; l++) {
                    sorted[n * (n * (n * i + j) + k) + l] = T[n * (n * (n * i + k) + j) + l];
                }
            }
        }
    }
    auto const m = static_cast<blas::int_t>(n * n);
    blas::gemv('N', m, m, -1.0, sorted, m, D, 1, 0.0, K, 1);
}

double median_ms(std::function<void()> const &f, int trials) {
    f(); // warm-up: first touch, plan caches, a graph's first-replay setup
    std::vector<double> t(trials);
    for (int r = 0; r < trials; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        f();
        auto const t1 = std::chrono::steady_clock::now();
        t[r]          = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(t.begin(), t.end());
    return trials % 2 ? t[trials / 2] : 0.5 * (t[trials / 2 - 1] + t[trials / 2]);
}

double max_rel_diff(double const *x, double const *ref, std::int64_t count) {
    double num = 0.0, den = 0.0;
    for (std::int64_t i = 0; i < count; i++) {
        num = std::max(num, std::abs(x[i] - ref[i]));
        den = std::max(den, std::abs(ref[i]));
    }
    return den > 0.0 ? num / den : num;
}

int einsums_main(int argc, char **argv) {
    int  n = 100, trials = 20;
    bool csv = false;
    for (int a = 1; a < argc; a++) {
        if (std::strcmp(argv[a], "-n") == 0 && a + 1 < argc) {
            n = std::atoi(argv[++a]);
        } else if (std::strcmp(argv[a], "-t") == 0 && a + 1 < argc) {
            trials = std::atoi(argv[++a]);
        } else if (std::strcmp(argv[a], "-c") == 0) {
            csv = true;
        }
    }
    std::int64_t const nn = n, n2 = nn * nn;

    Tensor<double, 4> TEI("TEI", n, n, n, n), sorted("sorted TEI", n, n, n, n);
    Tensor<double, 2> D("D", n, n), J("J", n, n), K("K", n, n);
    fill(TEI, D);

    // Reference: the serial loops.
    std::vector<double> J_ref(n2), K_ref(n2);
    loop_j(J_ref.data(), D.data(), TEI.data(), nn, false);
    loop_k(K_ref.data(), D.data(), TEI.data(), nn, false);

    struct Row {
        std::string name;
        double      j_ms, k_ms;
    };
    std::vector<Row> rows;
    bool             all_ok = true;

    auto check = [&](char const *name, char const *which, double const *x, std::vector<double> const &ref) {
        double const err = max_rel_diff(x, ref.data(), n2);
        if (!(err < 1e-10)) {
            std::fprintf(stderr, "MISMATCH %s %s: max relative difference %.3e\n", name, which, err);
            all_ok = false;
        }
    };

    auto run = [&](char const *name, std::function<void()> const &fj, std::function<void()> const &fk, double const *j_out,
                   double const *k_out) {
        std::memset(J.data(), 0, n2 * sizeof(double));
        std::memset(K.data(), 0, n2 * sizeof(double));
        double const tj = median_ms(fj, trials);
        double const tk = median_ms(fk, trials);
        check(name, "J", j_out, J_ref);
        check(name, "K", k_out, K_ref);
        rows.push_back({name, tj, tk});
    };

    double       *j = J.data(), *k = K.data();
    double const *d = D.data(), *t = TEI.data();

    run("c_loops", [&] { loop_j(j, d, t, nn, false); }, [&] { loop_k(k, d, t, nn, false); }, j, k);
    run("c_omp_loops", [&] { loop_j(j, d, t, nn, true); }, [&] { loop_k(k, d, t, nn, true); }, j, k);
    run("blas_loop", [&] { blas_j(j, d, t, nn); }, [&] { loop_k(k, d, t, nn, false); }, j, k);
    run("blas_permute", [&] { blas_j(j, d, t, nn); }, [&] { blas_permute_k(k, d, t, sorted.data(), nn); }, j, k);

    run(
        "einsum", [&] { einsum(0.0, Indices{mu, nu}, &J, 2.0, Indices{mu, nu, lambda, sigma}, TEI, Indices{lambda, sigma}, D); },
        [&] { einsum(0.0, Indices{mu, nu}, &K, -1.0, Indices{mu, lambda, nu, sigma}, TEI, Indices{lambda, sigma}, D); }, j, k);
    run(
        "einsum_permute", [&] { einsum(0.0, Indices{mu, nu}, &J, 2.0, Indices{mu, nu, lambda, sigma}, TEI, Indices{lambda, sigma}, D); },
        [&] {
            permute(0.0, Indices{mu, nu, lambda, sigma}, &sorted, 1.0, Indices{mu, lambda, nu, sigma}, TEI);
            einsum(0.0, Indices{mu, nu}, &K, -1.0, Indices{mu, nu, lambda, sigma}, sorted, Indices{lambda, sigma}, D);
        },
        j, k);

    // The string engine and the graph read RuntimeTensors.
    RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<double> J_rt("J", std::vector<size_t>{size_t(n), size_t(n)}), K_rt("K", std::vector<size_t>{size_t(n), size_t(n)});
    std::string           route_j, route_k;

    run(
        "string_eager",
        [&] {
            cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
            route_j = cg::dispatch::last_dispatch_route();
        },
        [&] {
            cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
            route_k = cg::dispatch::last_dispatch_route();
        },
        J_rt.data(), K_rt.data());

    cg::Graph g_j("J"), g_k("K");
    {
        cg::CaptureGuard const guard(g_j);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
    }
    {
        cg::CaptureGuard const guard(g_k);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
    }
    g_j.optimize();
    g_k.optimize();
    run("graph", [&] { g_j.execute(); }, [&] { g_k.execute(); }, J_rt.data(), K_rt.data());

    cg::Graph g_jk("JK");
    {
        cg::CaptureGuard const guard(g_jk);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
    }
    g_jk.optimize();
    run("graph_fused", [&] { g_jk.execute(); }, [] {}, J_rt.data(), K_rt.data());
    rows.back().k_ms = 0.0;

    if (csv) {
        std::printf("strategy,j_ms,k_ms\n");
        for (auto const &r : rows) {
            std::printf("%s,%.4f,%.4f\n", r.name.c_str(), r.j_ms, r.k_ms);
        }
    } else {
        std::printf("n=%d, %d trials, median wall time\n", n, trials);
        for (auto const &r : rows) {
            std::printf("  %-16s J %9.2f ms   K %9.2f ms   total %9.2f ms\n", r.name.c_str(), r.j_ms, r.k_ms, r.j_ms + r.k_ms);
        }
    }
    std::fprintf(stderr, "string routes: J %s, K %s; fused graph nodes: %zu\n", route_j.c_str(), route_k.c_str(), g_jk.num_nodes());

    // The same weighted sum profile_fort_loop prints, so its row can be checked.
    auto checksum = [&](std::vector<double> const &x) {
        double s = 0.0;
        for (std::int64_t b = 0; b < nn; b++) {
            for (std::int64_t a = 0; a < nn; a++) {
                s += x[a + nn * b] * static_cast<double>(1 + a + 3 * b);
            }
        }
        return s;
    };
    std::fprintf(stderr, "checksum,%.15e,%.15e\n", checksum(J_ref), checksum(K_ref));

    einsums::finalize();
    return all_ok ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
    einsums::initialize(argc, argv);
    return einsums_main(argc, argv);
}
