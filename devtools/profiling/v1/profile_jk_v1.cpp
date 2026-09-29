//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The einsum rows of profile_jk_breakdown, built against an Einsums 1.x install,
// so the J/K figure can show what v2 changed:
//
//   J(mu,nu) =  2 sum_{lam,sig} (mu nu|lam sig) D(lam,sig)
//   K(mu,nu) = -1 sum_{lam,sig} (mu lam|nu sig) D(lam,sig)
//
//   v1_einsum          two templated einsum calls, automatic dispatch
//   v1_einsum_permute  templated einsum for J; an HPTT permute of the TEI into
//                      J's order, then a J-shaped einsum, for K
//
// The operands are the deterministic symmetric values profile_jk_breakdown and
// profile_fort_loop fill, written through each tensor's own strides (1.x
// tensors default to row-major, 2.x to column-major), so every build computes
// the same J and K, and the checksums printed on stderr match theirs. Each row
// is also checked against a serial loop before its time is printed. Times are
// wall clock, the median of the trials after one untimed warm-up call.
//
// Usage: profile_jk_v1 -n <norbs> -t <trials> [-c]
//   -c prints a header line and one CSV line per strategy: strategy,j_ms,k_ms

#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

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

namespace {

// Identical to profile_jk_breakdown.cpp and profile_fort_loop.f03.
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

void fill(Tensor<double, 4> &TEI, Tensor<double, 2> &D) {
    auto const n  = static_cast<std::int64_t>(D.dim(0));
    auto const s0 = static_cast<std::int64_t>(TEI.stride(0)), s1 = static_cast<std::int64_t>(TEI.stride(1));
    auto const s2 = static_cast<std::int64_t>(TEI.stride(2)), s3 = static_cast<std::int64_t>(TEI.stride(3));
    double    *t = TEI.data();
#pragma omp parallel for
    for (std::int64_t d = 0; d < n; d++) {
        for (std::int64_t c = 0; c < n; c++) {
            for (std::int64_t b = 0; b < n; b++) {
                for (std::int64_t a = 0; a < n; a++) {
                    t[a * s0 + b * s1 + c * s2 + d * s3] = tei_value(a, b, c, d);
                }
            }
        }
    }
    for (std::int64_t b = 0; b < n; b++) {
        for (std::int64_t a = 0; a < n; a++) {
            D.data()[a * D.stride(0) + b * D.stride(1)] = density_value(a, b);
        }
    }
}

double median_ms(std::function<void()> const &f, int trials) {
    f(); // warm-up: first touch, plan caches
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

// Element (a, b) of a rank-2 tensor through its strides.
double at(Tensor<double, 2> const &X, std::int64_t a, std::int64_t b) {
    return X.data()[a * X.stride(0) + b * X.stride(1)];
}

} // namespace

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
    std::int64_t const nn = n;

    Tensor<double, 4> TEI("TEI", n, n, n, n), sorted("sorted TEI", n, n, n, n);
    Tensor<double, 2> D("D", n, n), J("J", n, n), K("K", n, n);
    fill(TEI, D);

    // Reference: serial loops through the strides, so layout cannot matter.
    std::vector<double> J_ref(nn * nn), K_ref(nn * nn);
    {
        auto const    s0 = static_cast<std::int64_t>(TEI.stride(0)), s1 = static_cast<std::int64_t>(TEI.stride(1));
        auto const    s2 = static_cast<std::int64_t>(TEI.stride(2)), s3 = static_cast<std::int64_t>(TEI.stride(3));
        double const *t = TEI.data();
#pragma omp parallel for collapse(2)
        for (std::int64_t i = 0; i < nn; i++) {
            for (std::int64_t j = 0; j < nn; j++) {
                double sj = 0.0, sk = 0.0;
                for (std::int64_t l = 0; l < nn; l++) {
                    for (std::int64_t k = 0; k < nn; k++) {
                        double const d = at(D, k, l);
                        sj += d * t[i * s0 + j * s1 + k * s2 + l * s3];
                        sk += d * t[i * s0 + k * s1 + j * s2 + l * s3];
                    }
                }
                J_ref[i + nn * j] = 2.0 * sj;
                K_ref[i + nn * j] = -sk;
            }
        }
    }

    struct Row {
        std::string name;
        double      j_ms, k_ms;
    };
    std::vector<Row> rows;
    bool             all_ok = true;

    auto check = [&](char const *name, char const *which, Tensor<double, 2> const &X, std::vector<double> const &ref) {
        double num = 0.0, den = 0.0;
        for (std::int64_t j = 0; j < nn; j++) {
            for (std::int64_t i = 0; i < nn; i++) {
                num = std::max(num, std::abs(at(X, i, j) - ref[i + nn * j]));
                den = std::max(den, std::abs(ref[i + nn * j]));
            }
        }
        if (!(num <= 1e-10 * den)) {
            std::fprintf(stderr, "MISMATCH %s %s: max relative difference %.3e\n", name, which, den > 0 ? num / den : num);
            all_ok = false;
        }
    };

    auto run = [&](char const *name, std::function<void()> const &fj, std::function<void()> const &fk) {
        J.zero();
        K.zero();
        double const tj = median_ms(fj, trials);
        double const tk = median_ms(fk, trials);
        check(name, "J", J, J_ref);
        check(name, "K", K, K_ref);
        rows.push_back({name, tj, tk});
    };

    auto const build_j = [&] { einsum(0.0, Indices{mu, nu}, &J, 2.0, Indices{mu, nu, lambda, sigma}, TEI, Indices{lambda, sigma}, D); };

    run("v1_einsum", build_j,
        [&] { einsum(0.0, Indices{mu, nu}, &K, -1.0, Indices{mu, lambda, nu, sigma}, TEI, Indices{lambda, sigma}, D); });
    run("v1_einsum_permute", build_j, [&] {
        permute(0.0, Indices{mu, nu, lambda, sigma}, &sorted, 1.0, Indices{mu, lambda, nu, sigma}, TEI);
        einsum(0.0, Indices{mu, nu}, &K, -1.0, Indices{mu, nu, lambda, sigma}, sorted, Indices{lambda, sigma}, D);
    });

    if (csv) {
        std::printf("strategy,j_ms,k_ms\n");
        for (auto const &r : rows) {
            std::printf("%s,%.4f,%.4f\n", r.name.c_str(), r.j_ms, r.k_ms);
        }
    } else {
        std::printf("n=%d, %d trials, median wall time\n", n, trials);
        for (auto const &r : rows) {
            std::printf("  %-18s J %9.2f ms   K %9.2f ms   total %9.2f ms\n", r.name.c_str(), r.j_ms, r.k_ms, r.j_ms + r.k_ms);
        }
    }

    // The same weighted sum the v2 driver and the Fortran program print.
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

    finalize();
    return all_ok ? 0 : 1;
}

int main(int argc, char **argv) {
    return start(einsums_main, argc, argv);
}
