//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Fock-build (G = 2J - K from the TEI and the density) through four execution
// strategies, one line each in the "Why Einsums?" figure:
//
//   forloop  hand-written OpenMP loop nests, index arithmetic by hand
//   eager    the same math as two einsum calls; automatic dispatch picks the
//            engines (a well-ordered GEMV for J, the measured-best route for
//            the scrambled K)
//   lccf     captured ComputeGraph + LinearCombinationContractionFolding:
//            algebraic folding - one contraction over a materialized
//            operand-sized linear combination L = 2*TEI - P(TEI)
//   stream   captured ComputeGraph + StreamContractionFusion: loop fusion -
//            ONE storage-order pass over the TEI feeding both accumulators
//
// Usage: profile_strategies -n <norbs> -t <trials> [-c]
//   -c prints one CSV line: n,forloop_ms,eager_ms,lccf_ms,stream_ms

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

template <typename F>
double best_ms(F &&f, int trials) {
    double best = 1e300;
    for (int r = 0; r < trials; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best    = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

// Hand-written loop nests: the "before notation" baseline, written the way a
// careful C programmer would - J and K fused in one nest, column-major
// element offsets, unit-stride inner loop over mu, parallel over nu so each
// thread owns disjoint G columns. This is strong hand code on purpose: the
// figure's claim is that the notation and the graph passes match and then
// beat it, not that they beat a strawman.
void fock_forloop(Tensor<double, 2> &G, Tensor<double, 4> const &TEI, Tensor<double, 2> const &D) {
    ptrdiff_t const n  = static_cast<ptrdiff_t>(G.dim(0));
    ptrdiff_t const n2 = n * n, n3 = n2 * n;
    double         *g = G.data();
    double const   *t = TEI.data();
    double const   *d = D.data();
#pragma omp parallel for
    for (ptrdiff_t nu = 0; nu < n; nu++) {
        double *gcol = g + nu * n;
        for (ptrdiff_t mu = 0; mu < n; mu++) {
            gcol[mu] = 0.0;
        }
        for (ptrdiff_t sig = 0; sig < n; sig++) {
            for (ptrdiff_t lam = 0; lam < n; lam++) {
                double const  dj     = 2.0 * d[lam + sig * n];
                double const  dk     = d[lam + sig * n];
                double const *tj_col = t + nu * n + lam * n2 + sig * n3; // TEI(:, nu, lam, sig)
                double const *tk_col = t + lam * n + nu * n2 + sig * n3; // TEI(:, lam, nu, sig)
                for (ptrdiff_t mu = 0; mu < n; mu++) {
                    gcol[mu] += dj * tj_col[mu] - dk * tk_col[mu];
                }
            }
        }
    }
}

} // namespace

int einsums_main(int argc, char **argv) {
    int  n = 60, trials = 3;
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

    auto TEI = create_random_tensor<double>("TEI", n, n, n, n);
    auto D   = create_random_tensor<double>("D", n, n);

    // 1. for-loops
    Tensor<double, 2> G_loop("G_loop", n, n);
    auto const        t_forloop = best_ms([&] { fock_forloop(G_loop, TEI, D); }, trials);

    // 2. eager einsum
    Tensor<double, 2> G_eager("G_eager", n, n);
    auto const        t_eager = best_ms(
        [&] {
            einsum(0.0, Indices{mu, nu}, &G_eager, 2.0, Indices{mu, nu, lambda, sigma}, TEI, Indices{lambda, sigma}, D);
            einsum(1.0, Indices{mu, nu}, &G_eager, -1.0, Indices{mu, lambda, nu, sigma}, TEI, Indices{lambda, sigma}, D);
        },
        trials);

    // Captured graphs share operand tensors.
    RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<double> G_rt("G", std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(n)});
    G_rt.zero();

    auto capture = [&](cg::Graph &g) {
        cg::CaptureGuard const guard(g);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &G_rt, 2.0, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 1.0, &G_rt, -1.0, TEI_rt, D_rt);
    };

    // 3. LCCF-folded graph
    cg::Graph g_lccf("lccf");
    capture(g_lccf);
    g_lccf.apply<cg::passes::LinearCombinationContractionFolding>();
    int const  lccf_trials = (n >= 80) ? 2 : trials; // its L build is expensive at large n
    auto const t_lccf      = best_ms([&] { g_lccf.execute(); }, lccf_trials);

    // 4. stream-fused graph
    cg::Graph g_stream("stream");
    capture(g_stream);
    g_stream.apply<cg::passes::StreamContractionFusion>();
    auto const t_stream = best_ms([&] { g_stream.execute(); }, trials);

    if (csv) {
        std::printf("%d,%.4f,%.4f,%.4f,%.4f\n", n, t_forloop, t_eager, t_lccf, t_stream);
    } else {
        std::printf("n=%4d  for-loops %10.2f ms  eager einsum %10.2f ms  lccf graph %10.2f ms  stream graph %10.2f ms\n", n, t_forloop,
                    t_eager, t_lccf, t_stream);
    }

    einsums::finalize();
    return 0;
}

int main(int argc, char **argv) {
    einsums::initialize(argc, argv);
    return einsums_main(argc, argv);
}
