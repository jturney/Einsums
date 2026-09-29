//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Should 1m be the default for complex contractions on x86?
//
// Times three arms per case, interleaved round by round (A B C A B C ...), so a drift in the
// machine's state during the run lands on every arm alike:
//
//   block   the packed engine as it ships: one vendor GEMM per cache block, then a scatter
//   1m      the same call with --einsums:packed-gemm:complex-1m, on the rung's real tile
//   vendor  one plain vendor GEMM of the same M, N and K on contiguous operands, the control
//
// The packed arms call try_packed_gemm directly with allow_scatter = true, as the string engine
// and ComputeGraph do. Every case is a scatter (C's M and N index groups interleave), so the
// packed arms take the packed loops rather than handing the shape to a vendor GEMM. The engine
// each packed arm ran is printed from last_packed_engine(), and the two packed arms' results are
// compared, so a mislabelled or broken arm shows up in the output.
//
//   complex_1m_probe z|c [rounds]
//
// Prints one line per case and arm: median, min and max GF/s over the rounds, counting 8 real
// flops per complex multiply-add. Thread count and pinning come from the environment
// (OMP_NUM_THREADS, OMP_PROC_BIND, OMP_PLACES, numactl).

#include <Einsums/BLAS.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace einsums;
using namespace einsums::index;

namespace {

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stats {
    double median, lo, hi;
};

Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

/// One case: a scatter contraction with M, N and K flat extents, run three ways.
template <typename T>
struct Arms {
    std::string name;
    int64_t     M, N, K;
    // Runs the packed contraction once; returns the engine it ran.
    std::function<std::string()> packed;
    // Copies C out, for comparing the two packed arms.
    std::function<std::vector<T>()> result;
};

template <typename T>
void run_case(Arms<T> const &c, int rounds) {
    // The control: a plain vendor GEMM with the same flat extents on contiguous operands.
    std::vector<T> Ag(static_cast<size_t>(c.M * c.K), T{0.5}), Bg(static_cast<size_t>(c.K * c.N), T{0.25}),
        Cg(static_cast<size_t>(c.M * c.N));
    auto vendor      = [&] { blas::gemm<T>('n', 'n', c.M, c.N, c.K, T{1}, Ag.data(), c.M, Bg.data(), c.K, T{0}, Cg.data(), c.M); };
    auto packed_with = [&](bool flag) {
        config::set(option::PackedGemmComplex1m, flag);
        return c.packed();
    };

    // Warm every arm once (plans, thread-local buffers, the vendor's own state), and check the two
    // packed engines agree before timing them.
    std::string const block_engine = packed_with(false);
    auto const        block_c      = c.result();
    std::string const onem_engine  = packed_with(true);
    auto const        onem_c       = c.result();
    vendor();
    double max_diff = 0.0, max_mag = 0.0;
    for (size_t e = 0; e < block_c.size(); e++) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(block_c[e] - onem_c[e])));
        max_mag  = std::max(max_mag, static_cast<double>(std::abs(block_c[e])));
    }

    std::vector<double> t_block, t_1m, t_vendor;
    for (int r = 0; r < rounds; r++) {
        double t0 = now();
        packed_with(false);
        t_block.push_back(now() - t0);
        t0 = now();
        packed_with(true);
        t_1m.push_back(now() - t0);
        t0 = now();
        vendor();
        t_vendor.push_back(now() - t0);
    }
    config::set(option::PackedGemmComplex1m, false);

    double const flops = 8.0 * static_cast<double>(c.M) * static_cast<double>(c.N) * static_cast<double>(c.K);
    auto const   gf    = [&](std::vector<double> const &t) {
        auto s = stats(t);
        // Fastest time is the highest rate, so lo/hi swap.
        return Stats{flops / s.median * 1e-9, flops / s.hi * 1e-9, flops / s.lo * 1e-9};
    };
    Stats const b = gf(t_block), o = gf(t_1m), v = gf(t_vendor);
    std::printf("%-22s M=%-5lld N=%-5lld K=%-5lld | block[%s] %7.2f (%7.2f-%7.2f) | 1m[%s] %7.2f (%7.2f-%7.2f) | vendor %7.2f "
                "(%7.2f-%7.2f) | 1m/block %.3f 1m/vendor %.3f | agree %.1e\n",
                c.name.c_str(), static_cast<long long>(c.M), static_cast<long long>(c.N), static_cast<long long>(c.K), block_engine.c_str(),
                b.median, b.lo, b.hi, onem_engine.c_str(), o.median, o.lo, o.hi, v.median, v.lo, v.hi, o.median / b.median,
                o.median / v.median, max_diff / std::max(max_mag, 1.0));
    std::fflush(stdout);
}

/// C(a,b,c,d) = A(a,c,k) * B(k,b,d): a GEMM's arithmetic (M = a*c, N = b*d, K = k) on the scatter
/// path, since C's M and N groups interleave.
template <typename T>
void gemm_like(char const *name, int64_t na, int64_t nc, int64_t nk, int64_t nb, int64_t nd, int rounds) {
    auto A = create_random_tensor<T>("A", na, nc, nk);
    auto B = create_random_tensor<T>("B", nk, nb, nd);
    auto C = create_random_tensor<T>("C", na, nb, nc, nd);
    run_case<T>({name, na * nc, nb * nd, nk,
                 [&] {
                     bool const ok = tensor_algebra::detail::try_packed_gemm_indices<false, false>(
                         T{0}, Indices{a, b, c, d}, &C, T{1}, Indices{a, c, k}, A, Indices{k, b, d}, B);
                     return ok && std::string(packed_gemm::last_contraction_route()) == "packed"
                                ? std::string(packed_gemm::last_packed_engine())
                                : std::string("NOT-PACKED");
                 },
                 [&] { return std::vector<T>(C.data(), C.data() + C.size()); }},
                rounds);
}

/// The CCSD ring term C(a,b,i,j) = A(a,e,i,m) * B(e,b,m,j): M = N = K = v*o.
template <typename T>
void ring(char const *name, int64_t v, int64_t o, int rounds) {
    auto A = create_random_tensor<T>("A", v, v, o, o);
    auto B = create_random_tensor<T>("B", v, v, o, o);
    auto C = create_random_tensor<T>("C", v, v, o, o);
    run_case<T>({name, v * o, v * o, v * o,
                 [&] {
                     bool const ok = tensor_algebra::detail::try_packed_gemm_indices<false, false>(
                         T{0}, Indices{a, b, i, j}, &C, T{1}, Indices{a, e, i, m}, A, Indices{e, b, m, j}, B);
                     return ok && std::string(packed_gemm::last_contraction_route()) == "packed"
                                ? std::string(packed_gemm::last_packed_engine())
                                : std::string("NOT-PACKED");
                 },
                 [&] { return std::vector<T>(C.data(), C.data() + C.size()); }},
                rounds);
}

template <typename T>
void run_all(int rounds) {
    // Square GEMM-like work: M = N = K = n, with M and N each split over two interleaved indices.
    gemm_like<T>("square n=128", 8, 16, 128, 8, 16, rounds);
    gemm_like<T>("square n=256", 16, 16, 256, 16, 16, rounds);
    gemm_like<T>("square n=512", 16, 32, 512, 16, 32, rounds);
    gemm_like<T>("square n=1024", 32, 32, 1024, 32, 32, rounds);
    gemm_like<T>("square n=2048", 32, 64, 2048, 32, 64, rounds);
    // The CCSD ring at the size the M4 measurements used (v = 48, o = 16), and a larger one.
    ring<T>("ring v=48 o=16", 48, 16, rounds);
    ring<T>("ring v=64 o=24", 64, 24, rounds);
    // Small K: M = N = 1024 with k = 16, where packing and the scatter dominate.
    gemm_like<T>("small-k n=1024 k=16", 32, 32, 16, 32, 32, rounds);
}

} // namespace

int main(int argc, char **argv) {
    char const dtype  = argc > 1 ? argv[1][0] : 'z';
    int const  rounds = argc > 2 ? std::atoi(argv[2]) : 7;
    char      *fake[] = {argv[0], nullptr};
    einsums::initialize(1, fake);
    std::printf("dtype=%s rounds=%d OMP_NUM_THREADS=%s\n", dtype == 'c' ? "complex<float>" : "complex<double>", rounds,
                std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "unset");
    if (dtype == 'c') {
        run_all<std::complex<float>>(rounds);
    } else {
        run_all<std::complex<double>>(rounds);
    }
    einsums::finalize();
    return 0;
}
