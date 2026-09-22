//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// TTGT probe: one TCB case, three ways.
//   packed : cg::einsum exactly as the harness calls it (reversed row-major layout)
//   ttgt   : HPTT permute A and B to [M..,K..] / [N..,K..], one vendor GEMM, permute C back if needed
//   gemm   : the reference GEMM of the same m,n,k on contiguous operands
#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename F>
static double best_of(int reps, F &&f) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        double t0 = now();
        f();
        best = std::min(best, now() - t0);
    }
    return best;
}

static std::vector<size_t> dims_of(std::string const &idx, std::map<char, size_t> const &ext) {
    std::vector<size_t> d;
    for (char c : idx)
        d.push_back(ext.at(c));
    return d;
}
static size_t elements(std::vector<size_t> const &d) {
    size_t n = 1;
    for (auto x : d)
        n *= x;
    return n;
}

template <typename T>
static void fill(T *p, size_t n, unsigned seed) {
    unsigned s = seed * 2654435761u + 1;
    for (size_t i = 0; i < n; ++i) {
        s    = s * 1664525u + 1013904223u;
        p[i] = static_cast<T>((s >> 9) & 0xffff) / T{65536} - T{0.5};
    }
}

template <typename T>
static void run(std::string const &c_idx, std::string const &a_idx, std::string const &b_idx, std::map<char, size_t> const &ext, int reps) {
    using einsums::blas::int_t;
    // Column-major view of the problem (leftmost index stride 1), as the benchmark defines it.
    std::string M, N, K, Bt;
    for (char c : c_idx) {
        bool in_a = a_idx.find(c) != std::string::npos, in_b = b_idx.find(c) != std::string::npos;
        if (in_a && in_b)
            Bt.push_back(c);
        else if (in_a)
            M.push_back(c);
        else if (in_b)
            N.push_back(c);
    }
    for (char c : a_idx)
        if (b_idx.find(c) != std::string::npos && c_idx.find(c) == std::string::npos)
            K.push_back(c);
    if (!Bt.empty()) {
        std::printf("batch indices present, skipping\n");
        return;
    }

    size_t m = elements(dims_of(M, ext)), n = elements(dims_of(N, ext)), k = elements(dims_of(K, ext));
    double flops = 2.0 * m * n * k;
    std::printf("case %s-%s-%s %s  M=%s(%zu) N=%s(%zu) K=%s(%zu)\n", c_idx.c_str(), a_idx.c_str(), b_idx.c_str(),
                sizeof(T) == 4 ? "s" : "d", M.c_str(), m, N.c_str(), n, K.c_str(), k);

    auto           ad = dims_of(a_idx, ext), bd = dims_of(b_idx, ext), cd = dims_of(c_idx, ext);
    std::vector<T> a(elements(ad)), b(elements(bd)), c_ref(elements(cd));
    fill(a.data(), a.size(), 3);
    fill(b.data(), b.size(), 7);

    // ---- packed: the harness's reversed row-major call ----
    auto rev = [](std::string s) {
        std::reverse(s.begin(), s.end());
        return s;
    };
    auto mk = [&](std::string const &idx) {
        auto d = dims_of(idx, ext);
        std::reverse(d.begin(), d.end());
        return einsums::RuntimeTensor<T>(std::string("op"), d, /*row_major=*/true);
    };
    einsums::RuntimeTensor<T> A = mk(a_idx), B = mk(b_idx), C = mk(c_idx);
    std::copy(a.begin(), a.end(), A.data());
    std::copy(b.begin(), b.end(), B.data());
    std::string spec     = rev(c_idx) + " <- " + rev(a_idx) + " ; " + rev(b_idx);
    double      t_packed = best_of(reps, [&] { cg::einsum(cg::EinsumFormatString(std::string_view{spec}), T{0}, &C, T{1}, A, B); });
    std::printf("  packed  %8.3f s  %7.2f GF/s  route=%s/%s\n", t_packed, flops / t_packed * 1e-9, cgd::last_dispatch_route(),
                einsums::packed_gemm::last_contraction_route());
    std::copy(C.data(), C.data() + c_ref.size(), c_ref.begin());
    if (std::getenv("PROBE_PACKED_ONLY") != nullptr)
        return;

    // ---- ttgt ----
    // HPTT is column-major here: perm[i] = source dim feeding destination dim i.
    auto perm_of = [](std::string const &dst, std::string const &src) {
        std::vector<int> p;
        for (char ch : dst)
            p.push_back(static_cast<int>(src.find(ch)));
        return p;
    };
    std::string    a_s = M + K, b_s = N + K, c_s = M + N;
    bool           c_canonical = (c_s == c_idx);
    std::vector<T> As(a.size()), Bs(b.size()), Cs(c_canonical ? 0 : c_ref.size()), Cout(c_ref.size());
    auto           pa = perm_of(a_s, a_idx), pb = perm_of(b_s, b_idx), pc = perm_of(c_idx, c_s);
    double         t_pa = 0, t_pb = 0, t_g = 0, t_pc = 0;
    double         t_ttgt = best_of(reps, [&] {
        double t0 = now();
        einsums::packed_gemm::hptt_transpose(pa.data(), (int)ad.size(), a.data(), ad.data(), nullptr, As.data(), 1, false);
        double t1 = now();
        einsums::packed_gemm::hptt_transpose(pb.data(), (int)bd.size(), b.data(), bd.data(), nullptr, Bs.data(), 1, false);
        double t2   = now();
        T     *cptr = c_canonical ? Cout.data() : Cs.data();
        einsums::blas::gemm<T>('N', 'T', (int_t)m, (int_t)n, (int_t)k, T{1}, As.data(), (int_t)m, Bs.data(), (int_t)n, T{0}, cptr,
                               (int_t)m);
        double t3 = now();
        if (!c_canonical) {
            auto csd = dims_of(c_s, ext);
            einsums::packed_gemm::hptt_transpose(pc.data(), (int)cd.size(), Cs.data(), csd.data(), nullptr, Cout.data(), 1, false);
        }
        double t4 = now();
        t_pa      = t1 - t0;
        t_pb      = t2 - t1;
        t_g       = t3 - t2;
        t_pc      = t4 - t3;
    });
    double         maxdev = 0;
    for (size_t i = 0; i < c_ref.size(); ++i)
        maxdev = std::max(maxdev, std::abs((double)Cout[i] - (double)c_ref[i]) / std::max(1.0, std::abs((double)c_ref[i])));
    std::printf("  ttgt    %8.3f s  %7.2f GF/s  (permA %.3f permB %.3f gemm %.3f permC %.3f; C %s) maxdev=%.2e\n", t_ttgt,
                flops / t_ttgt * 1e-9, t_pa, t_pb, t_g, t_pc, c_canonical ? "canonical" : "permuted back", maxdev);

    // ---- reference gemm ----
    std::vector<T> ga(m * k), gb(k * n), gc(m * n);
    fill(ga.data(), ga.size(), 3);
    fill(gb.data(), gb.size(), 7);
    double t_gemm = best_of(reps, [&] {
        einsums::blas::gemm<T>('N', 'N', (int_t)m, (int_t)n, (int_t)k, T{1}, ga.data(), (int_t)m, gb.data(), (int_t)k, T{0}, gc.data(),
                               (int_t)m);
    });
    std::printf("  gemm    %8.3f s  %7.2f GF/s\n", t_gemm, flops / t_gemm * 1e-9);
    std::printf("  %%GEMM: packed %.1f%%  ttgt %.1f%%\n", 100 * t_gemm / t_packed, 100 * t_gemm / t_ttgt);
}

int main(int argc, char **argv) {
    // usage: ttgt_probe s|d  abcd-aebf-dfce  "a:96;b:84;..."  [reps]
    einsums::initialize(argc, argv);
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s s|d name extents [reps]\n", argv[0]);
        return 2;
    }
    std::string            prec = argv[1], name = argv[2], exts = argv[3];
    int                    reps  = argc > 4 ? std::atoi(argv[4]) : 3;
    auto                   dash1 = name.find('-'), dash2 = name.find('-', dash1 + 1);
    std::string            c_idx = name.substr(0, dash1), a_idx = name.substr(dash1 + 1, dash2 - dash1 - 1), b_idx = name.substr(dash2 + 1);
    std::map<char, size_t> ext;
    size_t                 pos = 0;
    while (pos < exts.size()) {
        auto semi = exts.find(';', pos);
        if (semi == std::string::npos)
            semi = exts.size();
        std::string tok = exts.substr(pos, semi - pos);
        ext[tok[0]]     = std::stoul(tok.substr(2));
        pos             = semi + 1;
    }
    if (prec == "s")
        run<float>(c_idx, a_idx, b_idx, ext, reps);
    else
        run<double>(c_idx, a_idx, b_idx, ext, reps);
    einsums::finalize();
    return 0;
}
