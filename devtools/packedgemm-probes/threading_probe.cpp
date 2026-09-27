//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// How does the packed engine scale with threads?
//
// Runs each contraction through the eager string einsum (the call the head-to-head harness makes,
// which hands 94 of its 96 cases to PackedGemm) at every thread count in one process, and a plain
// vendor GEMM of the same M, N and K beside it as the yardstick. Thread counts are interleaved round
// by round, ascending in odd rounds and descending in even ones, so a drift in the machine's state
// lands on every count alike and no count always runs first after a case's setup.
//
//   threading_probe --cases FILE [--dtype real|complex|complex1m] [--precision s|d|sd]
//                   [--threads 1,2,4,6,12,24] [--rounds N] [--only TEXT] [--batched]
//
// --cases takes the head-to-head harness's cases.csv (group,case,raw,precision,extents,flops) and
// runs it in the harness's default "reversed" layout. --dtype complex runs the same shapes on
// complex<float> ('s') or complex<double> ('d'); complex1m does too with
// --einsums:packed-gemm:complex-1m set. --batched appends a few contractions with a batch index,
// which the suite has none of, so the engine's batch-parallel loop is exercised too.
//
// Prints one CSV row per case and thread count: the engine route, the median, min and max GF/s of
// the einsum over the rounds, and the median GF/s of the reference GEMM at that thread count.
// Memory placement comes from the environment (numactl), as for the other probes; keep it the same
// for every thread count, which a single process does by construction.

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" void openblas_set_num_threads(int);

namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;

namespace {

bool g_direct = false;

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Case {
    std::string            group, name;
    char                   precision;
    std::string            c_idx, a_idx, b_idx;
    std::map<char, size_t> extent;
};

std::vector<std::string> split(std::string const &s, char sep) {
    std::vector<std::string> out;
    std::stringstream        ss(s);
    std::string              f;
    while (std::getline(ss, f, sep)) {
        out.push_back(f);
    }
    return out;
}

std::vector<Case> read_cases(std::string const &path) {
    std::vector<Case> out;
    std::ifstream     in(path);
    std::string       line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("group,", 0) == 0) {
            continue;
        }
        auto const f = split(line, ',');
        Case       c;
        c.group     = f[0];
        c.name      = f[1];
        c.precision = f[3][0];
        auto idx    = split(c.name, '-');
        c.c_idx     = idx[0];
        c.a_idx     = idx[1];
        c.b_idx     = idx[2];
        for (auto const &kv : split(f[4], ';')) {
            c.extent[kv[0]] = std::stoul(kv.substr(2));
        }
        out.push_back(c);
    }
    return out;
}

/// Contractions with a batch index (one shared by A, B and C), which the suite lacks. The small
/// M * N ones take the engine's batch-parallel loop; the large one keeps the NC loop.
std::vector<Case> batched_cases() {
    std::vector<Case> out;
    auto              add = [&](char const *name, std::map<char, size_t> ext) {
        for (char p : {'s', 'd'}) {
            auto idx = split(name, '-');
            out.push_back({"batched", name, p, idx[0], idx[1], idx[2], ext});
        }
    };
    add("abx-akx-kbx", {{'a', 48}, {'b', 48}, {'k', 256}, {'x', 2048}}); // M*N = 2304 per slice: batch loop
    add("abx-akx-kbx", {{'a', 16}, {'b', 16}, {'k', 64}, {'x', 32768}}); // tiny slices: batch loop
    add("abx-akx-kbx", {{'a', 512}, {'b', 512}, {'k', 512}, {'x', 8}});  // big slices: NC loop per slice
    return out;
}

size_t product(std::string const &idx, Case const &c) {
    size_t p = 1;
    for (char ch : idx) {
        p *= c.extent.at(ch);
    }
    return p;
}

/// M, N, K as the GEMM of the same work sees them.
void mnk(Case const &c, size_t &m, size_t &n, size_t &k) {
    m = n = k = 1;
    for (auto const &[ch, e] : c.extent) {
        bool const in_c = c.c_idx.find(ch) != std::string::npos, in_a = c.a_idx.find(ch) != std::string::npos,
                   in_b = c.b_idx.find(ch) != std::string::npos;
        if (in_c && in_a && !in_b) {
            m *= e;
        } else if (in_c && in_b && !in_a) {
            n *= e;
        } else if (in_a && in_b && !in_c) {
            k *= e;
        } else if (in_a && in_b && in_c) {
            m *= e; // a batch index: count its work into M so the GEMM does the same flops
        }
    }
}

template <typename T>
einsums::RuntimeTensor<T> make(std::string const &idx, Case const &c) {
    std::vector<size_t> dims;
    for (char ch : idx) {
        dims.push_back(c.extent.at(ch));
    }
    std::reverse(dims.begin(), dims.end());
    return einsums::RuntimeTensor<T>(std::string("op"), dims, /*row_major=*/true);
}

template <typename T>
void fill(einsums::RuntimeTensor<T> &t, std::mt19937 &rng) {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    T                                     *p = t.data();
    for (size_t i = 0; i < t.size(); i++) {
        if constexpr (einsums::IsComplexV<T>) {
            p[i] = T{static_cast<typename T::value_type>(d(rng)), static_cast<typename T::value_type>(d(rng))};
        } else {
            p[i] = static_cast<T>(d(rng));
        }
    }
}

void set_threads(int t) {
    omp_set_num_threads(t);
    openblas_set_num_threads(t);
}

/// Evicts the caches between timed calls, as the harness does, so no call finds its operands
/// warm from the previous one.
void trash_caches() {
    static std::vector<char> buf(size_t{256} << 20);
    static char              v = 0;
    std::memset(buf.data(), ++v, buf.size());
}

struct Row {
    std::vector<double> einsum_s, gemm_s;
    std::string         route, engine;
};

template <typename T>
void run_case(Case const &c, std::vector<int> const &threads, int rounds, char const *dtype_label, std::mt19937 &rng) {
    auto A = make<T>(c.a_idx, c);
    auto B = make<T>(c.b_idx, c);
    auto C = make<T>(c.c_idx, c);
    fill(A, rng);
    fill(B, rng);
    std::string rev_c(c.c_idx.rbegin(), c.c_idx.rend()), rev_a(c.a_idx.rbegin(), c.a_idx.rend()), rev_b(c.b_idx.rbegin(), c.b_idx.rend());
    std::string const spec = rev_c + " <- " + rev_a + " ; " + rev_b;
    // --direct calls the engine's own entry with the spec the string engine would build, so the
    // engine templates are instantiated in THIS translation unit: an engine header change is then
    // measured by recompiling the probe alone, without rebuilding libEinsums, whose string einsum
    // carries its own instantiations.
    einsums::packed_gemm::ContractionSpec pspec;
    for (char ch : rev_c)
        pspec.c_indices.emplace_back(1, ch);
    for (char ch : rev_a)
        pspec.a_indices.emplace_back(1, ch);
    for (char ch : rev_b)
        pspec.b_indices.emplace_back(1, ch);
    for (char ch : rev_a) {
        if (rev_b.find(ch) != std::string::npos && rev_c.find(ch) == std::string::npos)
            pspec.link_indices.emplace_back(1, ch);
    }
    bool direct_declined = false;
    auto run             = [&] {
        if (g_direct) {
            if (einsums::packed_gemm::try_packed_gemm<einsums::RuntimeTensor<T>, einsums::RuntimeTensor<T>, einsums::RuntimeTensor<T>>(
                    pspec, T{0}, &C, T{1}, A, B, /*allow_scatter=*/true, nullptr)) {
                return;
            }
            direct_declined = true;
        }
        cg::einsum(cg::EinsumFormatString(std::string_view{spec}), T{0}, &C, T{1}, A, B);
    };

    size_t m, n, k;
    mnk(c, m, n, k);
    std::vector<T> Ag(m * k, T{0.5}), Bg(k * n, T{0.25}), Cg(m * n);
    auto           gemm = [&] {
        einsums::blas::gemm<T>('N', 'N', static_cast<einsums::blas::int_t>(m), static_cast<einsums::blas::int_t>(n),
                               static_cast<einsums::blas::int_t>(k), T{1}, Ag.data(), static_cast<einsums::blas::int_t>(m), Bg.data(),
                               static_cast<einsums::blas::int_t>(k), T{0}, Cg.data(), static_cast<einsums::blas::int_t>(m));
    };

    std::vector<Row> rows(threads.size());
    // Warm each thread count once: plans, thread-local packing buffers, the vendor's own state.
    for (size_t ti = 0; ti < threads.size(); ti++) {
        set_threads(threads[ti]);
        run();
        rows[ti].route = g_direct ? (direct_declined ? std::string("declined") : std::string("packed_gemm"))
                                  : (cgd::last_dispatch_route() != nullptr ? cgd::last_dispatch_route() : "unknown");
        // The engine slots are PackedGemm's, and only a contraction it ran writes them; read on any
        // other route they would name the previous case's engine.
        rows[ti].engine = rows[ti].route == "packed_gemm" ? std::string(einsums::packed_gemm::last_contraction_route()) : std::string("-");
        if (rows[ti].engine == "packed") {
            rows[ti].engine = std::string("packed:") + einsums::packed_gemm::last_packed_engine();
        }
        gemm();
    }
    for (int r = 0; r < rounds; r++) {
        for (size_t step = 0; step < threads.size(); step++) {
            size_t const ti = (r % 2 == 0) ? step : threads.size() - 1 - step;
            set_threads(threads[ti]);
            trash_caches();
            double t0 = now();
            run();
            rows[ti].einsum_s.push_back(now() - t0);
            trash_caches();
            t0 = now();
            gemm();
            rows[ti].gemm_s.push_back(now() - t0);
        }
    }

    double const flops = (einsums::IsComplexV<T> ? 8.0 : 2.0) * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    for (size_t ti = 0; ti < threads.size(); ti++) {
        auto &e = rows[ti].einsum_s;
        auto &g = rows[ti].gemm_s;
        std::sort(e.begin(), e.end());
        std::sort(g.begin(), g.end());
        double const med = flops / e[e.size() / 2] * 1e-9, lo = flops / e.back() * 1e-9, hi = flops / e.front() * 1e-9;
        double const gmed = flops / g[g.size() / 2] * 1e-9;
        std::printf("%s,%s,%s,%c,%d,%zu,%zu,%zu,%s,%s,%.3f,%.3f,%.3f,%.3f\n", dtype_label, c.group.c_str(), c.name.c_str(), c.precision,
                    threads[ti], m, n, k, rows[ti].route.c_str(), rows[ti].engine.c_str(), med, lo, hi, gmed);
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
    std::string      cases_path, dtype = "real", precision = "sd", only;
    std::vector<int> threads{1, 2, 4, 6, 12, 24};
    int              rounds  = 5;
    bool             batched = false;
    for (int i = 1; i < argc; i++) {
        std::string const a    = argv[i];
        auto              next = [&] { return std::string(argv[++i]); };
        if (a == "--cases") {
            cases_path = next();
        } else if (a == "--dtype") {
            dtype = next();
        } else if (a == "--precision") {
            precision = next();
        } else if (a == "--threads") {
            threads.clear();
            for (auto const &t : split(next(), ',')) {
                threads.push_back(std::stoi(t));
            }
        } else if (a == "--rounds") {
            rounds = std::stoi(next());
        } else if (a == "--only") {
            only = next();
        } else if (a == "--batched") {
            batched = true;
        } else if (a == "--direct") {
            g_direct = true;
        }
    }
    char *fake[] = {argv[0], nullptr};
    einsums::initialize(1, fake);
    if (dtype == "complex1m") {
        einsums::config::set(einsums::option::PackedGemmComplex1m, true);
    }

    std::vector<Case> cases = cases_path.empty() ? std::vector<Case>{} : read_cases(cases_path);
    if (batched) {
        auto b = batched_cases();
        cases.insert(cases.end(), b.begin(), b.end());
    }
    std::printf("dtype,group,case,precision,threads,m,n,k,route,engine,gflops_median,gflops_min,gflops_max,gemm_gflops_median\n");
    std::mt19937 rng(20260927);
    for (auto const &c : cases) {
        if (precision.find(c.precision) == std::string::npos ||
            (!only.empty() && c.name.find(only) == std::string::npos && c.group != only)) {
            continue;
        }
        if (dtype == "real") {
            if (c.precision == 's') {
                run_case<float>(c, threads, rounds, "real", rng);
            } else {
                run_case<double>(c, threads, rounds, "real", rng);
            }
        } else {
            if (c.precision == 's') {
                run_case<std::complex<float>>(c, threads, rounds, dtype.c_str(), rng);
            } else {
                run_case<std::complex<double>>(c, threads, rounds, dtype.c_str(), rng);
            }
        }
    }
    einsums::finalize();
    return 0;
}
