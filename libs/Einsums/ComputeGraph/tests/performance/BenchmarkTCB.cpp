//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The Tensor Contraction Benchmark (github.com/HPAC/tccg) through the string-spec einsum.
//
// The case table is the head-to-head harness's (EinsumsPaper/benchmarks/head-to-head), copied
// beside this file as tcb_cases.csv: four groups - ccsd, ccsd_t, intensli, ao2mo - each case in
// single and double precision. This measures Einsums alone, the way that harness calls it, so a
// regression on the shapes the engine is tuned against shows up in `einsums bench compare`;
// TBLIS and TCL stay in the harness.
//
// LAYOUT. The benchmark is column-major (the leftmost index of each operand is stride-1). As in
// the harness, each operand is a row-major RuntimeTensor over the REVERSED dimensions with the
// labels reversed too, which is byte-identical to the column-major original.
//
// VERIFICATION. A layout mistake does not fail, it measures a different contraction. So each
// case first runs at small extents, distinct per index, and is checked against
// einsums::testing::reference_einsum; a case that disagrees is a failed check and is not timed.
//
// TIMING. The harness's protocol: trash the cache, call once, repeat, keep the minimum. The
// minimum is the published value. The reference GEMM of the equivalent M, N, K is timed the
// same way, so each case also reports its percentage of GEMM.
//
// THREADS. One by default, as the harness runs its baseline: a serial number is the one that
// compares across machines and against TBLIS's serial figures. EINSUMS_TCB_THREADS sets another
// count, which then appears in each label, so serial and threaded results are never compared as
// one benchmark. Pinning is the caller's: run under `taskset -c <core>` on a memory-attached core
// (on a Threadripper WX, half the dies have no memory controller).
//
// Environment (`einsums bench run` passes no arguments):
//   EINSUMS_TCB_ONLY        only cases whose name contains this, or whose group is this
//   EINSUMS_TCB_PRECISION   s, d or sd (default sd)
//   EINSUMS_TCB_REPS        timed repetitions (default 3, the harness's)
//   EINSUMS_TCB_TRASH_MB    cache-trash buffer, total MiB (default 800, the harness's)
//   EINSUMS_TCB_THREADS     OpenMP and BLAS threads (default 1)
//   EINSUMS_TCB_CASES       another case table

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;
using einsums::RuntimeTensor;

namespace {

// ── The case table ───────────────────────────────────────────────────────────

struct Case {
    std::string            group, name, raw;
    char                   precision = 'd';
    std::string            c_idx, a_idx, b_idx;
    std::map<char, size_t> extent;
    double                 flops = 0;
};

std::vector<std::string> split(std::string const &s, char sep) {
    std::vector<std::string> out;
    std::stringstream        ss(s);
    for (std::string field; std::getline(ss, field, sep);) {
        out.push_back(field);
    }
    return out;
}

std::vector<Case> read_cases(std::string const &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open the case table " + path);
    }
    std::vector<Case> cases;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#' || line.rfind("group,", 0) == 0) {
            continue;
        }
        auto const f = split(line, ',');
        if (f.size() < 6) {
            throw std::runtime_error("malformed case row: " + line);
        }
        auto const operands = split(f[1], '-');
        if (operands.size() != 3) {
            throw std::runtime_error("case name is not C-A-B: " + f[1]);
        }
        Case c{.group     = f[0],
               .name      = f[1],
               .raw       = f[2],
               .precision = f[3][0],
               .c_idx     = operands[0],
               .a_idx     = operands[1],
               .b_idx     = operands[2],
               .flops     = std::stod(f[5])};
        for (auto const &pair : split(f[4], ';')) {
            if (auto const kv = split(pair, ':'); kv.size() == 2) {
                c.extent[kv[0][0]] = std::stoull(kv[1]);
            }
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

std::string env_or(char const *name, std::string fallback) {
    char const *v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::string{v} : fallback;
}

/// The M, N, K of the GEMM a case is equivalent to. None of the cases has a batch index or an
/// index summed away within one operand; the checks say so rather than assume it.
struct MatrixShape {
    size_t m = 1, n = 1, k = 1;
};

MatrixShape matrix_shape(Case const &c) {
    MatrixShape s;
    for (char ch : c.c_idx) {
        bool const in_a = c.a_idx.find(ch) != std::string::npos;
        bool const in_b = c.b_idx.find(ch) != std::string::npos;
        if (in_a == in_b) {
            throw std::runtime_error("case " + c.name + ": output index '" + std::string(1, ch) + "' is a batch index or in neither input");
        }
        (in_a ? s.m : s.n) *= c.extent.at(ch);
    }
    for (char ch : c.a_idx) {
        bool const in_b = c.b_idx.find(ch) != std::string::npos;
        if (c.c_idx.find(ch) == std::string::npos) {
            if (!in_b) {
                throw std::runtime_error("case " + c.name + ": index '" + std::string(1, ch) + "' is summed away within A");
            }
            s.k *= c.extent.at(ch);
        }
    }
    return s;
}

// ── Operands, fill, cache trashing, timing ───────────────────────────────────

std::string reversed(std::string s) {
    std::reverse(s.begin(), s.end());
    return s;
}

/// Row-major over the reversed dimensions: the same bytes as the column-major original.
template <typename T>
RuntimeTensor<T> operand(std::string const &idx, std::map<char, size_t> const &extent, int multiplier) {
    std::vector<size_t> dims;
    for (char ch : reversed(idx)) {
        dims.push_back(extent.at(ch));
    }
    RuntimeTensor<T> t(std::string{"op"}, dims, /*row_major=*/true);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<T>((((i + 1) * static_cast<size_t>(multiplier)) % 909) / 908.0); // the harness's fill
    }
    return t;
}

/// "C <- A ; B" with the labels reversed to match the dimensions.
std::string spec_of(Case const &c) {
    return reversed(c.c_idx) + " <- " + reversed(c.a_idx) + " ; " + reversed(c.b_idx);
}

struct CacheTrasher {
    std::vector<float> a, b;
    explicit CacheTrasher(size_t total_mb) : a(total_mb * 1024 * 1024 / sizeof(float) / 2, 1.0F), b(a.size(), 2.0F) {}
    void operator()() {
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] += 0.99F * b[i];
        }
    }
};

/// Trash, call once, repeat; microseconds, with the minimum as the published value.
template <typename Fn>
einsums::performance::TimingStats time_min(Fn &&call, int reps, CacheTrasher &trash) {
    std::vector<double> us;
    for (int rep = 0; rep < reps; ++rep) {
        trash();
        auto const t0 = std::chrono::steady_clock::now();
        call();
        auto const t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double const lo   = *std::ranges::min_element(us);
    double const hi   = *std::ranges::max_element(us);
    double const mean = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(us.size());
    double       ss   = 0;
    for (double t : us) {
        ss += (t - mean) * (t - mean);
    }
    double const stddev = us.size() > 1 ? std::sqrt(ss / static_cast<double>(us.size() - 1)) : 0.0;
    return {lo, lo, hi, stddev, 0.0, reps};
}

// ── One case ─────────────────────────────────────────────────────────────────

/// Small, distinct extents, so a transposed index order cannot coincide with the right one.
Case toy_version(Case const &c) {
    Case   toy  = c;
    size_t next = 2;
    toy.extent.clear();
    for (char ch : c.c_idx + c.a_idx + c.b_idx) {
        if (!toy.extent.contains(ch)) {
            toy.extent[ch] = next++;
        }
    }
    return toy;
}

template <typename T>
double toy_deviation(Case const &c) {
    Case const       toy  = toy_version(c);
    auto const       spec = spec_of(toy);
    RuntimeTensor<T> A = operand<T>(toy.a_idx, toy.extent, 3), B = operand<T>(toy.b_idx, toy.extent, 7);
    RuntimeTensor<T> C = operand<T>(toy.c_idx, toy.extent, 11), want = operand<T>(toy.c_idx, toy.extent, 11);
    cg::einsum(cg::EinsumFormatString(std::string_view{spec}), T{0}, &C, T{1}, A, B);
    einsums::testing::reference_einsum(spec, &want, A, B);
    double worst = 0;
    for (size_t i = 0; i < C.size(); ++i) {
        double const scale = std::max(1.0, std::abs(static_cast<double>(want.data()[i])));
        worst              = std::max(worst, std::abs(static_cast<double>(C.data()[i]) - static_cast<double>(want.data()[i])) / scale);
    }
    return worst;
}

template <typename T>
void run_case(Case const &c, int reps, int threads, CacheTrasher &trash) {
    std::string const label =
        fmt::format("tcb {} {} {}{}", c.group, c.name, c.precision, threads == 1 ? std::string{} : fmt::format(" threads={}", threads));
    LabeledSectionRuntime(label); // the zone the annotations below attach to

    double const deviation = toy_deviation<T>(c);
    INFO(label << ": toy-extent deviation " << deviation);
    CHECK(deviation < 1e-4); // loose enough for a different summation order, tight enough to catch a wrong contraction
    if (!(deviation < 1e-4)) {
        fmt::println("[TCB] {:<44} FAILED verification (deviation {:.2e}); not timed", label, deviation);
        return;
    }

    MatrixShape const shape = matrix_shape(c);
    auto const        spec  = spec_of(c);
    RuntimeTensor<T>  A = operand<T>(c.a_idx, c.extent, 3), B = operand<T>(c.b_idx, c.extent, 7), C = operand<T>(c.c_idx, c.extent, 11);
    auto const  einsum = time_min([&] { cg::einsum(cg::EinsumFormatString(std::string_view{spec}), T{0}, &C, T{1}, A, B); }, reps, trash);
    char const *route  = cgd::last_dispatch_route() != nullptr ? cgd::last_dispatch_route() : "unknown";

    using einsums::blas::int_t;
    std::vector<T> ga(shape.m * shape.k, T{1}), gb(shape.k * shape.n, T{1}), gc(shape.m * shape.n, T{0});
    auto const     gemm = time_min(
        [&] {
            einsums::blas::gemm<T>('N', 'N', static_cast<int_t>(shape.m), static_cast<int_t>(shape.n), static_cast<int_t>(shape.k), T{1},
                                   ga.data(), static_cast<int_t>(shape.m), gb.data(), static_cast<int_t>(shape.k), T{0}, gc.data(),
                                   static_cast<int_t>(shape.m));
        },
        reps, trash);

    double const gflops = c.flops / (einsum.min * 1e3);
    double const gemm_gflops =
        2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) * static_cast<double>(shape.k) / (gemm.min * 1e3);
    double const pct_of_gemm = gemm_gflops > 0 ? 100.0 * gflops / gemm_gflops : 0.0;
    fmt::println("[TCB] {:<44} {:9.1f} us {:8.2f} GF/s {:6.1f}% of GEMM  {}", label, einsum.min, gflops, pct_of_gemm, route);

    ProfileAnnotate("gflops", gflops);
    ProfileAnnotate("pct_of_gemm", pct_of_gemm);
    ProfileAnnotate("route", std::string_view{route});
    ProfileAnnotate("raw", std::string_view{c.raw});
    ProfileAnnotate("m", static_cast<int64_t>(shape.m));
    ProfileAnnotate("n", static_cast<int64_t>(shape.n));
    ProfileAnnotate("k", static_cast<int64_t>(shape.k));
    einsums::performance::publish_benchmark_result(label.c_str(), "t_einsum", einsum);
    ProfileAnnotate("gflops", gemm_gflops);
    einsums::performance::publish_benchmark_result(label.c_str(), "t_gemm", gemm);
}

} // namespace

EINSUMS_TEST_CASE("TCB: the tensor contraction benchmark through einsum", "[ComputeGraph][benchmark][tcb]") {
    std::string const only      = env_or("EINSUMS_TCB_ONLY", "");
    std::string const precision = env_or("EINSUMS_TCB_PRECISION", "sd");
    int const         reps      = std::max(1, std::stoi(env_or("EINSUMS_TCB_REPS", "3")));
    size_t const      trash_mb  = std::stoull(env_or("EINSUMS_TCB_TRASH_MB", "800"));
    int const         threads   = std::max(1, std::stoi(env_or("EINSUMS_TCB_THREADS", "1")));

    // Both knobs: OpenMP's governs an OpenMP-threaded BLAS (conda-forge's OpenBLAS) and every
    // parallel region in the engine; the per-thread one reaches a BLAS with its own runtime (MKL).
    einsums::hardware::set_num_threads(threads);
    einsums::blas::set_num_threads_this_thread(threads);

    std::vector<Case> cases;
    for (auto &c : read_cases(env_or("EINSUMS_TCB_CASES", EINSUMS_TCB_CASES))) {
        bool const picked = only.empty() || c.group == only || c.name.find(only) != std::string::npos;
        if (picked && precision.find(c.precision) != std::string::npos) {
            cases.push_back(std::move(c));
        }
    }
    REQUIRE_FALSE(cases.empty());
    fmt::println("[TCB] {} cases, {} reps, {} MiB cache trash, {} thread(s)", cases.size(), reps, trash_mb, threads);

    CacheTrasher trash(trash_mb);
    einsums::performance::progress_init(static_cast<int>(2 * cases.size()));
    for (auto const &c : cases) {
        if (c.precision == 's') {
            run_case<float>(c, reps, threads, trash);
        } else {
            run_case<double>(c, reps, threads, trash);
        }
    }
}
