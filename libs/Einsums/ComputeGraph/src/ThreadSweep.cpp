//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ThreadSweep.cpp
/// @brief The threading half of the hardware calibration: `e(w)` per kernel
///        family per size class.
///
/// Lives in the library rather than in the `calibrate_hardware` tool so that
/// the tool and the tests run the same sweep. The tool runs it with the default
/// caps; a test runs it with tiny ones to check the mechanics without spending
/// the seconds.

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Detail/WidthGuard.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Median wall time of @p run, in seconds, after @p warmup untimed runs.
template <typename F>
double median_seconds(size_t warmup, size_t repeats, F &&run) {
    for (size_t w = 0; w < warmup; w++) {
        run();
    }
    size_t const        n = std::max<size_t>(1, repeats);
    std::vector<double> times(n);
    for (size_t r = 0; r < n; r++) {
        auto const t0 = std::chrono::steady_clock::now();
        run();
        auto const t1 = std::chrono::steady_clock::now();
        times[r]      = std::chrono::duration<double>(t1 - t0).count();
    }
    std::ranges::sort(times);
    return times[n / 2];
}

/// One (family, size class) pair the sweep measures.
struct Cell {
    KernelFamily family;
    SizeClass    size_class;
};

/// The cells worth measuring.
///
/// A family is measured only where it exists: a small-GEMM cell at a streaming
/// working set is not a small GEMM, and a large-GEMM cell that fits in L1 is
/// not a large one. Unmeasured cells resolve through the nearest measured size
/// class in the same family. Fifteen cells at five rungs is what keeps the
/// sweep in seconds.
std::vector<Cell> sweep_cells() {
    return {
        {.family = KernelFamily::GemmSmall, .size_class = SizeClass::L1Resident},
        {.family = KernelFamily::GemmSmall, .size_class = SizeClass::L2Resident},
        {.family = KernelFamily::GemmLarge, .size_class = SizeClass::LLCResident},
        {.family = KernelFamily::GemmLarge, .size_class = SizeClass::Streaming},
        {.family = KernelFamily::BatchedGemm, .size_class = SizeClass::L2Resident},
        {.family = KernelFamily::BatchedGemm, .size_class = SizeClass::LLCResident},
        {.family = KernelFamily::BatchedGemm, .size_class = SizeClass::Streaming},
        {.family = KernelFamily::Permute, .size_class = SizeClass::L1Resident},
        {.family = KernelFamily::Permute, .size_class = SizeClass::L2Resident},
        {.family = KernelFamily::Permute, .size_class = SizeClass::LLCResident},
        {.family = KernelFamily::Permute, .size_class = SizeClass::Streaming},
        {.family = KernelFamily::Elementwise, .size_class = SizeClass::L1Resident},
        {.family = KernelFamily::Elementwise, .size_class = SizeClass::L2Resident},
        {.family = KernelFamily::Elementwise, .size_class = SizeClass::LLCResident},
        {.family = KernelFamily::Elementwise, .size_class = SizeClass::Streaming},
    };
}

/// Working set, in bytes, that lands a cell in @p size_class on this machine.
///
/// Half a level for the resident classes, so the data stays put with room for
/// the kernel's own footprint, and four times the last level for streaming.
/// @p cap can shrink a cell below the class it is labelled with; that is the
/// price of a bounded sweep, and only bites the deepest class.
size_t target_bytes(DeviceProfile const &profile, SizeClass size_class, size_t cap) {
    size_t const l1  = profile.cache_bytes(0, 32UL * 1024);
    size_t const l2  = std::max(l1 + 1, profile.cache_bytes(1, 1024UL * 1024));
    size_t const llc = std::max(l2 + 1, profile.cache_bytes(2, 8UL * 1024 * 1024));

    size_t bytes = 0;
    switch (size_class) {
    case SizeClass::L1Resident:
        bytes = l1 / 2;
        break;
    case SizeClass::L2Resident:
        bytes = l2 / 2;
        break;
    case SizeClass::LLCResident:
        bytes = llc / 2;
        break;
    case SizeClass::Streaming:
        bytes = 4 * llc;
        break;
    }
    return std::clamp<size_t>(bytes, 4096, std::max<size_t>(4096, cap));
}

/// Square extent of a GEMM whose three operands total @p bytes.
size_t gemm_extent(size_t bytes, size_t lo, size_t hi) {
    auto const n = static_cast<size_t>(std::sqrt(static_cast<double>(bytes) / (3.0 * sizeof(double))));
    return std::clamp<size_t>(n, lo, std::max(lo, hi));
}

void fill(double *data, size_t count) {
    for (size_t idx = 0; idx < count; idx++) {
        data[idx] = static_cast<double>(idx % 17) * 0.125 + 1.0;
    }
}

} // namespace

std::vector<EfficiencyCurve> measure_thread_efficiency(DeviceProfile const &profile, ThreadSweepOptions const &options) {
    unsigned const max_threads =
        options.max_threads > 0 ? options.max_threads : static_cast<unsigned>(std::max(1, einsums::hardware::get_max_threads()));
    if (max_threads <= 1) {
        return {};
    }

    auto const rungs = width_rungs_for(max_threads);

    std::vector<EfficiencyCurve> curves;
    curves.reserve(sweep_cells().size());

    for (auto const &cell : sweep_cells()) {
        size_t const bytes = target_bytes(profile, cell.size_class, options.max_bytes);

        // Build the operands once per cell and reuse them across the rungs, so
        // what varies between points is the width and nothing else.
        std::function<void()> run;

        einsums::Tensor<double, 2>              A2, B2, C2;
        einsums::Tensor<double, 1>              X1, Y1;
        std::vector<einsums::Tensor<double, 2>> batch_a, batch_b, batch_c;
        bool                                    blas_width = true;

        switch (cell.family) {
        case KernelFamily::GemmSmall:
        case KernelFamily::GemmLarge: {
            // The family split is a flop count, so the extents are held on the
            // correct side of it whatever the cache sizes say.
            size_t const n =
                (cell.family == KernelFamily::GemmSmall) ? gemm_extent(bytes, 8, 48) : gemm_extent(bytes, 72, options.max_gemm_extent);
            A2 = einsums::Tensor<double, 2>("sweep_a", n, n);
            B2 = einsums::Tensor<double, 2>("sweep_b", n, n);
            C2 = einsums::Tensor<double, 2>("sweep_c", n, n);
            fill(A2.data(), n * n);
            fill(B2.data(), n * n);
            C2.zero();
            run = [&A2, &B2, &C2] { einsums::linear_algebra::gemm<false, false>(1.0, A2, B2, 0.0, &C2); };
            break;
        }
        case KernelFamily::BatchedGemm: {
            size_t const entries = std::max<size_t>(2, options.batch_entries);
            size_t const n       = gemm_extent(bytes / entries, 8, 256);
            batch_a.reserve(entries);
            batch_b.reserve(entries);
            batch_c.reserve(entries);
            for (size_t e = 0; e < entries; e++) {
                batch_a.emplace_back("sweep_ba", n, n);
                batch_b.emplace_back("sweep_bb", n, n);
                batch_c.emplace_back("sweep_bc", n, n);
                fill(batch_a.back().data(), n * n);
                fill(batch_b.back().data(), n * n);
                batch_c.back().zero();
            }
            // The width goes to the loop over entries; each entry's BLAS call
            // stays serial, which is how grouped_batched_gemm runs.
            blas_width = false;
            run        = [&batch_a, &batch_b, &batch_c, entries] {
                auto const count = static_cast<long>(entries);
#pragma omp parallel for schedule(static)
                for (long e = 0; e < count; e++) {
                    auto const idx = static_cast<size_t>(e);
                    einsums::linear_algebra::gemm<false, false>(1.0, batch_a[idx], batch_b[idx], 0.0, &batch_c[idx]);
                }
            };
            break;
        }
        case KernelFamily::Permute: {
            auto const n = std::max<size_t>(8, static_cast<size_t>(std::sqrt(static_cast<double>(bytes) / (2.0 * sizeof(double)))));
            A2           = einsums::Tensor<double, 2>("sweep_pa", n, n);
            C2           = einsums::Tensor<double, 2>("sweep_pc", n, n);
            fill(A2.data(), n * n);
            C2.zero();
            run = [&A2, &C2] {
                using namespace einsums::index;
                einsums::tensor_algebra::permute(Indices{j, i}, &C2, Indices{i, j}, A2);
            };
            break;
        }
        case KernelFamily::Elementwise: {
            size_t const n = std::max<size_t>(1024, bytes / (2 * sizeof(double)));
            X1             = einsums::Tensor<double, 1>("sweep_x", n);
            Y1             = einsums::Tensor<double, 1>("sweep_y", n);
            fill(X1.data(), n);
            Y1.zero();
            run = [&X1, &Y1] { einsums::linear_algebra::axpy(1.0, X1, &Y1); };
            break;
        }
        }

        EfficiencyCurve curve;
        curve.family     = cell.family;
        curve.size_class = cell.size_class;

        double baseline = 0.0;
        for (unsigned const w : rungs) {
            // The vendor is left free to thread here (no moldable scope): a
            // GemmLarge cell is measuring exactly the vendor's own scaling, and
            // a clamp would measure the clamp instead.
            detail::WidthGuard const scope(static_cast<int>(w), blas_width, /*moldable_scope=*/false);
            double const             seconds = median_seconds(options.warmup, options.repeats, run);
            if (w == rungs.front()) {
                baseline = seconds;
                // A kernel too fast to time leaves no signal to divide by; the
                // cell is dropped rather than recorded as an infinite speedup.
                if (baseline <= 0.0) {
                    break;
                }
            }
            curve.widths.push_back(static_cast<std::uint32_t>(w));
            curve.speedup.push_back(seconds > 0.0 ? baseline / seconds : 1.0);
        }

        if (curve.valid()) {
            curves.push_back(std::move(curve));
        }
    }

    return curves;
}

EINSUMS_NAMESPACE_END(compute_graph)
