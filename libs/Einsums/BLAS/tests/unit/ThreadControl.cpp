//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// blas::set_num_threads_this_thread is a best-effort call whose observable
// effect depends on the linked vendor, so what is pinned here is the contract
// that holds for every vendor: it is safe, it is per-thread, and BLAS keeps
// computing the right answer after it.

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Testing.hpp>

TEST_CASE("blas thread control is safe on any vendor", "[blas][threads]") {
    // A vendor with no per-thread knob makes every one of these a no-op, which
    // is the point: callers do not have to know which vendor they linked.
    REQUIRE_NOTHROW(einsums::blas::set_num_threads_this_thread(1));
    REQUIRE_NOTHROW(einsums::blas::set_num_threads_this_thread(2));
    REQUIRE_NOTHROW(einsums::blas::set_num_threads_this_thread(1));

    // Nonsense counts are ignored rather than forwarded to the vendor.
    REQUIRE_NOTHROW(einsums::blas::set_num_threads_this_thread(0));
    REQUIRE_NOTHROW(einsums::blas::set_num_threads_this_thread(-4));

    // Whatever the answer, it must not change under repeated calls.
    bool const supported = einsums::blas::has_per_thread_control();
    REQUIRE(einsums::blas::has_per_thread_control() == supported);
}

TEST_CASE("blas thread control takes effect where the vendor supports it", "[blas][threads]") {
    // Where the vendor can be both set and read - MKL today - the request has
    // to actually land, otherwise TaskPool workers would still be dispatching
    // multi-threaded BLAS and this whole mechanism would be decorative. On
    // every other vendor the reader returns 0 and there is nothing to check.
#if defined(EINSUMS_TEST_BLAS_IS_MKL)
    // This build linked MKL, the one vendor whose threads the OpenMP ICVs do
    // not reach, so the symbol lookup failing is a real failure rather than an
    // absent feature. Asserted here because the skip below would otherwise let
    // a silently inert mechanism report success.
    REQUIRE(einsums::blas::has_per_thread_control());
#endif

    if (!einsums::blas::has_per_thread_control()) {
        SUCCEED("linked BLAS exposes no per-thread control; nothing to verify");
        return;
    }

    int const original = einsums::blas::get_num_threads_this_thread();

    einsums::blas::set_num_threads_this_thread(1);
    REQUIRE(einsums::blas::get_num_threads_this_thread() == 1);

    if (original > 1) {
        einsums::blas::set_num_threads_this_thread(original);
        REQUIRE(einsums::blas::get_num_threads_this_thread() == original);
    }
}

TEST_CASE("blas still computes correctly after limiting threads", "[blas][threads]") {
    // The failure this guards against is a vendor setter that leaves the
    // library in a state where later calls return garbage, which would
    // otherwise surface far away from here.
    constexpr int             n     = 8;
    constexpr size_t          elems = static_cast<size_t>(n) * n;
    std::vector<double> const a(elems, 2.0);
    std::vector<double> const b(elems, 3.0);
    std::vector<double>       c(elems, 0.0);

    einsums::blas::set_num_threads_this_thread(1);
    einsums::blas::gemm('N', 'N', n, n, n, 1.0, a.data(), n, b.data(), n, 0.0, c.data(), n);

    for (double const value : c) {
        REQUIRE_THAT(value, Catch::Matchers::WithinAbs(2.0 * 3.0 * n, 1.0e-12));
    }
}

TEST_CASE("blas thread control only affects the calling thread", "[blas][threads]") {
    // Per-thread is the whole reason this exists: TaskPool workers limit
    // themselves without stripping the threads from BLAS on the main thread.
    // A process-wide setter would make this test's GEMM wrong only under load,
    // so what is checked is that a limited worker leaves the main thread able
    // to compute, and that the call is safe from a foreign thread at all.
    std::thread worker([] {
        einsums::blas::set_num_threads_this_thread(1);

        constexpr int             n     = 4;
        constexpr size_t          elems = static_cast<size_t>(n) * n;
        std::vector<double> const a(elems, 1.0);
        std::vector<double> const b(elems, 1.0);
        std::vector<double>       c(elems, 0.0);
        einsums::blas::gemm('N', 'N', n, n, n, 1.0, a.data(), n, b.data(), n, 0.0, c.data(), n);
        REQUIRE_THAT(c[0], Catch::Matchers::WithinAbs(static_cast<double>(n), 1.0e-12));
    });
    worker.join();

    constexpr int             n     = 6;
    constexpr size_t          elems = static_cast<size_t>(n) * n;
    std::vector<double> const a(elems, 1.0);
    std::vector<double> const b(elems, 2.0);
    std::vector<double>       c(elems, 0.0);
    einsums::blas::gemm('N', 'N', n, n, n, 1.0, a.data(), n, b.data(), n, 0.0, c.data(), n);

    for (double const value : c) {
        REQUIRE_THAT(value, Catch::Matchers::WithinAbs(2.0 * n, 1.0e-12));
    }
}

#ifdef _OPENMP
TEST_CASE("a vendor call inside our own parallel region is bit-reproducible", "[blas][threads]") {
    // The regression this pins: a vendor carrying its OWN OpenMP runtime - MKL
    // - cannot see a region einsums opened, so its nested check never fires and
    // every member of our team forks a full vendor team. The widths then differ
    // per call, and on a tall-K shape the vendor splits K across whatever team
    // it happened to get, so the sum comes back in a different order and the
    // last bits move. That reached DLPNO as a residual that changed between two
    // evaluations at identical amplitudes, and a solve that diverged from it.
    //
    // Tall and thin on purpose: K is the only axis worth splitting here, so a
    // width that varies IS a summation order that varies. Values that are not
    // exactly representable, so a reordered sum actually lands somewhere else.
    constexpr int    m       = 4;
    constexpr int    n       = 4;
    constexpr int    k       = 300000;
    constexpr size_t a_elems = static_cast<size_t>(m) * k;
    constexpr size_t b_elems = static_cast<size_t>(k) * n;
    constexpr size_t c_elems = static_cast<size_t>(m) * n;

    std::vector<double> a(a_elems), b(b_elems);
    for (size_t i = 0; i < a_elems; i++) {
        a[i] = 1.0 / static_cast<double>(i + 3);
    }
    for (size_t i = 0; i < b_elems; i++) {
        b[i] = 1.0 / static_cast<double>(i + 7);
    }

    // The reference is taken on the main thread, outside any region, which is
    // the regime every vendor supports and where a wide GEMM is still wanted.
    std::vector<double> reference(c_elems, 0.0);
    einsums::blas::gemm('N', 'N', m, n, k, 1.0, a.data(), m, b.data(), k, 0.0, reference.data(), m);

    int const                        teams = std::max(2, std::min(8, omp_get_max_threads()));
    std::vector<std::vector<double>> results(static_cast<size_t>(teams), std::vector<double>(c_elems, 0.0));

#    pragma omp parallel for num_threads(teams) schedule(static)
    for (int t = 0; t < teams; t++) {
        einsums::blas::gemm('N', 'N', m, n, k, 1.0, a.data(), m, b.data(), k, 0.0, results[static_cast<size_t>(t)].data(), m);
    }

    for (int t = 0; t < teams; t++) {
        INFO("team member " << t);
        for (size_t i = 0; i < c_elems; i++) {
            INFO("element " << i);
            // Bit-identical, not close: a tolerance here would pass on exactly
            // the nondeterminism this exists to catch.
            REQUIRE(results[static_cast<size_t>(t)][i] == reference[i]);
        }
    }
}
#endif
