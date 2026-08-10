//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>

#include <complex>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using einsums::blas::int_t;

namespace {

/// One group's worth of parameters, plus storage for its members.
///
/// The whole point of the grouped entry point is that shapes vary between
/// groups, so the fixture below builds each group's operands at its own size
/// and then concatenates the pointers, which is the layout the call wants.
template <typename T>
struct Group {
    char  transa{'N'}, transb{'N'};
    int_t m{}, n{}, k{};
    int_t lda{}, ldb{}, ldc{};
    T     alpha{}, beta{};
    int_t size{};

    std::vector<std::vector<T>> a, b, c;
};

template <typename T>
T sample(std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    if constexpr (IsComplexV<T>) {
        return T{static_cast<RemoveComplexT<T>>(dist(rng)), static_cast<RemoveComplexT<T>>(dist(rng))};
    } else {
        return static_cast<T>(dist(rng));
    }
}

/// Fill a group's operands. Leading dimensions are deliberately padded past the
/// logical extent so that a kernel indexing by ``lda`` rather than by ``m``
/// is caught.
template <typename T>
void fill(Group<T> &g, std::mt19937 &rng) {
    int_t const a_rows = (g.transa == 'N') ? g.m : g.k;
    int_t const a_cols = (g.transa == 'N') ? g.k : g.m;
    int_t const b_rows = (g.transb == 'N') ? g.k : g.n;
    int_t const b_cols = (g.transb == 'N') ? g.n : g.k;

    g.lda = a_rows + 2;
    g.ldb = b_rows + 1;
    g.ldc = g.m + 3;

    for (int_t i = 0; i < g.size; ++i) {
        g.a.emplace_back(static_cast<size_t>(g.lda) * a_cols);
        g.b.emplace_back(static_cast<size_t>(g.ldb) * b_cols);
        g.c.emplace_back(static_cast<size_t>(g.ldc) * g.n);
        for (auto &v : g.a.back()) {
            v = sample<T>(rng);
        }
        for (auto &v : g.b.back()) {
            v = sample<T>(rng);
        }
        for (auto &v : g.c.back()) {
            v = sample<T>(rng);
        }
    }
}

/// The oracle: a plain serial triple loop, written straight from the BLAS
/// definition so it shares no code with what it is checking.
template <typename T>
void reference_gemm(Group<T> const &g, std::vector<T> const &a, std::vector<T> const &b, std::vector<T> &c) {
    auto conj_if = [](T v, bool conj) {
        if constexpr (IsComplexV<T>) {
            return conj ? std::conj(v) : v;
        } else {
            return v;
        }
    };
    auto at = [&](int_t i, int_t p) {
        T const v = (g.transa == 'N') ? a[i + p * g.lda] : a[p + i * g.lda];
        return conj_if(v, g.transa == 'C');
    };
    auto bt = [&](int_t p, int_t j) {
        T const v = (g.transb == 'N') ? b[p + j * g.ldb] : b[j + p * g.ldb];
        return conj_if(v, g.transb == 'C');
    };

    for (int_t j = 0; j < g.n; ++j) {
        for (int_t i = 0; i < g.m; ++i) {
            T acc{};
            for (int_t p = 0; p < g.k; ++p) {
                acc += at(i, p) * bt(p, j);
            }
            c[i + j * g.ldc] = g.alpha * acc + g.beta * c[i + j * g.ldc];
        }
    }
}

/// Flatten the groups and issue one grouped call.
template <typename T>
void run_grouped(std::vector<Group<T>> &groups) {
    std::vector<char>      transa, transb;
    std::vector<int_t>     m, n, k, lda, ldb, ldc, size;
    std::vector<T>         alpha, beta;
    std::vector<T const *> a_ptrs, b_ptrs;
    std::vector<T *>       c_ptrs;

    for (auto &g : groups) {
        transa.push_back(g.transa);
        transb.push_back(g.transb);
        m.push_back(g.m);
        n.push_back(g.n);
        k.push_back(g.k);
        alpha.push_back(g.alpha);
        beta.push_back(g.beta);
        lda.push_back(g.lda);
        ldb.push_back(g.ldb);
        ldc.push_back(g.ldc);
        size.push_back(g.size);
        for (int_t i = 0; i < g.size; ++i) {
            a_ptrs.push_back(g.a[i].data());
            b_ptrs.push_back(g.b[i].data());
            c_ptrs.push_back(g.c[i].data());
        }
    }

    blas::gemm_batch_grouped<T>(transa.data(), transb.data(), m.data(), n.data(), k.data(), alpha.data(), a_ptrs.data(), lda.data(),
                                b_ptrs.data(), ldb.data(), beta.data(), c_ptrs.data(), ldc.data(), static_cast<int_t>(groups.size()),
                                size.data());
}

/// Every group's every member against the serial oracle.
template <typename T>
void check_against_reference(std::vector<Group<T>> groups) {
    std::mt19937 rng(20260810);
    for (auto &g : groups) {
        fill(g, rng);
    }

    auto expected = groups;
    for (auto &g : expected) {
        for (int_t i = 0; i < g.size; ++i) {
            reference_gemm(g, g.a[i], g.b[i], g.c[i]);
        }
    }

    run_grouped(groups);

    auto const tol = static_cast<RemoveComplexT<T>>(std::is_same_v<RemoveComplexT<T>, float> ? 1.0e-4 : 1.0e-12);
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        for (int_t i = 0; i < groups[gi].size; ++i) {
            for (size_t e = 0; e < groups[gi].c[i].size(); ++e) {
                INFO("group " << gi << " member " << i << " element " << e);
                REQUIRE_THAT(std::abs(groups[gi].c[i][e] - expected[gi].c[i][e]), Catch::Matchers::WithinAbs(0.0, tol));
            }
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("gemm_batch_grouped matches a serial reference", "[blas][gemm_batch]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T = TestType;

    SECTION("heterogeneous groups") {
        // Deliberately mixed: shapes above and below the inline kernel's
        // cutoff, both transpositions, a group of one, and prefactors that are
        // neither 1 nor 0 so an ignored alpha or beta cannot pass.
        std::vector<Group<T>> groups = {
            {'N', 'N', 7, 5, 9, 0, 0, 0, T{2}, T{0}, 4},     {'T', 'N', 32, 17, 24, 0, 0, 0, T{1}, T{1}, 3},
            {'N', 'T', 4, 4, 4, 0, 0, 0, T{-1}, T{3}, 1},    {'T', 'T', 11, 40, 6, 0, 0, 0, T{1}, T{0}, 5},
            {'N', 'N', 64, 33, 48, 0, 0, 0, T{1}, T{-2}, 2},
        };
        check_against_reference<T>(std::move(groups));
    }

    SECTION("conjugate transposes") {
        std::vector<Group<T>> groups = {
            {'C', 'N', 9, 6, 12, 0, 0, 0, T{1}, T{0}, 3},
            {'N', 'C', 20, 14, 8, 0, 0, 0, T{2}, T{1}, 2},
            {'C', 'C', 5, 7, 5, 0, 0, 0, T{1}, T{0}, 4},
        };
        check_against_reference<T>(std::move(groups));
    }

    SECTION("k == 0 still scales C by beta") {
        // Not an empty call: BLAS defines a zero inner dimension as C := beta*C.
        // A quick-return that lumped it in with m or n zero would leave C stale.
        std::vector<Group<T>> groups = {{'N', 'N', 6, 5, 0, 0, 0, 0, T{3}, T{2}, 3}};
        check_against_reference<T>(std::move(groups));
    }
}

TEST_CASE("gemm_batch_grouped agrees with gemm_batch on a uniform batch", "[blas][gemm_batch]") {
    // One group has to be exactly the uniform entry point, or the two paths a
    // caller can take through the same work disagree.
    constexpr int_t m = 23, n = 19, k = 17, batch = 6;

    std::mt19937  rng(4242);
    Group<double> g{'T', 'N', m, n, k, 0, 0, 0, 1.5, -0.5, batch};
    fill(g, rng);

    auto uniform = g;
    {
        std::vector<double const *> a_ptrs, b_ptrs;
        std::vector<double *>       c_ptrs;
        for (int_t i = 0; i < batch; ++i) {
            a_ptrs.push_back(uniform.a[i].data());
            b_ptrs.push_back(uniform.b[i].data());
            c_ptrs.push_back(uniform.c[i].data());
        }
        blas::gemm_batch<double>(g.transa, g.transb, m, n, k, g.alpha, a_ptrs.data(), g.lda, b_ptrs.data(), g.ldb, g.beta, c_ptrs.data(),
                                 g.ldc, batch);
    }

    std::vector<Group<double>> grouped = {g};
    run_grouped(grouped);

    for (int_t i = 0; i < batch; ++i) {
        for (size_t e = 0; e < grouped[0].c[i].size(); ++e) {
            INFO("member " << i << " element " << e);
            REQUIRE(grouped[0].c[i][e] == uniform.c[i][e]);
        }
    }
}

TEST_CASE("gemm_batch_grouped quick-returns on empty work", "[blas][gemm_batch]") {
    std::vector<double> c(16, 7.0);
    auto const          before = c;

    double const *a_ptr = c.data();
    double       *c_ptr = c.data();
    double const  one = 1.0, zero = 0.0;

    SECTION("no groups") {
        char const  t     = 'N';
        int_t const dim   = 4;
        int_t const zero_ = 0;
        blas::gemm_batch_grouped<double>(&t, &t, &dim, &dim, &dim, &one, &a_ptr, &dim, &a_ptr, &dim, &zero, &c_ptr, &dim, 0, &zero_);
        REQUIRE(c == before);
    }

    SECTION("a zero-extent group among live ones") {
        // The zero-extent group must be skipped without disturbing the live
        // group's place in the flattened pointer arrays.
        std::vector<Group<double>> groups = {
            {'N', 'N', 0, 5, 6, 0, 0, 0, 1.0, 1.0, 2},
            {'N', 'N', 8, 7, 6, 0, 0, 0, 1.0, 0.5, 3},
            {'N', 'N', 6, 0, 6, 0, 0, 0, 1.0, 1.0, 2},
            {'N', 'N', 5, 9, 4, 0, 0, 0, 2.0, 0.0, 1},
        };
        check_against_reference<double>(std::move(groups));
    }

    SECTION("a group of size zero") {
        std::vector<Group<double>> groups = {
            {'N', 'N', 6, 6, 6, 0, 0, 0, 1.0, 0.0, 0},
            {'N', 'N', 6, 6, 6, 0, 0, 0, 1.0, 0.0, 2},
        };
        check_against_reference<double>(std::move(groups));
    }
}
