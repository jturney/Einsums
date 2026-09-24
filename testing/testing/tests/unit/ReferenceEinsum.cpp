//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The reference einsum is only worth trusting if it is checked by something other than itself.
// Two checks here: small cases worked by hand, and agreement with the templated einsum engine on
// every spec class the Python fuzzers draw, while that engine still exists to agree with.

#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string_view>

#include <Einsums/Testing.hpp>

using einsums::testing::parse_reference_spec;
using einsums::testing::reference_einsum;
using einsums::testing::reference_permute;

namespace {

template <typename T>
T prefactor(double re, double im) {
    if constexpr (einsums::IsComplexV<T>) {
        return T{static_cast<typename T::value_type>(re), static_cast<typename T::value_type>(im)};
    } else {
        return static_cast<T>(re);
    }
}

template <typename TensorType>
void check_same(TensorType const &got, TensorType const &want) {
    REQUIRE(got.size() == want.size());
    double magnitude = 0.0;
    for (size_t n = 0; n < want.size(); ++n)
        magnitude = std::max(magnitude, static_cast<double>(std::abs(want.data()[n])));
    for (size_t n = 0; n < want.size(); ++n) {
        CAPTURE(n);
        CHECK_THAT(got.data()[n], einsums::CheckWithinMagnitude(want.data()[n], magnitude));
    }
}

/// Run the templated engine and the reference on the same operands and prefactors, and compare.
template <typename T, bool ConjA = false, typename CIdx, typename CT, typename AIdx, typename AT, typename BIdx, typename BT>
void agree(std::string_view spec, CIdx const &c_idx, CT C, AIdx const &a_idx, AT const &A, BIdx const &b_idx, BT const &B) {
    CAPTURE(spec);
    auto    expected = C; // deep copy: both start from the same C, which c_pf must scale
    T const c_pf     = prefactor<T>(0.5, 0.25);
    T const ab_pf    = prefactor<T>(1.5, -0.75);
    einsums::tensor_algebra::einsum<ConjA, false>(c_pf, c_idx, &C, ab_pf, a_idx, A, b_idx, B);
    reference_einsum(spec, c_pf, &expected, ab_pf, A, B, ConjA, false);
    check_same(C, expected);
}

} // namespace

// ── The grammar and the contract ────────────────────────────────────────────

TEST_CASE("reference einsum - spec grammar", "[Testing][ReferenceEinsum]") {
    auto const left = parse_reference_spec("ij <- ik ; kj");
    CHECK(left.c == std::vector<std::string>{"i", "j"});
    CHECK(left.a == std::vector<std::string>{"i", "k"});
    CHECK(left.b == std::vector<std::string>{"k", "j"});

    auto const right = parse_reference_spec("ik;kj->ij");
    CHECK(right.c == left.c);
    CHECK(right.a == left.a);
    CHECK(right.b == left.b);

    auto const multi = parse_reference_spec("mu,nu <- mu,rho ; rho,nu");
    CHECK(multi.c == std::vector<std::string>{"mu", "nu"});
    CHECK(multi.a == std::vector<std::string>{"mu", "rho"});

    CHECK(parse_reference_spec(" <- i ; i").c.empty());

    CHECK_THROWS_AS(parse_reference_spec("ij ik kj"), std::invalid_argument);            // no arrow
    CHECK_THROWS_AS(parse_reference_spec("ij <- ik"), std::invalid_argument);            // one input
    CHECK_THROWS_AS(parse_reference_spec("ij <- ik ; kl ; lj"), std::invalid_argument);  // three inputs
    CHECK_THROWS_AS(parse_reference_spec("ij <- conj(ik) ; kj"), std::invalid_argument); // operators are not grammar
    CHECK_THROWS_AS(parse_reference_spec("i,,j <- ik ; kj"), std::invalid_argument);     // empty name
}

TEST_CASE("reference einsum - operands must match the spec", "[Testing][ReferenceEinsum]") {
    auto A = einsums::create_random_tensor<double>("A", 3, 4);
    auto B = einsums::create_random_tensor<double>("B", 4, 5);
    auto C = einsums::create_zero_tensor<double>("C", 3, 5);

    CHECK_THROWS_AS(reference_einsum("ij <- ikl ; kj", &C, A, B), einsums::RankError);
    CHECK_THROWS_AS(reference_einsum("ij <- ik ; jk", &C, A, B), einsums::DimensionError); // k is 4 in A, 5 in B
    CHECK_THROWS_AS(reference_einsum("iq <- ik ; kj", &C, A, B), std::invalid_argument);   // q is in neither input
}

// ── Worked by hand ──────────────────────────────────────────────────────────

namespace {

/// A 2x2 tensor from its rows.
einsums::Tensor<double, 2> matrix(double a00, double a01, double a10, double a11) {
    auto M  = einsums::create_zero_tensor<double>("M", 2, 2);
    M(0, 0) = a00;
    M(0, 1) = a01;
    M(1, 0) = a10;
    M(1, 1) = a11;
    return M;
}

einsums::Tensor<double, 1> vector(std::initializer_list<double> values) {
    auto   v = einsums::create_zero_tensor<double>("v", values.size());
    size_t n = 0;
    for (double x : values)
        v(n++) = x;
    return v;
}

} // namespace

TEST_CASE("reference einsum - small cases worked by hand", "[Testing][ReferenceEinsum]") {
    auto const A = matrix(1, 2, 3, 4);

    SECTION("a matrix product, in every spelling of the spec") {
        auto const B = matrix(5, 6, 7, 8);
        for (auto spec : {"ij <- ik ; kj", "ik;kj->ij", "p,q <- p,r ; r,q"}) {
            CAPTURE(spec);
            auto C = einsums::create_zero_tensor<double>("C", 2, 2);
            reference_einsum(spec, &C, A, B);
            CHECK(C(0, 0) == 19);
            CHECK(C(0, 1) == 22);
            CHECK(C(1, 0) == 43);
            CHECK(C(1, 1) == 50);
        }
    }

    SECTION("a repeated letter takes the diagonal") {
        auto const b = vector({10, 100});
        auto       C = einsums::create_zero_tensor<double>("C", 2);
        reference_einsum("i <- ii ; i", &C, A, b);
        CHECK(C(0) == 10);  // A(0,0) * b(0)
        CHECK(C(1) == 400); // A(1,1) * b(1)
    }

    SECTION("a letter in one input alone is summed over that input") {
        auto const b = vector({10, 100});
        auto       C = einsums::create_zero_tensor<double>("C", 2);
        reference_einsum("i <- ij ; i", &C, A, b);
        CHECK(C(0) == 30);  // (1 + 2) * 10
        CHECK(C(1) == 700); // (3 + 4) * 100
    }

    SECTION("no output letters writes one element") {
        auto const x = vector({1, 2, 3});
        auto const y = vector({4, 5, 6});
        auto       d = einsums::create_zero_tensor<double>("d", 1);
        reference_einsum(" <- i ; i", &d, x, y);
        CHECK(d(0) == 32);
    }

    SECTION("an empty sum still scales C, once") {
        auto const E = einsums::create_zero_tensor<double>("E", 2, 0);
        auto const F = einsums::create_zero_tensor<double>("F", 0, 2);
        auto       C = matrix(1, 2, 3, 4);
        reference_einsum("ij <- ik ; kj", 2.0, &C, 1.0, E, F);
        CHECK(C(0, 0) == 2);
        CHECK(C(1, 1) == 8);
    }

    SECTION("c_pf == 0 overwrites C without reading it") {
        auto C = matrix(std::numeric_limits<double>::quiet_NaN(), 0, 0, 0);
        reference_einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, matrix(1, 0, 0, 1));
        CHECK(C(0, 0) == 1);
    }

    SECTION("a repeated output letter writes the diagonal, after scaling all of C") {
        auto const x = vector({2, 3});
        auto const y = vector({5, 7});
        auto       C = matrix(1, 1, 1, 1);
        reference_einsum("ii <- i ; i", 10.0, &C, 1.0, x, y);
        CHECK(C(0, 0) == 10 + 2 * 5);
        CHECK(C(1, 1) == 10 + 3 * 7);
        CHECK(C(0, 1) == 10);
        CHECK(C(1, 0) == 10);
    }

    SECTION("conjugation") {
        using Z = std::complex<double>;
        auto a  = einsums::create_zero_tensor<Z>("a", 1);
        auto b  = einsums::create_zero_tensor<Z>("b", 1);
        auto d  = einsums::create_zero_tensor<Z>("d", 1);
        a(0)    = Z{1, 2};
        b(0)    = Z{3, 0};
        reference_einsum(" <- i ; i", Z{0}, &d, Z{1}, a, b, /*conj_a=*/true);
        CHECK(d(0) == Z{3, -6});
    }
}

// ── Agreement with the templated engine ─────────────────────────────────────

TEMPLATE_TEST_CASE("reference einsum - agrees with the templated engine", "[Testing][ReferenceEinsum]", float, double, std::complex<float>,
                   std::complex<double>) {
    using namespace einsums::index;
    using einsums::create_random_tensor;
    using einsums::Indices;
    using T = TestType;

    SECTION("matrix product") {
        agree<T>("ij <- ik ; kj", Indices{i, j}, create_random_tensor<T>("C", 3, 4), Indices{i, k}, create_random_tensor<T>("A", 3, 5),
                 Indices{k, j}, create_random_tensor<T>("B", 5, 4));
    }
    SECTION("matrix product into a transposed output") {
        agree<T>("ji <- ik ; kj", Indices{j, i}, create_random_tensor<T>("C", 4, 3), Indices{i, k}, create_random_tensor<T>("A", 3, 5),
                 Indices{k, j}, create_random_tensor<T>("B", 5, 4));
    }
    SECTION("matrix product, row-major operands") {
        agree<T>("ij <- ik ; kj", Indices{i, j}, create_random_tensor<T>(true, "C", 3, 4), Indices{i, k},
                 create_random_tensor<T>(true, "A", 3, 5), Indices{k, j}, create_random_tensor<T>(true, "B", 5, 4));
    }
    SECTION("matrix times vector") {
        agree<T>("i <- ij ; j", Indices{i}, create_random_tensor<T>("C", 3), Indices{i, j}, create_random_tensor<T>("A", 3, 4), Indices{j},
                 create_random_tensor<T>("B", 4));
    }
    SECTION("outer product") {
        agree<T>("ij <- i ; j", Indices{i, j}, create_random_tensor<T>("C", 3, 4), Indices{i}, create_random_tensor<T>("A", 3), Indices{j},
                 create_random_tensor<T>("B", 4));
    }
    SECTION("outer product with interleaved output indices") {
        // A's indices are not contiguous in C, the ordering that once broke the outer-product path.
        agree<T>("abc <- ac ; b", Indices{a, b, c}, create_random_tensor<T>("C", 2, 3, 4), Indices{a, c},
                 create_random_tensor<T>("A", 2, 4), Indices{b}, create_random_tensor<T>("B", 3));
    }
    SECTION("elementwise product") {
        agree<T>("ij <- ij ; ij", Indices{i, j}, create_random_tensor<T>("C", 3, 4), Indices{i, j}, create_random_tensor<T>("A", 3, 4),
                 Indices{i, j}, create_random_tensor<T>("B", 3, 4));
    }
    SECTION("rank-4 contraction over two indices") {
        agree<T>("ijmn <- ijkl ; klmn", Indices{i, j, m, n}, create_random_tensor<T>("C", 2, 3, 2, 3), Indices{i, j, k, l},
                 create_random_tensor<T>("A", 2, 3, 4, 2), Indices{k, l, m, n}, create_random_tensor<T>("B", 4, 2, 2, 3));
    }
    SECTION("batched matrix product") {
        agree<T>("bij <- bik ; bkj", Indices{b, i, j}, create_random_tensor<T>("C", 2, 3, 4), Indices{b, i, k},
                 create_random_tensor<T>("A", 2, 3, 5), Indices{b, k, j}, create_random_tensor<T>("B", 2, 5, 4));
    }
    SECTION("a letter summed over one input alone") {
        agree<T>("i <- ij ; i", Indices{i}, create_random_tensor<T>("C", 3), Indices{i, j}, create_random_tensor<T>("A", 3, 4), Indices{i},
                 create_random_tensor<T>("B", 3));
    }
    SECTION("a repeated letter") {
        agree<T>("i <- ii ; i", Indices{i}, create_random_tensor<T>("C", 3), Indices{i, i}, create_random_tensor<T>("A", 3, 3), Indices{i},
                 create_random_tensor<T>("B", 3));
    }
    SECTION("a repeated output letter writes the diagonal") {
        agree<T>("ii <- ijk ; jik", Indices{i, i}, create_random_tensor<T>("C", 3, 3), Indices{i, j, k},
                 create_random_tensor<T>("A", 3, 4, 2), Indices{j, i, k}, create_random_tensor<T>("B", 4, 3, 2));
    }
    SECTION("a repeated output letter at rank 3") {
        agree<T>("iji <- iji ; jij", Indices{i, j, i}, create_random_tensor<T>("C", 3, 4, 3), Indices{i, j, i},
                 create_random_tensor<T>("A", 3, 4, 3), Indices{j, i, j}, create_random_tensor<T>("B", 4, 3, 4));
    }
    SECTION("a zero-extent contraction") {
        agree<T>("ij <- ik ; kj", Indices{i, j}, create_random_tensor<T>("C", 3, 4), Indices{i, k}, create_random_tensor<T>("A", 3, 0),
                 Indices{k, j}, create_random_tensor<T>("B", 0, 4));
    }
    if constexpr (einsums::IsComplexV<T>) {
        SECTION("conjugating A") {
            agree<T, true>("ij <- ik ; kj", Indices{i, j}, create_random_tensor<T>("C", 3, 4), Indices{i, k},
                           create_random_tensor<T>("A", 3, 5), Indices{k, j}, create_random_tensor<T>("B", 5, 4));
        }
    }
}

TEMPLATE_TEST_CASE("reference einsum - agrees with the templated engine on a dot product", "[Testing][ReferenceEinsum]", float, double,
                   std::complex<float>, std::complex<double>) {
    // The templated engine writes a spec with no output letters into a scalar; the reference into a
    // one-element tensor.
    using namespace einsums::index;
    using einsums::Indices;
    using T          = TestType;
    auto const x     = einsums::create_random_tensor<T>("x", 7);
    auto const y     = einsums::create_random_tensor<T>("y", 7);
    T const    c_pf  = prefactor<T>(0.5, 0.25);
    T const    ab_pf = prefactor<T>(1.5, -0.75);

    T    scalar = prefactor<T>(2.0, -1.0);
    auto d      = einsums::create_zero_tensor<T>("d", 1);
    d(0)        = scalar;
    einsums::tensor_algebra::einsum(c_pf, Indices{}, &scalar, ab_pf, Indices{i}, x, Indices{i}, y);
    reference_einsum(" <- i ; i", c_pf, &d, ab_pf, x, y);
    CHECK_THAT(scalar, einsums::CheckWithinMagnitude(d(0), static_cast<double>(std::abs(d(0)))));
}

// ── The reference permute ───────────────────────────────────────────────────

TEST_CASE("reference permute - worked by hand, and its contract", "[Testing][ReferenceEinsum]") {
    auto A  = einsums::create_zero_tensor<double>("A", 2, 3);
    A(0, 0) = 1, A(0, 1) = 2, A(0, 2) = 3;
    A(1, 0) = 4, A(1, 1) = 5, A(1, 2) = 6;

    SECTION("a transpose, in both spellings") {
        for (auto spec : {"ji <- ij", "ij -> ji"}) {
            CAPTURE(spec);
            auto C = einsums::create_zero_tensor<double>("C", 3, 2);
            reference_permute(spec, 0.0, &C, 1.0, A);
            CHECK(C(0, 1) == 4);
            CHECK(C(2, 0) == 3);
            CHECK(C(1, 1) == 5);
        }
    }

    SECTION("prefactors") {
        auto C = einsums::create_zero_tensor<double>("C", 3, 2);
        C.set_all(10.0);
        reference_permute("ji <- ij", 0.5, &C, 2.0, A);
        CHECK(C(2, 1) == 0.5 * 10.0 + 2.0 * 6.0);
    }

    SECTION("beta == 0 overwrites C without reading it") {
        auto C = einsums::create_zero_tensor<double>("C", 3, 2);
        C.set_all(std::numeric_limits<double>::quiet_NaN());
        reference_permute("ji <- ij", 0.0, &C, 1.0, A);
        CHECK(C(0, 0) == 1);
    }

    SECTION("a cyclic rank-3 permutation") {
        auto T = einsums::create_random_tensor<double>("T", 2, 3, 4);
        auto C = einsums::create_zero_tensor<double>("C", 4, 2, 3);
        reference_permute("kij <- ijk", 0.0, &C, 1.0, T);
        CHECK(C(3, 1, 2) == T(1, 2, 3));
        CHECK(C(0, 0, 1) == T(0, 1, 0));
    }

    SECTION("an empty operand") {
        auto E  = einsums::create_zero_tensor<double>("E", 0, 3);
        auto CE = einsums::create_zero_tensor<double>("CE", 3, 0);
        CHECK_NOTHROW(reference_permute("ji <- ij", 0.0, &CE, 1.0, E));
    }

    SECTION("the contract") {
        auto C  = einsums::create_zero_tensor<double>("C", 3, 2);
        auto C3 = einsums::create_zero_tensor<double>("C3", 3, 2, 1);
        CHECK_THROWS_AS(reference_permute("ji ; ij", 0.0, &C, 1.0, A), std::invalid_argument);    // no arrow
        CHECK_THROWS_AS(reference_permute("jk <- ij", 0.0, &C, 1.0, A), std::invalid_argument);   // k only in C
        CHECK_THROWS_AS(reference_permute("jj <- ij", 0.0, &C, 1.0, A), std::invalid_argument);   // repeated letter
        CHECK_THROWS_AS(reference_permute("ij <- ij", 0.0, &C, 1.0, A), einsums::DimensionError); // C is 3x2, A is 2x3
        CHECK_THROWS_AS(reference_permute("jik <- ij", 0.0, &C3, 1.0, A), einsums::RankError);    // three letters, two axes of A
    }
}

TEMPLATE_TEST_CASE("reference permute - agrees with the typed permute", "[Testing][ReferenceEinsum]", float, double, std::complex<float>,
                   std::complex<double>) {
    using namespace einsums::index;
    using einsums::Indices;
    using T          = TestType;
    T const    beta  = prefactor<T>(0.5, 0.25);
    T const    alpha = prefactor<T>(1.5, -0.75);
    bool const rows  = GENERATE(false, true);
    CAPTURE(rows);

    auto A        = einsums::create_random_tensor<T>(rows, "A", 2, 3, 4);
    auto C        = einsums::create_random_tensor<T>(rows, "C", 4, 2, 3);
    auto expected = C;
    einsums::tensor_algebra::permute(beta, Indices{k, i, j}, &C, alpha, Indices{i, j, k}, A);
    reference_permute("kij <- ijk", beta, &expected, alpha, A);
    check_same(C, expected);
}

// ── Mixed precision ─────────────────────────────────────────────────────────

namespace {

/// The templated engine against the reference, for operand types that differ. Both read the
/// operands at the promoted type, so they agree to well inside a float's precision.
template <typename TC, typename TA, typename TB>
void agree_mixed() {
    using namespace einsums::index;
    using einsums::Indices;
    auto const A        = einsums::create_random_tensor<TA>("A", 4, 5);
    auto const B        = einsums::create_random_tensor<TB>("B", 5, 3);
    auto       C        = einsums::create_random_tensor<TC>("C", 4, 3);
    auto       expected = C;

    // The templated engine takes both prefactors as one type; 0.5 and 1.5 are exact in all four.
    einsums::tensor_algebra::einsum(TC{0.5}, Indices{i, j}, &C, TC{1.5}, Indices{i, k}, A, Indices{k, j}, B);
    reference_einsum("ij <- ik ; kj", TC{0.5}, &expected, 1.5, A, B);
    for (size_t n = 0; n < C.size(); ++n) {
        CAPTURE(n);
        CHECK(std::abs(C.data()[n] - expected.data()[n]) <= 1.0e-5 * (1.0 + std::abs(expected.data()[n])));
    }
}

} // namespace

TEST_CASE("reference einsum - mixed precision agrees with the templated engine", "[Testing][ReferenceEinsum][MixedPrecision]") {
    // The combinations TensorAlgebra's MixedPrecision.cpp exercises.
    SECTION("d <- f * d") {
        agree_mixed<double, float, double>();
    }
    SECTION("d <- f * f") {
        agree_mixed<double, float, float>();
    }
    SECTION("f <- d * d") {
        agree_mixed<float, double, double>();
    }
    SECTION("cd <- cd * d") {
        agree_mixed<std::complex<double>, std::complex<double>, double>();
    }
}

TEST_CASE("reference einsum - complex<float> times double", "[Testing][ReferenceEinsum][MixedPrecision]") {
    // No operator* exists between complex<float> and double, and the templated engine does not
    // compile for the pair. The reference reads both as complex<double>, so it keeps the imaginary
    // part and the double's precision.
    using cf = std::complex<float>;
    using cd = std::complex<double>;
    static_assert(std::is_same_v<einsums::testing::detail::AccumulatorT<cf, double>, cd>);
    static_assert(std::is_same_v<einsums::testing::detail::AccumulatorT<double, cf>, cd>);

    auto a = einsums::create_zero_tensor<cf>("a", 2);
    auto b = einsums::create_zero_tensor<double>("b", 2);
    auto c = einsums::create_zero_tensor<cd>("c", 1);
    a(0)   = cf{1.0f, 2.0f};
    a(1)   = cf{3.0f, -1.0f};
    b(0)   = 1.0 / 3.0; // not representable in float: a float accumulator would round it
    b(1)   = 2.0;
    reference_einsum(" <- i ; i", &c, a, b);
    CHECK(c(0).real() == Catch::Approx(1.0 / 3.0 + 6.0).epsilon(1e-15));
    CHECK(c(0).imag() == Catch::Approx(2.0 / 3.0 - 2.0).epsilon(1e-15));
}
