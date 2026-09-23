//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The character-index permute on rank-erased tensors. Every expected value here comes from index
// arithmetic over the raw storage, so nothing above this module is used to check it.

#include <Einsums/Errors/Error.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorPermute/Permute.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::detail::TensorImpl;
namespace tp = einsums::tensor_permute;

namespace {

template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

/// A distinct value for each flat position, with a nonzero imaginary part for complex types so
/// that conjugation is visible.
template <typename T>
T value_for(size_t n) {
    if constexpr (is_complex<T>::value) {
        return T{static_cast<typename T::value_type>(n + 1), static_cast<typename T::value_type>(0.5 * static_cast<double>(n) - 3.0)};
    } else {
        return static_cast<T>(n + 1);
    }
}

template <typename T>
void check_close(T got, T want) {
    if constexpr (is_complex<T>::value) {
        CHECK(got.real() == Catch::Approx(want.real()));
        CHECK(got.imag() == Catch::Approx(want.imag()));
    } else {
        CHECK(got == Catch::Approx(want));
    }
}

/// Storage for one operand plus a TensorImpl over it. @p pad adds unused elements to the leading
/// dimension, so the operand is a strided view into its buffer rather than contiguous.
template <typename T, size_t Rank>
struct Operand {
    std::vector<T>           storage;
    std::array<size_t, Rank> dims;
    std::array<size_t, Rank> strides;
    TensorImpl<T>            impl;

    Operand(std::array<size_t, Rank> d, bool row_major, size_t pad = 0) : dims(d), strides{}, impl() {
        size_t running = 1;
        if (row_major) {
            for (size_t k = Rank; k-- > 0;) {
                strides[k] = running;
                running *= dims[k] + (k == Rank - 1 ? pad : 0);
            }
        } else {
            for (size_t k = 0; k < Rank; ++k) {
                strides[k] = running;
                running *= dims[k] + (k == 0 ? pad : 0);
            }
        }
        storage.assign(running, T{0});
        impl = TensorImpl<T>(storage.data(), dims, strides);
    }

    T &at(std::array<size_t, Rank> const &index) {
        size_t offset = 0;
        for (size_t k = 0; k < Rank; ++k)
            offset += index[k] * strides[k];
        return storage[offset];
    }

    /// Visit every index of the operand in lexicographic order.
    template <typename F>
    void for_each(F &&f) {
        std::array<size_t, Rank> index{};
        size_t                   total = 1;
        for (size_t k = 0; k < Rank; ++k)
            total *= dims[k];
        for (size_t n = 0; n < total; ++n) {
            size_t rest = n;
            for (size_t k = Rank; k-- > 0;) {
                index[k] = rest % dims[k];
                rest /= dims[k];
            }
            f(index, n);
        }
    }
};

/// A(i, j, k) with distinct values, and C laid out for C(k, i, j).
template <typename T>
void check_cyclic(bool a_row_major, bool c_row_major, size_t a_pad, size_t c_pad) {
    Operand<T, 3> A({2, 3, 4}, a_row_major, a_pad);
    Operand<T, 3> C({4, 2, 3}, c_row_major, c_pad);
    A.for_each([&](auto const &idx, size_t n) { A.at(idx) = value_for<T>(n); });

    // "kij" <- "ijk" is a 3-cycle, so it differs from its own inverse ("jki" <- "ijk"). A kernel
    // that built the inverse permutation passed every transpose and failed here.
    tp::permute("kij <- ijk", T{0}, &C.impl, T{1}, A.impl);

    A.for_each([&](auto const &idx, size_t) { CHECK(C.at({idx[2], idx[0], idx[1]}) == A.at(idx)); });
}

} // namespace

TEMPLATE_TEST_CASE("TensorPermute - a cyclic permutation in every storage order", "[TensorPermute]", float, double, std::complex<float>,
                   std::complex<double>) {
    // Each pair takes a different branch of compile_permute.
    SECTION("column-major to column-major") {
        check_cyclic<TestType>(false, false, 0, 0);
    }
    SECTION("row-major to row-major") {
        check_cyclic<TestType>(true, true, 0, 0);
    }
    SECTION("row-major to column-major") {
        check_cyclic<TestType>(true, false, 0, 0);
    }
    SECTION("column-major to row-major") {
        check_cyclic<TestType>(false, true, 0, 0);
    }
}

TEMPLATE_TEST_CASE("TensorPermute - padded operands", "[TensorPermute]", float, double, std::complex<float>, std::complex<double>) {
    // A leading dimension wider than the extent makes each operand a view into its buffer.
    SECTION("column-major") {
        check_cyclic<TestType>(false, false, 3, 2);
    }
    SECTION("row-major") {
        check_cyclic<TestType>(true, true, 1, 5);
    }
}

TEMPLATE_TEST_CASE("TensorPermute - prefactors scale the output and the input", "[TensorPermute]", float, double, std::complex<float>,
                   std::complex<double>) {
    Operand<TestType, 2> A({3, 5}, false);
    Operand<TestType, 2> C({5, 3}, false);
    A.for_each([&](auto const &idx, size_t n) { A.at(idx) = value_for<TestType>(n); });
    C.for_each([&](auto const &idx, size_t n) { C.at(idx) = value_for<TestType>(100 + n); });
    auto const before = C.storage;

    TestType const beta{0.5};
    TestType const alpha{2};
    tp::permute("ji <- ij", beta, &C.impl, alpha, A.impl);

    C.for_each([&](auto const &idx, size_t) {
        size_t const flat = idx[0] * C.strides[0] + idx[1] * C.strides[1];
        check_close(C.at(idx), beta * before[flat] + alpha * A.at({idx[1], idx[0]}));
    });
}

TEMPLATE_TEST_CASE("TensorPermute - conjugating the input", "[TensorPermute]", std::complex<float>, std::complex<double>) {
    Operand<TestType, 2> A({3, 4}, false);
    Operand<TestType, 2> C({4, 3}, false);
    A.for_each([&](auto const &idx, size_t n) { A.at(idx) = value_for<TestType>(n); });

    tp::permute<true>("ji <- ij", TestType{0}, &C.impl, TestType{1}, A.impl);

    A.for_each([&](auto const &idx, size_t) { CHECK(C.at({idx[1], idx[0]}) == std::conj(A.at(idx))); });
}

TEMPLATE_TEST_CASE("TensorPermute - a compiled plan runs again on new data", "[TensorPermute]", float, double) {
    Operand<TestType, 2> A({3, 4}, false);
    Operand<TestType, 2> C({4, 3}, false);
    auto                 plan = tp::compile_permute("ji <- ij", TestType{0}, &C.impl, TestType{1}, A.impl);

    // Compiling does not run it.
    C.for_each([&](auto const &idx, size_t) { CHECK(C.at(idx) == TestType{0}); });

    Operand<TestType, 2> A2({3, 4}, false);
    Operand<TestType, 2> C2({4, 3}, false);
    A2.for_each([&](auto const &idx, size_t n) { A2.at(idx) = value_for<TestType>(n); });
    tp::permute(&C2.impl, A2.impl, plan);

    A2.for_each([&](auto const &idx, size_t) { CHECK(C2.at({idx[1], idx[0]}) == A2.at(idx)); });
}

TEMPLATE_TEST_CASE("TensorPermute - transpose", "[TensorPermute]", float, double, std::complex<float>, std::complex<double>) {
    Operand<TestType, 2> A({3, 5}, false);
    A.for_each([&](auto const &idx, size_t n) { A.at(idx) = value_for<TestType>(n); });

    SECTION("computes the transpose") {
        Operand<TestType, 2> C({5, 3}, false);
        tp::transpose(&C.impl, A.impl);
        A.for_each([&](auto const &idx, size_t) { CHECK(C.at({idx[1], idx[0]}) == A.at(idx)); });
    }

    SECTION("rejects an output too small for the transposed input") {
        Operand<TestType, 2> C({3, 5}, false);
        CHECK_THROWS_AS(tp::transpose(&C.impl, A.impl), einsums::DimensionError);
    }

    SECTION("rejects operands that are not matrices") {
        Operand<TestType, 3> T3({3, 5, 2}, false);
        Operand<TestType, 2> C({5, 3}, false);
        CHECK_THROWS_AS(tp::transpose(&C.impl, T3.impl), einsums::RankError);
    }
}

TEST_CASE("TensorPermute - the index strings are checked before HPTT sees them", "[TensorPermute]") {
    // These used to get past the check: difference() erased from the first match to the end of the
    // string, so "ij" less "ki" came out empty and the mismatch reached HPTT as a permutation with a
    // -1 in it, which HPTT answered by ending the process.
    Operand<double, 2> A({3, 4}, false);
    Operand<double, 2> C({4, 3}, false);

    SECTION("a letter only in the input") {
        CHECK_THROWS_AS(tp::permute("ki <- ij", 0.0, &C.impl, 1.0, A.impl), einsums::RankError);
    }
    SECTION("a letter only in the output") {
        CHECK_THROWS_AS(tp::permute("ji <- ik", 0.0, &C.impl, 1.0, A.impl), einsums::RankError);
    }
    SECTION("more letters than axes") {
        CHECK_THROWS_AS(tp::permute("jik <- ijk", 0.0, &C.impl, 1.0, A.impl), einsums::RankError);
    }
    SECTION("fewer letters than axes") {
        CHECK_THROWS_AS(tp::permute("j <- i", 0.0, &C.impl, 1.0, A.impl), einsums::RankError);
    }
}

TEMPLATE_TEST_CASE("TensorPermute - empty operands", "[TensorPermute]", float, double, std::complex<float>, std::complex<double>) {
    // Zero extents are valid tensors. HPTT rejects them, so the kernel must not hand them over.
    Operand<TestType, 3> A({2, 0, 4}, false);
    Operand<TestType, 3> C({4, 2, 0}, false);

    SECTION("permute does nothing") {
        CHECK_NOTHROW(tp::permute("kij <- ijk", TestType{0}, &C.impl, TestType{1}, A.impl));
    }

    SECTION("compile_permute has no plan, and running it does nothing") {
        auto plan = tp::compile_permute("kij <- ijk", TestType{0}, &C.impl, TestType{1}, A.impl);
        CHECK(plan == nullptr);
        CHECK_NOTHROW(tp::permute(&C.impl, A.impl, plan));
    }

    SECTION("transpose does nothing") {
        Operand<TestType, 2> M({0, 3}, false);
        Operand<TestType, 2> MT({3, 0}, false);
        CHECK_NOTHROW(tp::transpose(&MT.impl, M.impl));
    }
}

namespace {

/// The smallest thing that counts as a tensor here: a ValueType and an impl(). Real tensors come
/// from a module above this one, which these tests do not use.
template <typename T, size_t Rank>
struct FakeTensor {
    using ValueType = T;
    Operand<T, Rank> operand;

    explicit FakeTensor(std::array<size_t, Rank> dims) : operand(dims, false) {}
    TensorImpl<T>       &impl() { return operand.impl; }
    TensorImpl<T> const &impl() const { return operand.impl; }
};

} // namespace

TEMPLATE_TEST_CASE("TensorPermute - tensors are taken as they are, without impl() at the call site", "[TensorPermute]", float, double,
                   std::complex<float>, std::complex<double>) {
    static_assert(tp::HasTensorImpl<FakeTensor<TestType, 2>>);
    static_assert(!tp::HasTensorImpl<TensorImpl<TestType>>);

    FakeTensor<TestType, 3> A({2, 3, 4});
    A.operand.for_each([&](auto const &idx, size_t n) { A.operand.at(idx) = value_for<TestType>(n); });

    SECTION("permute") {
        FakeTensor<TestType, 3> C({4, 2, 3});
        tp::permute("kij <- ijk", TestType{0}, &C, TestType{1}, A);
        A.operand.for_each([&](auto const &idx, size_t) { CHECK(C.operand.at({idx[2], idx[0], idx[1]}) == A.operand.at(idx)); });
    }

    SECTION("transpose") {
        FakeTensor<TestType, 2> M({3, 5}), MT({5, 3});
        M.operand.for_each([&](auto const &idx, size_t n) { M.operand.at(idx) = value_for<TestType>(n); });
        tp::transpose(&MT, M);
        M.operand.for_each([&](auto const &idx, size_t) { CHECK(MT.operand.at({idx[1], idx[0]}) == M.operand.at(idx)); });
    }
}

TEST_CASE("TensorPermute - the spec grammar", "[TensorPermute]") {
    using einsums::tensor_permute::detail::parse_permute_spec;
    // Each side becomes one character per axis; names are re-encoded in order of appearance in A.
    CHECK(parse_permute_spec("ji <- ij") == std::pair<std::string, std::string>{"ba", "ab"});
    CHECK(parse_permute_spec("ij -> ji") == std::pair<std::string, std::string>{"ba", "ab"});
    CHECK(parse_permute_spec("nu,mu <- mu,nu") == std::pair<std::string, std::string>{"ba", "ab"});
    CHECK(parse_permute_spec(" k i j<-i j k ") == std::pair<std::string, std::string>{"cab", "abc"});

    CHECK_THROWS_AS(parse_permute_spec("ji ij"), std::invalid_argument);           // no arrow
    CHECK_THROWS_AS(parse_permute_spec("j(i) <- ij"), std::invalid_argument);      // not a name
    CHECK_THROWS_AS(parse_permute_spec("mu,,nu <- mu,nu"), std::invalid_argument); // empty name

    // The spec forms reach the same kernel; the multi-character spelling transposes too.
    Operand<double, 2> A({3, 4}, false);
    Operand<double, 2> C({4, 3}, false);
    A.for_each([&](auto const &idx, size_t n) { A.at(idx) = value_for<double>(n); });
    tp::permute("nu,mu <- mu,nu", 0.0, &C.impl, 1.0, A.impl);
    A.for_each([&](auto const &idx, size_t) { CHECK(C.at({idx[1], idx[0]}) == A.at(idx)); });
}

TEMPLATE_TEST_CASE("TensorPermute - the overwrite shorthand", "[TensorPermute]", float, double) {
    FakeTensor<TestType, 2> M({3, 5}), MT({5, 3});
    M.operand.for_each([&](auto const &idx, size_t n) { M.operand.at(idx) = value_for<TestType>(n); });
    MT.operand.for_each([&](auto const &idx, size_t) { MT.operand.at(idx) = TestType{99}; });
    tp::permute("ji <- ij", &MT, M);
    M.operand.for_each([&](auto const &idx, size_t) { CHECK(MT.operand.at({idx[1], idx[0]}) == M.operand.at(idx)); });
}
