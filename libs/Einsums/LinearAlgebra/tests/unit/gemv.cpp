//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

#include <Einsums/Testing.hpp>
#include <catch2/catch_all.hpp>

using namespace einsums;

template <typename T>
void test_gemv() {
    Tensor A = create_tensor<T>("A", 3, 3);
    Tensor x = create_tensor<T>("x", 3);
    Tensor y = create_tensor<T>("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));
    std::vector<T> temp = std::vector<T>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    A.vector_data() = temp;

    REQUIRE(A.vector_data().data() == A.impl().data());
    temp            = std::vector<T>{11.0, 22.0, 33.0};
    x.vector_data() = temp;

    einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
    if (A.impl().is_column_major()) {
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<T>{154.0, 352.0, 550.0}));
    } else {
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<T>{330.0, 396.0, 462.0}));
    }

    einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
    if (A.impl().is_column_major()) {
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<T>{330.0, 396.0, 462.0}));
    } else {
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<T>{154.0, 352.0, 550.0}));
    }
}

TEMPLATE_TEST_CASE("gemv", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    SECTION("Column Major") {
        Tensor A = create_tensor<TestType>(false, "A", 3, 3);
        Tensor x = create_tensor<TestType>(false, "x", 3);
        Tensor y = create_tensor<TestType>(false, "y", 3);

        REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
        REQUIRE((x.dim(0) == 3));
        REQUIRE((y.dim(0) == 3));

        A.vector_data() = {TestType{1.0}, TestType{2.0}, TestType{3.0}, TestType{4.0}, TestType{5.0},
                           TestType{6.0}, TestType{7.0}, TestType{8.0}, TestType{9.0}};

        x.vector_data() = {TestType{11.0}, TestType{22.0}, TestType{33.0}};

        einsums::linear_algebra::gemv<true>(TestType{1.0}, A, x, TestType{0.0}, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{TestType{154.0}, TestType{352.0}, TestType{550.0}}));

        einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{TestType{330.0}, TestType{396.0}, TestType{462.0}}));
    }

    SECTION("Row Major") {
        Tensor A = create_tensor<TestType>(true, "A", 3, 3);
        Tensor x = create_tensor<TestType>(true, "x", 3);
        Tensor y = create_tensor<TestType>(true, "y", 3);

        REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
        REQUIRE((x.dim(0) == 3));
        REQUIRE((y.dim(0) == 3));
        std::vector<TestType> temp = std::vector{TestType{1.0}, TestType{2.0}, TestType{3.0}, TestType{4.0}, TestType{5.0},
                                                 TestType{6.0}, TestType{7.0}, TestType{8.0}, TestType{9.0}};

        A.vector_data() = temp;

        temp            = std::vector<TestType>{TestType{11.0}, TestType{22.0}, TestType{33.0}};
        x.vector_data() = temp;

        einsums::linear_algebra::gemv<false>(TestType{1.0}, A, x, TestType{0.0}, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{TestType{154.0}, TestType{352.0}, TestType{550.0}}));

        einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{TestType{330.0}, TestType{396.0}, TestType{462.0}}));
    }

    SECTION("Disk") {
        constexpr int size = 10;
        auto          A    = create_random_tensor<TestType>("A", size, size);
        auto          x    = create_random_tensor<TestType>("x", size);
        auto          y    = create_tensor<TestType>("y", size);
        auto          y2   = create_tensor<TestType>("y2", size);

        DiskTensor<TestType, 2> A_disk(fmt::format("/test/gemv/{}/A", type_name<TestType>()), size, size);
        DiskTensor<TestType, 1> x_disk(fmt::format("/test/gemv/{}/x", type_name<TestType>()), size);
        DiskTensor<TestType, 1> y_disk(fmt::format("/test/gemv/{}/y", type_name<TestType>()), size);

        A_disk.write(A);
        x_disk.write(x);

        einsums::linear_algebra::gemv('n', TestType{1.0}, A, x, TestType{0.0}, &y);
        einsums::linear_algebra::gemv('n', TestType{1.0}, A_disk, x, TestType{0.0}, &y2);

        for (int i = 0; i < size; i++) {
            REQUIRE_THAT(y2(i), einsums::CheckWithinRel(y(i)));
        }

        einsums::linear_algebra::gemv('n', TestType{1.0}, A_disk, x_disk, TestType{0.0}, &y2);

        for (int i = 0; i < size; i++) {
            REQUIRE_THAT(y2(i), einsums::CheckWithinRel(y(i)));
        }

        einsums::linear_algebra::gemv('n', TestType{1.0}, A_disk, x, TestType{0.0}, &y_disk);

        {
            auto y_view = y_disk.get();

            for (int i = 0; i < size; i++) {
                REQUIRE_THAT(y_view(i), einsums::CheckWithinRel(y(i)));
            }
        }

        einsums::linear_algebra::gemv('n', TestType{1.0}, A_disk, x_disk, TestType{0.0}, &y_disk);

        {
            auto y_view = y_disk.get();
            for (int i = 0; i < size; i++) {
                REQUIRE_THAT(y_view(i), einsums::CheckWithinRel(y(i)));
            }
        }
    }
}

template <NotComplex T>
void gemv_test2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A  = create_incremented_tensor<T>("A", 3, 3);
    auto b  = create_incremented_tensor<T>("b", 3);
    auto fy = create_incremented_tensor<T>("y", 3);
    auto ty = create_incremented_tensor<T>("y", 3);

    // Perform basic matrix-vector
    einsums::linear_algebra::gemv<false>(1.0, A, b, 0.0, &fy);
    einsums::linear_algebra::gemv<true>(1.0, A, b, 0.0, &ty);

    REQUIRE(fy(0) == 5.0);
    REQUIRE(fy(1) == 14.0);
    REQUIRE(fy(2) == 23.0);

    REQUIRE(ty(0) == 15.0);
    REQUIRE(ty(1) == 18.0);
    REQUIRE(ty(2) == 21.0);
}

template <Complex T>
void gemv_test2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A  = create_incremented_tensor<T>("A", 3, 3);
    auto b  = create_incremented_tensor<T>("b", 3);
    auto fy = create_incremented_tensor<T>("y", 3);
    auto ty = create_incremented_tensor<T>("y", 3);

    // Perform basic matrix-vector
    einsums::linear_algebra::gemv<false>(T{1.0, 1.0}, A, b, T{0.0, 0.0}, &fy);
    einsums::linear_algebra::gemv<true>(T{1.0, 1.0}, A, b, T{0.0, 0.0}, &ty);

    REQUIRE(fy(0) == T{-10, 10});
    REQUIRE(fy(1) == T{-28, 28});
    REQUIRE(fy(2) == T{-46, 46});

    REQUIRE(ty(0) == T{-30, 30});
    REQUIRE(ty(1) == T{-36, 36});
    REQUIRE(ty(2) == T{-42, 42});
}

TEST_CASE("gemv2") {
    SECTION("double") {
        gemv_test2<double>();
    }
    SECTION("float") {
        gemv_test2<float>();
    }
    SECTION("complex<double>") {
        gemv_test2<std::complex<double>>();
    }
    SECTION("complex<float>") {
        gemv_test2<std::complex<float>>();
    }
}

// Correctness of gemv on a 2D view whose strides are BOTH non-unit (a slab of
// a 3D tensor, e.g. R[0,:,:] of a column-major (2,d1,d2)). is_gemmable() is
// false so this takes impl_gemv_noncontiguous. Regression context: that path
// used to ``collapse(2)`` over (link, target), racing on ``Y[target] += ...``.
// The race only manifests when the kernel runs multi-threaded — which depends
// on this TU's OpenMP/instantiation context, so this is a correctness check,
// not a reliable race detector (the deterministic race guard is the looped
// Python test test_dropped_view_matvec_repeated, exercising the _core path
// that actually parallelizes).
template <typename T>
void test_gemv_noncontiguous_view() {
    constexpr size_t d0 = 2, d1 = 192, d2 = 96;
    Tensor<T, 1>     backing = create_tensor<T>("backing", d0 * d1 * d2);
    for (size_t i = 0; i < d0 * d1 * d2; ++i)
        backing.data()[i] = T(static_cast<double>((i % 7) - 3)); // small integers, mixed-sign

    TensorView<T, 2> A{backing.data(), Dim<2>{d1, d2}, Stride<2>{d0, d0 * d1}};
    REQUIRE_FALSE(A.impl().is_gemmable()); // must take the noncontiguous path

    Tensor<T, 1> x = create_tensor<T>("x", d2);
    for (size_t k = 0; k < d2; ++k)
        x.data()[k] = T(static_cast<double>(k % 5) - 2.0);

    std::vector<T> ref(d1);
    for (size_t i = 0; i < d1; ++i) {
        T acc{};
        for (size_t k = 0; k < d2; ++k)
            acc += A(i, k) * x(k);
        ref[i] = acc;
    }

    Tensor<T, 1> y = create_tensor<T>("y", d1);
    for (size_t i = 0; i < d1; ++i)
        y.data()[i] = T(-12345.0); // poison: a missed write is detectable
    einsums::linear_algebra::gemv<false>(T(1.0), A, x, T(0.0), &y);
    for (size_t i = 0; i < d1; ++i) {
        INFO("element " << i);
        REQUIRE(y(i) == ref[i]);
    }
}

TEST_CASE("gemv noncontiguous view") {
    SECTION("double") {
        test_gemv_noncontiguous_view<double>();
    }
    SECTION("float") {
        test_gemv_noncontiguous_view<float>();
    }
    SECTION("complex<double>") {
        test_gemv_noncontiguous_view<std::complex<double>>();
    }
    SECTION("complex<float>") {
        test_gemv_noncontiguous_view<std::complex<float>>();
    }
}
