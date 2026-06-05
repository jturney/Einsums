//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("gemm", "[linear-algebra]", float, double) {
    using namespace einsums;

    Tensor A = create_tensor<TestType>(true, "A", 3, 3);
    Tensor B = create_tensor<TestType>(true, "B", 3, 3);
    Tensor C = create_tensor<TestType>(true, "C", 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));
    REQUIRE(C.stride(0) == 3);
    REQUIRE(C.stride(1) == 1);

    auto temp       = std::vector<TestType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    A.vector_data() = temp;
    temp            = std::vector<TestType>{11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};
    B.vector_data() = temp;

    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(),
               Catch::Matchers::Equals(std::vector<TestType>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0}));

    einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(),
               Catch::Matchers::Equals(std::vector<TestType>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0}));

    einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(),
               Catch::Matchers::Equals(std::vector<TestType>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0}));

    einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(),
               Catch::Matchers::Equals(std::vector<TestType>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0}));
}

template <typename T>
void gemm_test_1() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A    = create_incremented_tensor<T>("a", 3, 3);
    auto B    = create_incremented_tensor<T>("b", 3, 3);
    auto Cff  = create_tensor<T>("c", 3, 3);
    auto C0ff = create_tensor<T>("C0", 3, 3);
    zero(Cff);
    zero(C0ff);
    auto Cft  = create_tensor<T>("c", 3, 3);
    auto C0ft = create_tensor<T>("C0", 3, 3);
    zero(Cft);
    zero(C0ft);
    auto Ctf  = create_tensor<T>("c", 3, 3);
    auto C0tf = create_tensor<T>("C0", 3, 3);
    zero(Ctf);
    zero(C0tf);
    auto Ctt  = create_tensor<T>("c", 3, 3);
    auto C0tt = create_tensor<T>("C0", 3, 3);
    zero(Ctt);
    zero(C0tt);

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(T{1.0}, A, B, T{0.0}, &Cff);
    einsums::linear_algebra::gemm<false, true>(T{1.0}, A, B, T{0.0}, &Cft);
    einsums::linear_algebra::gemm<true, false>(T{1.0}, A, B, T{0.0}, &Ctf);
    einsums::linear_algebra::gemm<true, true>(T{1.0}, A, B, T{0.0}, &Ctt);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                C0ff(i, j) += A(i, k) * B(k, j);
                C0ft(i, j) += A(i, k) * B(j, k);
                C0tf(i, j) += A(k, i) * B(k, j);
                C0tt(i, j) += A(k, i) * B(j, k);
            }
        }
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(C0ff(i, j) == Cff(i, j));
            REQUIRE(C0ft(i, j) == Cft(i, j));
            REQUIRE(C0tf(i, j) == Ctf(i, j));
            REQUIRE(C0tt(i, j) == Ctt(i, j));
        }
    }
}

TEST_CASE("gemm_1") {
    SECTION("float") {
        gemm_test_1<float>();
    }

    SECTION("double") {
        gemm_test_1<double>();
    }

    SECTION("complex float") {
        gemm_test_1<std::complex<float>>();
    }

    SECTION("complex double") {
        gemm_test_1<std::complex<double>>();
    }
}

template <einsums::NotComplex T>
void gemm_test_2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A    = create_incremented_tensor<T>("a", 3, 3);
    auto B    = create_incremented_tensor<T>("b", 3, 3);
    auto C0ff = create_tensor<T>("C0", 3, 3);
    auto C0ft = create_tensor<T>("C0", 3, 3);
    auto C0tf = create_tensor<T>("C0", 3, 3);
    auto C0tt = create_tensor<T>("C0", 3, 3);
    zero(C0ff);
    zero(C0tf);
    zero(C0ft);
    zero(C0tt);

    // Perform basic matrix multiplication
    auto Cff = einsums::linear_algebra::gemm<false, false>(1.0, A, B);
    auto Cft = einsums::linear_algebra::gemm<false, true>(1.0, A, B);
    auto Ctf = einsums::linear_algebra::gemm<true, false>(1.0, A, B);
    auto Ctt = einsums::linear_algebra::gemm<true, true>(1.0, A, B);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                C0ff(i, j) += A(i, k) * B(k, j);
                C0ft(i, j) += A(i, k) * B(j, k);
                C0tf(i, j) += A(k, i) * B(k, j);
                C0tt(i, j) += A(k, i) * B(j, k);
            }
        }
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(C0ff(i, j) == Cff(i, j));
            REQUIRE(C0ft(i, j) == Cft(i, j));
            REQUIRE(C0tf(i, j) == Ctf(i, j));
            REQUIRE(C0tt(i, j) == Ctt(i, j));
        }
    }
}

template <einsums::Complex T>
void gemm_test_2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A    = create_incremented_tensor<T>("a", 3, 3);
    auto B    = create_incremented_tensor<T>("b", 3, 3);
    auto C0ff = create_tensor<T>("C0", 3, 3);
    auto C0ft = create_tensor<T>("C0", 3, 3);
    auto C0tf = create_tensor<T>("C0", 3, 3);
    auto C0tt = create_tensor<T>("C0", 3, 3);
    zero(C0ff);
    zero(C0tf);
    zero(C0ft);
    zero(C0tt);

    // Perform basic matrix multiplication
    auto Cff = einsums::linear_algebra::gemm<false, false>(T{1.0, 1.0}, A, B);
    auto Cft = einsums::linear_algebra::gemm<false, true>(T{1.0, 1.0}, A, B);
    auto Ctf = einsums::linear_algebra::gemm<true, false>(T{1.0, 1.0}, A, B);
    auto Ctt = einsums::linear_algebra::gemm<true, true>(T{1.0, 1.0}, A, B);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                C0ff(i, j) += T{1.0, 1.0} * A(i, k) * B(k, j);
                C0ft(i, j) += T{1.0, 1.0} * A(i, k) * B(j, k);
                C0tf(i, j) += T{1.0, 1.0} * A(k, i) * B(k, j);
                C0tt(i, j) += T{1.0, 1.0} * A(k, i) * B(j, k);
            }
        }
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(C0ff(i, j) == Cff(i, j));
            REQUIRE(C0ft(i, j) == Cft(i, j));
            REQUIRE(C0tf(i, j) == Ctf(i, j));
            REQUIRE(C0tt(i, j) == Ctt(i, j));
        }
    }
}

TEST_CASE("gemm_2") {
    SECTION("double") {
        gemm_test_2<double>();
    }
    SECTION("float") {
        gemm_test_2<float>();
    }
    SECTION("complex<float>") {
        gemm_test_2<std::complex<float>>();
    }
    SECTION("complex<double>") {
        gemm_test_2<std::complex<double>>();
    }
}

TEMPLATE_TEST_CASE("Disk gemm", "[linear-algebra]", double, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 3;

    auto A = einsums::create_random_tensor<TestType>("A", size, size);
    auto B = einsums::create_random_tensor<TestType>("B", size, size);
    auto C = einsums::create_tensor<TestType>("C", size, size);

    DiskTensor<TestType, 2> A_disk(fmt::format("/test/gemm/{}/A", type_name<TestType>()), size, size);
    DiskTensor<TestType, 2> B_disk(fmt::format("/test/gemm/{}/B", type_name<TestType>()), size, size);
    DiskTensor<TestType, 2> C_disk(fmt::format("/test/gemm/{}/C", type_name<TestType>()), size, size);

    A_disk.write(A);
    B_disk.write(B);

    {
        auto &singleton = einsums::GlobalConfigMap::get_singleton();

        auto lock = std::lock_guard(singleton);

        singleton.set_string("buffer-size", "4GB");
    }

    SECTION("nn") {
        gemm('n', 'n', TestType{1.0}, A, B, TestType{0.0}, &C);
        gemm('n', 'n', TestType{1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);

        auto &C_tens = C_disk.get_update();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                REQUIRE_THAT(C_tens(i, j), einsums::CheckWithinRel(C(i, j)));
            }
        }
    }

    SECTION("nt") {
        gemm('n', 't', TestType{1.0}, A, B, TestType{0.0}, &C);
        gemm('n', 't', TestType{1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);

        auto &C_tens = C_disk.get_update();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                REQUIRE_THAT(C_tens(i, j), einsums::CheckWithinRel(C(i, j)));
            }
        }
    }

    SECTION("tn") {
        gemm('t', 'n', TestType{1.0}, A, B, TestType{0.0}, &C);
        gemm('t', 'n', TestType{1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);

        auto &C_tens = C_disk.get_update();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                REQUIRE_THAT(C_tens(i, j), einsums::CheckWithinRel(C(i, j)));
            }
        }
    }

    SECTION("tt") {
        gemm('t', 't', TestType{1.0}, A, B, TestType{0.0}, &C);
        gemm('t', 't', TestType{1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);

        auto &C_tens = C_disk.get_update();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                REQUIRE_THAT(C_tens(i, j), einsums::CheckWithinRel(C(i, j)));
            }
        }
    }
}
