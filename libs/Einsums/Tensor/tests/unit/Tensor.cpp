//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/ARange.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

#include <atomic>

#include <Einsums/Testing.hpp>

TEST_CASE("Tensor creation", "[tensor]") {
    using namespace einsums;

    Tensor A(true, "A", 1, 1);
    Tensor B(true, "B", 1, 1);

    REQUIRE((A.dim(0) == 1 && A.dim(1) == 1));
    REQUIRE((B.dim(0) == 1 && B.dim(1) == 1));

    A.resize(einsums::Dim{3, 3});
    B.resize(einsums::Dim{3, 3});

    auto C = create_tensor<double>(true, "C", 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    A.zero();
    B.zero();
    C.zero();

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    CHECK_THAT(B.vector_data(), Catch::Matchers::Equals(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));

    // Set A and B to identity
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    CHECK_THAT(B.vector_data(), Catch::Matchers::Equals(std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
}

TEST_CASE("Tensor GEMMs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor B("B", 3, 3);
    einsums::Tensor C("C", 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    A.vector_data() = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    B.vector_data() = {11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};

    if (einsums::GlobalConfigMap::get_singleton().get_bool("row-major")) {
        einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0}));

        einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0}));

        einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0}));

        einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0}));
    } else {
        einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0}));

        einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0}));

        einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0}));

        einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
        CHECK_THAT(C.vector_data(),
                   Catch::Matchers::Equals(std::vector<double>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0}));
    }
}

TEST_CASE("Tensor GEMVs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor x("x", 3);
    einsums::Tensor y("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));

    A.vector_data() = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    x.vector_data() = {11.0, 22.0, 33.0};

    if (einsums::GlobalConfigMap::get_singleton().get_bool("row-major")) {
        einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0}));

        einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0}));
    } else {
        einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{154.0, 352.0, 550.0}));

        einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
        CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double>{330.0, 396.0, 462.0}));
    }
}

TEST_CASE("Tensor SYEVs", "[tensor]") {
    einsums::Tensor A(true, "A", 3, 3);
    einsums::Tensor x(true, "x", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));

    A.vector_data() = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    einsums::linear_algebra::syev(&A, &x);

    CHECK_THAT(x(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(x(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(x(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));
}

TEST_CASE("Tensor Invert") {
    einsums::Tensor A(true, "A", 3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 3.0;
    A(1, 1) = 2.0;
    A(1, 2) = 1.0;
    A(2, 0) = 2.0;
    A(2, 1) = 1.0;
    A(2, 2) = 3.0;

    einsums::linear_algebra::invert(&A);

    CHECK_THAT(A.vector_data(), Catch::Matchers::Approx(std::vector<double>{-5.0 / 12, 0.25, 1.0 / 3.0, 7.0 / 12.0, 0.25, -2.0 / 3.0,
                                                                            1.0 / 12.0, -0.25, 1.0 / 3.0})
                                    .margin(0.00001));
}

TEST_CASE("TensorView creation", "[tensor]") {
    using namespace einsums;
    // With the aid of deduction guides we can choose to not specify the rank on the tensor
    einsums::Tensor     A("A", 3, 3, 3);
    einsums::TensorView viewA(A, einsums::Dim{3, 9});

    // Since we are changing the underlying datatype to float the deduction guides will not work.
    einsums::Tensor           fA("A", 3, 3, 3);
    einsums::TensorView const fviewA(fA, einsums::Dim{3, 9});

    for (int i = 0, ijk = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++, ijk++)
                A(i, j, k) = ijk;

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
    REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

    if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
        for (int i = 0, ij = 0; i < 3; i++)
            for (int j = 0; j < 9; j++, ij++)
                REQUIRE(viewA(i, j) == A(i, j / 3, j % 3));
    } else {
        for (int i = 0, ij = 0; i < 3; i++)
            for (int j = 0; j < 9; j++, ij++)
                REQUIRE(viewA(i, j) == A(i, j % 3, j / 3));
    }

    auto *array = new double[100];

    for (int i = 0; i < 100; i++) {
        array[i] = i;
    }

    // Drop down in scope to make sure the view is deleted before the array it is viewing.
    {
        TensorView<double, 2>       view1{array, Dim<2>{10, 10}}, view2{array, Dim{10, 10}, Stride{10, 1}};
        TensorView<double, 2> const const_view1{static_cast<double const *>(array), Dim<2>{10, 10}},
            const_view2{static_cast<double const *>(array), Dim{10, 10}, Stride{10, 1}};

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
                    REQUIRE(view1(j, i) == array[i + j * 10]);
                    REQUIRE(const_view1(j, i) == array[i + j * 10]);
                } else {
                    REQUIRE(view1(i, j) == array[i + j * 10]);
                    REQUIRE(const_view1(i, j) == array[i + j * 10]);
                }
                REQUIRE(view2(i, j) == array[10 * i + j]);
                REQUIRE(const_view2(i, j) == array[10 * i + j]);
            }
        }
    }

    delete[] array;
}

// TEST_CASE("Tensor-2D HDF5") {
//     using namespace einsums;

//     einsums::Tensor A("A", 3, 3);

//     for (int i = 0, ij = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++, ij++) {
//             A(i, j) = ij;
//         }
//     }

//     {
//         h5::fd_t fd = h5::create("tensor.h5", H5F_ACC_TRUNC);
//         h5::ds_t ds = h5::create<double>(fd, "Matrix A", h5::current_dims{10, 20}, h5::max_dims{10, 20});
//         h5::write(ds, A, h5::count{2, 3}, h5::offset{2, 2}, h5::stride{1, 3});
//     }

//     {
//         auto B = h5::read<einsums::Tensor<double, 2>>("tensor.h5", "Matrix A");

//         REQUIRE((B.dim(0) == 10 && B.dim(1) == 20));
//         REQUIRE(B(2, 2) == 0.0);
//         REQUIRE(B(2, 5) == 1.0);
//         REQUIRE(B(2, 8) == 2.0);
//         REQUIRE(B(3, 2) == 3.0);
//         REQUIRE(B(3, 5) == 4.0);
//         REQUIRE(B(3, 8) == 5.0);
//     }
// }

// TEST_CASE("Tensor-1D HDF5") {
//     auto A = einsums::create_random_tensor("A", 3);

//     {
//         h5::fd_t fd = h5::create("tensor-1d.h5", H5F_ACC_TRUNC);
//         h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3}, h5::max_dims{3});
//         h5::write(ds, A);
//     }

//     {
//         auto B = h5::read<einsums::Tensor<double, 1>>("tensor-1d.h5", "A");

//         REQUIRE(A(0) == B(0));
//         REQUIRE(A(1) == B(1));
//         REQUIRE(A(2) == B(2));
//     }
// }

// TEST_CASE("Tensor-3D HDF5") {
//     auto A = einsums::create_random_tensor("A", 3, 2, 1);

//     {
//         h5::fd_t fd = h5::create("tensor-3d.h5", H5F_ACC_TRUNC);
//         h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3, 2, 1});
//         h5::write(ds, A);
//     }

//     {
//         auto B = h5::read<einsums::Tensor<double, 3>>("tensor-3d.h5", "A");

//         REQUIRE(B.dim(0) == 3);
//         REQUIRE(B.dim(1) == 2);
//         REQUIRE(B.dim(2) == 1);
//     }
// }

// TEST_CASE("TensorView-2D HDF5") {
//     SECTION("Subview Offset{0,0,0}") {
//         auto                           A = einsums::create_random_tensor("A", 3, 3, 3);
//         einsums::TensorView<double, 2> viewA(A, einsums::Dim{3, 9});

//         REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
//         REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

//         {
//             h5::fd_t fd = h5::create("tensorview-2d.h5", H5F_ACC_TRUNC);
//             h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3, 9});
//             h5::write(ds, viewA);
//         }

//         {
//             auto B = h5::read<einsums::Tensor<double, 2>>("tensorview-2d.h5", "A");
//             for (int i = 0; i < 3; i++)
//                 for (int j = 0; j < 9; j++)
//                     REQUIRE(viewA(i, j) == B(i, j));
//         }
//     }
// }

TEST_CASE("TensorView Ranges") {
    using namespace einsums;

    SECTION("Subviews") {
        auto                C = einsums::create_random_tensor("C", 3, 3);
        einsums::TensorView viewC(C, einsums::Dim{2, 2}, einsums::Offset{1, 1}, einsums::Stride{3, 1});

        // einsums::println("C strides: %zu %zu\n", C.strides()[0], C.strides()[1]);

        if (C.is_row_major()) {
            REQUIRE(C(1, 1) == viewC(0, 0));
            REQUIRE(C(2, 1) == viewC(1, 0));
            REQUIRE(C(1, 2) == viewC(0, 1));
            REQUIRE(C(2, 2) == viewC(1, 1));
        } else {
            REQUIRE(C(1, 1) == viewC(0, 0));
            REQUIRE(C(1, 2) == viewC(1, 0));
            REQUIRE(C(2, 1) == viewC(0, 1));
            REQUIRE(C(2, 2) == viewC(1, 1));
        }
    }

    SECTION("Subviews 2") {
        auto C = einsums::create_random_tensor("C", 3, 3);
        // std::array<einsums::Range, 2> test;
        einsums::TensorView viewC = C(einsums::Range{1, 3}, einsums::Range{1, 3});

        // einsums::println(C);
        // einsums::println(viewC);

        REQUIRE(C(1, 1) == viewC(0, 0));
        REQUIRE(C(2, 1) == viewC(1, 0));
        REQUIRE(C(1, 2) == viewC(0, 1));
        REQUIRE(C(2, 2) == viewC(1, 1));
    }

    // SECTION("Subviews 3") {
    //     auto C = create_random_tensor("C", 3, 3, 3, 3);
    //     auto viewC = C(0, 0, Range{1, 3}, Range{1, 3});

    //     println(C);
    //     println(viewC);
    // }
}

// TEST_CASE("Tensor 2D - HDF5 wrapper") {
//     auto A = einsums::create_random_tensor("A", 3, 3);

//     h5::fd_t fd = h5::create("tensor-wrapper.h5", H5F_ACC_TRUNC);

//     einsums::write(fd, A);

//     auto B = einsums::read<double, 2>(fd, "A");

//     for (int i = 0; i < 3; i++)
//         for (int j = 0; j < 3; j++)
//             REQUIRE(A(i, j) == B(i, j));
// }

TEST_CASE("reshape") {
    SECTION("1") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        REQUIRE_NOTHROW(einsums::Tensor<double, 2>(std::move(C), "D", 10, -1));
        // NOTE: At this point tensor C is no longer valid.
    }

    SECTION("2") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        auto D = einsums::Tensor<double, 2>(std::move(C), "D", 100, 10);
        // NOTE: At this point tensor C is no longer valid.

        // println(C); // <- This will cause a segfault when println tries to print the tensor elements
        // println(D); // <- This succeeds.
    }

    SECTION("3") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        REQUIRE_THROWS(einsums::Tensor<double, 2>(std::move(C), "D", -1, -1));
        // NOTE: At this point tensor C is no longer valid.
    }

    SECTION("4") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        REQUIRE_THROWS(einsums::Tensor<double, 2>(std::move(C), "D", 9, 9));
        // NOTE: At this point tensor C is no longer valid.
    }
}

namespace {
template <typename Destination, typename Source>
void types_test() {
    using namespace einsums;

    auto A = create_random_tensor<Source>("A", 10, 10);
    auto B = create_random_tensor<Destination>("B", 10, 10);

    B = A;
}
} // namespace

TEST_CASE("types") {
    SECTION("float->double") {
        types_test<double, float>();
    }

    SECTION("float->complex<double>") {
        types_test<std::complex<double>, float>();
    }

    SECTION("double->complex<float>") {
        types_test<std::complex<float>, double>();
    }
}

namespace {
template <typename T>
void test_tensor_from_tensorview() {
    using namespace einsums;

    auto   A  = create_incremented_tensor("A", 10, 10);
    auto   vA = TensorView(A, Dim{2, 2}, Offset{4, 4});
    Tensor B  = vA;

    // vA owns an independent mutex from A (both inherit Lockable<recursive_mutex>
    // separately). The point of these blocks is to prove the view's lock isn't
    // hijacked by the parent's lock; try_lock on vA must succeed even while A
    // is held, from at least one thread. The original code did `locked++` (not
    // atomic) inside an `omp parallel` and asserted == 1, which only held by
    // accident: the test binary used to wrap session.run() in
    // `#pragma omp parallel { #pragma omp single { ... } }`, so this nested
    // parallel ran with one thread (max_active_levels=1). With the wrap dropped
    // (for TSan cleanliness) the inner team is real, every thread races the
    // counter, and the literal "==1" no longer means anything. Make the counter
    // atomic and assert "at least one thread acquired the view's lock", which
    // is what the test was actually trying to prove.
    A.lock();

    std::atomic<int> locked = 0;

#pragma omp parallel shared(locked)
    {
        if (vA.try_lock()) {
            locked.fetch_add(1, std::memory_order_relaxed);
            vA.unlock();
        }
    }
    REQUIRE(locked.load() >= 1);

    A.unlock();

    vA.lock();

    locked.store(0);

#pragma omp parallel shared(locked)
    {
        if (A.try_lock()) {
            locked.fetch_add(1, std::memory_order_relaxed);
            A.unlock();
        }
    }
    REQUIRE(locked.load() >= 1);

    vA.unlock();

    REQUIRE(B(0, 0) == A(4, 4));
    REQUIRE(B(0, 1) == A(4, 5));
    REQUIRE(B(1, 0) == A(5, 4));
    REQUIRE(B(1, 1) == A(5, 5));
}
} // namespace

TEST_CASE("tensor_tensorview") {
    test_tensor_from_tensorview<float>();
    test_tensor_from_tensorview<double>();
    test_tensor_from_tensorview<std::complex<float>>();
    test_tensor_from_tensorview<std::complex<double>>();
}

namespace {
void test_tensorview() {
    using namespace einsums;

    auto A = create_incremented_tensor("A", 10, 10);
    //    auto B = A(Dim{2, 2}, Offset{4, 4});
}

template <einsums::NotComplex T>
void arange_test() {
    auto A = einsums::arange<T>(0, 10);

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
}
} // namespace

TEST_CASE("arange") {
    arange_test<double>();
    arange_test<float>();
}