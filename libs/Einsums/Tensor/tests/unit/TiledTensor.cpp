//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("TiledTensor creation", "[tensor]") {
    using namespace einsums;

    TiledTensor<double, 2> A("A", std::array{1, 0, 2});
    TiledTensor<double, 2> B("B", std::array{1, 0, 2});
    TiledTensor<double, 2> C("C", std::array{1, 0, 2});

    REQUIRE(A.dim(0) == 3);
    REQUIRE(A.dim(1) == 3);
    REQUIRE(B.dim(0) == 3);
    REQUIRE(B.dim(1) == 3);
    REQUIRE(A.tile_offset(0)[0] == 0);
    REQUIRE(A.tile_offset(0)[1] == 1);
    REQUIRE(A.tile_offset(0)[2] == 1);
    REQUIRE(B.tile_offset(0)[0] == 0);
    REQUIRE(B.tile_offset(0)[1] == 1);
    REQUIRE(B.tile_offset(0)[2] == 1);
    REQUIRE(A.tile_offset(1)[0] == 0);
    REQUIRE(A.tile_offset(1)[1] == 1);
    REQUIRE(A.tile_offset(1)[2] == 1);
    REQUIRE(B.tile_offset(1)[0] == 0);
    REQUIRE(B.tile_offset(1)[1] == 1);
    REQUIRE(B.tile_offset(1)[2] == 1);

    A.zero();
    B.zero();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(fabs(A(i, j)) < 1e-11);
            CHECK(fabs(B(i, j)) < 1e-11);
        }
    }

    // Set A and B to identity
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                CHECK(A(i, j) == 1.0);
                CHECK(B(i, j) == 1.0);
                CHECK(C(i, j) == 1.0);
            } else {
                CHECK(A(i, j) == 0.0);
                CHECK(B(i, j) == 0.0);
                CHECK(C(i, j) == 0.0);
            }
        }
    }
}

TEST_CASE("TiledTensor GEMMs", "[tensor]") {
    using namespace einsums;

    TiledTensor<double, 2> A("A", std::array{1, 0, 2});
    TiledTensor<double, 2> B("B", std::array{1, 0, 2});
    TiledTensor<double, 2> C("C", std::array{1, 0, 2});

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = 3 * i + j + 1;
            B(i, j) = 33 * i + 11 * j + 11;
        }
    }

    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    println(A);
    println(B);
    println(C);
    auto res = einsums::VectorData<double>{330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }

    einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
    res = einsums::VectorData<double>{726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }

    einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
    res = einsums::VectorData<double>{154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }

    einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
    res = einsums::VectorData<double>{330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(C(i, j) == res[3 * i + j]);
        }
    }
}

TEST_CASE("TiledTensor GEMVs", "[tensor]") {
    using namespace einsums;

    TiledTensor<double, 2> A("A", std::array{1, 0, 2});
    einsums::Tensor        x("x", 3);
    einsums::Tensor        y("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = 3 * i + j + 1;
        }
        x(i) = 11 * i + 11;
    }

    einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
    auto res = einsums::VectorData<double>{154.0, 352.0, 550.0};

    for (int i = 0; i < 3; i++) {
        CHECK(y(i) == res[i]);
    }

    einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
    res = einsums::VectorData<double>{330.0, 396.0, 462.0};
    for (int i = 0; i < 3; i++) {
        CHECK(y(i) == res[i]);
    }
}

/*
TEST_CASE("Tensor Invert") {
    einsums::Tensor A("A", 3, 3);
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

    CHECK_THAT(A.vector_data(), Catch::Matchers::Approx(einsums::VectorData<double>{-5.0 / 12, 0.25, 1.0 / 3.0, 7.0 / 12.0, 0.25,
                                                                                    -2.0 / 3.0, 1.0 / 12.0, -0.25, 1.0 / 3.0})
                                    .margin(0.00001));
}

TEST_CASE("TensorView creation", "[tensor]") {
    // With the aid of deduction guides we can choose to not specify the rank on the tensor
    einsums::Tensor     A("A", 3, 3, 3);
    einsums::TensorView viewA(A, einsums::Dim{3, 9});

    // Since we are changing the underlying datatype to float the deduction guides will not work.
    einsums::Tensor     fA("A", 3, 3, 3);
    einsums::TensorView fviewA(fA, einsums::Dim{3, 9});

    for (int i = 0, ijk = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++, ijk++)
                A(i, j, k) = ijk;

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
    REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

    for (int i = 0, ij = 0; i < 3; i++)
        for (int j = 0; j < 9; j++, ij++)
            REQUIRE(viewA(i, j) == ij);
}

TEST_CASE("Tensor-2D HDF5") {
    einsums::Tensor A("A", 3, 3);

    for (int i = 0, ij = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++, ij++) {
            A(i, j) = ij;
        }
    }

    {
        h5::fd_t fd = h5::create("tensor.h5", H5F_ACC_TRUNC);
        h5::ds_t ds = h5::create<double>(fd, "Matrix A", h5::current_dims{10, 20}, h5::max_dims{10, 20});
        h5::write(ds, A, h5::count{2, 3}, h5::offset{2, 2}, h5::stride{1, 3});
    }

    {
        auto B = h5::read<einsums::Tensor<double, 2>>("tensor.h5", "Matrix A");

        REQUIRE((B.dim(0) == 10 && B.dim(1) == 20));
        REQUIRE(B(2, 2) == 0.0);
        REQUIRE(B(2, 5) == 1.0);
        REQUIRE(B(2, 8) == 2.0);
        REQUIRE(B(3, 2) == 3.0);
        REQUIRE(B(3, 5) == 4.0);
        REQUIRE(B(3, 8) == 5.0);
    }
}

TEST_CASE("Tensor-1D HDF5") {
    auto A = einsums::create_random_tensor("A", 3);

    {
        h5::fd_t fd = h5::create("tensor-1d.h5", H5F_ACC_TRUNC);
        h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3}, h5::max_dims{3});
        h5::write(ds, A);
    }

    {
        auto B = h5::read<einsums::Tensor<double, 1>>("tensor-1d.h5", "A");

        REQUIRE(A(0) == B(0));
        REQUIRE(A(1) == B(1));
        REQUIRE(A(2) == B(2));
    }
}

TEST_CASE("Tensor-3D HDF5") {
    auto A = einsums::create_random_tensor("A", 3, 2, 1);

    {
        h5::fd_t fd = h5::create("tensor-3d.h5", H5F_ACC_TRUNC);
        h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3, 2, 1});
        h5::write(ds, A);
    }

    {
        auto B = h5::read<einsums::Tensor<double, 3>>("tensor-3d.h5", "A");

        REQUIRE(B.dim(0) == 3);
        REQUIRE(B.dim(1) == 2);
        REQUIRE(B.dim(2) == 1);
    }
}

TEST_CASE("TensorView-2D HDF5") {
    SECTION("Subview Offset{0,0,0}") {
        auto                           A = einsums::create_random_tensor("A", 3, 3, 3);
        einsums::TensorView<double, 2> viewA(A, einsums::Dim{3, 9});

        REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
        REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

        {
            h5::fd_t fd = h5::create("tensorview-2d.h5", H5F_ACC_TRUNC);
            h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3, 9});
            h5::write(ds, viewA);
        }

        {
            auto B = h5::read<einsums::Tensor<double, 2>>("tensorview-2d.h5", "A");
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 9; j++)
                    REQUIRE(viewA(i, j) == B(i, j));
        }
    }
}

TEST_CASE("TensorView Ranges") {
    using namespace einsums;

    SECTION("Subviews") {
        auto                C = einsums::create_random_tensor("C", 3, 3);
        einsums::TensorView viewC(C, einsums::Dim{2, 2}, einsums::Offset{1, 1}, einsums::Stride{3, 1});

        // einsums::println("C strides: %zu %zu\n", C.strides()[0], C.strides()[1]);

        REQUIRE(C(1, 1) == viewC(0, 0));
        REQUIRE(C(1, 2) == viewC(0, 1));
        REQUIRE(C(2, 1) == viewC(1, 0));
        REQUIRE(C(2, 2) == viewC(1, 1));
    }

    SECTION("Subviews 2") {
        auto C = einsums::create_random_tensor("C", 3, 3);
        // std::array<einsums::Range, 2> test;
        einsums::TensorView viewC = C(einsums::Range{1, 3}, einsums::Range{1, 3});

        // einsums::println(C);
        // einsums::println(viewC);

        REQUIRE(C(1, 1) == viewC(0, 0));
        REQUIRE(C(1, 2) == viewC(0, 1));
        REQUIRE(C(2, 1) == viewC(1, 0));
        REQUIRE(C(2, 2) == viewC(1, 1));
    }

    // SECTION("Subviews 3") {
    //     auto C = create_random_tensor("C", 3, 3, 3, 3);
    //     auto viewC = C(0, 0, Range{1, 3}, Range{1, 3});

    //     println(C);
    //     println(viewC);
    // }
}

TEST_CASE("Tensor 2D - HDF5 wrapper") {
    auto A = einsums::create_random_tensor("A", 3, 3);

    h5::fd_t fd = h5::create("tensor-wrapper.h5", H5F_ACC_TRUNC);

    einsums::write(fd, A);

    auto B = einsums::read<double, 2>(fd, "A");

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            REQUIRE(B(i, j) == B(i, j));
}

TEST_CASE("reshape") {
    SECTION("1") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        REQUIRE_NOTHROW(einsums::Tensor{std::move(C), "D", 10, -1});
        // NOTE: At this point tensor C is no longer valid.
    }

    SECTION("2") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        auto D = einsums::Tensor{std::move(C), "D", 100, 10};
        // NOTE: At this point tensor C is no longer valid.

        // println(C); // <- This will cause a segfault when println tries to print the tensor elements
        // println(D); // <- This succeeds.
    }

    SECTION("3") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        REQUIRE_THROWS(einsums::Tensor{std::move(C), "D", -1, -1});
        // NOTE: At this point tensor C is no longer valid.
    }

    SECTION("4") {
        auto C = einsums::create_incremented_tensor("C", 10, 10, 10);
        REQUIRE_THROWS(einsums::Tensor{std::move(C), "D", 9, 9});
        // NOTE: At this point tensor C is no longer valid.
    }
}

template <typename Destination, typename Source>
void types_test() {
    using namespace einsums;

    auto A = create_random_tensor<Source>("A", 10, 10);
    auto B = create_random_tensor<Destination>("B", 10, 10);

    B = A;
}

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

template <typename T>
void test_tensor_from_tensorview() {
    using namespace einsums;

    auto   A  = create_incremented_tensor("A", 10, 10);
    auto   vA = TensorView(A, Dim{2, 2}, Offset{4, 4});
    Tensor B  = vA;

    REQUIRE(B(0, 0) == A(4, 4));
    REQUIRE(B(0, 1) == A(4, 5));
    REQUIRE(B(1, 0) == A(5, 4));
    REQUIRE(B(1, 1) == A(5, 5));
}

TEST_CASE("tensor_tensorview") {
    test_tensor_from_tensorview<float>();
    test_tensor_from_tensorview<double>();
    test_tensor_from_tensorview<std::complex<float>>();
    test_tensor_from_tensorview<std::complex<double>>();
}

void test_tensorview() {
    using namespace einsums;

    auto A = create_incremented_tensor("A", 10, 10);
    //    auto B = A(Dim{2, 2}, Offset{4, 4});
}

template <einsums::NotComplex T>
void arange_test() {
    auto A = einsums::arange<T>(0, 10);

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(einsums::VectorData<T>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
}

TEST_CASE("arange") {
    arange_test<double>();
    arange_test<float>();
}
*/

TEST_CASE("tiled einsum1", "[tensor]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    SECTION("ik=ij,jk") {
        TiledTensor<double, 2> A("A", std::array{1, 0, 2});
        TiledTensor<double, 2> B("B", std::array{1, 0, 2});
        TiledTensor<double, 2> C("C", std::array{1, 0, 2});

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
                B(i, j) = ij;
            }
        }
        C.zero();

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B));

        // println(A);
        // println(B);
        // println(C);

        /*[[ 30,  36,  42],
           [ 66,  81,  96],
           [102, 126, 150]]*/
        CHECK(C(0, 0) == 30.0);
        CHECK(C(0, 1) == 36.0);
        CHECK(C(0, 2) == 42.0);
        CHECK(C(1, 0) == 66.0);
        CHECK(C(1, 1) == 81.0);
        CHECK(C(1, 2) == 96.0);
        CHECK(C(2, 0) == 102.0);
        CHECK(C(2, 1) == 126.0);
        CHECK(C(2, 2) == 150.0);
    }

    SECTION("il=ijk,jkl") {
        TiledTensor<double, 3> A("A", std::array{1, 0, 2});
        TiledTensor<double, 3> B("B", std::array{1, 0, 2});
        TiledTensor<double, 2> C("C", std::array{1, 0, 2});

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                    B(i, j, k) = ij;
                }
            }
        }
        C.zero();

        // println(A);
        // println(B);
        // println(C);

        // einsum("il=ijk,jkl", &C, A, B);
        REQUIRE_NOTHROW(
            einsum(Indices{index::i, index::l}, &C, Indices{index::i, index::j, index::k}, A, Indices{index::j, index::k, index::l}, B));

        // println(C);

        // array([[ 765.,  810.,  855.],
        //        [1818., 1944., 2070.],
        //        [2871., 3078., 3285.]])
        REQUIRE(C(0, 0) == 765.0);
        REQUIRE(C(0, 1) == 810.0);
        REQUIRE(C(0, 2) == 855.0);
        REQUIRE(C(1, 0) == 1818.0);
        REQUIRE(C(1, 1) == 1944.0);
        REQUIRE(C(1, 2) == 2070.0);
        REQUIRE(C(2, 0) == 2871.0);
        REQUIRE(C(2, 1) == 3078.0);
        REQUIRE(C(2, 2) == 3285.0);
    }

    SECTION("ik=block ij,jk") {
        BlockTensor<double, 2> A("A", 1, 0, 2);
        TiledTensor<double, 2> B("B", std::array{1, 0, 2});
        TiledTensor<double, 2> C("C", std::array{1, 0, 2});

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
                B(i, j) = ij;
            }
        }
        C.zero();

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B));

        // println(A);
        // println(B);
        // println(C);

        /*[[ 1,  2,  3],
           [ 62,  73,  84],
           [ 95, 112, 129]]*/
        CHECK(C(0, 0) == 1.0);
        CHECK(C(0, 1) == 2.0);
        CHECK(C(0, 2) == 3.0);
        CHECK(C(1, 0) == 62.0);
        CHECK(C(1, 1) == 73.0);
        CHECK(C(1, 2) == 84.0);
        CHECK(C(2, 0) == 95.0);
        CHECK(C(2, 1) == 112.0);
        CHECK(C(2, 2) == 129.0);
    }

    SECTION("ik=ij,block jk") {
        BlockTensor<double, 2> A("A", 1, 0, 2);
        TiledTensor<double, 2> B("B", std::array{1, 0, 2});
        TiledTensor<double, 2> C("C", std::array{1, 0, 2});

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
                B(i, j) = ij;
            }
        }
        C.zero();

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, B, Indices{index::k, index::j}, A));

        // println(A);
        // println(B);
        // println(C);

        /*[[ 1,  34,  39],
           [ 4,  73,  84],
           [ 7, 112, 129]]*/
        CHECK(C(0, 0) == 1.0);
        CHECK(C(0, 1) == 34.0);
        CHECK(C(0, 2) == 39.0);
        CHECK(C(1, 0) == 4.0);
        CHECK(C(1, 1) == 73.0);
        CHECK(C(1, 2) == 84.0);
        CHECK(C(2, 0) == 7.0);
        CHECK(C(2, 1) == 112.0);
        CHECK(C(2, 2) == 129.0);
    }
}
