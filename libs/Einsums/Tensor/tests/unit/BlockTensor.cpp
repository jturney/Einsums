//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Block Tensor creation", "[tensor][block-tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    BlockTensor<TestType, 2> A("A", 3, 3);
    BlockTensor<TestType, 2> B("B", 3, 3);
    BlockTensor<TestType, 2> C;

    REQUIRE((A.dim(0) == 6 && A.dim(1) == 6));
    REQUIRE((B.dim(0) == 6 && B.dim(1) == 6));

    A.zero();
    B.zero();

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            CHECK(A(i, j) == TestType(0.0));
            CHECK(B(i, j) == TestType(0.0));
        }
    }

    // Set A and B to identity
    A(0, 0) = TestType(1.0);
    A(1, 1) = TestType(1.0);
    A(2, 2) = TestType(1.0);
    A(3, 3) = TestType(1.0);
    A(4, 4) = TestType(1.0);
    A(5, 5) = TestType(1.0);

    B(0, 0) = TestType(1.0);
    B(1, 1) = TestType(1.0);
    B(2, 2) = TestType(1.0);
    B(3, 3) = TestType(1.0);
    B(4, 4) = TestType(1.0);
    B(5, 5) = TestType(1.0);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(A(i, j) == TestType(1.0));
                CHECK(B(i, j) == TestType(1.0));
            } else {
                CHECK(A(i, j) == TestType(0.0));
                CHECK(B(i, j) == TestType(0.0));
            }
        }
    }

    // Set up C using a different method.
    auto block1 = Tensor<TestType, 2>("block1", 3, 3), block2 = Tensor<TestType, 2>("block2", 3, 3);
    C.push_block(block2);
    C.insert_block(0, block1);
    C.set_name("C");
    C.zero();

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(C(i, j) == TestType(1.0));
            } else {
                CHECK(C(i, j) == TestType(0.0));
            }
        }
    }

    // Check by blocks.
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                CHECK(A[0](i, j) == TestType(1.0));
                CHECK(B[0](i, j) == TestType(1.0));
                CHECK(A[1](i, j) == TestType(1.0));
                CHECK(B[1](i, j) == TestType(1.0));
                CHECK(C["block1"](i, j) == TestType(1.0));
                CHECK(C["block2"](i, j) == TestType(1.0));
            } else {
                CHECK(A[0](i, j) == TestType(0.0));
                CHECK(B[0](i, j) == TestType(0.0));
                CHECK(A[1](i, j) == TestType(0.0));
                CHECK(B[1](i, j) == TestType(0.0));
                CHECK(C["block1"](i, j) == TestType(0.0));
                CHECK(C["block2"](i, j) == TestType(0.0));
            }
        }
    }

    CHECK_THROWS(C["block3"]);

    A.set_all(TestType(2.0));

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i / 3 == j / 3) {
                REQUIRE(A.block_of(i) == A.block_of(j));
            } else {
                REQUIRE(A.block_of(i) != A.block_of(j));
            }
            if (A.block_of(i) == A.block_of(j)) {
                CHECK(A(i, j) == TestType(2.0));
            } else {
                CHECK(A(i, j) == TestType(0.0));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Block Tensor operations", "[tensor][block-tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    BlockTensor<TestType, 2> A("A", 3, 3);
    BlockTensor<TestType, 2> B("B", 3, 3);

    A.zero();
    B.zero();

    // Set A and B to identity
    A(0, 0) = TestType(1.0);
    A(1, 1) = TestType(1.0);
    A(2, 2) = TestType(1.0);
    A(3, 3) = TestType(1.0);
    A(4, 4) = TestType(1.0);
    A(5, 5) = TestType(1.0);

    B(0, 0) = TestType(1.0);
    B(1, 1) = TestType(1.0);
    B(2, 2) = TestType(1.0);
    B(3, 3) = TestType(1.0);
    B(4, 4) = TestType(1.0);
    B(5, 5) = TestType(1.0);

    A *= TestType(2);
    A += TestType(2);
    A /= TestType(2);
    A -= TestType(2);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(A(i, j) == TestType(0.0));
            } else if (A.block_of(i) == A.block_of(j)) {
                CHECK(A(i, j) == TestType(-1.0));
            } else {
                CHECK(A(i, j) == TestType(0.0));
            }
        }
    }

    A += B; // Swap here to avoid division by zero.
    B *= A;
    B /= A; // Either way there would be a division by zero. Do the addition first to not.
    B -= A;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                CHECK(A(i, j) == TestType(1.0));
                CHECK(B(i, j) == TestType(0.0));
            } else if (A.block_of(i) == A.block_of(j)) {
                CHECK(A(i, j) == TestType(-1.0));
                CHECK(B(i, j) == TestType(1.0));
            } else {
                CHECK(A(i, j) == TestType(0.0));
                CHECK(B(i, j) == TestType(0.0));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Block Tensor Info", "[tensor][block-tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    BlockTensor<TestType, 2> A("A", 3, 3);

    for (int i = 0; i < 6; i++) {
        CHECK(A.block_of(i) == i / 3);
    }

    CHECK_THROWS(A.block_of(6));
    CHECK_THROWS(A.block_of(-1));
}

/*
TEST_CASE("Tensor GEMMs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor B("B", 3, 3);
    einsums::Tensor C("C", 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    A.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    B.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};

    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                    330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0}));

    einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                    726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0}));

    einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                    154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0}));

    einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                    330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0}));
}

TEST_CASE("Tensor GEMVs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor x("x", 3);
    einsums::Tensor y("y", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));
    REQUIRE((y.dim(0) == 3));

    A.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    x.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{11.0, 22.0, 33.0};

    einsums::linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
    CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{154.0, 352.0, 550.0}));

    einsums::linear_algebra::gemv<true>(1.0, A, x, 0.0, &y);
    CHECK_THAT(y.vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{330.0, 396.0, 462.0}));
}

TEST_CASE("Tensor SYEVs", "[tensor]") {
    einsums::Tensor A("A", 3, 3);
    einsums::Tensor x("x", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));

    A.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    einsums::linear_algebra::syev(&A, &x);

    CHECK_THAT(x(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(x(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(x(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));
}

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

    CHECK_THAT(A.vector_data(),
               Catch::Matchers::Approx(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                           -5.0 / 12, 0.25, 1.0 / 3.0, 7.0 / 12.0, 0.25, -2.0 / 3.0, 1.0 / 12.0, -0.25, 1.0 / 3.0})
                   .margin(0.00001));
}

TEST_CASE("TensorView creation", "[tensor]") {
    // With the aid of deduction guides we can choose to not specify the rank on the tensor
    einsums::Tensor     A("A", 3, 3, 3);
    einsums::TensorView viewA(A, einsums::Dim<2>{3, 9});

    // Since we are changing the underlying datatype to float the deduction guides will not work.
    einsums::Tensor     fA("A", 3, 3, 3);
    einsums::TensorView fviewA(fA, einsums::Dim<2>{3, 9});

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
        einsums::TensorView<double, 2> viewA(A, einsums::Dim<2>{3, 9});

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
        einsums::TensorView viewC(C, einsums::Dim<2>{2, 2}, einsums::Offset<2>{1, 1}, einsums::Stride<2>{3, 1});

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
*/
