//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorUtilities/BlockViews.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Block tensor views", "[tensor]", float, double, std::complex<float>, std::complex<double>) {

    using namespace einsums;

    SECTION("10x10 all same") {

        auto A = BlockTensor<TestType, 2>("Test tensor", 5, 5);

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                A(i, j)         = i + 5 * j + 1;
                A(i + 5, j + 5) = i + 5 * j + 26;
            }
        }

        auto A_view = apply_view(A, std::vector<Range>{Range{0, 3}, Range{1, 4}});

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(A(i, j) == A_view(i, j));
                REQUIRE(A(i + 6, j + 6) == A_view(i + 3, j + 3));
            }
        }

        // Modify.
        auto B = A;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A_view(i, j)         = 0;
                A_view(i + 3, j + 3) = 0;
            }
        }

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (i < 3 && j < 3) {
                    REQUIRE(A(i, j) == TestType{0.0});
                    REQUIRE(A(i, j) != B(i, j));
                } else {
                    REQUIRE(A(i, j) != TestType{0.0});
                    REQUIRE(A(i, j) == B(i, j));
                }

                if (0 < i && i < 4 && 0 < j && j < 4) {
                    REQUIRE(A(i + 5, j + 5) == TestType{0.0});
                    REQUIRE(A(i + 5, j + 5) != B(i + 5, j + 5));
                } else {
                    REQUIRE(A(i + 5, j + 5) != TestType{0.0});
                    REQUIRE(A(i + 5, j + 5) == B(i + 5, j + 5));
                }
            }
        }
    }

    SECTION("STO-3G water coefficients") {
        BlockTensor<TestType, 2> C_matrix{"Overlap", 4, 0, 1, 2};

        C_matrix[0] = create_random_tensor<TestType>("A1", 4, 4);
        C_matrix[1].set_name("A2");
        C_matrix[2] = create_random_tensor<TestType>("B1", 1, 1);
        C_matrix[3] = create_random_tensor<TestType>("B2", 2, 2);

        auto C_occ = apply_view(C_matrix, std::vector<Range>{Range{0, 4}, Range{0, 0}, Range{0, 1}, Range{0, 2}},
                                std::vector<Range>{Range{0, 3}, Range{0, 0}, Range{0, 1}, Range{0, 1}});

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(C_matrix[0](i, j) == C_occ.tile(0, 0)(i, j));
            }
        }

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix[2](i, j) == C_occ.tile(2, 2)(i, j));
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix[3](i, j) == C_occ.tile(3, 3)(i, j));
            }
        }

        C_occ.zero_no_clear();

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(C_matrix[0](i, j) == TestType{0.0});
            }
            REQUIRE(C_matrix[0](i, 3) != TestType{0.0});
        }

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix[2](i, j) == TestType{0.0});
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix[3](i, j) == TestType{0.0});
            }
            REQUIRE(C_matrix[3](i, 1) != TestType{0.0});
        }
    }

    SECTION("STO-3G water but tiled") {
        TiledTensor<TestType, 2> C_matrix{"Overlap", std::array{4, 0, 1, 2}};

        C_matrix.tile(0, 0) = create_random_tensor<TestType>("A1", 4, 4);
        C_matrix.tile(1, 1).set_name("A2");
        C_matrix.tile(2, 2) = create_random_tensor<TestType>("B1", 1, 1);
        C_matrix.tile(3, 3) = create_random_tensor<TestType>("B2", 2, 2);

        auto C_occ = apply_view(C_matrix, std::vector<Range>{Range{0, 4}, Range{0, 0}, Range{0, 1}, Range{0, 2}},
                                std::vector<Range>{Range{0, 3}, Range{0, 0}, Range{0, 1}, Range{0, 1}});

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(C_matrix.tile(0, 0)(i, j) == C_occ.tile(0, 0)(i, j));
            }
        }

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix.tile(2, 2)(i, j) == C_occ.tile(2, 2)(i, j));
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix.tile(3, 3)(i, j) == C_occ.tile(3, 3)(i, j));
            }
        }

        C_occ.zero_no_clear();

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(C_matrix.tile(0, 0)(i, j) == TestType{0.0});
            }
            REQUIRE(C_matrix.tile(0, 0)(i, 3) != TestType{0.0});
        }

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix.tile(2, 2)(i, j) == TestType{0.0});
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 1; j++) {
                REQUIRE(C_matrix.tile(3, 3)(i, j) == TestType{0.0});
            }
            REQUIRE(C_matrix.tile(3, 3)(i, 1) != TestType{0.0});
        }
    }
}
