//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("einsum5", "[tensor_algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        auto gMO0 = create_tensor<TestType>("g0", 3, 3, 3, 3);
        auto gMO1 = create_tensor<TestType>("g1", 3, 3, 3, 3);
        zero(gMO0);
        zero(gMO1);
        auto A = create_random_tensor<TestType>("A", 3, 3, 3, 3);
        auto B = create_random_tensor<TestType>("B", 3, 3);

        REQUIRE_NOTHROW(einsum(Indices{p, q, r, l}, &gMO0, Indices{p, q, r, s}, A, Indices{s, l}, B));

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, k0, p0) * B(p0, l0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), TestType{0.001}));
                    }
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{p, q, k, s}, &gMO0, Indices{p, q, r, s}, A, Indices{r, k}, B));

        gMO1.zero();
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, p0, l0) * B(p0, k0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), TestType{0.001}));
                    }
                }
            }
        }
    }
}
