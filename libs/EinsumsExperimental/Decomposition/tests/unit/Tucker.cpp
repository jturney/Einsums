//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Decomposition/Tucker.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/TensorUtilities/RMSD.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("TUCKER 1", "[decomposition]", float, double) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor test1        = create_tensor<TestType>("test 1", 3, 3, 3);
    test1.vector_data() = {0.94706517, 0.3959549,  0.14122476, 0.83665482, 0.27340639, 0.29811429, 0.1823041,  0.66556282, 0.73178046,
                           0.72504222, 0.58360409, 0.68301135, 0.8316929,  0.66955444, 0.25182224, 0.24108674, 0.09582611, 0.93056666,
                           0.60919366, 0.97363788, 0.24531965, 0.23757898, 0.43426057, 0.64763913, 0.61224901, 0.86068415, 0.12051599};

    std::vector<size_t> ranks{2, 2, 2};
    auto                result   = tucker_ho_svd(test1, ranks);
    auto                g_tensor = std::get<0>(result);
    auto                factors  = std::get<1>(result);

    auto test1_ho_svd = tucker_reconstruct(g_tensor, factors);

    TestType diff = rmsd(test1, test1_ho_svd);

    REQUIRE(diff >= 0.0);
    REQUIRE(diff <= 0.178837);

    result   = tucker_ho_oi(test1, ranks, 50, 1.0e-6);
    g_tensor = std::get<0>(result);
    factors  = std::get<1>(result);

    auto test1_ho_oi = tucker_reconstruct(g_tensor, factors);

    diff = rmsd(test1, test1_ho_oi);

    REQUIRE(diff >= 0.0);
    REQUIRE(diff <= 0.173911);
}

TEMPLATE_TEST_CASE("TUCKER 2", "[decomposition]", float, double) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor test2        = create_tensor<TestType>("test 2", 3, 4, 2);
    test2.vector_data() = {0.29945093, 0.0090937,  0.99788559, 0.0821231,  0.29625705, 0.80278977, 0.15189681, 0.35832086,
                           0.09648153, 0.39398175, 0.49662056, 0.83101396, 0.84288292, 0.48603425, 0.93286471, 0.47101289,
                           0.32736096, 0.50067919, 0.49932342, 0.91922942, 0.44777189, 0.23009644, 0.34874549, 0.19356636};

    std::vector<size_t> ranks{2, 3, 2};
    auto                result   = tucker_ho_svd(test2, ranks);
    auto                g_tensor = std::get<0>(result);
    auto                factors  = std::get<1>(result);

    auto test2_ho_svd = tucker_reconstruct(g_tensor, factors);

    TestType diff = rmsd(test2, test2_ho_svd);

    REQUIRE(diff >= 0.0);
    REQUIRE(diff <= 0.110250);

    result   = tucker_ho_oi(test2, ranks, 50, 1.0e-6);
    g_tensor = std::get<0>(result);
    factors  = std::get<1>(result);

    auto test2_ho_oi = tucker_reconstruct(g_tensor, factors);

    diff = rmsd(test2, test2_ho_oi);

    REQUIRE(diff >= 0.0);
    REQUIRE(diff <= 0.108301);
}

TEMPLATE_TEST_CASE("TUCKER 3", "[decomposition]", float, double) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor test3 = create_tensor<TestType>("test 3", 3, 2, 3, 2);

    test3.vector_data() = {0.37001224, 0.77676895, 0.17589323, 0.02762156, 0.21037116, 0.83686174, 0.35042434, 0.19117270, 0.58095640,
                           0.99220655, 0.33536840, 0.15210615, 0.95033534, 0.73212124, 0.31346639, 0.83961596, 0.15418801, 0.58927303,
                           0.46744825, 0.44001279, 0.50372353, 0.09696069, 0.96449749, 0.71151666, 0.72334792, 0.98646368, 0.13764230,
                           0.95949904, 0.07774470, 0.18239083, 0.82591821, 0.40939436, 0.22088749, 0.90281597, 0.37465773, 0.02541923};

    std::vector<size_t> ranks{2, 2, 2, 2};
    auto                result   = tucker_ho_svd(test3, ranks);
    auto                g_tensor = std::get<0>(result);
    auto                factors  = std::get<1>(result);

    auto test3_ho_svd = tucker_reconstruct(g_tensor, factors);

    TestType diff = rmsd(test3, test3_ho_svd);

    REQUIRE(diff >= 0.0);
    REQUIRE(diff <= 0.196843);

    result   = tucker_ho_oi(test3, ranks, 50, 1.0e-6);
    g_tensor = std::get<0>(result);
    factors  = std::get<1>(result);

    auto test3_ho_oi = tucker_reconstruct(g_tensor, factors);

    diff = rmsd(test3, test3_ho_oi);

    REQUIRE(diff >= 0.0);
    REQUIRE(diff <= 0.192402);
}
