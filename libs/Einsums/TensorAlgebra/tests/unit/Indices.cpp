//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Detail/Index.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Index strings", "[tensor-algebra]") {
    using namespace einsums;

    // Pick several indices.
    std::ostringstream x_stream, X_stream, alpha_stream, Alpha_stream, a_stream, A_stream;

    x_stream << index::x;
    X_stream << index::X;
    alpha_stream << index::alpha;
    Alpha_stream << index::Alpha;
    a_stream << index::a;
    A_stream << index::A;

    auto x_str = x_stream.str(), X_str = X_stream.str(), alpha_str = alpha_stream.str(), Alpha_str = Alpha_stream.str(),
         a_str = a_stream.str(), A_str = A_stream.str();

    REQUIRE(x_str == "x");
    REQUIRE(X_str == "X");
    REQUIRE(alpha_str == "alpha");
    REQUIRE(Alpha_str == "Alpha");
    REQUIRE(a_str == "a");
    REQUIRE(A_str == "A");
}
