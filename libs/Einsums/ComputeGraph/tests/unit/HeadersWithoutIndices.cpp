//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// ComputeGraph's public header must not reach the compile-time index machinery. The
// CompileTimeIndices check reads the module's own text; this reads what the compiler actually
// includes, so it also catches TensorAlgebra/Detail/Index.hpp arriving through another module's
// header. It fails at compile time, which is the only time it can.

#include <Einsums/ComputeGraph.hpp>

#if defined(EINSUMS_TENSOR_ALGEBRA_DETAIL_INDEX_HPP)
#    error                                                                                                                                 \
        "<Einsums/ComputeGraph.hpp> includes TensorAlgebra/Detail/Index.hpp. Find the include with -H and route it through a header that carries no index types."
#endif

#include <Einsums/Testing.hpp>

TEST_CASE("ComputeGraph's header does not include the compile-time index machinery", "[ComputeGraph]") {
    SUCCEED("checked by the preprocessor when this file compiled");
}
