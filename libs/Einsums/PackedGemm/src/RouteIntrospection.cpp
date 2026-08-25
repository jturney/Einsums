//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

// One slot per thread for the whole PROCESS, which is why these are not inline.
// An inline function's thread-local is duplicated per shared object under
// hidden visibility, so a contraction run inside libEinsums would write a slot
// the test executable cannot read. See the declarations for the full note.

char const *&last_contraction_route() {
    thread_local char const *route = "none";
    return route;
}

KernelRoute &last_route_pin() {
    thread_local KernelRoute pin = KernelRoute::Adaptive;
    return pin;
}

EINSUMS_NAMESPACE_END(packed_gemm)
