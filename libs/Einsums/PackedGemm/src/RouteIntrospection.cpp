//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

// Out of line, so the process has one slot per thread (inline thread_locals duplicate per library).

char const *&last_contraction_route() {
    thread_local char const *route = "none";
    return route;
}

char const *&last_packed_engine() {
    thread_local char const *engine = "none";
    return engine;
}

int &last_team_size() {
    thread_local int size = 1;
    return size;
}

PackedBlocking &last_packed_blocking() {
    thread_local PackedBlocking blocking{};
    return blocking;
}

KernelRoute &last_route_pin() {
    thread_local KernelRoute pin = KernelRoute::Adaptive;
    return pin;
}

EINSUMS_NAMESPACE_END(packed_gemm)
