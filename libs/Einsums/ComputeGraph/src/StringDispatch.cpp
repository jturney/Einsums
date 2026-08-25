//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

// One slot per thread for the whole PROCESS, which is why this is not inline.
// See the declaration in StringDispatch.hpp for why that matters.
char const *&last_dispatch_route() {
    thread_local char const *route = "none";
    return route;
}

EINSUMS_NAMESPACE_END(compute_graph::dispatch)
