//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/GroupedBatchedGemm.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

bool grouped_gemm_group_profiling() {
    // Read fresh rather than caching in a static: a capture can happen before
    // or after the option is set, and freezing the first answer would make the
    // flag depend on which graph was built first. Guarded because the config
    // singleton is only there once the runtime is up, and building a graph
    // without one is legal.
    try {
        return GlobalConfigMap::get_singleton().get_bool("graph-profile-groups", false);
    } catch (...) {
        return false;
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
