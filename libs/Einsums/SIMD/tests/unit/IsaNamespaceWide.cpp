//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The second translation unit of IsaNamespace_test, compiled at the v3 rung's flags while the
// first is compiled at the project's baseline. See IsaNamespace.cpp.

#include <Einsums/SIMD/Vec.hpp>

#include "IsaNamespaceProbe.hpp"

IsaProbe isa_probe_wide() {
    return isa_probe_here();
}
