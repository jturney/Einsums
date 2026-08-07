//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/ABILayout.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(sealed)

// Declared in <Einsums/Config/ABI.hpp> and defined here, which is the one place
// that can see every type it measures. world() calls it lazily rather than
// reading a global, so this crossing of the module graph carries no static
// initialization order requirement with it.
std::uint64_t library_layout_fingerprint() noexcept {
    return layout_fingerprint();
}

EINSUMS_NAMESPACE_END(sealed)
