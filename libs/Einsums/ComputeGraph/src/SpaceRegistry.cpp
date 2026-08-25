//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

SpaceRegistry &global_space_registry() {
    // Function-local static: constructed on first use, so a space registered from a namespace-scope
    // initializer in some other translation unit cannot race the registry into existence. It is
    // never destroyed before the spaces that name it stop being queried, which is why this is a
    // reference rather than an object with a destructor ordered against other statics.
    static SpaceRegistry registry;
    return registry;
}

EINSUMS_NAMESPACE_END(compute_graph)
