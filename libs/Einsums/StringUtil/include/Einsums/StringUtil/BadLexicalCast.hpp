//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <typeinfo>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @struct bad_lexical_cast
 *
 * Indicates that a string could not be converted to a different type.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT bad_lexical_cast : std::bad_cast {
    using std::bad_cast::bad_cast;
};

EINSUMS_NAMESPACE_END()