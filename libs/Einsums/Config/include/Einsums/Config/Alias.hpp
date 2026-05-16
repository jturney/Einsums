//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
/**
 * @def ALIAS_TEMPLATE_FUNC
 *
 * Creates an alias of a template function.
 *
 * @param highLevelFunction The new name for the function.
 * @param lowLevelFunction The old name for the function.
 *
 * @versionadded{1.0.0}
 */
#define ALIAS_TEMPLATE_FUNC(highLevelFunction, lowLevelFunction)                                                                           \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args)->decltype(lowLevelFunction(std::forward<Args>(args)...)) {                               \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }
