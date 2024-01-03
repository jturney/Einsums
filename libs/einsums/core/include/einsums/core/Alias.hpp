//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define ALIAS_TEMPLATE_FUNCTION(highLevelFunction, lowLevelFunction)                                                                       \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args)->decltype(lowLevelFunction(std::forward<Args>(args)...)) {                               \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }
