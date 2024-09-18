//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_TRACY)
#    if defined(__has_include)
// Newer versions of tracy have Tracy.hpp in the subdirectory tracy
#        if __has_include(<tracy/Tracy.hpp>)
#            include <tracy/Tracy.hpp>
#        else
#            include <Tracy.hpp>
#        endif
// If we can't detect tracy's includes we assume it is new enough to use the
// tracy subdirectory
#    else
#        include <tracy/Tracy.hpp>
#    endif
#endif
