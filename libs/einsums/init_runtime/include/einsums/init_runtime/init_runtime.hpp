//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include "einsums/preprocessor/Stringize.hpp"

#include "einsums/type_support/Unused.hpp"

#if defined(EINSUMS_APPLICATION_NAME_DEFAULT) && !defined(EINSUMS_APPLICATION_NAME)
#    define EINSUMS_APPLICATION_NAME EINSUMS_APPLICATION_NAME_DEFAULT
#endif

#if !defined(EINSUMS_APPLICATION_STRING)
#    if defined(EINSUMS_APPLICATION_NAME)
#        define EINSUMS_APPLICATION_STRING EINSUMS_PP_STRINGIZE(EINSUMS_APPLICATION_NAME)
#    else
#        define EINSUMS_APPLICATION_STRING "unknown einsums application"
#    endif
#endif

namespace einsums {
namespace detail {

// Default params to initialize the init_params struct
EINSUMS_MAYBE_UNUSED static int    dummy_argc      = 1;
EINSUMS_MAYBE_UNUSED static char   app_name[]      = EINSUMS_APPLICATION_STRING;
static char                       *default_argv[2] = {app_name, nullptr};
EINSUMS_MAYBE_UNUSED static char **dummy_argv      = default_argv;

} // namespace detail

struct InitParams {};

} // namespace einsums