//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(EINSUMS_HAVE_CXX23_STATIC_CALL_OPERATOR) &&                                                                                    \
    (!defined(EINSUMS_HAVE_GPU_SUPPORT) || defined(EINSUMS_HAVE_CXX23_STATIC_CALL_OPERATOR_GPU))
#    define EINSUMS_STATIC_CALL_OPERATOR(...) static operator()(__VA_ARGS__)
#else
#    define EINSUMS_STATIC_CALL_OPERATOR(...) operator()(__VA_ARGS__) const
#endif
