//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(EINSUMS_HAVE_CXX_LAMBDA_CAPTURE_DECLTYPE)
#    define EINSUMS_FORWARD(T, ...) static_cast<decltype(__VA_ARGS__) &&>(__VA_ARGS__)
#else
#    define EINSUMS_FORWARD(T, ...) static_cast<T &&>(__VA_ARGS__)
#endif
