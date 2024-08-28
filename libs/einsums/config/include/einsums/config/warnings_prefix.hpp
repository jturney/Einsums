//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// This file can be included multiple times and does not use #pragma once.
#include <einsums/config/compiler_specific.hpp>

// suppress warnings about dependent classes not being exported from the dll
#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
#    pragma warning(push)
#    pragma warning(disable : 4251 4231 4275 4355 4660)
#    pragma warning(disable : 4355) // this used in base member initializer
#endif
