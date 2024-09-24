//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#if defined(__GNUC__)
#define EINSUMS_SUPER_PURE __attribute__((const))
#define EINSUMS_PURE __attribute__((pure))
#define EINSUMS_HOT __attribute__((hot))
#define EINSUMS_COLD __attribute__((cold))
#else
#define EINSUMS_SUPER_PURE
#define EINSUMS_PURE
#define EINSUMS_HOT
#define EINSUMS_COLD
#endif