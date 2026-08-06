//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// A stand-in for libEinsums, small enough to build twice.
///
/// The two-copies problem cannot be tested by building Einsums twice, so this
/// is the smallest thing that reproduces its shape: one source compiled into
/// two shared libraries that export the SAME symbol names and return DIFFERENT
/// values, plus a consumer linked against each. If a consumer ever reports the
/// other copy's value, the two libraries have interposed, which is precisely
/// the corruption the sealed-worlds work exists to prevent.
///
/// C linkage on purpose. C++ mangling would be a second variable, and the
/// symbols most at risk of silent interposition are exactly the ones no
/// namespace scheme can protect.

#pragma once

#if defined(_WIN32)
#    define MINIWORLD_EXPORT __declspec(dllexport)
#else
#    define MINIWORLD_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

/// A constant baked in at compile time, different in each copy. Which value a
/// caller observes says which copy it reached.
MINIWORLD_EXPORT int miniworld_value(void);

/// Address of a function-local static, one per loaded copy. The same trick
/// einsums::sealed::world() uses, and the reason it must not be inline.
MINIWORLD_EXPORT void const *miniworld_identity(void);
}
