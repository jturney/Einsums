//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file CXX23.hpp
/// @brief Umbrella header for C++23 backports.
///
/// Provides implementations of C++23 standard library features that are usable
/// in C++20 codebases. When compiling with C++23 or later (and the compiler
/// provides the feature), the standard library version is used automatically.
///
/// Available backports:
///   - einsums::expected<T, E>: a value-or-error type, the std::expected backport.
///   - einsums::unexpected<E>: the error tag for expected.

#include <Einsums/CXX23/Expected.hpp>
