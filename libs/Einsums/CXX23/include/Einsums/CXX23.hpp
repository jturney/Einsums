//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file CXX23.hpp
/// @brief Umbrella header for C++23 backports.
///
/// C++23 library features for C++20; the standard versions are used when available.
///
/// Available backports:
///   - einsums::expected<T, E>: a value-or-error type, the std::expected backport.
///   - einsums::unexpected<E>: the error tag for expected.

#include <Einsums/CXX23/Expected.hpp>
