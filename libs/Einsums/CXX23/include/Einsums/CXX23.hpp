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
///   - einsums::expected<T, E>  — value-or-error type (std::expected)
///   - einsums::unexpected<E>   — error tag for expected
///   - einsums::unreachable()   — mark code as unreachable (std::unreachable)
///   - einsums::flat_set<K>     — sorted set on contiguous storage (std::flat_set)
///   - einsums::flat_map<K, V>  — sorted map on contiguous storage (std::flat_map)

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/CXX23/FlatMap.hpp>
#include <Einsums/CXX23/FlatSet.hpp>
#include <Einsums/CXX23/Unreachable.hpp>
