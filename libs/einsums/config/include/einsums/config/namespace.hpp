//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>

#include <string>

/**
 * Handles setting up the namespace and creates a global std::string that is used
 * by the timing mechanism to track locations.
 *
 * @param x The name of the namespace to create and store the name of.
 *
 */
#define BEGIN_EINSUMS_NAMESPACE_HPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace detail {                                                                                                                     \
    EINSUMS_EXPORT const std::string &get_namespace();                                                                                     \
    }

/**
 * The matching macro for BEGIN_EINSUMS_NAMESPACE_HPP(x)
 */
#define END_EINSUMS_NAMESPACE_HPP(x) }

/**
 * The .cpp file equivalent of BEGIN_EINSUMS_NAMESPACE_HPP. Should only exist in one
 * source file; otherwise, multiple definition errors will occur.
 */
#define BEGIN_EINSUMS_NAMESPACE_CPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace detail {                                                                                                                     \
    namespace {                                                                                                                            \
    std::string s_Namespace = #x;                                                                                                          \
    }                                                                                                                                      \
    EINSUMS_EXPORT const std::string &get_namespace() {                                                                                    \
        return s_Namespace;                                                                                                                \
    }                                                                                                                                      \
    }

/**
 * Matching macro to BEGIN_EINSUMS_NAMESPACE_HPP(x)
 */
#define END_EINSUMS_NAMESPACE_CPP(x) }
