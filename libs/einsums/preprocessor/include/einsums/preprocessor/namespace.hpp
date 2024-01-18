//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>

#include <string>

/**
 * @brief Handles setting up the namespace and creates a global std::string that is used by the timing mechanism to
 * track locations.
 *
 */
#define BEGIN_EINSUMS_NAMESPACE_HPP(x)                                                                                 \
    namespace x {                                                                                                      \
    namespace detail {                                                                                                 \
    extern EINSUMS_EXPORT std::string s_Namespace;                                                                     \
    }

/**
 * @brief The matching macro for BEGIN_EINSUMS_NAMESPACE_HPP(x)
 *
 */
#define END_EINSUMS_NAMESPACE_HPP(x) }

/**
 * @brief The .cpp file equivalent of BEGIN_EINSUMS_NAMESPACE_HPP. Should only exist in one source file; otherwise,
 * multiple definition errors will occur.
 */
#define BEGIN_EINSUMS_NAMESPACE_CPP(x)                                                                                 \
    namespace x {                                                                                                      \
    namespace detail {                                                                                                 \
    EINSUMS_EXPORT std::string s_Namespace = #x;                                                                       \
    }

/**
 * @brief Matching macro to BEGIN_EINSUMS_NAMESPACE_HPP(x)
 */
#define END_EINSUMS_NAMESPACE_CPP(x) }
