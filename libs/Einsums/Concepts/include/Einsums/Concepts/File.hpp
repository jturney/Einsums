//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdio>
#include <ostream>
#include <type_traits>

namespace einsums {

/**
 * @concept IsFilePointer
 *
 * @brief Tests if the given type is a file pointer.
 *
 * Only returns true if the type is ``FILE *``.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
concept IsFilePointer = std::is_same_v<std::remove_cvref_t<T>, FILE *>;

/**
 * @concept IsOStream
 *
 * @brief Tests if the given type is a std::ostream.
 *
 * Also works on types that inherit std::ostream.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
concept IsOStream = std::is_base_of_v<std::ostream, std::remove_cvref_t<T>>;

/**
 * @concept FileOrOStream
 *
 * @brief Checks both IsFilePointer and IsOStream and returns true if either is true.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
concept FileOrOStream = IsFilePointer<T> || IsOStream<T>;

} // namespace einsums