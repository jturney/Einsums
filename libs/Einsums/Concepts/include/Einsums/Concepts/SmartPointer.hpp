//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>
#include <type_traits>

namespace einsums {
namespace details {
/**
 * @struct IsSmartPointerHelper
 *
 * @brief Checks if a type is a std::shared_ptr, std::unique_ptr, or std::weak_ptr.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
struct IsSmartPointerHelper : public std::false_type {};

template <typename T>
struct IsSmartPointerHelper<std::shared_ptr<T>> : public std::true_type {};

template <typename T>
struct IsSmartPointerHelper<std::unique_ptr<T>> : public std::true_type {};

template <typename T>
struct IsSmartPointerHelper<std::weak_ptr<T>> : public std::true_type {};
} // namespace details

/**
 * @struct IsSmartPointerHelper
 *
 * @brief Checks if a type is a std::shared_ptr, std::unique_ptr, or std::weak_ptr.
 *
 * This one removes const and volatile.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
struct IsSmartPointerHelper : public details::IsSmartPointerHelper<typename std::remove_cv<T>::type> {};

/**
 * @var IsSmartPointerV
 *
 * @brief Checks if a type is a std::shared_ptr, std::unique_ptr, or std::weak_ptr.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
inline constexpr bool IsSmartPointerV = IsSmartPointerHelper<T>::value;

/**
 * @concept SmartPointer
 *
 * @brief Checks if a type is a std::shared_ptr, std::unique_ptr, or std::weak_ptr.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
concept SmartPointer = IsSmartPointerV<T>;

/**
 * @concept NotASmartPointer
 *
 * @brief The opposite of SmartPointer.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
concept NotASmartPointer = !IsSmartPointerV<T>;

} // namespace einsums