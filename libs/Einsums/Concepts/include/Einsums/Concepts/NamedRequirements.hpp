//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <concepts>
#include <iterator>
#include <type_traits>
EINSUMS_NAMESPACE_BEGIN()

/**
 * @concept Container
 *
 * Checks that a type satisfies the Container requirement.
 *
 * @tparam T The type to check.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
concept Container = requires(T a, T b, T const ca, T const cb, T &ra, T &rb, T const &rca, T const &rcb, T &&rra, T &&rrb) {
    // Check that the types exist.
    typename T::value_type;
    typename T::reference;
    typename T::const_reference;
    typename T::iterator;
    typename T::const_iterator;
    typename T::difference_type;
    typename T::size_type;

    // Check the properties of the types.
    requires std::forward_iterator<typename T::iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::iterator>, typename T::value_type>;
    requires std::is_convertible_v<typename T::iterator, typename T::const_iterator>;
    requires std::forward_iterator<typename T::const_iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::const_iterator>, typename T::value_type>;
    requires std::is_integral_v<typename T::difference_type>;
    requires std::is_signed_v<typename T::difference_type>;
    requires std::is_integral_v<typename T::size_type>;
    requires std::is_unsigned_v<typename T::size_type>;
    requires sizeof(typename T::size_type) >= sizeof(typename T::difference_type);

    // Expressions.
    T();
    T(a);
    T(ca);
    a  = b;
    a  = ca;
    ra = a;
    ra = ca;
    a.~T();
    { a.begin() } -> std::same_as<typename T::iterator>;
    { ca.begin() } -> std::same_as<typename T::const_iterator>;
    { a.end() } -> std::same_as<typename T::iterator>;
    { ca.end() } -> std::same_as<typename T::const_iterator>;
    { a.cbegin() } -> std::same_as<typename T::const_iterator>;
    { a.cend() } -> std::same_as<typename T::const_iterator>;

    { a == b } -> std::same_as<bool>;
    { a == cb } -> std::same_as<bool>;
    { ca == b } -> std::same_as<bool>;
    { ca == cb } -> std::same_as<bool>;
    { a != b } -> std::same_as<bool>;
    { a != cb } -> std::same_as<bool>;
    { ca != b } -> std::same_as<bool>;
    { ca != cb } -> std::same_as<bool>;

    a.swap(b);
    ra.swap(b);
    a.swap(rb);
    ra.swap(rb);
    requires std::swappable<T>;

    { a.size() } -> std::same_as<typename T::size_type>;
    { ca.size() } -> std::same_as<typename T::size_type>;
    { a.max_size() } -> std::same_as<typename T::size_type>;
    { ca.max_size() } -> std::same_as<typename T::size_type>;

    { a.empty() } -> std::same_as<bool>;
    { ca.empty() } -> std::same_as<bool>;
};

/**
 * @concept ContainerOrInitializer
 *
 * Checks that a type satisfies the Container requirement or is an inizializer list.
 *
 * @tparam T The type to check
 *
 * @versionadded{1.1.0}
 */
template <typename T>
concept ContainerOrInitializer = requires(T a, T b, T const ca, T const cb, T &ra, T &rb, T const &rca, T const &rcb, T &&rra, T &&rrb) {
    // Check that the types exist.
    typename T::value_type;
    typename T::reference;
    typename T::const_reference;
    typename T::iterator;
    typename T::const_iterator;
    typename T::size_type;

    // Check the properties of the types.
    requires std::forward_iterator<typename T::iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::iterator>, typename T::value_type>;
    requires std::is_convertible_v<typename T::iterator, typename T::const_iterator>;
    requires std::forward_iterator<typename T::const_iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::const_iterator>, typename T::value_type>;
    requires std::is_integral_v<typename T::size_type>;
    requires std::is_unsigned_v<typename T::size_type>;

    // Expressions.
    T();
    { a.begin() } -> std::same_as<typename T::iterator>;
    { ca.begin() } -> std::same_as<typename T::const_iterator>;
    { a.end() } -> std::same_as<typename T::iterator>;
    { ca.end() } -> std::same_as<typename T::const_iterator>;

    { a.size() } -> std::same_as<typename T::size_type>;
    { ca.size() } -> std::same_as<typename T::size_type>;
    { a.max_size() } -> std::same_as<typename T::size_type>;
};

/**
 * @concept ContiguousContainer
 *
 * Check to see if a type satisfies the Container requirement and stores its data contiguously.
 * This means that vectors and arrays should satisfy, but things like linked lists should not.
 *
 * @tparam T The type to check.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
concept ContiguousContainer = requires(T a) {
    requires Container<T>;
    { a.data() } -> std::same_as<typename T::value_type *>;
};

/**
 * @concept ContiguousContainerOf
 *
 * Check to see if a type satisfies the Container requirement and stores its data contiguously.
 * Also, check that the container stores values of a specific type.
 * This means that vectors and arrays should satisfy, but things like linked lists should not.
 *
 * @tparam T The type to check.
 * @tparam Holds The type of objects the container should hold.
 *
 * @versionadded{2.0.0}
 */
template <typename T, typename Holds>
concept ContiguousContainerOf = requires {
    requires ContiguousContainer<T>;
    requires std::same_as<typename T::value_type, Holds>;
};

/**
 * @concept ContainerOf
 *
 * Check to see if a type satisfies the Container requirement and stores values of a certain type.
 *
 * @tparam T The type to check.
 * @tparam Holds The type of objects the container should hold.
 *
 * @versionadded{2.0.0}
 */
template <typename T, typename Holds>
concept ContainerOf = requires {
    requires Container<T>;
    requires std::same_as<typename T::value_type, Holds>;
};

/**
 * @concept ContainerOf
 *
 * Check to see if a type satisfies the Container requirement or is an initializer list and stores values of a certain type.
 *
 * @tparam T The type to check.
 * @tparam Holds The type of objects the container should hold.
 *
 * @versionadded{2.0.0}
 */
template <typename T, typename Holds>
concept ContainerOrInitializerOf = requires {
    requires ContainerOrInitializer<T>;
    requires std::same_as<typename T::value_type, Holds>;
};

EINSUMS_NAMESPACE_END()