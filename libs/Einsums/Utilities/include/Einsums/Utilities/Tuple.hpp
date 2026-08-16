//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

EINSUMS_NAMESPACE_BEGIN()
namespace detail {

template <typename T, std::size_t... Is>
constexpr auto create_tuple(T value, std::index_sequence<Is...>) {
    return std::tuple{(static_cast<void>(Is), value)...};
}

template <typename T, std::size_t... Is>
constexpr auto create_tuple_from_array(T const &arr, std::index_sequence<Is...>) {
    return std::tuple((arr[Is])...);
}

template <size_t NewN, typename T, size_t OldN, size_t... __I>
constexpr inline std::array<T, NewN> slice_array_impl(std::array<T, OldN> const &old_array, std::index_sequence<__I...> const &) {
    return {{old_array[__I]...}};
}

template <size_t NewN, typename T, size_t OldN, size_t... __I>
constexpr inline std::array<T, NewN> slice_array_impl(std::array<T, OldN> &&old_array, std::index_sequence<__I...> const &) {
    return {{std::move(old_array[__I])...}};
}

template <typename T, typename... TupleTypes, size_t... __I>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple_impl(std::tuple<TupleTypes...> const &tuple,
                                                                                   std::index_sequence<__I...> const &) {
    return {{std::get<__I>(tuple)...}};
}

template <typename T, typename... TupleTypes, size_t... __I>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple_impl(std::tuple<TupleTypes...> &&tuple,
                                                                                   std::index_sequence<__I...> const &) {
    return {{std::move(std::get<__I>(tuple))...}};
}

template <typename NewT, typename OldT, size_t N, size_t... __I>
constexpr inline std::array<std::remove_cv_t<NewT>, N> convert_array_impl(std::array<std::remove_cv_t<OldT>, N> const &arr,
                                                                          std::index_sequence<__I...> const &) {
    return {{static_cast<std::remove_cv_t<NewT>>(arr[__I])...}};
}

} // namespace detail

/**
 * @brief Creates a tuple consisting of copies of the passed value.
 *
 * @tparam N The number of copies to make.
 * @tparam T The type of the input.
 * @param value The value to copy.
 * @return A tuple consisting of copies of the passed value.
 *
 * @versionadded{1.1.0}
 */
template <size_t N, typename T>
constexpr auto create_tuple(T const &value) {
    return detail::create_tuple(value, std::make_index_sequence<N>());
}

/**
 * @brief Converts an array into a tuple with the same representation.
 *
 * @tparam N The number of elements in the array.
 * @tparam T The array type, such as std::array or std::vector.
 * @param arr The array to copy.
 * @return A tuple whose elements match the passed array.
 *
 * @versionadded{1.1.0}
 */
template <size_t N, typename T>
constexpr auto create_tuple_from_array(T const &arr) {
    return detail::create_tuple_from_array(arr, std::make_index_sequence<N>());
}

/**
 * @brief Slices an array.
 *
 * This function takes an array and returns a new one whose elements are the first several elements of the input.
 *
 * @tparam NewN The number of elements to extract.
 * @tparam T The datatype stored by the array.
 * @tparam OldN The size of the input array.
 * @param old_array The array to slice.
 * @return An array containing the first several elements of the input array.
 *
 * @versionadded{1.1.0}
 */
template <size_t NewN, typename T, size_t OldN>
    requires(OldN >= NewN)
constexpr std::array<T, NewN> slice_array(std::array<T, OldN> const &old_array) {
    return detail::slice_array_impl<NewN>(old_array, std::make_index_sequence<NewN>());
}

/**
 * @brief Slices an array.
 *
 * This function takes an array and returns a new one whose elements are the first several elements of the input.
 *
 * @tparam NewN The number of elements to extract.
 * @tparam T The datatype stored by the array.
 * @tparam OldN The size of the input array.
 * @param old_array The array to slice.
 * @return An array containing the first several elements of the input array.
 *
 * @versionadded{1.1.0}
 */
template <size_t NewN, typename T, size_t OldN>
    requires(OldN >= NewN)
constexpr std::array<T, NewN> slice_array(std::array<T, OldN> &&old_array) {
    return detail::slice_array_impl<NewN>(std::move(old_array), std::make_index_sequence<NewN>());
}

/**
 * @brief Converts a tuple whose elements can all be converted to the same type into an array.
 *
 * @tparam T The datatype of the array.
 * @tparam TupleTypes The types of the tuple's elements.
 * @param tuple The tuple to cast.
 * @return An array whose elements are cast from the input tuple's elements.
 *
 * @versionadded{1.1.0}
 */
template <typename T, typename... TupleTypes>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple(std::tuple<TupleTypes...> const &tuple) {
    return detail::create_array_from_tuple_impl<T>(tuple, std::make_index_sequence<sizeof...(TupleTypes)>());
}

/**
 * @brief Converts a tuple whose elements can all be converted to the same type into an array.
 *
 * @tparam T The datatype of the array.
 * @tparam TupleTypes The types of the tuple's elements.
 * @param tuple The tuple to cast.
 * @return An array whose elements are cast from the input tuple's elements.
 *
 * @versionadded{1.1.0}
 */
template <typename T, typename... TupleTypes>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple(std::tuple<TupleTypes...> &&tuple) {
    return detail::create_array_from_tuple_impl<T>(std::move(tuple), std::make_index_sequence<sizeof...(TupleTypes)>());
}

/**
 * @brief Converts the elements of an array to a new type.
 *
 * @tparam NewT The new type to convert to.
 * @tparam OldT The type of the elements in the input array.
 * @tparam N The number of elements.
 * @param arr The array to cast.
 * @return A new array whose elements have been cast from the input array.
 *
 * @versionadded{1.1.0}
 */
template <typename NewT, typename OldT, size_t N>
    requires(std::is_convertible_v<OldT, NewT>)
constexpr inline std::array<std::remove_cv_t<NewT>, N> convert_array(std::array<std::remove_cv_t<OldT>, N> const &arr) {
    return detail::convert_array_impl(arr, std::make_index_sequence<N>());
}

EINSUMS_NAMESPACE_END()