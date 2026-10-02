//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file EnumNames.hpp
 * @brief One table per enum, answering both directions of the name question.
 *
 * The spellings are held once, so naming and parsing cannot drift apart.
 */

#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief A value-to-spelling table for one enumeration.
 *
 * @tparam E The enumeration.
 * @tparam N How many enumerators the table names.
 *
 * Construct one at namespace scope with the enumerators and their spellings,
 * and the fallback @ref name returns for a value the table does not list:
 *
 * @code
 * constexpr EnumNames kBisectModeNames{std::array<std::pair<BisectMode, std::string_view>, 2>{{
 *     {BisectMode::Individual, "individual"},
 *     {BisectMode::Cumulative, "cumulative"},
 * }}, "individual"};
 * @endcode
 *
 * A linear scan, usable in constant expressions.
 *
 * @versionadded{2.0.0}
 */
template <typename E, std::size_t N>
struct EnumNames {
    /// The enumerators and their spellings, in the order a listing should show them.
    std::array<std::pair<E, std::string_view>, N> entries;

    /// What @ref name answers for a value the table does not list.
    std::string_view fallback;

    /// @brief The spelling of @p value.
    /// @param[in] value The enumerator to name.
    /// @return Its spelling, or @ref fallback when the table does not list it.
    [[nodiscard]] constexpr std::string_view name(E value) const noexcept {
        for (auto const &[key, spelling] : entries) {
            if (key == value) {
                return spelling;
            }
        }
        return fallback;
    }

    /// @brief The enumerator spelled @p spelling, if there is one.
    /// @param[in] spelling A spelling @ref name produces.
    /// @return The enumerator, or an empty optional so a loader can report the string.
    [[nodiscard]] constexpr std::optional<E> from_name(std::string_view spelling) const noexcept {
        for (auto const &[key, text] : entries) {
            if (text == spelling) {
                return key;
            }
        }
        return std::nullopt;
    }
};

// Deduce E and N from the table an EnumNames is built over.
template <typename E, std::size_t N>
EnumNames(std::array<std::pair<E, std::string_view>, N>, std::string_view) -> EnumNames<E, N>;

EINSUMS_NAMESPACE_END(compute_graph)
