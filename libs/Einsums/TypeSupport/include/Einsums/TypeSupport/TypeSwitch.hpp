//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums {

/**
 * @struct Case
 *
 * @brief Represents a case in a type switch.
 *
 * If the type switch matches any of the conditions, then the @c Result parameter will be used.
 *
 * @versionadded{2.0.0}
 */
template <typename Result, typename... Conditions>
struct Case final {};

/**
 * @struct Default
 *
 * @brief Represents the default case.
 *
 * @versionadded{2.0.0}
 */
template <typename Result>
struct Default final {};

/**
 * @brief Like a switch statement, but for types!
 *
 * This struct takes in cases and matches the switch to the conditions for each case.
 * Here's an example to transform a data type to its GPU equivalent, while mapping void to a type with a size.
 *
 * @code
 * Switch<T, Case<uint8_t, void, void const>, Case<hipFloatComplex, std::complex<float>>, Case<hipDoubleComplex, std::complex<double>>,
 * Default<T>>
 * @endcode
 *
 * @tparam SwitchType The type to compare.
 * @tparam Cases The cases for the switch statement.
 *
 * @versionadded{2.0.0}
 */
template <typename SwitchType, typename... Cases>
struct Switch final {
    /**
     * Function that will fail to compile because the current case is invalid.
     *
     * @versionadded{2.0.0}
     */
    static void func() { static_assert(false, "Type Switch needs cases, or all cases were false!"); }

    /**
     * The type that this switch evaluates into.
     *
     * @versionadded{2.0.0}
     */
    using type = void; // NOLINT(readability-identifier-naming)
};

// First possibility: We meet a case, but the switch does not match the first condition.
// In this case, we get what happens when we check the next condition.
template <typename SwitchType, typename Result, typename FirstCond, typename... Rest, typename... Cases>
struct Switch<SwitchType, Case<Result, FirstCond, Rest...>, Cases...> final {
    using type = typename Switch<SwitchType, Case<Result, Rest...>, Cases...>::type; // NOLINT(readability-identifier-naming)
};

// Second possibility: We meet a case where none of the conditions match. In this case,
// we check the next case.
template <typename SwitchType, typename Result, typename LastCond, typename... Cases>
struct Switch<SwitchType, Case<Result, LastCond>, Cases...> final {
    using type = typename Switch<SwitchType, Cases...>::type; // NOLINT(readability-identifier-naming)
};

// Third possibility: We meet a case where the switch matches the first condition.
// In this case, we return the result for the case.
template <typename SwitchType, typename Result, typename... Rest, typename... Cases>
struct Switch<SwitchType, Case<Result, SwitchType, Rest...>, Cases...> final {
    using type = Result; // NOLINT(readability-identifier-naming)
};

// Fourth possibility: We meet a default case, but there are still cases to process.
// This is an error.
template <typename SwitchType, typename Result, typename... Cases>
struct Switch<SwitchType, Default<Result>, Cases...> final {
    using type = void; // NOLINT(readability-identifier-naming)
    static void func() { static_assert(false, "Type Switch has a default case, but there are still cases left to process!"); }
};

// Fifth possibility: We meet a default case, and there are no more cases to process. We use the
// result of the default case.
template <typename SwitchType, typename Result>
struct Switch<SwitchType, Default<Result>> final {
    using type = Result; // NOLINT(readability-identifier-naming)
};

/**
 * @typedef SwitchT
 *
 * @brief Equivalent to <tt>typename Switch<...>::type</tt>.
 *
 * @versionadded{2.0.0}
 */
template <typename... Args>
using SwitchT = typename Switch<Args...>::type;

} // namespace einsums