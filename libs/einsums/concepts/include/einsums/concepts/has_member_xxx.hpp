//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/preprocessor/cat.hpp>

#include <type_traits>

/// This macro creates a boolean unary meta-function which result is
/// true if and only if its parameter type has member function with
/// MEMBER name (no matter static it is or not). The generated trait
/// ends up in a namespace where the macro itself has been placed.
#define EINSUMS_HAS_MEMBER_XXX_TRAIT_DEF(MEMBER)                                                                                           \
    namespace EINSUMS_PP_CAT(EINSUMS_PP_CAT(has_, MEMBER), _detail) {                                                                      \
    struct helper {                                                                                                                        \
        void MEMBER(...);                                                                                                                  \
    };                                                                                                                                     \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    struct helper_composed : T, helper {};                                                                                                 \
                                                                                                                                           \
    template <void (helper::*)(...)>                                                                                                       \
    struct member_function_holder {};                                                                                                      \
                                                                                                                                           \
    template <typename T, typename Ambiguous = member_function_holder<&helper::MEMBER>>                                                    \
    struct impl : std::true_type {};                                                                                                       \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    struct impl<T, member_function_holder<&helper_composed<T>::MEMBER>> : std::false_type {};                                              \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename T, typename Enable = void>                                                                                          \
    struct EINSUMS_PP_CAT(has_, MEMBER) : std::false_type {};                                                                              \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    struct EINSUMS_PP_CAT(has_, MEMBER)<T, std::enable_if_t<std::is_class_v<T>>>                                                           \
        : EINSUMS_PP_CAT(EINSUMS_PP_CAT(has_, MEMBER), _detail)::impl<T> {};                                                               \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    using EINSUMS_PP_CAT(EINSUMS_PP_CAT(has_, MEMBER), _t) = typename EINSUMS_PP_CAT(has_, MEMBER)<T>::type;                               \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    inline constexpr bool EINSUMS_PP_CAT(EINSUMS_PP_CAT(has_, MEMBER), _v) = EINSUMS_PP_CAT(has_, MEMBER)<T>::value;                       \
    /**/
