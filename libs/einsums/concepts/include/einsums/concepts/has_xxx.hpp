//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/preprocessor/cat.hpp>

#include <type_traits>

/// This macro creates a boolean unary meta-function such that for
/// any type X, has_name<X>::value == true if and only if X is a
/// class type and has a nested type member x::name. The generated
/// trait ends up in a namespace where the macro itself has been
/// placed.
#define EINSUMS_HAS_XXX_TRAIT_DEF(Name)                                                                                                    \
    template <typename T, typename Enable = void>                                                                                          \
    struct EINSUMS_PP_CAT(has_, Name) : std::false_type {};                                                                                \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    struct EINSUMS_PP_CAT(has_, Name)<T, std::void_t<typename T::Name>> : std::true_type {};                                               \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    using EINSUMS_PP_CAT(EINSUMS_PP_CAT(has_, Name), _t) = typename EINSUMS_PP_CAT(has_, Name)<T>::type;                                   \
                                                                                                                                           \
    template <typename T>                                                                                                                  \
    inline constexpr bool EINSUMS_PP_CAT(EINSUMS_PP_CAT(has_, Name), _v) = EINSUMS_PP_CAT(has_, Name)<T>::value;                           \
    /**/
