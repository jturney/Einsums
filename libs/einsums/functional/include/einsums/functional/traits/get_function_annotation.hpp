//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/modules/itt.hpp>

namespace einsums::detail {

// By default we don't know anything about the function's name
template <typename F, typename Enable = void>
struct get_function_annotation {
    static constexpr auto call(F const & /*f*/) noexcept -> char const * { return nullptr; }
};

#if EINSUMS_HAVE_ITTNOTIFY != 0 && !defined(EINSUMS_HAVE_APEX)
template <typename F, typename Enable = void>
struct get_function_annotation_itt {
    static auto call(F const &f) -> util::itt::string_handle {
        static util::itt::string_handle sh(get_function_annotation<F>::call(f));
        return sh;
    }
};
#endif

} // namespace einsums::detail