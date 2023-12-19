//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/Config.hpp>

#include <typeinfo>

// clang-format off
#include <einsums/config/WarningsPrefix.hpp>
// clang-format on

namespace einsums::detail {

class EINSUMS_EXPORT BadLexicalCast : public std::bad_cast {
    std::type_info const *_source;
    std::type_info const *_target;

  public:
    BadLexicalCast() noexcept : _source(&typeid(void)), _target(&typeid(void)) {}

    const char *what() const noexcept override;

    virtual ~BadLexicalCast() noexcept;

    BadLexicalCast(std::type_info const &source_type_arg, std::type_info const &target_type_arg) noexcept
        : _source(&source_type_arg), _target(&target_type_arg) {}

    std::type_info const &source_type() const noexcept { return *_source; }
    std::type_info const &target_type() const noexcept { return *_target; }
};

[[noreturn]] EINSUMS_EXPORT void throw_bad_lexical_cast(std::type_info const &source_type, std::type_info const &target_type);

template <typename Source, typename Target>
[[noreturn]] inline Target throw_bad_lexical_cast() {
    throw_bad_lexical_cast(typeid(Source), typeid(Target));
}

} // namespace einsums::detail

#include <einsums/config/WarningsSuffix.hpp>
