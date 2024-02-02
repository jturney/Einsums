//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <typeinfo>

// clang-format off
#include <einsums/config/warnings_prefix.hpp>
// clang-format on

namespace einsums::detail {

struct EINSUMS_EXPORT bad_lexical_cast : public std::bad_cast {
  private:
    std::type_info const *_source;
    std::type_info const *_target;

  public:
    bad_lexical_cast() noexcept : _source(&typeid(void)), _target(&typeid(void)) {}

    [[nodiscard]] auto what() const noexcept -> char const * override;

    virtual ~bad_lexical_cast() noexcept;

    bad_lexical_cast(std::type_info const &source_type_arg, std::type_info const &target_type_arg) noexcept
        : _source(&source_type_arg), _target(&target_type_arg) {}

    [[nodiscard]] auto source_type() const noexcept -> std::type_info const & { return *_source; }
    [[nodiscard]] auto target_type() const noexcept -> std::type_info const & { return *_target; }
};

[[noreturn]] EINSUMS_EXPORT void throw_bad_lexical_cast(std::type_info const &source_type,
                                                        std::type_info const &target_type);

template <typename Source, typename Target>
[[noreturn]] inline auto throw_bad_lexical_cast() -> Target {
    throw_bad_lexical_cast(typeid(Source), typeid(Target));
}
} // namespace einsums::detail

#include <einsums/config/warnings_suffix.hpp>