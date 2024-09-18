//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>
#include <einsums/type_support/unused.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>

namespace einsums::detail {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////
struct thread_description {
  public:
    enum data_type { data_type_description = 0, data_type_address = 1 };

  private:
    union data {
        char const *desc_; //-V117
        std::size_t addr_; //-V117
    };

    data_type type_;
    data      data_;

    EINSUMS_EXPORT void init_from_alternative_name(char const *altname);

  public:
    thread_description() noexcept : type_(data_type_description) { data_.desc_ = "<unknown>"; }

    thread_description(char const *desc) noexcept : type_(data_type_description) { data_.desc_ = desc ? desc : "<unknown>"; }

    /* The priority of description is name, altname, address */
    template <typename F, typename = std::enable_if_t<!std::is_same_v<F, thread_description>>>
    explicit thread_description(F const &f, char const *altname = nullptr) noexcept : type_(data_type_description) {
        char const *name = einsums::detail::get_function_annotation<F>::call(f);

        // If a name exists, use it, not the altname.
        if (name != nullptr) // -V547
        {
            altname = name;
        }

#    if defined(EINSUMS_HAVE_THREAD_DESCRIPTION_FULL)
        if (altname != nullptr) {
            data_.desc_ = altname;
        } else {
            type_       = data_type_address;
            data_.addr_ = einsums::detail::get_function_address<F>::call(f);
        }
#    else
        init_from_alternative_name(altname);
#    endif
    }

    constexpr data_type kind() const noexcept { return type_; }

    char const *get_description() const noexcept {
        EINSUMS_ASSERT(type_ == data_type_description);
        return data_.desc_;
    }

    std::size_t get_address() const noexcept {
        EINSUMS_ASSERT(type_ == data_type_address);
        return data_.addr_;
    }

    explicit operator bool() const noexcept { return valid(); }

    bool valid() const noexcept {
        if (type_ == data_type_description)
            return nullptr != data_.desc_;

        EINSUMS_ASSERT(type_ == data_type_address);
        return 0 != data_.addr_;
    }
};
#else
///////////////////////////////////////////////////////////////////////////
struct thread_description {
  public:
    enum data_type { data_type_description = 0, data_type_address = 1 };

  private:
    // expose for ABI compatibility reasons
    EINSUMS_EXPORT void init_from_alternative_name(char const *altname);

  public:
    thread_description() noexcept = default;

    constexpr thread_description(char const * /*desc*/) noexcept {}

    template <typename F, typename = typename std::enable_if_t<!std::is_same_v<F, thread_description>>>
    explicit constexpr thread_description(F const & /*f*/, char const * /*altname*/ = nullptr) noexcept {}

    constexpr data_type kind() const noexcept { return data_type_description; }

    constexpr char const *get_description() const noexcept { return "<unknown>"; }

    constexpr std::size_t get_address() const noexcept { return 0; }

    explicit constexpr operator bool() const noexcept { return valid(); }

    constexpr bool valid() const noexcept { return true; }
};
#endif

EINSUMS_EXPORT std::ostream &operator<<(std::ostream &, thread_description const &);
EINSUMS_EXPORT std::string as_string(thread_description const &desc);
} // namespace einsums::detail

namespace einsums::threads::detail {
///////////////////////////////////////////////////////////////////////////
/// The function get_thread_description is part of the thread related API
/// allows to query the description of one of the threads known to the
/// thread-manager.
///
/// \param id         [in] The thread id of the thread being queried.
/// \param ec         [in,out] this represents the error status on exit,
///                   if this is pre-initialized to \a einsums#throws
///                   the function will throw on error instead.
///
/// \returns          This function returns the description of the
///                   thread referenced by the \a id parameter. If the
///                   thread is not known to the thread-manager the return
///                   value will be the string "<unknown>".
///
/// \note             As long as \a ec is not pre-initialized to
///                   \a einsums#throws this function doesn't
///                   throw but returns the result code using the
///                   parameter \a ec. Otherwise it throws an instance
///                   of einsums#exception.
EINSUMS_EXPORT ::einsums::detail::thread_description get_thread_description(thread_id_type const &id, error_code &ec = throws);
EINSUMS_EXPORT ::einsums::detail::thread_description
set_thread_description(thread_id_type const                        &id,
                       ::einsums::detail::thread_description const &desc = ::einsums::detail::thread_description(),
                       error_code                                  &ec   = throws);

EINSUMS_EXPORT ::einsums::detail::thread_description get_thread_lco_description(thread_id_type const &id, error_code &ec = throws);
EINSUMS_EXPORT ::einsums::detail::thread_description
set_thread_lco_description(thread_id_type const                        &id,
                           ::einsums::detail::thread_description const &desc = ::einsums::detail::thread_description(),
                           error_code                                  &ec   = throws);
} // namespace einsums::threads::detail

template <>
struct fmt::formatter<einsums::detail::thread_description> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(einsums::detail::thread_description const &desc, FormatContext &ctx) const {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
        if (desc.kind() == einsums::detail::thread_description::data_type_description)
            return fmt::formatter<std::string>::format(desc ? desc.get_description() : "<unknown>", ctx);

        return fmt::formatter<std::string>::format(fmt::format("{}", desc.get_address()), ctx);
#else
        EINSUMS_UNUSED(desc);
        return fmt::formatter<std::string>::format("<unknown>", ctx);
#endif
    }
};
