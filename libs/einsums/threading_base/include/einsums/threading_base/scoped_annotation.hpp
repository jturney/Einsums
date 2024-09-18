//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
#    include <einsums/threading_base/thread_description.hpp>
#    include <einsums/threading_base/thread_helpers.hpp>
#endif

#include <string>
#include <type_traits>

namespace einsums {
namespace detail {
EINSUMS_EXPORT char const *store_function_annotation(std::string name);
} // namespace detail

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////
#    if defined(EINSUMS_COMPUTE_DEVICE_CODE)
struct [[nodiscard]] scoped_annotation {
    EINSUMS_NON_COPYABLE(scoped_annotation);

    explicit constexpr scoped_annotation(char const *) noexcept {}

    template <typename F>
    explicit EINSUMS_HOST_DEVICE constexpr scoped_annotation(F &&) noexcept {}

    // add empty (but non-trivial) destructor to silence warnings
    EINSUMS_HOST_DEVICE ~scoped_annotation() {}
};
#    elif EINSUMS_HAVE_ITTNOTIFY != 0
struct [[nodiscard]] scoped_annotation {
    EINSUMS_NON_COPYABLE(scoped_annotation);

    explicit scoped_annotation(char const *name) : task_(thread_domain_, einsums::util::itt::string_handle(name)) {}
    template <typename F>
    explicit scoped_annotation(F &&f) : task_(thread_domain_, einsums::detail::get_function_annotation_itt<std::decay_t<F>>::call(f)) {}

  private:
    einsums::util::itt::thread_domain thread_domain_;
    einsums::util::itt::task          task_;
};
#    elif defined(EINSUMS_HAVE_TRACY)
struct [[nodiscard]] scoped_annotation {
    EINSUMS_NON_COPYABLE(scoped_annotation);

    explicit scoped_annotation(char const *annotation) : annotation(annotation) {}

    explicit scoped_annotation(std::string annotation) : annotation(detail::store_function_annotation(EINSUMS_MOVE(annotation))) {}

    template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
    explicit scoped_annotation(F &&f) {
        if (auto f_annotation = einsums::detail::get_function_annotation<std::decay_t<F>>::call(f)) {
            annotation = f_annotation;
        }
    }

  private:
    char const *annotation = "<unknown>";

    // We don't use a Zone* macro from Tracy here because they are only
    // meant to be used in function scopes. The Zone* macros make use of
    // e.g.  __FUNCTION__ and other macros that are either unavailable in
    // this scope or are meaningless since they are not evaluated in the
    // scope of the scoped_annotation constructor. We instead manually
    // enable the ScopedZone only if TRACY_ENABLE is set.
#        if defined(TRACY_ENABLE)
    tracy::ScopedZone tracy_annotation{0, nullptr, 0, nullptr, 0, annotation, strlen(annotation), true};
#        endif
};
#    else
struct [[nodiscard]] scoped_annotation {
    EINSUMS_NON_COPYABLE(scoped_annotation);

    explicit scoped_annotation(char const *name) {
        auto *self = einsums::threads::detail::get_self_ptr();
        if (self != nullptr) {
            _desc = threads::detail::get_thread_id_data(self->get_thread_id())->set_description(name);
        }
    }

    explicit scoped_annotation(std::string name) {
        auto *self = einsums::threads::detail::get_self_ptr();
        if (self != nullptr) {
            char const *name_c_str = detail::store_function_annotation(std::move(name));
            _desc                  = threads::detail::get_thread_id_data(self->get_thread_id())->set_description(name_c_str);
        }
    }

    template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
    explicit scoped_annotation(F &&f) {
        einsums::detail::thread_description desc(f);

        auto *self = einsums::threads::detail::get_self_ptr();
        if (self != nullptr) {
            _desc = threads::detail::get_thread_id_data(self->get_thread_id())->set_description(desc);
        }
    }

    ~scoped_annotation() {
        auto *self = einsums::threads::detail::get_self_ptr();
        if (self != nullptr) {
            threads::detail::get_thread_id_data(self->get_thread_id())->set_description(_desc);
        }
    }

    einsums::detail::thread_description _desc;
};
#    endif

#else
///////////////////////////////////////////////////////////////////////////
struct [[nodiscard]] scoped_annotation {
    EINSUMS_NON_COPYABLE(scoped_annotation);

    explicit constexpr scoped_annotation(char const * /*name*/) noexcept {}

    template <typename F>
    explicit EINSUMS_HOST_DEVICE constexpr scoped_annotation(F && /*f*/) noexcept {}

    // add empty (but non-trivial) destructor to silence warnings
    EINSUMS_HOST_DEVICE ~scoped_annotation() {}
};
#endif
} // namespace einsums
