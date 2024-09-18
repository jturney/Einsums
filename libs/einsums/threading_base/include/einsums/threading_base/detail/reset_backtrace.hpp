//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#ifdef EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

#    include <einsums/errors/error_code.hpp>
#    include <einsums/threading_base/threading_base_fwd.hpp>

#    include <memory>
#    include <string>

namespace einsums::threads::detail {
struct reset_backtrace {
    EINSUMS_EXPORT explicit reset_backtrace(thread_id_type const &id, error_code &ec = throws);
    EINSUMS_EXPORT ~reset_backtrace();

    thread_id_type                                     _id;
    std::unique_ptr<einsums::debug::detail::backtrace> _backtrace;
#    ifdef EINSUMS_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    std::string _full_backtrace;
#    endif
    error_code &_ec;
};
} // namespace einsums::threads::detail

#endif // EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
