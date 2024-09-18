//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#ifdef EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

#    include <einsums/modules/debugging.hpp>
#    include <einsums/modules/errors.hpp>
#    include <einsums/threading_base/thread_helpers.hpp>
#    include <einsums/threading_base/threading_base_fwd.hpp>

namespace einsums::threads::detail {
reset_backtrace::reset_backtrace(thread_id_type const &id, error_code &ec)
    : id_(id), backtrace_(new einsums::debug::detail::backtrace())
#    ifdef EINSUMS_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
      ,
      full_backtrace_(backtrace_->trace())
#    endif
      ,
      ec_(ec) {
#    ifdef EINSUMS_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    threads::detail::set_thread_backtrace(id_, full_backtrace_.c_str(), ec_);
#    else
    threads::detail::set_thread_backtrace(id_, backtrace_.get(), ec_);
#    endif
}
reset_backtrace::~reset_backtrace() {
    threads::detail::set_thread_backtrace(id_, 0, ec_);
}
} // namespace einsums::threads::detail

#endif // EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
