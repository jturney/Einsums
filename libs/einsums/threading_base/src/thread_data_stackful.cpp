//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/allocator_support/internal_allocator.hpp>
#include <einsums/logging.hpp>
#include <einsums/threading_base/thread_data.hpp>

#include <fmt/format.h>

////////////////////////////////////////////////////////////////////////////////
namespace einsums::threads::detail {
einsums::detail::internal_allocator<thread_data_stackful> thread_data_stackful::thread_alloc_;

thread_data_stackful::~thread_data_stackful() {
    EINSUMS_LOG(debug, "~thread_data_stackful({}), description({}), phase({})", fmt::ptr(this), this->get_description(),
                this->get_thread_phase());
}
} // namespace einsums::threads::detail
