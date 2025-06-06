//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/ThreadsBase/SchedulerBase.hpp>

namespace einsums::threads::detail {

SchedulerBase::SchedulerBase(std::size_t num_threads, char const *description, SchedulerMode mode)
    : _parent_pool(nullptr), _description(description) {
}

} // namespace einsums::threads::detail