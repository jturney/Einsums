//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/ThreadsBase/ThreadPoolBase.hpp>

#include <functional>

namespace einsums::threads::detail {

template <typename Scheduler>
struct ScheduledThreadPool : ThreadPoolBase {
    ScheduledThreadPool(std::unique_ptr<Scheduler> scheduler, ThreadPoolInitParameters const &init);
    virtual ~ScheduledThreadPool();
};

} // namespace einsums::threads::detail