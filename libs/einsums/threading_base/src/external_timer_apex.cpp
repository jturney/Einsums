//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#ifdef EINSUMS_HAVE_APEX
#    include <einsums/assert.hpp>
#    include <einsums/threading_base/external_timer.hpp>
#    include <einsums/threading_base/thread_data.hpp>

#    include <cstdint>
#    include <memory>
#    include <string>

namespace einsums::detail::external_timer {
std::shared_ptr<task_wrapper> new_task(einsums::detail::thread_description const &description,
                                       threads::detail::thread_id_type            parent_task) {
    std::shared_ptr<task_wrapper> parent_wrapper = nullptr;
    if (parent_task != nullptr) {
        parent_wrapper = get_thread_id_data(parent_task)->get_timer_data();
    }

    if (description.kind() == einsums::detail::thread_description::data_type_description) {
        return external_timer::new_task(description.get_description(), UINTMAX_MAX, parent_wrapper);
    } else {
        EINSUMS_ASSERT(description.kind() == einsums::detail::thread_description::data_type_address);
        return external_timer::new_task(description.get_address(), UINTMAX_MAX, parent_wrapper);
    }
}

std::shared_ptr<task_wrapper> update_task(std::shared_ptr<task_wrapper> wrapper, einsums::detail::thread_description const &description) {
    if (wrapper == nullptr) {
        threads::detail::thread_id_type parent_task;
        return external_timer::new_task(description, parent_task);
    } else if (description.kind() == einsums::detail::thread_description::data_type_description) {
        // Disambiguate the call by making a temporary string object
        return external_timer::update_task(wrapper, std::string(description.get_description()));
    } else {
        EINSUMS_ASSERT(description.kind() == einsums::detail::thread_description::data_type_address);
        return external_timer::update_task(wrapper, description.get_address());
    }
}

scoped_timer::scoped_timer(std::shared_ptr<task_wrapper> data_ptr) : stopped(false), data_(nullptr) {
    // APEX internal actions are not timed. Otherwise, we would end
    // up with recursive timers. So it's possible to have a null
    // task wrapper pointer here.
    if (data_ptr != nullptr) {
        data_ = data_ptr;
        einsums::detail::external_timer::start(data_);
    }
}

scoped_timer::~scoped_timer() {
    stop();
}

void scoped_timer::stop() {
    if (!stopped) {
        stopped = true;
        // APEX internal actions are not timed. Otherwise, we would
        // end up with recursive timers. So it's possible to have a
        // null task wrapper pointer here.
        if (data_ != nullptr) {
            einsums::detail::external_timer::stop(data_);
        }
    }
}

void scoped_timer::yield() {
    if (!stopped) {
        stopped = true;
        // APEX internal actions are not timed. Otherwise, we would
        // end up with recursive timers. So it's possible to have a
        // null task wrapper pointer here.
        if (data_ != nullptr) {
            einsums::detail::external_timer::yield(data_);
        }
    }
}
} // namespace einsums::detail::external_timer
#endif
