//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/string_util/to_string.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/threading_base/thread_description.hpp>
#include <einsums/type_support/unused.hpp>

#include <fmt/format.h>

#include <ostream>
#include <string>

namespace einsums::detail {
std::ostream &operator<<(std::ostream &os, thread_description const &d) {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    if (d.kind() == thread_description::data_type_description) {
        os << d.get_description();
    } else {
        EINSUMS_ASSERT(d.kind() == thread_description::data_type_address);
        os << d.get_address(); //-V128
    }
#else
    EINSUMS_UNUSED(d);
    os << "<unknown>";
#endif
    return os;
}

std::string as_string(thread_description const &desc) {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    if (desc.kind() == detail::thread_description::data_type_description)
        return desc ? desc.get_description() : "<unknown>";

    return fmt::format("address: {:#x}", desc.get_address());
#else
    EINSUMS_UNUSED(desc);
    return "<unknown>";
#endif
}

/* The priority of description is altname, id::name, id::address */
void thread_description::init_from_alternative_name(char const *altname) {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION) && !defined(EINSUMS_HAVE_THREAD_DESCRIPTION_FULL)
    if (altname != nullptr) {
        type_       = data_type_description;
        data_.desc_ = altname;
        return;
    }
    einsums::threads::detail::thread_id_type id = einsums::threads::detail::get_self_id();
    if (id) {
        // get the current task description
        thread_description desc = einsums::threads::detail::get_thread_description(id);
        type_                   = desc.kind();
        // if the current task has a description, use it.
        if (type_ == data_type_description) {
            data_.desc_ = desc.get_description();
        } else {
            // otherwise, use the address of the task.
            EINSUMS_ASSERT(type_ == data_type_address);
            data_.addr_ = desc.get_address();
        }
    } else {
        type_       = data_type_description;
        data_.desc_ = "<unknown>";
    }
#else
    EINSUMS_UNUSED(altname);
#endif
}
} // namespace einsums::detail

namespace einsums::threads::detail {
::einsums::detail::thread_description get_thread_description(thread_id_type const &id, error_code & /* ec */) {
    return id ? get_thread_id_data(id)->get_description() : ::einsums::detail::thread_description("<unknown>");
}

::einsums::detail::thread_description set_thread_description(thread_id_type const &id, ::einsums::detail::thread_description const &desc,
                                                             error_code &ec) {
    if (EINSUMS_UNLIKELY(!id)) {
        EINSUMS_THROWS_IF(ec, einsums::error::null_thread_id, "einsums::threads::detail::set_thread_description",
                          "null thread id encountered");
        return ::einsums::detail::thread_description();
    }
    if (&ec != &throws)
        ec = make_success_code();

    return get_thread_id_data(id)->set_description(desc);
}

///////////////////////////////////////////////////////////////////////////
::einsums::detail::thread_description get_thread_lco_description(thread_id_type const &id, error_code &ec) {
    if (EINSUMS_UNLIKELY(!id)) {
        EINSUMS_THROWS_IF(ec, einsums::error::null_thread_id, "einsums::threads::get_thread_lco_description", "null thread id encountered");
        return nullptr;
    }

    if (&ec != &throws)
        ec = make_success_code();

    return get_thread_id_data(id)->get_lco_description();
}

::einsums::detail::thread_description set_thread_lco_description(thread_id_type const                        &id,
                                                                 ::einsums::detail::thread_description const &desc, error_code &ec) {
    if (EINSUMS_UNLIKELY(!id)) {
        EINSUMS_THROWS_IF(ec, einsums::error::null_thread_id, "einsums::threads::detail::set_thread_lco_description",
                          "null thread id encountered");
        return nullptr;
    }

    if (&ec != &throws)
        ec = make_success_code();

    return get_thread_id_data(id)->set_lco_description(desc);
}
} // namespace einsums::threads::detail
