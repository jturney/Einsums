//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/threading_base/print.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/thread_data.hpp>

#include <cstdint>
#include <thread>

#if defined(__linux) || defined(linux) || defined(__linux__)
#    include <linux/unistd.h>
#    include <sys/mman.h>
#    define EINSUMS_DEBUGGING_PRINT_LINUX
#endif

// ------------------------------------------------------------
/// \cond NODETAIL
namespace einsums::debug::detail {

std::ostream &operator<<(std::ostream &os, threadinfo<threads::detail::thread_data *> const &d) {
    os << ptr(d.data) << " \"" << ((d.data != nullptr) ? d.data->get_description() : "nullptr") << "\"";
    return os;
}

std::ostream &operator<<(std::ostream &os, threadinfo<threads::detail::thread_id_type *> const &d) {
    if (d.data == nullptr) {
        os << "nullptr";
    } else {
        os << threadinfo<threads::detail::thread_data *>(get_thread_id_data(*d.data));
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, threadinfo<threads::detail::thread_id_ref_type *> const &d) {
    if (d.data == nullptr) {
        os << "nullptr";
    } else {
        os << threadinfo<threads::detail::thread_data *>(get_thread_id_data(*d.data));
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, threadinfo<einsums::threads::detail::thread_init_data> const &d) {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    os << std::left << " \"" << d.data.description.get_description() << "\"";
#else
    os << "??? " << /*hex<8,uintptr_t>*/ (std::uintptr_t(&d.data));
#endif
    return os;
}

// ------------------------------------------------------------------
// helper class for printing thread ID, either std:: or einsums::
// ------------------------------------------------------------------
namespace detail {

void print_thread_info(std::ostream &os) {
    if (einsums::threads::detail::get_self_id() == einsums::threads::detail::invalid_thread_id) {
        os << "-------------- ";
    } else {
        einsums::threads::detail::thread_data *dummy = einsums::threads::detail::get_self_id_data();
        os << hex<12, std::uintptr_t>(reinterpret_cast<std::uintptr_t>(dummy)) << " ";
    }
    const char *pool = "--------";
    auto        tid  = einsums::threads::detail::get_self_id();
    if (tid != threads::detail::invalid_thread_id) {
        auto *p = get_thread_id_data(tid)->get_scheduler_base()->get_parent_pool();
        pool    = p->get_pool_name().c_str();
    }
    os << hex<12, std::thread::id>(std::this_thread::get_id()) << " " << debug::detail::str<8>(pool)

#ifdef EINSUMS_DEBUGGING_PRINT_LINUX
       << " cpu " << debug::detail::dec<3, int>(sched_getcpu()) << " ";
#else
       << " cpu " << "--- ";
#endif
}

struct current_thread_print_helper {
    current_thread_print_helper() { debug::detail::register_print_info(&detail::print_thread_info); }

    static current_thread_print_helper helper_;
};

current_thread_print_helper current_thread_print_helper::helper_{};
} // namespace detail
} // namespace einsums::debug::detail
/// \endcond
