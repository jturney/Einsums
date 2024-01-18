//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Section.hpp"

#include "einsums/Timer.hpp"
#include "einsums/core/Trim.hpp"

#if defined(EINSUMS_HAVE_ITTNOTIFY)
#    include <ittnotify.h>

__itt_domain *global_domain = __itt_domain_create("Einsums");
#endif

struct section::Impl {
    std::string name;
    bool        push_timer;
#if defined(EINSUMS_HAVE_ITTNOTIFY)
    __itt_domain        *domain;
    __itt_string_handle *section;
#endif
};

section::section(std::string const &name, bool pushTimer) : _impl{new section::Impl} {
    _impl->name       = einsums::trim_copy(name);
    _impl->push_timer = pushTimer;

#if defined(EINSUMS_HAVE_ITTNOTIFY)
    _impl->domain  = global_domain;
    _impl->section = __itt_string_handle_create(name.c_str());
#endif

    begin();
}

section::section(std::string const &name, std::string const &domain, bool pushTimer) : _impl{new section::Impl} {
    _impl->name       = einsums::trim_copy(name);
    _impl->push_timer = pushTimer;

#if defined(EINSUMS_HAVE_ITTNOTIFY)
    _impl->domain  = __itt_domain_create(domain.c_str());
    _impl->section = __itt_string_handle_create(name.c_str());
#endif

    begin();
}

section::~section() {
    end();
}

void section::begin() {
    if (_impl->push_timer)
        einsums::timer::push(_impl->name);

#if defined(EINSUMS_HAVE_ITTNOTIFY)
    __itt_task_begin(_impl->domain, __itt_null, __itt_null, _impl->section);
#endif
}

void section::end() {
    if (_impl) {
#if defined(EINSUMS_HAVE_ITTNOTIFY)
        __itt_task_end(_impl->domain);
#endif
        if (_impl->push_timer)
            einsums::timer::pop();
    }

    _impl = nullptr;
}