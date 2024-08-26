// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/itt_notify.hpp>
#include <einsums/profile/section.hpp>
#include <einsums/profile/timer.hpp>
#include <einsums/string_utils/trim.hpp>

___itt_domain *global_domain = itt_domain_create("Einsums");

namespace einsums::profile {

struct section::impl {
    std::string           name;
    bool                  push_timer{false};
    ___itt_domain        *domain{nullptr};
    ___itt_string_handle *section{nullptr};
};

section::section(const std::string &name, bool push_timer) : _impl{std::make_unique<impl>()} {
    _impl->name       = name;
    _impl->push_timer = push_timer;
    _impl->domain     = global_domain;
    _impl->section    = itt_string_handle_create(name.c_str());

    begin();
}

section::section(const std::string &name, const std::string &domain, bool pushTimer) : _impl{std::make_unique<impl>()} {
    _impl->name       = einsums::string_utils::trim_copy(name);
    _impl->push_timer = pushTimer;

    _impl->domain  = itt_domain_create(domain.c_str());
    _impl->section = itt_string_handle_create(name.c_str());

    begin();
}

section::~section() {
    end();
}

void section::begin() {
    if (_impl->push_timer) {
        einsums::profile::timer::push(_impl->name);
    }

    itt_task_begin(_impl->domain, _impl->section);
}

void section::end() {
    if (_impl) {
        itt_task_end(_impl->domain);
        if (_impl->push_timer) {
            einsums::profile::timer::pop();
        }
    }

    _impl = nullptr;
}

} // namespace einsums::profile