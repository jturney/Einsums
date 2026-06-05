//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>

namespace einsums::cl {

Registry &Registry::instance() {
    static Registry R;
    return R;
}

std::shared_ptr<ExclusiveCategory> make_yes_no(Flag &yes_flag, Flag &no_flag, bool default_value) {
    auto out = std::make_shared<ExclusiveCategory>();

    yes_flag.exclusions = out.get();
    no_flag.exclusions  = out.get();

    out->options.push_back(&yes_flag);
    out->options.push_back(&no_flag);

    yes_flag.set_on_unseen = false;
    no_flag.set_on_unseen  = false;

    yes_flag.value = true;
    no_flag.value  = false;

    if (default_value) {
        if (yes_flag.bound)
            *yes_flag.bound = true;
        if (yes_flag.setter)
            yes_flag.setter(true);
    } else {
        if (no_flag.bound)
            *no_flag.bound = true;
        if (no_flag.setter)
            no_flag.setter(true);
    }

    return out;
}

} // namespace einsums::cl