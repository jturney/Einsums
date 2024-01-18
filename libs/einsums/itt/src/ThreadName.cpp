//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/itt/ThreadName.hpp>

#include <string>

namespace einsums::detail {

auto thread_name() -> std::string & {
    static thread_local std::string thread_name;
    return thread_name;
}

} // namespace einsums::detail