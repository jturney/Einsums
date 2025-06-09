//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/Singleton.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include "InitModule.hpp"

namespace einsums {
namespace detail {

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

struct EINSUMS_EXPORT Einsums_CommandLine_vars final : design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_CommandLine_vars)

    // Put module-global variables here.

private:
    explicit Einsums_CommandLine_vars() = default;
};

}
}