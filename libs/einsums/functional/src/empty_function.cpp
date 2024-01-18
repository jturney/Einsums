//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/functional/detail/empty_function.hpp>
#include <einsums/modules/errors.hpp>

namespace einsums::util::detail {

[[noreturn]] void throw_bad_function_call() {
    einsums::throw_exception(einsums::error::bad_function_call, "empty function object should not be used",
                             "empty_function::operator()");
}

} // namespace einsums::util::detail