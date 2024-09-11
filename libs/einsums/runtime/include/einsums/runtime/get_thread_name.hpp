//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <string>

namespace einsums::detail {

/**
 * \brief Return the name of the calling thread.
 *
 * This function returns the name of the calling thread. This name uniquely identifies the thread in the context of einsums. If the function
 * is called while no einsum runtime is active, the result will be "<unknown>".
 */
EINSUMS_EXPORT std::string get_thread_name();

} // namespace einsums::detail