//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
#    include <einsums/coroutines/detail/posix_utility.hpp>

namespace einsums::threads::coroutines::detail ::posix {
///////////////////////////////////////////////////////////////////////
// this global (urghhh) variable is used to control whether guard pages
// will be used or not
EINSUMS_EXPORT bool use_guard_pages = true;
} // namespace einsums::threads::coroutines::detail::posix
#endif
