//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)

#    include <einsums/errors/error_code.hpp>
#    include <einsums/threading_base/thread_description.hpp>
#    include <einsums/threading_base/threading_base_fwd.hpp>

namespace einsums::threads::detail {
struct reset_lco_description {
    EINSUMS_EXPORT reset_lco_description(thread_id_type const &id, ::einsums::detail::thread_description const &description,
                                         error_code &ec = throws);
    EINSUMS_EXPORT ~reset_lco_description();

    thread_id_type                        _id;
    ::einsums::detail::thread_description _old_desc;
    error_code                           &_ec;
};
} // namespace einsums::threads::detail

#endif // EINSUMS_HAVE_THREAD_DESCRIPTION
