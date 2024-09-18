//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)

#    include <einsums/errors/error_code.hpp>
#    include <einsums/threading_base/detail/reset_lco_description.hpp>
#    include <einsums/threading_base/thread_description.hpp>
#    include <einsums/threading_base/threading_base_fwd.hpp>

namespace einsums::threads::detail {
reset_lco_description::reset_lco_description(thread_id_type const &id, ::einsums::detail::thread_description const &description,
                                             error_code &ec)
    : _id(id), _ec(ec) {
    _old_desc = set_thread_lco_description(_id, description, _ec);
}

reset_lco_description::~reset_lco_description() {
    set_thread_lco_description(_id, _old_desc, _ec);
}
} // namespace einsums::threads::detail

#endif // EINSUMS_HAVE_THREAD_DESCRIPTION
