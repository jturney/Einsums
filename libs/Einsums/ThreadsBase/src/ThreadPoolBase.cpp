//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ThreadsBase/ThreadPoolBase.hpp>

namespace einsums::threads::detail {

ThreadPoolBase::ThreadPoolBase(ThreadPoolInitParameters const &init)
    : _id(init._index, init._name), _thread_offset(init._thread_offset), _affinity_data(init._affinity_data), _notifier(init._notifier) {
}

} // namespace einsums::threads::detail