//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/InitModule.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <H5Ipublic.h>
#include <atomic>

namespace einsums::detail {

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

// NOLINTNEXTLINE(readability-identifier-naming)
class EINSUMS_EXPORT Einsums_Tensor_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_Tensor_vars)

  public:
    // Put module-global variables here.
    hid_t hdf5_file;
    hid_t link_property_list;

    hid_t double_complex_type;
    hid_t float_complex_type;

    // Used for making temporary disk tensors.
    std::atomic_int64_t volatile temp_counter;

  private:
    explicit Einsums_Tensor_vars() = default;
};

} // namespace einsums::detail
