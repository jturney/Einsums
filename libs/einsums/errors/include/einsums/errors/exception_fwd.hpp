// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>

namespace einsums {

struct error_code;
struct EINSUMS_EXPORT exception;
struct EINSUMS_EXPORT thread_interrupted;

enum class throw_mode : std::uint8_t { plain = 0, rethrow = 1, lightweight = 0x80, lightweight_rethrow = lightweight | rethrow };

EINSUMS_EXPORT extern error_code throws;

} // namespace einsums