//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/threading_base/callback_notifier.hpp>

namespace einsums::detail {
/// Retrieve the currently installed start handler function. This is a
/// function that will be called by einsums for each newly created thread that
/// is made known to the runtime. einsums stores exactly one such function
/// reference, thus the caller needs to make sure any newly registered
/// start function chains into the previous one (see
/// \a register_thread_on_start_func).
///
/// \returns The currently installed error handler function.
///
/// \note This function can be called before the einsums runtime is initialized.
///
EINSUMS_EXPORT einsums::threads::callback_notifier::on_startstop_type get_thread_on_start_func();

/// Retrieve the currently installed stop handler function. This is a
/// function that will be called by einsums for each newly created thread that
/// is made known to the runtime. einsums stores exactly one such function
/// reference, thus the caller needs to make sure any newly registered
/// stop function chains into the previous one (see
/// \a register_thread_on_stop_func).
///
/// \returns The currently installed error handler function.
///
/// \note This function can be called before the einsums runtime is initialized.
///
EINSUMS_EXPORT einsums::threads::callback_notifier::on_startstop_type get_thread_on_stop_func();

/// Retrieve the currently installed error handler function. This is a
/// function that will be called by einsums for each newly created thread that
/// is made known to the runtime. einsums stores exactly one such function
/// reference, thus the caller needs to make sure any newly registered
/// error function chains into the previous one (see
/// \a register_thread_on_error_func).
///
/// \returns The currently installed error handler function.
///
/// \note This function can be called before the einsums runtime is initialized.
///
EINSUMS_EXPORT einsums::threads::callback_notifier::on_error_type get_thread_on_error_func();

/// Set the currently installed start handler function. This is a
/// function that will be called by einsums for each newly created thread that
/// is made known to the runtime. einsums stores exactly one such function
/// reference, thus the caller needs to make sure any newly registered
/// start function chains into the previous one (see
/// \a get_thread_on_start_func).
///
/// \param f The function to install as the new start handler.
///
/// \returns The previously registered function of this category. It is
///          the user's responsibility to call that function if the
///          callback is invoked by einsums.
///
/// \note This function can be called before the einsums runtime is initialized.
///
EINSUMS_EXPORT einsums::threads::callback_notifier::on_startstop_type
               register_thread_on_start_func(einsums::threads::callback_notifier::on_startstop_type &&f);

/// Set the currently installed stop handler function. This is a
/// function that will be called by einsums for each newly created thread that
/// is made known to the runtime. einsums stores exactly one such function
/// reference, thus the caller needs to make sure any newly registered
/// stop function chains into the previous one (see
/// \a get_thread_on_stop_func).
///
/// \param f The function to install as the new stop handler.
///
/// \returns The previously registered function of this category. It is
///          the user's responsibility to call that function if the
///          callback is invoked by einsums.
///
/// \note This function can be called before the einsums runtime is initialized.
///
EINSUMS_EXPORT einsums::threads::callback_notifier::on_startstop_type
               register_thread_on_stop_func(einsums::threads::callback_notifier::on_startstop_type &&f);

/// Set the currently installed error handler function. This is a
/// function that will be called by einsums for each newly created thread that
/// is made known to the runtime. einsums stores exactly one such function
/// reference, thus the caller needs to make sure any newly registered
/// error function chains into the previous one (see
/// \a get_thread_on_error_func).
///
/// \param f The function to install as the new error handler.
///
/// \returns The previously registered function of this category. It is
///          the user's responsibility to call that function if the
///          callback is invoked by einsums.
///
/// \note This function can be called before the einsums runtime is initialized.
///
EINSUMS_EXPORT einsums::threads::callback_notifier::on_error_type
               register_thread_on_error_func(einsums::threads::callback_notifier::on_error_type &&f);
} // namespace einsums::detail
