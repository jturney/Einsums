//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config/ExportDefinitions.hpp>
#include <einsums/config/Version.hpp>
#include <einsums/preprocessor/Stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace einsums {
EINSUMS_EXPORT char const EINSUMS_CHECK_VERSION[]       = EINSUMS_PP_STRINGIZE(EINSUMS_CHECK_VERSION);
EINSUMS_EXPORT char const EINSUMS_CHECK_BOOST_VERSION[] = EINSUMS_PP_STRINGIZE(EINSUMS_CHECK_BOOST_VERSION);
} // namespace einsums
