//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine/Declare.hpp>
#include <Einsums/CommandLine/Parse.hpp>
#include <Einsums/CommandLine/Source.hpp>

/*
 * The whole option system in one include, for callers that want all of it and
 * for source compatibility with the single header this module used to be.
 *
 * Prefer the piece you actually need:
 *
 *   Source.hpp   the vocabulary types; everyone gets these transitively
 *   Declare.hpp  the option types and named-argument tags - registration TUs
 *   Parse.hpp    argv walking, config files, help rendering - the init driver
 */
