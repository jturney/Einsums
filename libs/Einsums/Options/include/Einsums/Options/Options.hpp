//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Options/Declare.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Options/Parse.hpp>
#include <Einsums/Options/Source.hpp>

/*
 * The whole option system in one include, for callers that want all of it and
 * for source compatibility with the single header this module used to be.
 * <Einsums/CommandLine.hpp> forwards here for one release.
 *
 * Prefer the piece you actually need:
 *
 *   Source.hpp   the vocabulary types; everyone gets these transitively
 *   Get.hpp      descriptors and typed reads - what a consuming module wants
 *   Declare.hpp  the option types and named-argument tags - registration TUs
 *   Parse.hpp    argv walking, config files, help rendering - the init driver
 */
