//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Options/Declare.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Options/Parse.hpp>
#include <Einsums/Options/Source.hpp>

/*
 * The module this header named is now called Options, because a module whose
 * job is options-from-all-sources - defaults, a config file, the environment,
 * and the command line - was undersold by a name that describes one of them.
 *
 * Include <Einsums/Options.hpp>, or better, the piece you need:
 *
 *   Options/Source.hpp   the vocabulary types
 *   Options/Get.hpp      descriptors and typed reads
 *   Options/Declare.hpp  the option types and named-argument tags
 *   Options/Parse.hpp    argv walking, config files, help rendering
 *
 * This header forwards for one release.
 */
