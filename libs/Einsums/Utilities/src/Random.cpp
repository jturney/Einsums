//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <chrono>
#include <random>

EINSUMS_NAMESPACE_BEGIN()
std::default_random_engine random_engine;

void seed_random(std::default_random_engine::result_type seed) {
    random_engine.seed(seed);
}

EINSUMS_NAMESPACE_END()
