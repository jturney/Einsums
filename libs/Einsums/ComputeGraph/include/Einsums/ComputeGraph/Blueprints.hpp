//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Blueprints.hpp
 * @brief Umbrella header for all ComputeGraph blueprints.
 *
 * Blueprints are reusable tensor algebra patterns that compose standard
 * ComputeGraph operations. They work both inside capture blocks (recorded
 * into the graph) and outside (executed immediately).
 *
 * @code
 * #include <Einsums/ComputeGraph/Blueprints.hpp>
 *
 * // Inside a graph capture:
 * cg::blueprints::symmetrize(&A);
 * cg::blueprints::orthogonalize(&X, S);
 * @endcode
 */

#include <Einsums/ComputeGraph/Blueprints/Orthogonalization.hpp>
#include <Einsums/ComputeGraph/Blueprints/TensorAlgebra.hpp>
