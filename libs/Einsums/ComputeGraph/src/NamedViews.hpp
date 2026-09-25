//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file NamedViews.hpp
 * @brief The views a factorization or projection reads its inputs through.
 *
 * Private to the sources that capture a fitting or a projection. What such a capture reads
 * becomes an interface tensor of the transformed graph, bound by name, so each view here carries
 * a name, and the name is the point.
 */

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>

#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// A view of @p tensor carrying its name, which the implicit conversion drops.
template <typename TensorType>
RuntimeTensorView<double> named_view(TensorType const &tensor) {
    RuntimeTensorView<double> view{tensor};
    view.set_name(tensor.name());
    return view;
}

/// @p view under a name of its own, for a fit that reads the tensor the caller TAGGED.
///
/// A fitting is captured, so whatever it reads becomes an interface tensor of the transformed
/// graph, bound by name at every later bind. When the fit reads a tensor the caller's own algebra
/// also holds, that is TWO interface tensors over one buffer: the pass and the fitting each hold
/// their own handle for it, and a manifest that binds by name refuses two entries sharing one.
/// Naming the fitting's copy is what makes the pair expressible, and it says what it is: a caller
/// rebinding the graph supplies the same tensor under both names, and both handles are repointed.
/// Without it a grid-fitted graph whose fit reads its own tagged tensor could not be saved at all.
inline RuntimeTensorView<double> fit_input_view(RuntimeTensorView<double> view) {
    std::string const stem = view.name();
    view.set_name(fmt::format("{}@fit", stem.empty() ? std::string{"tagged"} : stem));
    return view;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
