//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// The layout half of the ABI fingerprint.
///
/// `<Einsums/Config/ABI.hpp>` folds the build toggles, which is everything the
/// Config module can see. Config sits below every module that declares a type,
/// so the sizes of the types that actually cross a stage boundary have to be
/// measured from up here, where they are complete.
///
/// The split matters for what each half catches. A toggle mismatch means the
/// two sides disagree about how the library was configured; a layout mismatch
/// means they disagree about what a `RuntimeTensor` IS, which is the failure
/// that corrupts memory rather than merely misbehaving.

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/ABI.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cstdint>

namespace einsums::sealed {

/// Fold of the sizes and alignments of every type that crosses a stage
/// boundary.
///
/// Computed in the header so a stage module computes it from ITS headers while
/// the library carries what it saw when it was compiled; the two are compared
/// at load time. Add a type here when it starts appearing in a stage signature,
/// and only then: every entry is a thing that must match exactly, so measuring
/// types nobody passes across turns compatible builds into refusals.
///
/// `blas::int_t` earns its place despite not being a tensor: LP64 against ILP64
/// is a silent, catastrophic mismatch, and it is a plausible way for an
/// out-of-tree build to differ from the library it links.
[[nodiscard]] constexpr std::uint64_t layout_fingerprint() noexcept {
    std::uint64_t h = detail::fnv1a("einsums.abi.layout.1");

    h = detail::fnv1a_value(sizeof(blas::int_t), h);

    h = detail::fnv1a_value(sizeof(::einsums::detail::TensorImpl<double>), h);
    h = detail::fnv1a_value(alignof(::einsums::detail::TensorImpl<double>), h);

    h = detail::fnv1a_value(sizeof(RuntimeTensor<double>), h);
    h = detail::fnv1a_value(alignof(RuntimeTensor<double>), h);
    h = detail::fnv1a_value(sizeof(RuntimeTensor<std::complex<double>>), h);

    h = detail::fnv1a_value(sizeof(Tensor<double, 2>), h);
    h = detail::fnv1a_value(alignof(Tensor<double, 2>), h);

    h = detail::fnv1a_value(sizeof(compute_graph::Graph), h);
    h = detail::fnv1a_value(alignof(compute_graph::Graph), h);

    return h;
}

} // namespace einsums::sealed
