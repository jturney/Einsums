//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>

#include "Record.hpp"
#include "ScanKernels.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

ScanOp scan_op_from_name(char const *api, std::string_view name) {
    if (auto const op = kScanOpNames.from_name(name)) {
        return *op;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: '{}' is not a scan operator; the operators are sum, prod, max, min", api, name);
}

template <typename T>
void eager_scan(ScanDescriptor const &desc, Impl<T> const &A, Impl<T> &out) {
    WAGGLE_ZONE("scan eager");
    run_scan<T>(desc, A, out);
}

void capture_scan(CaptureContext &ctx, ScanDescriptor desc, TensorId a_id, TensorId out_id, packed_gemm::ScalarType dtype,
                  std::size_t rank) {
    WAGGLE_ZONE("scan capture");
    // The result is overwritten in full, so it is the output only; an in-place scan lists the
    // tensor once as each.
    auto label = fmt::format("scan({}, axis {}{}{})", kScanOpNames.name(desc.op), desc.axis, desc.exclusive ? ", exclusive" : "",
                             desc.reverse ? ", reverse" : "");
    ctx.record_built(OpKind::Scan, std::move(label), dtype, rank, desc, std::span<TensorId const>{&a_id, 1},
                     std::span<TensorId const>{&out_id, 1}, {a_id}, {out_id});
}

#define EINSUMS_SCAN_OPERATIONS(T) template EINSUMS_EXPORT void eager_scan<T>(ScanDescriptor const &, Impl<T> const &, Impl<T> &);

EINSUMS_CG_ELEMENT_TYPES(EINSUMS_SCAN_OPERATIONS)
#undef EINSUMS_SCAN_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
