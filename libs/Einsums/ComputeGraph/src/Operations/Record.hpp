//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// The node-recording helpers the src/Operations translation units share. Private to them: each
// records an OpKind::Custom node whose executor reaches its operands through the graph's slots on
// every run, so the node follows rebind() and the memory planner moving a tensor's storage.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/CommonTensorTypes.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cstddef>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

/// Record a one-in one-out node calling ``apply(dst, src)`` on the operands' live impls, the way the
/// eager entry calls it. @p reads_dst lists the destination as an input too, for an op that reads
/// it before writing.
template <typename TD, typename TS, typename Fn>
void record_unary(CaptureContext &ctx, char const *name, char const *execute_label, SlotRef dst, SlotRef src, Fn apply,
                  bool reads_dst = false) {
    OperandAccessor const d_access(dst.second, packed_gemm::get_scalar_type<TD>());
    OperandAccessor const s_access(src.second, packed_gemm::get_scalar_type<TS>());
    auto                  executor = [d_access, s_access, apply, execute_label]() {
        LabeledSection(execute_label);
        apply(*d_access.impl<TD>(), *s_access.impl<TS>());
    };
    std::vector<TensorId> inputs{src.first};
    if (reads_dst) {
        inputs.push_back(dst.first);
    }
    ctx.record(OpKind::Custom, name, std::move(inputs), {dst.first}, std::move(executor));
}

/// Record a one-in one-out node writing ``reduce(src)`` into the first element of @p r.
template <typename TR, typename TA, typename Fn>
void record_reduction(CaptureContext &ctx, char const *name, char const *execute_label, SlotRef r, SlotRef a, Fn reduce) {
    OperandAccessor const r_access(r.second, packed_gemm::get_scalar_type<TR>());
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<TA>());
    auto                  executor = [r_access, a_access, reduce, execute_label]() {
        LabeledSection(execute_label);
        r_access.impl<TR>()->data()[0] = reduce(*a_access.impl<TA>());
    };
    ctx.record(OpKind::Custom, name, {a.first}, {r.first}, std::move(executor));
}

/// An accessor per slot, all of element type @p T.
template <typename T>
std::vector<OperandAccessor> accessors(std::vector<SlotRef> const &refs) {
    std::vector<OperandAccessor> out;
    out.reserve(refs.size());
    for (SlotRef const &r : refs) {
        out.emplace_back(r.second, packed_gemm::get_scalar_type<T>());
    }
    return out;
}

/// Visit every multi-index of an index space of extents @p dims, calling ``visit(idx)`` once each:
/// axis 0 fastest, or the last axis fastest when @p LastFastest. Nothing is visited when any
/// extent is zero. Each caller turns the index into offsets through its own operands' strides, which
/// is what makes a strided view come out right.
template <bool LastFastest = false, typename Visit>
void for_each_index(std::vector<std::size_t> const &dims, Visit &&visit) {
    std::size_t total = 1;
    for (std::size_t const d : dims) {
        total *= d;
    }
    std::vector<std::size_t> idx(dims.size(), 0);
    for (std::size_t count = 0; count < total; ++count) {
        visit(std::as_const(idx));
        for (std::size_t n = 0; n < dims.size(); ++n) {
            std::size_t const k = LastFastest ? dims.size() - 1 - n : n;
            if (++idx[k] < dims[k]) {
                break;
            }
            idx[k] = 0;
        }
    }
}

/// The byte-free element offset of multi-index @p idx under @p strides.
inline std::size_t offset_of(std::vector<std::size_t> const &idx, std::vector<std::size_t> const &strides) {
    std::size_t off = 0;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        off += idx[k] * strides[k];
    }
    return off;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
