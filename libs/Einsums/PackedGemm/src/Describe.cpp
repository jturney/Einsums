//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/Describe.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>

#include <complex>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

namespace {

/// Calls @p f with a value of the element type @p type names, or returns @p otherwise for an
/// unknown type.
template <typename F, typename R>
R with_scalar_type(ScalarType type, R otherwise, F &&f) {
    switch (type) {
    case ScalarType::Float32:
        return f(float{});
    case ScalarType::Float64:
        return f(double{});
    case ScalarType::Complex64:
        return f(std::complex<float>{});
    case ScalarType::Complex128:
        return f(std::complex<double>{});
    case ScalarType::Unknown:
        break;
    }
    return otherwise;
}

/// Elements in one contiguous run of an operand whose packer reads @p outer and @p inner: the
/// extent of whichever coalesced dim has a unit stride, or one when neither group has one.
int64_t operand_run(std::vector<DimSpec> const &outer, std::vector<DimSpec> const &inner) {
    int64_t run = 1;
    for (auto const *group : {&outer, &inner}) {
        for (auto const &ds : *group) {
            if (ds.tensor_stride == 1 && ds.size > run) {
                run = ds.size;
            }
        }
    }
    return run;
}

/// The contiguous run the C scatter writes, walking the side @ref blis_contraction_body walks
/// innermost: a whole M segment by a whole N segment when the two compose into one dense span,
/// else the unit-stride dim of the inner side, else single elements.
int64_t scatter_run(PackingPlan const &plan) {
    DimSpec const &m_fast = plan.c_m_dims.back();
    DimSpec const &n_fast = plan.c_n_dims.back();
    if (m_fast.tensor_stride == 1 && m_fast.size > 1 && n_fast.tensor_stride == m_fast.size) {
        return m_fast.size * n_fast.size;
    }
    bool const     n_inner = n_fast.tensor_stride != 0 && (m_fast.tensor_stride == 0 || n_fast.tensor_stride < m_fast.tensor_stride);
    DimSpec const &inner   = n_inner ? n_fast : m_fast;
    return inner.tensor_stride == 1 ? inner.size : 1;
}

} // namespace

PackingPlan prepare_packing_plan(ContractionKey const &key) {
    PackingPlan plan = compute_packing_topology(key);
    if (!plan.valid) {
        return plan;
    }
    return with_scalar_type(key.spec.scalar_type, PackingPlan{}, [&]<typename T>(T) {
        fill_strides(plan, key);
        sort_k_dims_for_packing(plan);
        coalesce_plan(plan, static_cast<int64_t>(sizeof(T)));
        // Which operand takes the kernel's M role is a property of the contraction and the
        // resolved kernel, so it is settled with the rest of the plan.
        constexpr bool is_complex = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;
        if (mn_roles_should_swap(plan, micro_kernel_shape<T>(), is_complex)) {
            plan = transposed_plan(plan);
        }
        return plan;
    });
}

PackedContractionShape describe_packed_contraction(ContractionKey const &key_in) {
    // The derived index lists, filled the way try_packed_gemm fills them for a caller that left
    // them out.
    ContractionKey key  = key_in;
    auto          &spec = key.spec;
    if (spec.target_indices.empty()) {
        spec.target_indices = unique_ordered(spec.c_indices);
    }
    if (spec.link_indices.empty()) {
        spec.link_indices = compute_link_indices(spec.a_indices, spec.b_indices, spec.target_indices);
    }
    if (spec.all_indices.empty()) {
        spec.all_indices = spec.target_indices;
        spec.all_indices.insert(spec.all_indices.end(), spec.link_indices.begin(), spec.link_indices.end());
    }
    spec.scalar_output = spec.c_indices.empty();

    PackingPlan const plan = prepare_packing_plan(key);
    if (!plan.valid) {
        return {};
    }
    return with_scalar_type(spec.scalar_type, PackedContractionShape{}, [&]<typename T>(T) {
        PackedContractionShape out;
        out.valid = true;
        out.M     = plan.M_total;
        out.N     = plan.N_total;
        out.K     = plan.K_total;
        out.batch = plan.batch_total;
        // The predicate try_packed_gemm takes the scatter path on.
        out.c_scattered = plan.c_m_dims.size() > 1 || plan.c_n_dims.size() > 1 ||
                          (plan.c_m_dims[0].tensor_stride != 1 && plan.c_n_dims[0].tensor_stride != 1);
        out.c_run       = out.c_scattered ? scatter_run(plan) : 0;
        out.a_run       = operand_run(plan.m_dims, plan.k_dims_in_a);
        out.b_run       = operand_run(plan.n_dims, plan.k_dims_in_b);
        // The split blis_contraction takes, cut for the machine's width whatever the call's.
        MicroKernelShape const shape = micro_kernel_shape<T>();
        auto const blk = compute_blocking(static_cast<int64_t>(sizeof(T)), shape.mr, shape.nr, plan.M_total, plan.N_total, plan.K_total);
        out.k_shares   = std::max(1, choose_k_split<T>(plan, k_split_width(), shape.mr, shape.nr, blk.NC, blk.KC).shares);
        return out;
    });
}

PackedContractionShape describe_packed_contraction(OperandLayout const &C, OperandLayout const &A, OperandLayout const &B,
                                                   ScalarType type) {
    for (auto const *layout : {&C, &A, &B}) {
        if (layout->dims.size() != layout->indices.size() || layout->strides.size() != layout->indices.size()) {
            return {};
        }
    }
    ContractionKey key;
    key.spec.c_indices      = C.indices;
    key.spec.a_indices      = A.indices;
    key.spec.b_indices      = B.indices;
    key.spec.target_indices = unique_ordered(C.indices);
    key.spec.link_indices   = compute_link_indices(A.indices, B.indices, key.spec.target_indices);
    key.spec.scalar_type    = type;
    key.a_desc              = {.rank = A.indices.size(), .dtype = type, .strides = A.strides};
    key.b_desc              = {.rank = B.indices.size(), .dtype = type, .strides = B.strides};
    key.c_desc              = {.rank = C.indices.size(), .dtype = type, .strides = C.strides};
    auto const extent_in    = [](OperandLayout const &layout, std::string const &index) {
        auto const at = std::ranges::find(layout.indices, index);
        return layout.dims[static_cast<size_t>(at - layout.indices.begin())];
    };
    // As try_packed_gemm reads them: a target's extent from C, a link's from A.
    for (auto const &index : key.spec.target_indices) {
        key.target_dims.push_back(extent_in(C, index));
    }
    for (auto const &index : key.spec.link_indices) {
        key.link_dims.push_back(extent_in(A, index));
    }
    return describe_packed_contraction(key);
}

EINSUMS_NAMESPACE_END(packed_gemm)
