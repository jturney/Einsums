//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstdint>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

/// @brief What the packed engine would do with a contraction, read from the plan it would run.
///
/// For a caller that prices a contraction without running it: the cost of a packed contraction
/// depends on more than its M, N and K. A C whose indices interleave the two operands' is built in
/// a scratch block and scattered into place, at a cost per element set by how long C's contiguous
/// runs are; an operand whose contracted indices are split by a stride break is gathered into its
/// panel the same way. Everything here comes from @ref prepare_packing_plan, the plan
/// @ref try_packed_gemm runs, so the description and the engine cannot disagree.
struct PackedContractionShape {
    /// False when the engine has no plan for the contraction: repeated or broadcast indices, or a
    /// scalar output. Every other field is then zero.
    bool    valid{false};
    int64_t M{0};
    int64_t N{0};
    int64_t K{0};
    int64_t batch{1};
    /// True when C is built in a scratch block and scattered into place.
    bool c_scattered{false};
    /// True when one vendor GEMM can take the contraction whole: C written in place, no batch, and
    /// the M, N and K groups each one dim in every operand. Any other contraction runs the packed
    /// engine's own loops, whichever route a caller prefers.
    bool flat{false};
    /// Elements in one contiguous run of the scatter's inner walk over C. Zero when C is not
    /// scattered.
    int64_t c_run{0};
    /// Elements in one contiguous run of A, and of B, along the dims their packers read
    /// innermost: one when neither group has a unit stride.
    int64_t a_run{0};
    int64_t b_run{0};
    /// Shares the contracted extent is cut into (see @c choose_k_split); one when it is not cut.
    int k_shares{1};
};

/// @brief What the packed engine would do with the contraction @p key describes.
///
/// @p key carries the spec, the three operands' strides in elements and the target and link
/// extents, as @ref try_packed_gemm builds it; @c key.spec.scalar_type picks the micro-kernel.
[[nodiscard]] EINSUMS_EXPORT PackedContractionShape describe_packed_contraction(ContractionKey const &key);

/// @brief One operand of a contraction as a caller without the tensor in hand describes it: its
/// index letters, and the extent and element stride of each position.
struct OperandLayout {
    std::vector<std::string> indices;
    std::vector<int64_t>     dims;
    std::vector<int64_t>     strides;
};

/// @brief What the packed engine would do with C = A B on operands laid out as @p C, @p A and @p B,
/// of element type @p type.
///
/// Builds the key @ref try_packed_gemm would build from tensors of these layouts. Invalid when a
/// layout's three vectors disagree in length, or when the engine has no plan.
[[nodiscard]] EINSUMS_EXPORT PackedContractionShape describe_packed_contraction(OperandLayout const &C, OperandLayout const &A,
                                                                                OperandLayout const &B, ScalarType type);

EINSUMS_NAMESPACE_END(packed_gemm)
