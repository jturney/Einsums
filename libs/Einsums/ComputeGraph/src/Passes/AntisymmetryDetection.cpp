//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryDetection.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// The generators one operator makes worth testing, and the shape they address.
struct Candidate {
    SymmetryDescriptor       within;     ///< antisymmetry inside each group
    SymmetryDescriptor       invariance; ///< invariance under the terms the operator sums
    std::vector<std::size_t> dims;       ///< extents a tensor must match to be addressed
};

/// Map an operator's letters onto the output's axis positions.
std::optional<std::vector<std::vector<int>>> axis_groups(PermutationOperator const &op, std::vector<std::string> const &c_indices) {
    std::vector<std::vector<int>> groups;
    for (auto const &group : op.groups) {
        std::vector<int> axes;
        for (auto const &letter : group) {
            auto const first = std::ranges::find(c_indices, letter);
            if (first == c_indices.end() || std::count(c_indices.begin(), c_indices.end(), letter) != 1) {
                return std::nullopt;
            }
            auto const position = static_cast<int>(first - c_indices.begin());
            if (position >= kMaxSymmetryRank) {
                return std::nullopt;
            }
            axes.push_back(position);
        }
        std::ranges::sort(axes);
        groups.push_back(std::move(axes));
    }
    return groups;
}

/// Antisymmetry WITHIN each group, which is what a coset operator's operand has
/// to carry before the operator's output carries anything.
SymmetryDescriptor within_group_antisymmetry(std::vector<std::vector<int>> const &groups) {
    SymmetryDescriptor desc;
    for (auto const &axes : groups) {
        for (std::size_t k = 0; k + 1 < axes.size(); ++k) {
            desc.add(SymmetryOp::swap(axes[k], axes[k + 1], -1));
        }
    }
    return desc;
}

/// INVARIANCE under the axes an operator permutes, which is what a divisor needs
/// before the quotient keeps a numerator's antisymmetry.
///
/// Spelled as adjacent transpositions of the operator's whole letter set, with
/// sign +1, which is the SAME generator set the antisymmetry uses and the reason
/// R2 can match them by lookup.
///
/// The first version tested the operator's coset representatives instead, since
/// those are literally the permutations it sums over. That set is not closed
/// under the transpositions the antisymmetry is written with: `P(i/jk)` sums over
/// {e, (ij), (ik)} and never mentions (jk), so a divisor invariant under all
/// three was recorded as invariant under two, and R2 dropped the antisymmetry it
/// could not match. Generating the group's adjacent transpositions covers the
/// coset reps as compositions and is cheaper besides, two probes rather than
/// N-1.
SymmetryDescriptor operator_invariance(std::vector<std::vector<int>> const &groups) {
    std::vector<int> axes;
    for (auto const &group : groups) {
        axes.insert(axes.end(), group.begin(), group.end());
    }
    std::ranges::sort(axes);

    SymmetryDescriptor desc;
    for (std::size_t k = 0; k + 1 < axes.size(); ++k) {
        desc.add(SymmetryOp::swap(axes[k], axes[k + 1], +1));
    }
    return desc;
}

/// Whether @p desc holds in @p handle's data, dispatched on its element type.
bool holds_in_data(TensorHandle const &handle, SymmetryDescriptor const &desc) {
    if (!handle.impl_fn || desc.empty()) {
        return false;
    }
    bool result = false;
    detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
        using Impl       = ::einsums::detail::TensorImpl<T>;
        auto const *impl = static_cast<Impl const *>(handle.impl_fn());
        if (impl == nullptr || impl->data() == nullptr) {
            result = false;
            return;
        }
        RuntimeTensorView<T> const view{*impl};
        result = check_symmetry(view, desc);
    });
    return result;
}

} // namespace

std::vector<std::string> AntisymmetryDetection::explain() const {
    if (_num_found == 0) {
        return {};
    }
    return {fmt::format("AntisymmetryDetection: found {} symmetry generator(s) on {} bound input(s), from {} probe(s)", _num_found,
                        _num_tensors, _num_probed)};
}

void AntisymmetryDetection::reset_stats() {
    _num_probed  = 0;
    _num_found   = 0;
    _num_tensors = 0;
}

bool AntisymmetryDetection::run(Graph &graph) {
    // A tensor written by any node holds, right now, whatever it was initialized
    // to. Graph scratch is typically zero, and a zero tensor satisfies every
    // symmetry there is, so tagging one records a fact true of the buffer and
    // false of the value the graph will compute into it. Only tensors NOTHING
    // writes are candidates: those are the bound inputs, and what is in them now
    // is what the arithmetic will read.
    std::unordered_set<TensorId> written;
    for (auto const &node : graph.nodes()) {
        if (is_lifecycle(node.kind)) {
            continue;
        }
        for (auto const out : node.outputs) {
            written.insert(graph.resolve_alias(out));
        }
    }

    // Every operator the graph names, reduced to the generators worth testing
    // and the extents a tensor has to match for the permutation to address it.
    std::vector<Candidate> candidates;
    for (auto const &node : graph.nodes()) {
        std::vector<PermutationOperator> ops;
        std::vector<std::string>         c_indices;
        if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data); desc != nullptr && node.kind == OpKind::Einsum) {
            bool const live = desc->indices != nullptr;
            ops             = live ? desc->indices->spec.operators : desc->operators;
            c_indices       = live ? desc->indices->spec.c_indices : desc->spec.c_indices;
        } else if (auto const *pdesc = std::get_if<PermuteDescriptor>(&node.op_data); pdesc != nullptr && node.kind == OpKind::Permute) {
            ops       = pdesc->operators;
            c_indices = pdesc->c_indices;
        }
        if (ops.empty() || node.outputs.empty()) {
            continue;
        }
        auto const *out_handle = graph.find_tensor(graph.resolve_alias(node.outputs[0]));
        if (out_handle == nullptr) {
            continue;
        }
        for (auto const &op : ops) {
            auto groups = axis_groups(op, c_indices);
            if (!groups.has_value()) {
                continue;
            }
            candidates.push_back(Candidate{
                .within = within_group_antisymmetry(*groups), .invariance = operator_invariance(*groups), .dims = out_handle->dims});
        }
    }

    if (candidates.empty()) {
        return false;
    }

    for (auto &[tid, handle] : graph.tensors_map()) {
        if (written.contains(graph.resolve_alias(tid)) || !handle.impl_fn) {
            continue;
        }

        SymmetryDescriptor found;
        for (auto const &candidate : candidates) {
            if (handle.dims != candidate.dims) {
                continue; // the operator's permutation does not address this tensor
            }
            // Generator by generator, not descriptor by descriptor: a tensor can
            // be invariant under some of an operator's terms and not others, and
            // recording only the all-or-nothing answer would throw away the part
            // a later rule could have used.
            for (auto const *group : {&candidate.within, &candidate.invariance}) {
                for (auto const &op : group->ops) {
                    if (std::ranges::find(found.ops, op) != found.ops.end()) {
                        continue; // two operators can ask for the same generator
                    }
                    ++_num_probed;
                    SymmetryDescriptor one;
                    one.add(op);
                    if (holds_in_data(handle, one)) {
                        found.add(op);
                        ++_num_found;
                    }
                }
            }
        }

        if (found.empty()) {
            continue;
        }

        // The graph's metadata only. handle.set_symmetry_fn would push this onto
        // the user's own tensor, and a pass that looked at someone's data has no
        // business writing a declaration back onto it.
        if (handle.symmetry_hint != nullptr) {
            for (auto const &op : handle.symmetry_hint->ops) {
                if (std::ranges::find(found.ops, op) == found.ops.end()) {
                    found.add(op);
                }
            }
        }
        handle.symmetry_hint = std::make_shared<SymmetryDescriptor>(std::move(found));
        ++_num_tensors;
    }

    // Annotation only.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
