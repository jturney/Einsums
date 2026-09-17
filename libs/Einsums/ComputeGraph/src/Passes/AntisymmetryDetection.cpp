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

namespace {

/// One operator's letter groups, as the graph writes them.
struct OperatorGroups {
    std::vector<std::vector<std::string>> groups;
};

/// Read the operators and output index list a node carries.
bool read_operators(Node const &node, std::vector<PermutationOperator> &ops, std::vector<std::string> &c_indices) {
    if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data); desc != nullptr && node.kind == OpKind::Einsum) {
        bool const live = desc->indices != nullptr;
        ops             = live ? desc->indices->spec.operators : desc->operators;
        c_indices       = live ? desc->indices->spec.c_indices : desc->spec.c_indices;
    } else if (auto const *pdesc = std::get_if<PermuteDescriptor>(&node.op_data); pdesc != nullptr && node.kind == OpKind::Permute) {
        ops       = pdesc->operators;
        c_indices = pdesc->c_indices;
    } else {
        return false;
    }
    return !ops.empty();
}

/// The index lists of an einsum node, live where it has them.
bool read_einsum_indices(Node const &node, std::vector<std::string> &a, std::vector<std::string> &b, std::vector<std::string> &c) {
    if (node.kind != OpKind::Einsum) {
        return false;
    }
    auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
    if (desc == nullptr) {
        return false;
    }
    bool const live = desc->indices != nullptr;
    a               = live ? desc->indices->spec.a_indices : desc->spec.a_indices;
    b               = live ? desc->indices->spec.b_indices : desc->spec.b_indices;
    c               = live ? desc->indices->spec.c_indices : desc->spec.c_indices;
    return true;
}

/// Where a letter sits in a list, when it sits there exactly once.
std::optional<int> sole_position(std::vector<std::string> const &list, std::string const &letter) {
    if (std::count(list.begin(), list.end(), letter) != 1) {
        return std::nullopt;
    }
    auto const first = std::ranges::find(list, letter);
    auto const at    = static_cast<int>(first - list.begin());
    return at < kMaxSymmetryRank ? std::optional<int>{at} : std::nullopt;
}

} // namespace

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

    // Candidate generators, per tensor. Two sources feed this, and keeping them
    // in one map is what lets the probe loop below stay single and the dedup be
    // automatic.
    std::map<TensorId, std::vector<SymmetryOp>> wanted;
    auto const                                  want = [&](TensorId id, SymmetryOp const &op) {
        TensorId const resolved = graph.resolve_alias(id);
        if (written.contains(resolved)) {
            return;
        }
        auto &ops = wanted[resolved];
        if (std::ranges::find(ops, op) == ops.end()) {
            ops.push_back(op);
        }
    };

    // The operator groups the graph names, which is what makes every probe below
    // a question the graph actually asked rather than a search.
    std::vector<OperatorGroups> operator_groups;

    // SOURCE ONE: a tensor of the operator's own output shape. That is the
    // operator's operand, which the conditional arm asks about, and a divisor of
    // the same shape, which R2 asks about.
    for (auto const &node : graph.nodes()) {
        std::vector<PermutationOperator> ops;
        std::vector<std::string>         c_indices;
        if (!read_operators(node, ops, c_indices) || node.outputs.empty()) {
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
            operator_groups.push_back(OperatorGroups{.groups = op.groups});

            SymmetryDescriptor const within     = within_group_antisymmetry(*groups);
            SymmetryDescriptor const invariance = operator_invariance(*groups);
            for (auto &[tid, handle] : graph.tensors_map()) {
                if (handle.dims != out_handle->dims) {
                    continue; // the operator's permutation does not address it
                }
                for (auto const &generator : within.ops) {
                    want(tid, generator);
                }
                for (auto const &generator : invariance.ops) {
                    want(tid, generator);
                }
            }
        }
    }

    // SOURCE TWO: an einsum's OPERAND. The chain a residual needs starts at the
    // amplitudes, which are a different rank from the operator's output and so
    // are never addressed by source one. What R3 asks is whether the operand
    // carrying two of the operator's grouped letters is antisymmetric in the
    // slots it carries them in, and the einsum's index lists say which slots
    // those are.
    for (auto const &node : graph.nodes()) {
        std::vector<std::string> a_idx;
        std::vector<std::string> b_idx;
        std::vector<std::string> c_idx;
        if (!read_einsum_indices(node, a_idx, b_idx, c_idx) || node.inputs.size() < 2) {
            continue;
        }
        for (auto const &og : operator_groups) {
            for (auto const &group : og.groups) {
                for (std::size_t x = 0; x + 1 < group.size(); ++x) {
                    for (std::size_t y = x + 1; y < group.size(); ++y) {
                        std::string const &p = group[x];
                        std::string const &q = group[y];
                        if (!sole_position(c_idx, p).has_value() || !sole_position(c_idx, q).has_value()) {
                            continue;
                        }
                        // Exactly one operand may carry BOTH, and the other must
                        // carry neither, or swapping them is not a swap of that
                        // operand's slots alone.
                        for (int which = 0; which < 2; ++which) {
                            auto const &carrier = which == 0 ? a_idx : b_idx;
                            auto const &other   = which == 0 ? b_idx : a_idx;
                            auto const  at_p    = sole_position(carrier, p);
                            auto const  at_q    = sole_position(carrier, q);
                            if (!at_p.has_value() || !at_q.has_value()) {
                                continue;
                            }
                            if (std::ranges::find(other, p) != other.end() || std::ranges::find(other, q) != other.end()) {
                                continue;
                            }
                            want(node.inputs[static_cast<std::size_t>(which)], SymmetryOp::swap(*at_p, *at_q, -1));
                        }
                    }
                }
            }
        }
    }

    if (wanted.empty()) {
        return false;
    }

    for (auto &[tid, generators] : wanted) {
        auto *handle = graph.find_tensor(tid);
        if (handle == nullptr || !handle->impl_fn) {
            continue;
        }

        SymmetryDescriptor found;
        for (auto const &generator : generators) {
            // Generator by generator, not descriptor by descriptor: a tensor can
            // be invariant under some of an operator's terms and not others, and
            // recording only the all-or-nothing answer would throw away the part
            // a later rule could have used.
            ++_num_probed;
            SymmetryDescriptor one;
            one.add(generator);
            if (holds_in_data(*handle, one)) {
                found.add(generator);
                ++_num_found;
            }
        }

        if (found.empty()) {
            continue;
        }

        // The graph's metadata only. handle.set_symmetry_fn would push this onto
        // the user's own tensor, and a pass that looked at someone's data has no
        // business writing a declaration back onto it.
        if (handle->symmetry_hint != nullptr) {
            for (auto const &op : handle->symmetry_hint->ops) {
                if (std::ranges::find(found.ops, op) == found.ops.end()) {
                    found.add(op);
                }
            }
        }
        handle->symmetry_hint = std::make_shared<SymmetryDescriptor>(std::move(found));
        ++_num_tensors;
    }

    // Annotation only.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
