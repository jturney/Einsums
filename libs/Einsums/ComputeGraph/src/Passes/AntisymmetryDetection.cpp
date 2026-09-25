//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
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
#include <vector>

#include "AntisymmetryRules.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

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
    antisymmetry::add_adjacent_swaps(desc, axes, +1);
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
    if (auto const *desc = node.op_data.get_if<EinsumDescriptor>(); desc != nullptr && node.kind == OpKind::Einsum) {
        auto const lists = live_index_lists(*desc);
        ops              = lists.operators;
        c_indices        = lists.c;
    } else if (auto const *pdesc = node.op_data.get_if<PermuteDescriptor>(); pdesc != nullptr && node.kind == OpKind::Permute) {
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
    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (desc == nullptr) {
        return false;
    }
    auto const lists = live_index_lists(*desc);
    a                = lists.a;
    b                = lists.b;
    c                = lists.c;
    return true;
}

} // namespace

bool AntisymmetryDetection::run(Graph &graph) {
    // A tensor written by any node holds, right now, whatever it was initialized
    // to. Graph scratch is typically zero, and a zero tensor satisfies every
    // symmetry there is, so tagging one records a fact true of the buffer and
    // false of the value the graph will compute into it. Only tensors NOTHING
    // writes are candidates: those are the bound inputs, and what is in them now
    // is what the arithmetic will read.
    auto const writers = EscapeAnalysis::over(graph);

    // Candidate generators, per tensor. Two sources feed this, and keeping them
    // in one map is what lets the probe loop below stay single and the dedup be
    // automatic.
    std::map<TensorId, std::vector<SymmetryOp>> wanted;
    auto const                                  want = [&](TensorId id, SymmetryOp const &op) {
        TensorId const resolved = graph.resolve_alias(id);
        if (writers.writer_count(resolved) != 0) {
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

    /// One operator's generators and the shape they address.
    struct Shape {
        SymmetryDescriptor       within;
        SymmetryDescriptor       invariance;
        std::vector<std::size_t> dims;
    };
    std::vector<Shape> shapes;

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
            auto groups = antisymmetry::axis_groups(op, c_indices);
            if (!groups.has_value()) {
                continue;
            }
            operator_groups.push_back(OperatorGroups{.groups = op.groups});

            // DISTINCT (generators, shape) pairs only. An unrolled loop writes
            // the same operator once per iteration - the blocked (T) of
            // examples/toy has a hundred and twenty copies of P(a/bc) - and the
            // tensor scan below is over every tensor the graph knows, which for
            // that graph is a couple of thousand cached slices. Walking the
            // product of the two cost 27 ms of candidate collection before a
            // single byte of data was read.
            shapes.push_back(Shape{
                .within = antisymmetry::within_groups(*groups), .invariance = operator_invariance(*groups), .dims = out_handle->dims});
        }
    }

    // Source two walks every einsum against every operator group, so the same
    // duplication that inflated the shape scan inflates it quadratically: the
    // blocked (T) has 120 copies of P(a/bc) and 720 contractions, and the pair
    // loop underneath runs `std::count` over index lists. Deduping the groups is
    // what actually removed the 27 ms, not deduping the shapes.
    {
        std::vector<OperatorGroups> distinct_groups;
        for (auto const &og : operator_groups) {
            bool const seen = std::ranges::any_of(distinct_groups, [&](OperatorGroups const &other) { return other.groups == og.groups; });
            if (!seen) {
                distinct_groups.push_back(og);
            }
        }
        operator_groups = std::move(distinct_groups);
    }

    {
        std::vector<Shape> distinct;
        for (auto const &shape : shapes) {
            bool const seen = std::ranges::any_of(distinct, [&](Shape const &other) {
                return other.dims == shape.dims && other.within == shape.within && other.invariance == shape.invariance;
            });
            if (!seen) {
                distinct.push_back(shape);
            }
        }
        shapes = std::move(distinct);
    }

    for (auto const &shape : shapes) {
        for (auto &[tid, handle] : graph.tensors_map()) {
            if (handle.dims != shape.dims) {
                continue; // the operator's permutation does not address it
            }
            for (auto const &generator : shape.within.ops) {
                want(tid, generator);
            }
            for (auto const &generator : shape.invariance.ops) {
                want(tid, generator);
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
                        if (!antisymmetry::sole_position(c_idx, p).has_value() || !antisymmetry::sole_position(c_idx, q).has_value()) {
                            continue;
                        }
                        if (auto const carrier = antisymmetry::sole_carrier(a_idx, b_idx, p, q)) {
                            want(node.inputs[carrier->operand], SymmetryOp::swap(carrier->at_p, carrier->at_q, -1));
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
