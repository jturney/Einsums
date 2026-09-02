//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LayoutAssignment.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Where a tensor sits in a contraction. The array index everywhere below.
enum Role : std::size_t { RoleA = 0, RoleB = 1, RoleC = 2, RoleCount = 3 };

/// @brief A permutation of a tensor's axes: new axis @c i holds the axis capture put at
///        ``order[i]``.
///
/// Always expressed against the CAPTURED order rather than against the previous choice, so the
/// solver can revisit a tensor without composing permutations and getting one of them backwards.
using AxisOrder = std::vector<std::size_t>;

/// @brief The identity order of @p rank axes.
AxisOrder identity_order(std::size_t rank) {
    AxisOrder order(rank);
    std::iota(order.begin(), order.end(), std::size_t{0});
    return order;
}

/// @brief Read @p captured through @p order.
std::vector<std::string> relabel(std::vector<std::string> const &captured, AxisOrder const &order) {
    std::vector<std::string> out;
    out.reserve(order.size());
    for (auto axis : order) {
        out.push_back(captured[axis]);
    }
    return out;
}

/// @brief The order that reads @p captured as @p wanted.
/// @return The order, or empty when @p wanted is not a duplicate-free permutation of @p captured.
AxisOrder order_reading(std::vector<std::string> const &captured, std::vector<std::string> const &wanted) {
    if (captured.size() != wanted.size()) {
        return {};
    }
    std::unordered_map<std::string, std::size_t> position;
    position.reserve(captured.size());
    for (std::size_t axis = 0; axis < captured.size(); axis++) {
        if (!position.emplace(captured[axis], axis).second) {
            return {}; // a repeated letter has no unique axis
        }
    }
    AxisOrder order;
    order.reserve(wanted.size());
    for (auto const &letter : wanted) {
        auto const it = position.find(letter);
        if (it == position.end()) {
            return {};
        }
        order.push_back(it->second);
    }
    return order;
}

/// @brief The members of @p group that appear in @p indices, in @p indices' own order.
std::vector<std::string> subsequence(std::vector<std::string> const &indices, std::unordered_set<std::string> const &group) {
    std::vector<std::string> out;
    for (auto const &letter : indices) {
        if (group.count(letter) != 0) {
            out.push_back(letter);
        }
    }
    return out;
}

/// @brief One contraction, reduced to what a layout decision needs from it.
///
/// The four letter groups are properties of the SPEC and not of any layout: relaying an operand
/// out permutes its axes and leaves its letter set alone. That is what makes the groups constant
/// through the search, and it is why the solver only ever recomputes sequences.
struct Site {
    std::size_t                                     node{0};
    std::array<TensorId, RoleCount>                 ids{};
    std::array<std::vector<std::string>, RoleCount> captured{}; ///< Index lists as captured.
    std::array<std::size_t, RoleCount>              bytes{};
    std::array<std::size_t, RoleCount>              rank{};
    std::unordered_set<std::string>                 links, m_group, n_group, batch;
    bool                                            modelled{false};
};

/// @brief How one operand's axes decompose, under a chosen order.
struct Reading {
    std::vector<std::string> first;  ///< The first of the operand's two groups, in axis order.
    std::vector<std::string> second; ///< The other one.
    std::vector<std::string> batch;  ///< The batched letters, in axis order.
    bool                     contiguous{false};
};

/// @brief Decompose @p indices into (@p g1, @p g2, batch), and say whether it reads flat.
///
/// Flat means the two groups each occupy a contiguous run and the batched letters occupy exactly
/// the trailing positions. Trailing is not arbitrary: tensors here are column-major, so the last
/// axes carry the largest strides, and a batched contraction walks whole matrices apart by a
/// fixed stride. A batch letter anywhere else interleaves the matrices and there is no flat
/// matrix to hand a vendor call.
///
/// Which group comes first is deliberately not constrained; that is the transpose flag, and BLAS
/// takes either reading for free.
Reading read_as(std::vector<std::string> const &indices, std::unordered_set<std::string> const &g1,
                std::unordered_set<std::string> const &g2, std::unordered_set<std::string> const &batch) {
    Reading reading;
    reading.first  = subsequence(indices, g1);
    reading.second = subsequence(indices, g2);
    reading.batch  = subsequence(indices, batch);

    std::size_t const head = indices.size() - reading.batch.size();
    for (std::size_t pos = 0; pos < indices.size(); pos++) {
        if ((batch.count(indices[pos]) != 0) != (pos >= head)) {
            return reading; // a batched letter is not in the trailing block
        }
    }
    std::vector<std::string> const head_indices(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(head));
    reading.contiguous = !link_placement(head_indices, reading.first).split();
    return reading;
}

/// @brief The layout state the solver mutates: one axis order per candidate tensor.
using Assignment = std::unordered_map<TensorId, AxisOrder>;

/// @brief This tensor's index list at @p site under @p assignment.
std::vector<std::string> indices_of(Site const &site, Role role, Assignment const &assignment) {
    auto const it = assignment.find(site.ids[role]);
    if (it == assignment.end()) {
        return site.captured[role];
    }
    return relabel(site.captured[role], it->second);
}

/// @brief The copies @p site must make, one bit per role.
///
/// C names the order of the free groups and A names the order of the contracted one. That
/// convention is arbitrary in the way a choice of origin is arbitrary, and it has to be MADE:
/// two operands that each read flat on their own but disagree on which letter of a multi-index
/// group varies fastest cannot be walked in lockstep, so one of them is copied, and a model that
/// blamed neither would price a real copy at zero.
std::array<bool, RoleCount> copies_at(Site const &site, Assignment const &assignment) {
    std::array<bool, RoleCount> copies{false, false, false};

    auto const a = indices_of(site, RoleA, assignment);
    auto const b = indices_of(site, RoleB, assignment);
    auto const c = indices_of(site, RoleC, assignment);

    Reading const ra = read_as(a, site.m_group, site.links, site.batch);
    Reading const rb = read_as(b, site.n_group, site.links, site.batch);
    Reading const rc = read_as(c, site.m_group, site.n_group, site.batch);

    copies[RoleC] = !rc.contiguous;
    copies[RoleA] = !ra.contiguous || ra.first != rc.first || ra.batch != rc.batch;
    copies[RoleB] = !rb.contiguous || rb.first != rc.second || rb.second != ra.second || rb.batch != rc.batch;
    return copies;
}

/// @brief Everything one ``run()`` decides, built and discarded inside it.
struct Plan {
    std::vector<Site>                                      sites;
    Assignment                                             assignment;
    std::unordered_map<TensorId, std::vector<std::size_t>> uses;  ///< Candidate tensor -> its sites.
    std::vector<TensorId>                                  order; ///< Candidates, ascending, for determinism.
};

} // namespace

LayoutAssignment::LayoutAssignment() : _cost_model(CostModel::detect_default()) {
}

LayoutAssignment::LayoutAssignment(CostModel cost_model) : _cost_model(std::move(cost_model)) {
}

void LayoutAssignment::reset_stats() {
    _num_relaid_out      = 0;
    _num_copies_removed  = 0;
    _estimated_saving_us = 0.0;
}

bool LayoutAssignment::run(Graph &graph) {
    PassCounter const relaid_out{_num_relaid_out};

    auto &nodes = graph.nodes();

    // ── Collect the contractions, and with them the tensors that might move ────────────────
    //
    // Program order throughout, never the tensor map: an unordered walk would pick a different
    // tensor to move first on a different run, and every layout downstream of it would follow.
    Plan                         plan;
    std::unordered_set<TensorId> pinned;

    auto pin = [&pinned](TensorId id) {
        if (id != 0) {
            pinned.insert(id);
        }
    };

    for (std::size_t nd = 0; nd < nodes.size(); nd++) {
        Node const &node = nodes[nd];

        // The lifecycle kinds name a tensor without indexing it, so they say nothing about its
        // axes. Everything else that is not a contraction pins what it touches: this pass
        // rewrites index lists, and a node with no index list to rewrite would keep reading the
        // captured order after the tensor stopped being stored in it.
        if (node.kind == OpKind::Alloc || node.kind == OpKind::Free || node.kind == OpKind::Materialize ||
            node.kind == OpKind::Initialize) {
            continue;
        }
        if (node.kind != OpKind::Einsum) {
            for (auto id : node.inputs) {
                pin(id);
            }
            for (auto id : node.outputs) {
                pin(id);
            }
            continue;
        }

        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr || desc->indices == nullptr || node.inputs.size() < 2 || node.outputs.size() != 1) {
            for (auto id : node.inputs) {
                pin(id);
            }
            for (auto id : node.outputs) {
                pin(id);
            }
            note_skip("contraction carries no rewritable index state", fmt::format("node {}", node.id));
            continue;
        }

        Site site;
        site.node            = nd;
        site.ids[RoleA]      = node.inputs[0];
        site.ids[RoleB]      = node.inputs[1];
        site.ids[RoleC]      = node.outputs[0];
        site.captured[RoleA] = desc->indices->spec.a_indices;
        site.captured[RoleB] = desc->indices->spec.b_indices;
        site.captured[RoleC] = desc->indices->spec.c_indices;

        // Letter groups. A letter this classification does not place is one the model has no
        // reading for - a trace, a diagonal, a lone summed index - and the site is declined
        // rather than modelled with a hole in it.
        bool                                                   placeable = true;
        std::array<std::unordered_set<std::string>, RoleCount> present;
        for (std::size_t role = 0; role < RoleCount; role++) {
            for (auto const &letter : site.captured[role]) {
                if (!present[role].insert(letter).second) {
                    placeable = false; // a repeated letter is a diagonal, not an axis
                }
            }
        }
        if (placeable) {
            for (auto const &letter : site.captured[RoleA]) {
                bool const in_b = present[RoleB].count(letter) != 0;
                bool const in_c = present[RoleC].count(letter) != 0;
                if (in_b && in_c) {
                    site.batch.insert(letter);
                } else if (in_b) {
                    site.links.insert(letter);
                } else if (in_c) {
                    site.m_group.insert(letter);
                } else {
                    placeable = false;
                }
            }
            for (auto const &letter : site.captured[RoleB]) {
                bool const in_a = present[RoleA].count(letter) != 0;
                bool const in_c = present[RoleC].count(letter) != 0;
                if (in_a) {
                    continue; // already classified from A's side
                }
                if (in_c) {
                    site.n_group.insert(letter);
                } else {
                    placeable = false;
                }
            }
            for (auto const &letter : site.captured[RoleC]) {
                if (present[RoleA].count(letter) == 0 && present[RoleB].count(letter) == 0) {
                    placeable = false; // an output letter neither operand supplies
                }
            }
        }

        site.modelled = placeable;
        if (!placeable) {
            for (std::size_t role = 0; role < RoleCount; role++) {
                pin(site.ids[role]);
            }
            note_skip("contraction has a letter this pass has no flat reading for", fmt::format("node {} ({})", node.id, node.label));
            continue;
        }

        for (std::size_t role = 0; role < RoleCount; role++) {
            TensorHandle const *handle = graph.find_tensor(site.ids[role]);
            if (handle == nullptr) {
                site.modelled = false;
                continue;
            }
            site.rank[role] = handle->rank;
            site.bytes[role] =
                handle->element_size * std::accumulate(handle->dims.begin(), handle->dims.end(), std::size_t{1}, std::multiplies<>{});
        }
        if (!site.modelled) {
            for (std::size_t role = 0; role < RoleCount; role++) {
                pin(site.ids[role]);
            }
            continue;
        }

        plan.sites.push_back(std::move(site));
    }

    if (plan.sites.empty()) {
        return false;
    }

    // ── Decide which tensors may move ──────────────────────────────────────────────────────
    auto const                 escapes = EscapeAnalysis::over(graph);
    std::unordered_set<NodeId> whole_graph;
    whole_graph.reserve(nodes.size());
    for (auto const &node : nodes) {
        whole_graph.insert(node.id);
    }

    auto eligible = [&](TensorId id) -> bool {
        if (pinned.count(id) != 0) {
            return false;
        }
        TensorHandle const *handle = graph.find_tensor(id);
        if (handle == nullptr) {
            return false;
        }
        // Rank two is two readings of one matrix and BLAS takes either through `transa`, so
        // there is nothing here to win. It is also what keeps this pass clear of every GemmHint
        // in the graph, since a hint exists only where all three operands are rank two.
        if (handle->rank < 3) {
            return false;
        }
        if (handle->is_tiled || handle->is_distributed || handle->symmetry_hint != nullptr) {
            note_skip("tensor's axes carry a meaning beyond their extents", fmt::format("tensor '{}'", handle->name));
            return false;
        }
        // Deferred, because re-laying out an unallocated shell is a change of declaration while
        // re-laying out a live buffer is a data movement this pass does not perform. Whether the
        // tensor is the GRAPH's is not asked here: the escape verdict below reports a
        // user-visible tensor as `UserOwned`, and asking twice is how the two answers drift.
        if (handle->alloc_state != AllocState::Deferred || !handle->resize_deferred_fn) {
            note_skip("tensor's storage is already allocated", fmt::format("tensor '{}'", handle->name));
            return false;
        }
        if (Escape const verdict = escapes.classify(id, whole_graph); verdict != Escape::Dissolvable) {
            note_skip(std::string{escape_reason(verdict)}, fmt::format("tensor '{}'", handle->name));
            return false;
        }
        return true;
    };

    for (auto const &site : plan.sites) {
        for (std::size_t role = 0; role < RoleCount; role++) {
            TensorId const id = site.ids[role];
            if (plan.assignment.count(id) != 0 || !eligible(id)) {
                continue;
            }
            plan.assignment.emplace(id, identity_order(site.rank[role]));
            plan.order.push_back(id);
        }
    }
    if (plan.order.empty()) {
        return false;
    }
    std::ranges::sort(plan.order);
    for (std::size_t s = 0; s < plan.sites.size(); s++) {
        for (std::size_t role = 0; role < RoleCount; role++) {
            if (plan.assignment.count(plan.sites[s].ids[role]) != 0) {
                plan.uses[plan.sites[s].ids[role]].push_back(s);
            }
        }
    }

    // ── Search ─────────────────────────────────────────────────────────────────────────────
    auto cost_of_site = [&](Site const &site, Assignment const &assignment) -> double {
        auto const copies = copies_at(site, assignment);
        double     total  = 0.0;
        for (std::size_t role = 0; role < RoleCount; role++) {
            if (copies[role]) {
                total += _cost_model.estimate_permute_time_us(site.bytes[role], site.rank[role], Target::CPU);
            }
        }
        return total;
    };
    auto cost_of_tensor = [&](TensorId id, Assignment const &assignment) -> double {
        double total = 0.0;
        for (auto s : plan.uses.at(id)) {
            total += cost_of_site(plan.sites[s], assignment);
        }
        return total;
    };

    // Candidate orders for one tensor: the readings some participant asked for. A layout nobody
    // wants cannot beat one at least one node does, and enumerating r! orders per tensor to
    // discover that would make the pass's own cost a function of rank.
    auto candidates_for = [&](TensorId id, Assignment const &assignment) {
        std::vector<AxisOrder> out;
        auto                   offer = [&out](AxisOrder order) {
            if (!order.empty() && std::ranges::find(out, order) == out.end()) {
                out.push_back(std::move(order));
            }
        };
        for (auto s : plan.uses.at(id)) {
            Site const &site = plan.sites[s];
            auto const  a    = indices_of(site, RoleA, assignment);
            auto const  b    = indices_of(site, RoleB, assignment);
            auto const  c    = indices_of(site, RoleC, assignment);
            for (std::size_t role = 0; role < RoleCount; role++) {
                if (site.ids[role] != id) {
                    continue;
                }
                // The two groups this operand carries, taken from the operands that name them,
                // plus the operand's own reading of each so a tensor can still move when the
                // reference itself is the odd one out.
                std::vector<std::vector<std::string>> firsts, seconds;
                std::vector<std::string> const        own_batch = subsequence(site.captured[role], site.batch);
                std::vector<std::string> const        c_batch   = subsequence(c, site.batch);
                switch (role) {
                case RoleA:
                    firsts  = {subsequence(c, site.m_group), subsequence(site.captured[role], site.m_group)};
                    seconds = {subsequence(b, site.links), subsequence(site.captured[role], site.links)};
                    break;
                case RoleB:
                    firsts  = {subsequence(c, site.n_group), subsequence(site.captured[role], site.n_group)};
                    seconds = {subsequence(a, site.links), subsequence(site.captured[role], site.links)};
                    break;
                default:
                    firsts  = {subsequence(a, site.m_group), subsequence(site.captured[role], site.m_group)};
                    seconds = {subsequence(b, site.n_group), subsequence(site.captured[role], site.n_group)};
                    break;
                }
                for (auto const &tail : {c_batch, own_batch}) {
                    for (auto const &g1 : firsts) {
                        for (auto const &g2 : seconds) {
                            std::vector<std::string> wanted = g1;
                            wanted.insert(wanted.end(), g2.begin(), g2.end());
                            wanted.insert(wanted.end(), tail.begin(), tail.end());
                            offer(order_reading(site.captured[role], wanted));

                            std::vector<std::string> reversed = g2;
                            reversed.insert(reversed.end(), g1.begin(), g1.end());
                            reversed.insert(reversed.end(), tail.begin(), tail.end());
                            offer(order_reading(site.captured[role], reversed));
                        }
                    }
                }
            }
        }
        return out;
    };

    // Coordinate descent. Bounded rather than run to a fixpoint: each sweep is a full re-costing
    // of every site each candidate touches, and a pass in the default pipeline does not get to
    // spend unbounded time on a graph it may well not improve at all.
    constexpr int kMaxSweeps = 4;
    for (int sweep = 0; sweep < kMaxSweeps; sweep++) {
        bool improved = false;
        for (auto id : plan.order) {
            double    best_cost  = cost_of_tensor(id, plan.assignment);
            AxisOrder best_order = plan.assignment.at(id);
            for (auto &candidate : candidates_for(id, plan.assignment)) {
                AxisOrder const saved  = plan.assignment.at(id);
                plan.assignment.at(id) = candidate;
                double const trial     = cost_of_tensor(id, plan.assignment);
                plan.assignment.at(id) = saved;
                if (trial < best_cost) {
                    best_cost  = trial;
                    best_order = candidate;
                }
            }
            if (best_order != plan.assignment.at(id)) {
                plan.assignment.at(id) = best_order;
                improved               = true;
            }
        }
        if (!improved) {
            break;
        }
    }

    // ── What the assignment is worth ───────────────────────────────────────────────────────
    Assignment const captured_assignment = [&] {
        Assignment identity;
        for (auto id : plan.order) {
            identity.emplace(id, identity_order(plan.assignment.at(id).size()));
        }
        return identity;
    }();

    std::size_t copies_before = 0, copies_after = 0;
    double      cost_before = 0.0, cost_after = 0.0;
    for (auto const &site : plan.sites) {
        auto const was = copies_at(site, captured_assignment);
        auto const now = copies_at(site, plan.assignment);
        for (std::size_t role = 0; role < RoleCount; role++) {
            double const price = _cost_model.estimate_permute_time_us(site.bytes[role], site.rank[role], Target::CPU);
            copies_before += was[role] ? 1 : 0;
            copies_after += now[role] ? 1 : 0;
            cost_before += was[role] ? price : 0.0;
            cost_after += now[role] ? price : 0.0;
        }
    }

    std::vector<TensorId> moved;
    for (auto id : plan.order) {
        if (plan.assignment.at(id) != captured_assignment.at(id)) {
            moved.push_back(id);
        }
    }
    if (moved.empty()) {
        note_skip("no reordering of an eligible intermediate removes a copy",
                  fmt::format("{} candidate tensor(s) over {} contraction(s)", plan.order.size(), plan.sites.size()));
        return false;
    }
    // The search only ever takes a strict improvement per tensor, so a total that went the other
    // way means the model disagrees with itself. Decline rather than commit: this is the phase
    // whose output a save keeps.
    if (cost_after >= cost_before) {
        note_skip("the assignment found is no cheaper than the captured one",
                  fmt::format("{:.3f} us -> {:.3f} us", cost_before, cost_after));
        return false;
    }

    // ── Commit ─────────────────────────────────────────────────────────────────────────────
    for (auto id : moved) {
        AxisOrder const          &order    = plan.assignment.at(id);
        TensorHandle             *handle   = graph.find_tensor(id);
        std::vector<size_t> const was_dims = handle->dims;
        std::vector<size_t>       dims;
        dims.reserve(order.size());
        for (auto axis : order) {
            dims.push_back(was_dims[axis]);
        }

        // Every per-axis annotation moves with the axis it describes. A missed one is not a
        // wrong number today; it is a dim symbol naming the wrong extent at the next bind, which
        // is the failure the whole symbolic-extent machinery exists to catch.
        auto permute_axis_data = [&order](auto &vec) {
            if (vec.size() != order.size()) {
                return;
            }
            std::remove_reference_t<decltype(vec)> moved_vec;
            moved_vec.reserve(order.size());
            for (auto axis : order) {
                moved_vec.push_back(vec[axis]);
            }
            vec = std::move(moved_vec);
        };
        permute_axis_data(handle->spaces);
        permute_axis_data(handle->dim_symbols);

        handle->resize_deferred_fn(dims);
        handle->dims = dims;
        handle->strides.assign(dims.size(), 1);
        for (std::size_t axis = 1; axis < dims.size(); axis++) {
            handle->strides[axis] = handle->strides[axis - 1] * dims[axis - 1]; // column-major
        }
        if (TensorSlot *slot = graph.find_slot(id); slot != nullptr) {
            slot->dims = dims;
        }
        _num_relaid_out++;
        report(2, fmt::format("store '{}' as ({}) instead of ({})", handle->name, fmt::join(dims, ","), fmt::join(was_dims, ",")));
    }

    // The derived lists on the descriptor - `link_indices`, `target_indices`, `all_indices` and
    // the live `EinsumIndices::link_indices` - are functions of the letter SETS, which a
    // permutation of an operand's axes leaves alone. So they are correct after this loop without
    // being touched by it, and rebuilding them would only invite the two spellings to disagree.
    for (auto const &site : plan.sites) {
        auto *desc    = std::get_if<EinsumDescriptor>(&nodes[site.node].op_data);
        bool  touched = false;
        for (std::size_t role = 0; role < RoleCount; role++) {
            auto const it = plan.assignment.find(site.ids[role]);
            if (it == plan.assignment.end() || it->second == captured_assignment.at(site.ids[role])) {
                continue;
            }
            auto const rewritten = relabel(site.captured[role], it->second);
            // Both spellings, always. `spec` is the snapshot analysis passes read and `indices`
            // is what the executor dereferences on the next replay; a rewrite that moved only
            // one of them would compute one thing and be reported as another.
            switch (role) {
            case RoleA:
                desc->spec.a_indices          = rewritten;
                desc->indices->spec.a_indices = rewritten;
                break;
            case RoleB:
                desc->spec.b_indices          = rewritten;
                desc->indices->spec.b_indices = rewritten;
                break;
            default:
                desc->spec.c_indices          = rewritten;
                desc->indices->spec.c_indices = rewritten;
                break;
            }
            touched = true;
        }
        if (!touched) {
            continue;
        }
        // The raw spelling is a display string, but it is the one an execute-time diagnostic
        // quotes, so it is regenerated in the same form the IR loader writes rather than left
        // describing the captured axes.
        desc->indices->spec.raw = fmt::format("{} <- {} ; {}", fmt::join(desc->spec.c_indices, ","), fmt::join(desc->spec.a_indices, ","),
                                              fmt::join(desc->spec.b_indices, ","));
        report(2, fmt::format("node {} now reads {}", nodes[site.node].id, desc->indices->spec.raw));
    }

    // Signed, because a cheaper assignment can make MORE copies: one large operand traded for
    // two small ones is a win on the quantity being minimized and a loss on the count. The
    // counter reports copies and the search minimizes time, so they are allowed to disagree and
    // an unsigned subtraction here would turn that into a very large number.
    auto const copies_delta = static_cast<std::ptrdiff_t>(copies_before) - static_cast<std::ptrdiff_t>(copies_after);
    _num_copies_removed += static_cast<std::size_t>(std::max<std::ptrdiff_t>(0, copies_delta));
    _estimated_saving_us += cost_before - cost_after;

    // A descriptor-only rewrite still invalidates every plan keyed on an index list or an
    // extent, which is what the counter is read for; see Graph::note_structural_change.
    graph.note_structural_change();

    report(1, fmt::format("re-laid out {} intermediate(s), removing {} operand copy(ies) worth {:.3f} us per replay", relaid_out.delta(),
                          copies_delta, cost_before - cost_after));
    EINSUMS_LOG_INFO("LayoutAssignment: {} intermediate(s) re-laid out, {} operand copies removed", relaid_out.delta(), copies_delta);
    return true;
}

std::vector<std::string> LayoutAssignment::explain() const {
    if (_num_relaid_out == 0) {
        return {};
    }
    return {fmt::format("LayoutAssignment: re-laid out {} intermediate(s), removing {} operand copy(ies) modelled at {:.3f} us per replay",
                        _num_relaid_out, _num_copies_removed, _estimated_saving_us)};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
