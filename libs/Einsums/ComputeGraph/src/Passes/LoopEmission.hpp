//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file LoopEmission.hpp
 * @brief Emitting a loop that walks blocks of one or more axes, one block per iteration.
 *
 * Private to the pass sources. A pass that rewrites a statement or a region into a loop over
 * blocks of an axis needs the same four things whatever it slices: the walk each loop takes over
 * each axis (full blocks, and a remainder loop of its own when the block does not divide the
 * axis), how many iterations that is, the parameters a body's views read their bounds from on
 * each iteration, and the spelling of a contraction to capture again. AxisTiling streams
 * intermediates through them and SlabStreaming streams backed tensors through them.
 */

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::detail)

/// How far one view bound has advanced through the slice sequence.
///
/// One cursor per bound rather than one shared by the whole body. Every bound's node runs
/// exactly once per iteration, so independent cursors stay in lockstep whatever order the
/// schedule puts them in, where a shared counter would depend on which node read it first.
/// A replay starts the sweep over rather than continuing it without the counter being reset,
/// because the decode below is periodic in the slice count: the second replay's counts are the
/// first replay's plus a whole sweep, and a whole sweep is what the decode divides out.
struct TileCursor {
    std::size_t calls{0};
};

/// The bound one axis of one chunk member takes, resolved afresh on every iteration: the
/// member's position along the axis, @p count positions @p step apart, plus @p bias.
inline std::function<std::int64_t()> slice_bound(std::shared_ptr<TileCursor> cursor, std::size_t depth, std::size_t member,
                                                 std::size_t stride, std::size_t count, std::size_t step, std::int64_t bias) {
    return [cursor = std::move(cursor), depth, member, stride, count, step, bias]() -> std::int64_t {
        std::size_t const slice = (cursor->calls++ * depth) + member;
        return static_cast<std::int64_t>(((slice / stride) % count) * step) + bias;
    };
}

/// How one emitted loop walks one sliced axis.
struct AxisWalk {
    std::size_t count{1};  ///< Positions the loop visits along the axis.
    std::size_t step{1};   ///< Distance between two consecutive positions.
    std::size_t offset{0}; ///< The first position.
    std::size_t length{1}; ///< How much of the axis the body holds at a position.
    bool        range{false};
};

/// The walks one axis of extent @p extent takes: every point when @p block is one or less, else
/// its full blocks and, when the block does not divide the axis, one remainder.
inline std::vector<AxisWalk> axis_walks(std::size_t extent, std::size_t block) {
    std::vector<AxisWalk> out;
    if (block <= 1) {
        out.push_back(AxisWalk{.count = extent});
        return out;
    }
    if (extent / block > 0) {
        out.push_back(AxisWalk{.count = extent / block, .step = block, .length = block, .range = true});
    }
    if (extent % block != 0) {
        out.push_back(AxisWalk{.count = 1, .step = block, .offset = (extent / block) * block, .length = extent % block, .range = true});
    }
    return out;
}

/// The loops a set of axes needs: every combination of one walk per axis, so a body is declared
/// at one fixed shape and a remainder needs no guard inside it.
inline std::vector<std::vector<AxisWalk>> combine_walks(std::vector<std::vector<AxisWalk>> const &per_axis) {
    std::vector<std::vector<AxisWalk>> loops{{}};
    for (auto const &choices : per_axis) {
        std::vector<std::vector<AxisWalk>> next;
        for (auto const &prefix : loops) {
            for (auto const &walk : choices) {
                next.push_back(prefix);
                next.back().push_back(walk);
            }
        }
        loops = std::move(next);
    }
    return loops;
}

/// How many iterations one loop runs.
inline std::size_t walk_iterations(std::vector<AxisWalk> const &walks) {
    std::size_t out = 1;
    for (auto const &walk : walks) {
        out *= walk.count;
    }
    return out;
}

/// The parameters a loop body's views read their bounds from: per member, per axis, the name of
/// the start and, for a range axis, the name of the end (empty for a point).
struct WalkBounds {
    std::vector<std::vector<std::string>> start;
    std::vector<std::vector<std::string>> end;
};

/**
 * @brief Record, in the body being captured, the parameters a loop's bounds are read from.
 *
 * Must run inside a capture of @p body. Names are ``<prefix>:<axis>:<member>`` (with ``<tag>:``
 * after the prefix when @p tag is set) and ``…:end`` for a range axis's end, because a bound is a
 * name or a constant and not a sum of the two. The table is seeded with the first iteration's
 * values, so a view recorded afterwards carries the SLICE's shape rather than the parent's and
 * the capture-time shape checks see the program that will run; the parameter writes overwrite
 * them on every iteration.
 *
 * @param[in] body The loop body.
 * @param[in] depth Members per iteration; one for a range loop.
 * @param[in] walks One walk per axis.
 * @param[in] prefix What every name starts with, unique to the pass, since every body shares the
 *            root's parameter table.
 * @param[in] tag What keeps the names of one loop apart from another's, or empty.
 */
inline WalkBounds declare_walk_bounds(Graph &body, std::size_t depth, std::vector<AxisWalk> const &walks, std::string const &prefix,
                                      std::string const &tag) {
    std::size_t const axes = walks.size();

    // Row-major strides over the positions the loop visits, so one integer names one slice.
    std::vector<std::size_t> stride(axes, 1);
    for (std::size_t k = axes; k-- > 0;) {
        stride[k] = (k + 1 < axes) ? stride[k + 1] * walks[k + 1].count : 1;
    }

    auto const bound = [&](std::string const &name, std::size_t member, AxisWalk const &walk, std::size_t k, std::size_t bias) {
        write_param(name, slice_bound(std::make_shared<TileCursor>(), depth, member, stride[k], walk.count, walk.step,
                                      static_cast<std::int64_t>(bias)));
        std::size_t const first = (((member / stride[k]) % walk.count) * walk.step) + bias;
        body.params_ptr()->set(name, static_cast<std::int64_t>(first));
    };

    WalkBounds out;
    out.start.resize(depth);
    out.end.resize(depth);
    for (std::size_t member = 0; member < depth; ++member) {
        for (std::size_t k = 0; k < axes; ++k) {
            auto const &walk = walks[k];
            auto const name = tag.empty() ? fmt::format("{}:{}:{}", prefix, k, member) : fmt::format("{}:{}:{}:{}", prefix, tag, k, member);
            bound(name, member, walk, k, walk.offset);
            out.start[member].push_back(name);
            if (walk.range) {
                auto const end = name + ":end";
                bound(end, member, walk, k, walk.offset + walk.length);
                out.end[member].push_back(end);
            } else {
                out.end[member].emplace_back();
            }
        }
    }
    return out;
}

/// The bound one axis takes when the loop's positions come from a table rather than a grid: on
/// iteration @c t, block @c table[t][axis] of the walk, @p step apart, plus @p bias. Periodic in the
/// table's length for the same reason @ref slice_bound is periodic in the slice count.
inline std::function<std::int64_t()> table_bound(std::shared_ptr<TileCursor>                                  cursor,
                                                 std::shared_ptr<std::vector<std::vector<std::size_t>> const> table, std::size_t axis,
                                                 std::size_t step, std::int64_t bias) {
    return [cursor = std::move(cursor), table = std::move(table), axis, step, bias]() -> std::int64_t {
        auto const &row = (*table)[cursor->calls++ % table->size()];
        return (static_cast<std::int64_t>(row[axis]) * static_cast<std::int64_t>(step)) + bias;
    };
}

/**
 * @brief Record the parameters a loop's bounds are read from, visiting the positions @p table lists.
 *
 * The grid form, @ref declare_walk_bounds, visits every combination of one position per axis. A
 * loop that visits only some of them, the diagonal of two axes or the blocks above it, lists them:
 * row @c t holds, for each axis, which of that axis's @c walks[k].count positions iteration @c t is
 * at. Depth one, names as @ref declare_walk_bounds spells them for member zero.
 *
 * @param[in] body The loop body.
 * @param[in] walks One walk per axis: its step, offset and length.
 * @param[in] table One row per iteration, one position per axis.
 * @param[in] prefix What every name starts with.
 * @param[in] tag What keeps the names of one loop apart from another's, or empty.
 */
inline WalkBounds declare_table_bounds(Graph &body, std::vector<AxisWalk> const &walks, std::vector<std::vector<std::size_t>> table,
                                       std::string const &prefix, std::string const &tag) {
    auto const shared = std::make_shared<std::vector<std::vector<std::size_t>> const>(std::move(table));

    auto const bound = [&](std::string const &name, std::size_t k, std::size_t bias) {
        write_param(name, table_bound(std::make_shared<TileCursor>(), shared, k, walks[k].step, static_cast<std::int64_t>(bias)));
        body.params_ptr()->set(name, (static_cast<std::int64_t>((*shared)[0][k]) * static_cast<std::int64_t>(walks[k].step)) +
                                         static_cast<std::int64_t>(bias));
    };

    WalkBounds out;
    out.start.resize(1);
    out.end.resize(1);
    for (std::size_t k = 0; k < walks.size(); ++k) {
        auto const &walk = walks[k];
        auto const  name = tag.empty() ? fmt::format("{}:{}:0", prefix, k) : fmt::format("{}:{}:{}:0", prefix, tag, k);
        bound(name, k, walk.offset);
        out.start[0].push_back(name);
        if (walk.range) {
            auto const end = name + ":end";
            bound(end, k, walk.offset + walk.length);
            out.end[0].push_back(end);
        } else {
            out.end[0].emplace_back();
        }
    }
    return out;
}

/// The permutation operators a spec applies, each followed by a space, in the prefix position
/// the arrow form takes them in.
inline std::string operator_prefix(std::vector<PermutationOperator> const &operators) {
    std::string out;
    for (auto const &op : operators) {
        out += op.render() + " ";
    }
    return out;
}

/// Rename, alike in every list, each label the spec parser would refuse.
///
/// A node's letters are data, and the algebra names some axes with labels no spec may hold: an
/// elementwise op's axes by position ('#0'), a letter a rewrite renamed ('~0'), a quadrature axis
/// ('laplace_t'). They are also the variables of the node's symbolic cost, so they stay as they are
/// on the node, and a body that spells the node as text renames them there instead, to labels none
/// of the lists spells.
inline void make_spellable(std::vector<std::vector<std::string> *> const &lists, std::vector<PermutationOperator> *operators = nullptr) {
    auto const spellable = [](std::string const &letter) {
        return !letter.empty() && std::ranges::all_of(letter, [](unsigned char ch) { return std::isalnum(ch) != 0; });
    };
    std::set<std::string> used;
    for (auto const *list : lists) {
        used.insert(list->begin(), list->end());
    }
    std::map<std::string, std::string> renamed;
    std::size_t                        fresh   = 0;
    auto const                         respell = [&](std::string &letter) {
        if (spellable(letter)) {
            return;
        }
        auto [entry, inserted] = renamed.try_emplace(letter);
        if (inserted) {
            // A single letter where one is free: a list of one label is spelled without commas,
            // and the parser reads a comma-less list a character at a time.
            do {
                entry->second = fresh < 52 ? std::string(1, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[fresh])
                                           : fmt::format("x{}", fresh - 52);
                ++fresh;
            } while (used.contains(entry->second));
            used.insert(entry->second);
        }
        letter = entry->second;
    };
    for (auto *list : lists) {
        std::ranges::for_each(*list, respell);
    }
    if (operators != nullptr) {
        for (auto &op : *operators) {
            for (auto &group : op.groups) {
                std::ranges::for_each(group, respell);
            }
        }
    }
}

/// The canonical spelling of a contraction, which is what the capture entry point parses back
/// into the same lists, conjugation flags and operators, with every label made one it accepts.
inline std::string einsum_spec_text(std::vector<std::string> a, std::vector<std::string> b, std::vector<std::string> c, bool conj_a,
                                    bool conj_b, std::vector<PermutationOperator> operators) {
    make_spellable({&a, &b, &c}, &operators);
    auto const side = [](std::vector<std::string> const &indices, bool conjugated) {
        std::string const joined = fmt::format("{}", fmt::join(indices, ","));
        return conjugated ? fmt::format("conj({})", joined) : joined;
    };
    return fmt::format("{} <- {}{} ; {}", fmt::join(c, ","), operator_prefix(operators), side(a, conj_a), side(b, conj_b));
}

/// The spelling of a permutation, with every label made one the parser accepts.
inline std::string permute_spec_text(std::vector<std::string> c, std::vector<std::string> a,
                                     std::vector<PermutationOperator> operators = {}) {
    make_spellable({&c, &a}, &operators);
    return fmt::format("{} <- {}{}", fmt::join(c, ","), operator_prefix(operators), fmt::join(a, ","));
}

EINSUMS_NAMESPACE_END(compute_graph::passes::detail)
