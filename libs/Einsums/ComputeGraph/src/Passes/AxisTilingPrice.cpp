//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// AxisTiling: choosing block extents and pricing a plan.

#include <Einsums/Config/Namespace.hpp>

#include "AxisTilingDetail.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::axis_tiling)

std::size_t chunk_depth(std::size_t slices, std::size_t bytes_per_slice, std::size_t cap) {
    if (bytes_per_slice == 0) {
        return slices;
    }
    std::size_t const ceiling = std::max<std::size_t>(1, std::min<std::size_t>(slices, cap / bytes_per_slice));
    for (std::size_t depth = ceiling; depth > 1; --depth) {
        if (slices % depth == 0) {
            return depth;
        }
    }
    return 1;
}

std::vector<std::size_t> balanced_blocks(std::size_t extent) {
    std::vector<std::size_t> out;
    for (std::size_t count = 1; count <= extent; ++count) {
        std::size_t const block = (extent + count - 1) / count;
        if (out.empty() || out.back() != block) {
            out.push_back(block);
        }
    }
    return out;
}

bool same_price(double lhs, double rhs) {
    return std::abs(lhs - rhs) <= 1e-6 * std::max(std::abs(lhs), std::abs(rhs));
}

bool better(BlockChoice const &lhs, BlockChoice const &rhs) {
    if (!same_price(lhs.price_us, rhs.price_us)) {
        return lhs.price_us < rhs.price_us;
    }
    if (lhs.traffic != rhs.traffic) {
        return lhs.traffic < rhs.traffic;
    }
    if (lhs.ranged() != rhs.ranged()) {
        return lhs.ranged() < rhs.ranged();
    }
    if (lhs.largest != rhs.largest) {
        return lhs.largest < rhs.largest;
    }
    return std::ranges::lexicographical_compare(std::views::reverse(rhs.blocks), std::views::reverse(lhs.blocks));
}

std::vector<std::vector<AxisWalk>> loop_walks(Plan const &plan) {
    std::vector<std::vector<AxisWalk>> per_axis(plan.extents.size());
    for (std::size_t k = 0; k < plan.extents.size(); ++k) {
        per_axis[k] = axis_walks(plan.extents[k], plan.range ? plan.blocks[k] : 1);
    }
    return combine_walks(per_axis);
}

std::vector<TriangleLoop> triangle_loops(std::size_t extent, std::size_t block) {
    std::size_t const full = extent / block;
    std::size_t const rest = extent % block;
    AxisWalk const    whole{.count = full, .step = block, .length = block, .range = true};
    AxisWalk const    tail{.count = 1, .step = block, .offset = full * block, .length = rest, .range = true};

    std::vector<TriangleLoop> out;
    if (full > 0) {
        TriangleLoop diagonal{.walks = {whole, whole}};
        for (std::size_t b = 0; b < full; ++b) {
            diagonal.table.push_back({b, b});
        }
        out.push_back(std::move(diagonal));
    }
    if (full > 1) {
        TriangleLoop above{.walks = {whole, whole}, .mirror = true};
        for (std::size_t b = 0; b < full; ++b) {
            for (std::size_t c = b + 1; c < full; ++c) {
                above.table.push_back({b, c});
            }
        }
        out.push_back(std::move(above));
    }
    if (full > 0 && rest > 0) {
        TriangleLoop column{.walks = {whole, tail}, .mirror = true};
        for (std::size_t b = 0; b < full; ++b) {
            column.table.push_back({b, 0});
        }
        out.push_back(std::move(column));
    }
    if (rest > 0) {
        out.push_back(TriangleLoop{.walks = {tail, tail}, .table = {{0, 0}}});
    }
    return out;
}

DeviceProfile const &pricing_profile() {
    static CostModel const model = CostModel::detect_default();
    return model.device(Target::CPU);
}

namespace {

/// The arithmetic-only price of a contraction the packed engine has no plan for (a repeated or
/// broadcast letter): its letters sorted into batch, M, N and K at the tile lengths @p length.
double gemm_price(Graph const &graph, NodePlan const &op, EinsumIndexLists const &lists, std::vector<std::size_t> const &length,
                  DeviceProfile const &profile) {
    std::map<std::string, std::size_t> extent;
    auto const                         learn = [&](std::vector<std::string> const &letters, SlotPlan const &slot) {
        auto const &dims = graph.tensor(slot.tid).dims;
        for (std::size_t p = 0; p < letters.size() && p < dims.size(); ++p) {
            bool const sliced  = p < slot.labels.size() && slot.labels[p] != kFree;
            extent[letters[p]] = sliced ? length[static_cast<std::size_t>(slot.labels[p])] : dims[p];
        }
    };
    learn(lists.c, op.slots[0]);
    learn(lists.a, op.slots[1]);
    learn(lists.b, op.slots[2]);
    auto const in = [](std::vector<std::string> const &letters, std::string const &l) {
        return std::ranges::find(letters, l) != letters.end();
    };
    std::size_t m = 1, n = 1, k = 1, batch = 1;
    for (auto const &[letter, e] : extent) {
        bool const c = in(lists.c, letter), a = in(lists.a, letter), b = in(lists.b, letter);
        if (a && b && c) {
            batch *= e;
        } else if (a && b) {
            k *= e;
        } else if (a) {
            m *= e;
        } else if (b) {
            n *= e;
        }
    }
    return static_cast<double>(batch) * profile.estimate_gemm_time_us(m, n, k);
}

/// One operand of a statement as a tile at the lengths @p length (one per sliced axis) reads it:
/// its @p letters, the tile's extent at each position, and the strides the engine sees. A slice of
/// a tensor keeps that tensor's strides. With @p dense set - an intermediate the body re-declares
/// at slice extents, or the scratch tile a mirrored tile is computed into - the tile is its own
/// buffer, laid out densely in the order of the tensor's strides.
///
/// A sliced axis @p point marks is taken a point at a time, and the body drops it from the operand
/// (see @ref surviving), so the tile has no position for it.
packed_gemm::OperandLayout tile_layout(Graph const &graph, SlotPlan const &slot, std::vector<std::string> const &letters,
                                       std::vector<std::size_t> const &length, std::vector<bool> const &point, bool dense) {
    auto const              &handle = graph.tensor(slot.tid);
    std::vector<std::size_t> kept;
    for (std::size_t p = 0; p < letters.size(); ++p) {
        bool const sliced = p < slot.labels.size() && slot.labels[p] != kFree;
        if (!sliced || point.empty() || !point[static_cast<std::size_t>(slot.labels[p])]) {
            kept.push_back(p);
        }
    }
    packed_gemm::OperandLayout out;
    for (std::size_t const p : kept) {
        bool const sliced = p < slot.labels.size() && slot.labels[p] != kFree;
        out.indices.push_back(letters[p]);
        out.dims.push_back(static_cast<int64_t>(sliced ? length[static_cast<std::size_t>(slot.labels[p])] : handle.dims[p]));
    }
    bool const has_strides = handle.strides.size() == letters.size();
    if (!dense && has_strides) {
        for (std::size_t const p : kept) {
            out.strides.push_back(static_cast<int64_t>(handle.strides[p]));
        }
        return out;
    }
    std::vector<std::size_t> order(kept.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (has_strides) {
        std::ranges::stable_sort(order, {}, [&](std::size_t q) { return handle.strides[kept[q]]; });
    }
    out.strides.assign(kept.size(), 0);
    int64_t step = 1;
    for (std::size_t const q : order) {
        out.strides[q] = step;
        step *= out.dims[q];
    }
    return out;
}

} // namespace

double tile_price(Graph const &graph, Plan const &plan, std::vector<std::size_t> const &length, bool mirror, DeviceProfile const &profile,
                  std::vector<bool> const &point) {
    double     total = 0.0;
    auto const owned = [&](TensorId tid) {
        return std::ranges::any_of(plan.body_owned, [&](auto const &entry) { return entry.first == tid; });
    };
    for (std::size_t o = 0; o < plan.ops.size(); ++o) {
        auto const &op = plan.ops[o];
        if (op.kind != OpKind::Einsum) {
            continue;
        }
        auto const lists   = live_index_lists(op.op_data.get<EinsumDescriptor>());
        bool const scratch = mirror && plan.triangle.has_value() && o == plan.triangle->writer;
        auto const shape   = packed_gemm::describe_packed_contraction(
            tile_layout(graph, op.slots[0], lists.c, length, point, scratch || owned(op.slots[0].tid)),
            tile_layout(graph, op.slots[1], lists.a, length, point, owned(op.slots[1].tid)),
            tile_layout(graph, op.slots[2], lists.b, length, point, owned(op.slots[2].tid)), plan.dtype);
        if (shape.valid) {
            total += profile.estimate_contraction_time_us(shape, 1, graph.tensor(op.slots[0].tid).element_size);
        } else {
            total += gemm_price(graph, op, lists, length, profile);
        }
        if (scratch) {
            std::size_t elements = 1;
            auto const &dims     = graph.tensor(op.slots[0].tid).dims;
            for (std::size_t p = 0; p < dims.size(); ++p) {
                bool const sliced = p < op.slots[0].labels.size() && op.slots[0].labels[p] != kFree;
                elements *= sliced ? length[static_cast<std::size_t>(op.slots[0].labels[p])] : dims[p];
            }
            // The direct add and the mirrored permute each read the scratch and read and write a block.
            total += profile.estimate_memory_time_us(6 * elements * graph.tensor(op.slots[0].tid).element_size);
        }
    }
    return total;
}

namespace {

/// What a loop pays on every iteration beyond its contractions' calls: the loop's own step, and
/// one call of each other statement in the body.
double iteration_price(Plan const &plan, DeviceProfile const &profile) {
    auto const others = std::ranges::count_if(plan.ops, [](NodePlan const &op) { return op.kind != OpKind::Einsum; });
    return profile.kernel_launch_overhead_us * static_cast<double>(1 + others);
}

} // namespace

double range_price(Graph const &graph, Plan const &plan, long double unpriced, DeviceProfile const &profile) {
    std::vector<bool> point;
    for (std::size_t const b : plan.blocks) {
        point.push_back(b <= 1);
    }
    double const per_iteration = iteration_price(plan, profile);
    double       total         = profile.estimate_memory_time_us(static_cast<std::size_t>(std::max(unpriced, 0.0L)));
    for (auto const &walks : loop_walks(plan)) {
        std::vector<std::size_t> length;
        for (auto const &walk : walks) {
            length.push_back(walk.length);
        }
        total += static_cast<double>(walk_iterations(walks)) * (tile_price(graph, plan, length, false, profile, point) + per_iteration);
    }
    return total;
}

double grouped_price(Graph const &graph, Plan const &plan, long double unpriced, DeviceProfile const &profile) {
    std::vector<bool> const        point(plan.extents.size(), true);
    std::vector<std::size_t> const length(plan.extents.size(), 1);
    double const                   step  = profile.kernel_launch_overhead_us;
    double const                   calls = iteration_price(plan, profile) - step;
    return profile.estimate_memory_time_us(static_cast<std::size_t>(std::max(unpriced, 0.0L))) +
           static_cast<double>(plan.slices) * (tile_price(graph, plan, length, false, profile, point) + calls) +
           static_cast<double>(plan.iterations) * step;
}

std::string describe_chunks(Plan const &plan) {
    if (plan.triangle.has_value()) {
        return fmt::format("a triangle of blocks of {} in {} loop(s) of {} iteration(s) in all", fmt::join(plan.blocks, " x "),
                           triangle_loops(plan.extents[0], plan.blocks[0]).size(), plan.iterations);
    }
    if (plan.range) {
        return fmt::format("blocks of {} in {} loop(s) of {} iteration(s) in all", fmt::join(plan.blocks, " x "), loop_walks(plan).size(),
                           plan.iterations);
    }
    return fmt::format("chunks of {}, {} iteration(s)", plan.depth, plan.iterations);
}

std::vector<std::vector<std::size_t>> candidate_sets(std::size_t rank) {
    std::vector<std::vector<std::size_t>> out;
    if (rank == 0 || rank > 20) {
        return out;
    }
    for (std::size_t mask = 1; mask < (std::size_t{1} << rank); ++mask) {
        std::vector<std::size_t> positions;
        for (std::size_t p = 0; p < rank; ++p) {
            if ((mask & (std::size_t{1} << p)) != 0) {
                positions.push_back(p);
            }
        }
        if (positions.size() <= kMaxSlicedAxes) {
            out.push_back(std::move(positions));
        }
    }
    // Fewest axes first, then by position, so a tie between two equally cheap sets is broken
    // the same way in every process.
    std::ranges::stable_sort(
        out, [](auto const &lhs, auto const &rhs) { return lhs.size() != rhs.size() ? lhs.size() < rhs.size() : lhs < rhs; });
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::axis_tiling)
