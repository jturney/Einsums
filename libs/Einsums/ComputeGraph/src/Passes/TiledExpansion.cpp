//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// The tile grid of one tiled operand, read through its type-erased handle.
struct TiledView {
    size_t                               rank{0};
    std::vector<std::vector<int>> const *sizes{nullptr};
    /// Presence test and per-tile TensorId minting both need the concrete type,
    /// so they are captured as type-erased callables built under a dtype dispatch.
    std::function<bool(std::vector<int> const &)>     has_tile;
    std::function<TensorId(std::vector<int> const &)> tile_id;
};

/// Row-major strides over a grid, so one linear index enumerates every combination.
std::vector<size_t> grid_strides(std::vector<int> const &grid) {
    size_t const        n = grid.size();
    std::vector<size_t> stride(n, 1);
    for (size_t i = n; i-- > 1;) {
        stride[i - 1] = stride[i] * static_cast<size_t>(grid[i]);
    }
    return stride;
}

} // namespace

TiledExpansion::TiledExpansion(size_t max_nodes) : _max_nodes(max_nodes) {
}

void TiledExpansion::reset_stats() {
    _num_expanded   = 0;
    _num_tile_nodes = 0;
    _num_declined   = 0;
}

bool TiledExpansion::run(Graph &graph) {
    auto        &nodes = graph.nodes();
    size_t const n     = nodes.size();
    if (n == 0) {
        return false;
    }

    // Tensors written by some node in THIS graph. A tiled operand that is produced
    // rather than pre-built has a tile set that is not known until execution, so we
    // cannot decide output sparsity for it at pass time and must decline.
    std::unordered_set<TensorId> produced;
    for (auto const &node : nodes) {
        for (auto tid : node.outputs) {
            produced.insert(tid);
        }
    }

    // One TensorId per (tiled tensor, tile coord). Registering a tile twice would
    // give one buffer two ids and defeat every aliasing analysis downstream.
    std::map<std::pair<TensorId, std::vector<int>>, TensorId> tile_ids;

    std::vector<std::pair<size_t, std::vector<Node>>> inserts;
    std::vector<bool>                                 remove(n, false);
    bool                                              modified = false;

    for (size_t ni = 0; ni < n; ++ni) {
        auto const *tdesc = std::get_if<TiledEinsumDescriptor>(&nodes[ni].op_data);
        if (tdesc == nullptr || !tdesc->indices || !tdesc->params) {
            continue;
        }
        Node const &src = nodes[ni];
        if (src.inputs.size() != 2 || src.outputs.size() != 1) {
            continue;
        }
        TensorId const a_id = src.inputs[0];
        TensorId const b_id = src.inputs[1];
        TensorId const c_id = src.outputs[0];

        auto const &a_h = graph.tensor(a_id);
        auto const &b_h = graph.tensor(b_id);
        auto const &c_h = graph.tensor(c_id);
        if (!a_h.is_tiled || !b_h.is_tiled || !c_h.is_tiled) {
            continue; // mixed operands are rejected at capture; nothing to do here
        }
        if (a_h.dtype != c_h.dtype || b_h.dtype != c_h.dtype) {
            continue;
        }

        auto decline = [&](std::string_view why) {
            ++_num_declined;
            report(2, fmt::format("declining '{}': {}", src.label, why));
            EINSUMS_LOG_DEBUG("TiledExpansion: declining node {} - {}", src.id, why);
        };

        // Operand tile sets must be known now. An output produced elsewhere in this
        // graph is fine to WRITE into, but a produced *input* is not analyzable.
        if (produced.contains(a_id) || produced.contains(b_id)) {
            decline("a tiled operand is produced by another node, so its tile set is unknown at pass time");
            continue;
        }

        auto const &cidx = tdesc->indices->c_indices;
        auto const &aidx = tdesc->indices->a_indices;
        auto const &bidx = tdesc->indices->b_indices;
        if (cidx.empty()) {
            decline("scalar output (full reduction) over tiled operands is unsupported");
            continue;
        }

        // Unique index letters, in the same stable order the runtime uses (C, then
        // new from A, then new from B) so the enumeration matches exactly.
        std::vector<std::string> unique;
        auto                     add_unique = [&unique](std::vector<std::string> const &v) {
            for (auto const &s : v) {
                if (std::ranges::find(unique, s) == unique.end()) {
                    unique.push_back(s);
                }
            }
        };
        add_unique(cidx);
        add_unique(aidx);
        add_unique(bidx);
        size_t const nu  = unique.size();
        auto         pos = [&unique](std::string const &s) { return static_cast<size_t>(std::ranges::find(unique, s) - unique.begin()); };

        // Per-index grid extent and partition, validated for alignment across every
        // operand carrying that index. The runtime throws here; we decline.
        std::vector<int>              grid(nu, -1);
        std::vector<std::vector<int>> part(nu);
        bool                          aligned = true;
        auto                          absorb  = [&](std::vector<std::string> const &idx, std::vector<std::vector<int>> const &sizes) {
            if (idx.size() != sizes.size()) {
                aligned = false;
                return;
            }
            for (size_t ax = 0; ax < idx.size() && aligned; ++ax) {
                size_t const u = pos(idx[ax]);
                auto const  &p = sizes[ax];
                if (grid[u] < 0) {
                    grid[u] = static_cast<int>(p.size());
                    part[u] = p;
                } else if (grid[u] != static_cast<int>(p.size()) || part[u] != p) {
                    aligned = false;
                }
            }
        };

        // Reach the concrete tiled objects to read grids, test tile presence, and
        // mint per-tile TensorIds. Everything type-dependent happens in here.
        TiledView av, bv, cv;
        bool      usable = true;
        detail::dispatch_scalar_type(c_h.dtype, [&]<typename T>(T /*tag*/) {
            using Tiled = TiledRuntimeTensor<T>;
            auto *A     = static_cast<Tiled *>(a_h.tensor_ptr);
            auto *B     = static_cast<Tiled *>(b_h.tensor_ptr);
            auto *C     = static_cast<Tiled *>(c_h.tensor_ptr);
            if (A == nullptr || B == nullptr || C == nullptr) {
                usable = false;
                return;
            }
            auto bind = [&](Tiled *t, TensorId owner) {
                TiledView v;
                v.rank     = t->rank();
                v.sizes    = &t->tile_sizes();
                v.has_tile = [t](std::vector<int> const &co) { return t->has_tile(co); };
                v.tile_id  = [t, owner, &tile_ids, &graph](std::vector<int> const &co) {
                    auto key = std::make_pair(owner, co);
                    auto it  = tile_ids.find(key);
                    if (it != tile_ids.end()) {
                        return it->second;
                    }
                    auto &tile = t->tile(co); // infer-and-create, matching the runtime
                    tile.materialize();
                    TensorId const id = graph.register_tensor(make_handle(tile, 0));
                    tile_ids.emplace(std::move(key), id);
                    return id;
                };
                return v;
            };
            av = bind(A, a_id);
            bv = bind(B, b_id);
            cv = bind(C, c_id);
        });
        if (!usable) {
            decline("an operand has no backing tiled object");
            continue;
        }

        absorb(aidx, *av.sizes);
        absorb(bidx, *bv.sizes);
        absorb(cidx, *cv.sizes);
        if (!aligned || std::ranges::any_of(grid, [](int g) { return g <= 0; })) {
            decline("tile partitions for a shared index disagree across operands");
            continue;
        }

        // Enumerate the unique-index grid exactly as the runtime does.
        auto const   stride = grid_strides(grid);
        size_t const total  = stride.empty() ? 0 : stride[0] * static_cast<size_t>(grid[0]);
        if (total == 0) {
            continue;
        }
        if (total > _max_nodes) {
            decline(fmt::format("projected {} tile combinations exceeds the {}-node budget", total, _max_nodes));
            continue;
        }

        std::vector<size_t> a_tab(aidx.size()), b_tab(bidx.size()), c_tab(cidx.size());
        for (size_t ax = 0; ax < aidx.size(); ++ax) {
            a_tab[ax] = pos(aidx[ax]);
        }
        for (size_t ax = 0; ax < bidx.size(); ++ax) {
            b_tab[ax] = pos(bidx[ax]);
        }
        for (size_t ax = 0; ax < cidx.size(); ++ax) {
            c_tab[ax] = pos(cidx[ax]);
        }

        // Which output tiles already exist decides how c_pf is applied, so snapshot
        // it BEFORE tile_id() starts creating tiles.
        auto const c_pf  = tdesc->params->c_pf;
        auto const ab_pf = tdesc->params->ab_pf;

        ParsedEinsumSpec per_tile;
        per_tile.c_indices = cidx;
        per_tile.a_indices = aidx;
        per_tile.b_indices = bidx;
        per_tile.raw       = fmt::format("{} <- {} ; {}", fmt::join(cidx, ","), fmt::join(aidx, ","), fmt::join(bidx, ","));

        std::vector<Node>                emitted;
        std::map<std::vector<int>, bool> written; // output coord -> already written once
        std::vector<std::vector<int>>    pre_existing;
        for (size_t s = 0; s < total; ++s) {
            size_t           rem = s;
            std::vector<int> ucoord(nu);
            for (size_t u = 0; u < nu; ++u) {
                ucoord[u] = static_cast<int>(rem / stride[u]);
                rem %= stride[u];
            }
            std::vector<int> acoord(aidx.size()), bcoord(bidx.size()), ccoord(cidx.size());
            for (size_t ax = 0; ax < aidx.size(); ++ax) {
                acoord[ax] = ucoord[a_tab[ax]];
            }
            for (size_t ax = 0; ax < bidx.size(); ++ax) {
                bcoord[ax] = ucoord[b_tab[ax]];
            }
            if (!av.has_tile(acoord) || !bv.has_tile(bcoord)) {
                continue; // structural zero
            }
            for (size_t ax = 0; ax < cidx.size(); ++ax) {
                ccoord[ax] = ucoord[c_tab[ax]];
            }

            bool const first = !written[ccoord];
            if (first && cv.has_tile(ccoord)) {
                pre_existing.push_back(ccoord);
            }
            // First write to a PRE-EXISTING tile carries the real c_pf (the runtime's
            // up-front scale); first write to a tile we are about to create carries 0,
            // a pure overwrite, so this does not rely on the new tile being zeroed.
            PrefactorScalar const node_c_pf = first ? (cv.has_tile(ccoord) ? c_pf : PrefactorScalar{double{0}}) //
                                                    : PrefactorScalar{double{1}};

            emitted.push_back(graph.make_einsum_node(av.tile_id(acoord), bv.tile_id(bcoord), cv.tile_id(ccoord), per_tile, node_c_pf, ab_pf,
                                                     /*conj_a=*/false, /*conj_b=*/false,
                                                     fmt::format("tile_einsum({})", fmt::join(ccoord, ","))));
            written[ccoord] = true;
        }

        // Pre-existing output tiles that received NO contribution are still scaled by
        // c_pf in the runtime. Emitting only the contributing nodes would leave them
        // untouched - a silent numerical difference - so scale them explicitly.
        std::vector<std::vector<int>> existing_coords;
        detail::dispatch_scalar_type(c_h.dtype, [&]<typename T>(T /*tag*/) {
            auto *C = static_cast<TiledRuntimeTensor<T> *>(c_h.tensor_ptr);
            for (auto const &kv : C->tiles()) {
                existing_coords.push_back(kv.first);
            }
        });
        for (auto const &coord : existing_coords) {
            if (written.contains(coord)) {
                continue;
            }
            TensorId const tid = cv.tile_id(coord);
            Node           sc;
            sc.id      = graph.reserve_node_id();
            sc.kind    = OpKind::Scale;
            sc.label   = fmt::format("tile_scale({})", fmt::join(coord, ","));
            sc.inputs  = {tid};
            sc.outputs = {tid};
            if (is_zero(c_pf)) {
                sc.execute = graph.make_zero_executor(tid);
            } else {
                Graph *g   = &graph;
                auto   pf  = c_pf;
                auto   dt  = c_h.dtype;
                sc.execute = [g, tid, pf, dt]() {
                    detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                        auto *t = static_cast<GeneralRuntimeTensor<T, std::allocator<T>> *>(g->tensor(tid).tensor_ptr);
                        *t *= as<T>(pf);
                    });
                };
            }
            // Only describe the scale when the factor is representable: a
            // ScaleDescriptor carries a plain double, so a complex prefactor would
            // be silently truncated and ScaleAbsorption would then fold a wrong
            // value. Leaving op_data empty keeps the node opaque but honest.
            if (is_real_valued(c_pf)) {
                ScaleDescriptor sd;
                sd.factor  = as_real<double>(c_pf);
                sc.op_data = sd;
            }
            emitted.push_back(std::move(sc));
        }

        if (emitted.empty()) {
            // Every tile pair was a structural zero and nothing pre-existed: the
            // original node is a no-op, but leave it rather than silently changing
            // what the graph contains.
            continue;
        }

        _num_tile_nodes += emitted.size();
        ++_num_expanded;
        remove[ni] = true;
        inserts.emplace_back(ni, std::move(emitted));
        modified = true;
    }

    if (!modified) {
        return false;
    }

    graph.erase_nodes(remove);

    // Positions were recorded pre-erase; shift each down by the number of erased
    // nodes below it. The expanded node's own slot IS erased, so the shifted
    // position lands where it used to be and the replacements take its place.
    std::vector<size_t> erased_below(remove.size() + 1, 0);
    for (size_t i = 0; i < remove.size(); ++i) {
        erased_below[i + 1] = erased_below[i] + (remove[i] ? 1 : 0);
    }
    for (auto &[position, group] : inserts) {
        position -= erased_below[position];
    }
    graph.insert_node_groups(std::move(inserts));
    graph.topological_sort();

    EINSUMS_LOG_INFO("TiledExpansion: expanded {} tiled einsum(s) into {} per-tile nodes ({} declined)", _num_expanded, _num_tile_nodes,
                     _num_declined);
    report(1,
           fmt::format("expanded {} tiled einsum(s) into {} per-tile node(s), declined {}", _num_expanded, _num_tile_nodes, _num_declined));
    return true;
}

} // namespace einsums::compute_graph::passes
