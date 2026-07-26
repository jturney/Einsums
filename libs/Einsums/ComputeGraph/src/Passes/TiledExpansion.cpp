//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// The tile grid of one tiled operand, read through its type-erased handle.
struct TiledView {
    size_t                               rank{0};
    std::vector<std::vector<int>> const *sizes{nullptr};
    /// Presence test, stored-coord listing and per-tile TensorId minting all need
    /// the concrete type, so they are captured as type-erased callables built
    /// under a dtype dispatch.
    std::function<bool(std::vector<int> const &)>     has_tile;
    std::function<std::vector<std::vector<int>>()>    coords;
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
    // Planning below walks the nodes in order and carries each tiled tensor's tile
    // set forward across them, so the vector has to be in an order the executor
    // will actually use.
    graph.topological_sort();

    auto        &nodes = graph.nodes();
    size_t const n     = nodes.size();
    if (n == 0) {
        return false;
    }

    // One TensorId per (tiled tensor, tile coord). Registering a tile twice would
    // give one buffer two ids and defeat every aliasing analysis downstream.
    std::map<std::pair<TensorId, std::vector<int>>, TensorId> tile_ids;

    // Which tiles each tiled tensor will hold at this point in the program, given
    // the expansions planned so far. A tensor produced by an earlier node does not
    // have its tiles yet at pass time, so its sparsity cannot be read off the
    // object; it has to be carried forward from whoever writes it. Seeded from the
    // stored tiles and updated as each producer is planned.
    //
    // Sound despite being computed before the stranding fixpoint runs: if a
    // producer ends up rejected it stays in the graph still naming its output, so
    // that tensor is stranded and every consumer whose prediction depended on it is
    // rejected too. Nothing is created from a prediction either, since minting
    // happens only during emission.
    std::unordered_map<TensorId, std::set<std::vector<int>>> predicted;
    auto tiles_of = [&predicted](TensorId tid, TiledView const &v) -> std::set<std::vector<int>> & {
        auto it = predicted.find(tid);
        if (it == predicted.end()) {
            auto const cs = v.coords();
            it            = predicted.emplace(tid, std::set<std::vector<int>>(cs.begin(), cs.end())).first;
        }
        return it->second;
    };

    // Bind a type-erased view of one tiled operand: grid, presence test, stored
    // coords, and a memoized per-tile TensorId minter that infer-and-creates the
    // tile exactly as the runtime does.
    auto bind_tiled = [&graph, &tile_ids](TensorHandle const &h, TensorId owner, TiledView &out) {
        bool ok = true;
        detail::dispatch_scalar_type(h.dtype, [&]<typename T>(T /*tag*/) {
            auto *t = static_cast<TiledRuntimeTensor<T> *>(h.tensor_ptr);
            if (t == nullptr) {
                ok = false;
                return;
            }
            out.rank     = t->rank();
            out.sizes    = &t->tile_sizes();
            out.has_tile = [t](std::vector<int> const &co) { return t->has_tile(co); };
            out.coords   = [t]() {
                std::vector<std::vector<int>> cs;
                cs.reserve(t->tiles().size());
                for (auto const &kv : t->tiles()) {
                    cs.push_back(kv.first);
                }
                return cs;
            };
            out.tile_id = [t, owner, &tile_ids, &graph](std::vector<int> const &co) {
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
        });
        return ok;
    };

    // One dense in-place scale of a single tile. Used both for a tiled scale and
    // for the output tiles a tiled einsum scales but never accumulates into.
    auto emit_tile_scale = [&graph](TensorId tid, PrefactorScalar pf, packed_gemm::ScalarType dt, std::string label) {
        Node sc;
        sc.id      = graph.reserve_node_id();
        sc.kind    = OpKind::Scale;
        sc.label   = std::move(label);
        sc.inputs  = {tid};
        sc.outputs = {tid};
        if (is_zero(pf)) {
            sc.execute = graph.make_zero_executor(tid);
        } else {
            Graph *g   = &graph;
            sc.execute = [g, tid, pf, dt]() {
                detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                    auto *t = static_cast<GeneralRuntimeTensor<T, std::allocator<T>> *>(g->tensor(tid).tensor_ptr);
                    *t *= as<T>(pf);
                });
            };
        }
        // Only describe the scale when the factor is representable: a
        // ScaleDescriptor carries a plain double, so a complex prefactor would be
        // silently truncated and ScaleAbsorption would then fold a wrong value.
        // Leaving op_data empty keeps the node opaque but honest.
        if (is_real_valued(pf)) {
            ScaleDescriptor sd;
            sd.factor  = as_real<double>(pf);
            sc.op_data = sd;
        }
        return sc;
    };

    // One dense `y_tile += alpha * x_tile`. Recorded as a real OpKind::Axpy with
    // the destination among its inputs, so the liveness and hoisting passes read
    // it as the accumulation it is.
    auto emit_tile_axpy = [&graph](TensorId xt, TensorId yt, PrefactorScalar alpha, packed_gemm::ScalarType dt, std::string label) {
        Node nd;
        nd.id      = graph.reserve_node_id();
        nd.kind    = OpKind::Axpy;
        nd.label   = std::move(label);
        nd.inputs  = {xt, yt};
        nd.outputs = {yt};
        Graph *g   = &graph;
        nd.execute = [g, xt, yt, alpha, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense      = GeneralRuntimeTensor<T, std::allocator<T>>;
                auto const *xptr = static_cast<Dense const *>(g->tensor(xt).tensor_ptr);
                auto       *yptr = static_cast<Dense *>(g->tensor(yt).tensor_ptr);
                linear_algebra::axpy(as<T>(alpha), *xptr, yptr);
            });
        };
        return nd;
    };

    std::vector<std::pair<size_t, std::vector<Node>>> inserts;
    std::vector<bool>                                 remove(n, false);
    bool                                              modified = false;

    // What a candidate WOULD expand into. Planning is separated from emission
    // because minting a per-tile TensorId creates the tile as a side effect, and
    // the stranding fixpoint below can still reject a candidate after it has been
    // planned. Deciding first and creating second keeps a rejected candidate from
    // leaving new tiles behind, which would change how the runtime applies c_pf.
    struct Plan {
        enum class Kind : std::uint8_t { Einsum, Scale, Axpy };

        size_t                  index{0};
        Kind                    kind{Kind::Scale};
        std::vector<TensorId>   touched; ///< whole-tensor tiled ids this node uses
        packed_gemm::ScalarType dtype{};
        /// Source views. Einsum uses all three; Scale uses `dst`; Axpy uses `src_a`
        /// for X and `dst` for Y.
        TiledView src_a, src_b, dst;
        // Einsum
        ParsedEinsumSpec              spec;
        PrefactorScalar               ab_pf{double{1}};
        std::vector<std::vector<int>> a_coords, b_coords, c_coords;
        std::vector<PrefactorScalar>  c_pfs;
        std::vector<std::vector<int>> leftover; ///< output tiles that are only scaled
        PrefactorScalar               leftover_pf{double{0}};
        // Scale / Axpy
        PrefactorScalar               alpha{double{1}};
        std::vector<std::vector<int>> coords;
    };
    std::vector<Plan> plans;

    for (size_t ni = 0; ni < n; ++ni) {
        Node const &src = nodes[ni];

        auto decline = [&](std::string_view why) {
            ++_num_declined;
            report(2, fmt::format("declining '{}': {}", src.label, why));
            EINSUMS_LOG_DEBUG("TiledExpansion: declining node {} - {}", src.id, why);
        };

        // ── Elementwise: tiled scale and tiled axpy ──────────────────────────
        if (auto const *edesc = std::get_if<TiledElementwiseDescriptor>(&src.op_data)) {
            if (!edesc->params) {
                continue;
            }
            PrefactorScalar const alpha = edesc->params->alpha;

            if (edesc->op == TiledElementwiseOp::Scale) {
                if (src.inputs.size() != 1 || src.outputs.size() != 1 || src.inputs[0] != src.outputs[0]) {
                    continue;
                }
                TensorId const a_id = src.outputs[0];
                auto const    &a_h  = graph.tensor(a_id);
                if (!a_h.is_tiled) {
                    continue;
                }
                TiledView av;
                if (!bind_tiled(a_h, a_id, av)) {
                    decline("the operand has no backing tiled object");
                    continue;
                }
                // tiled_scale scales whichever tiles exist AT EXECUTION, which is the
                // predicted set here, not merely the ones stored now: an earlier
                // producer in this graph may still be about to add more.
                auto const &pa     = tiles_of(a_id, av);
                auto const  coords = std::vector<std::vector<int>>(pa.begin(), pa.end());
                if (coords.empty()) {
                    continue; // no stored tiles: the op is a no-op, leave it alone
                }
                if (coords.size() > _max_nodes) {
                    decline(fmt::format("projected {} tiles exceeds the {}-node budget", coords.size(), _max_nodes));
                    continue;
                }
                Plan p;
                p.index   = ni;
                p.kind    = Plan::Kind::Scale;
                p.touched = {a_id};
                p.dtype   = a_h.dtype;
                p.dst     = av;
                p.alpha   = alpha;
                p.coords  = coords;
                plans.push_back(std::move(p));
                continue;
            }

            // Axpy: Y += alpha*X, one dense axpy per tile STORED IN X. A tile absent
            // from X contributes nothing; a tile present in X but not in Y is created
            // zeroed, so the accumulation is still correct.
            if (src.inputs.size() != 2 || src.outputs.size() != 1) {
                continue;
            }
            TensorId const x_id = src.inputs[0];
            TensorId const y_id = src.outputs[0];
            auto const    &x_h  = graph.tensor(x_id);
            auto const    &y_h  = graph.tensor(y_id);
            if (!x_h.is_tiled || !y_h.is_tiled || x_h.dtype != y_h.dtype) {
                continue;
            }
            TiledView xv, yv;
            if (!bind_tiled(x_h, x_id, xv) || !bind_tiled(y_h, y_id, yv)) {
                decline("an operand has no backing tiled object");
                continue;
            }
            if (*xv.sizes != *yv.sizes) {
                // The runtime throws on a grid mismatch. Declining preserves that:
                // the opaque node stays and still throws at execute time.
                decline("X and Y do not share a tile grid");
                continue;
            }
            auto const &px     = tiles_of(x_id, xv);
            auto const  coords = std::vector<std::vector<int>>(px.begin(), px.end());
            if (coords.empty()) {
                continue;
            }
            if (coords.size() > _max_nodes) {
                decline(fmt::format("projected {} tiles exceeds the {}-node budget", coords.size(), _max_nodes));
                continue;
            }
            Plan p;
            p.index   = ni;
            p.kind    = Plan::Kind::Axpy;
            p.touched = {x_id, y_id};
            p.dtype   = y_h.dtype;
            p.src_a   = xv;
            p.dst     = yv;
            p.alpha   = alpha;
            p.coords  = coords;
            plans.push_back(std::move(p));
            // Y gains every tile X has: tiled_axpy creates the matching Y tile when
            // it is absent.
            tiles_of(y_id, yv).insert(coords.begin(), coords.end());
            continue;
        }

        // ── Contraction ──────────────────────────────────────────────────────
        auto const *tdesc = std::get_if<TiledEinsumDescriptor>(&src.op_data);
        if (tdesc == nullptr || !tdesc->indices || !tdesc->params) {
            continue;
        }
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
        if (!bind_tiled(a_h, a_id, av) || !bind_tiled(b_h, b_id, bv) || !bind_tiled(c_h, c_id, cv)) {
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

        auto const c_pf  = tdesc->params->c_pf;
        auto const ab_pf = tdesc->params->ab_pf;

        // Operand sparsity and, crucially, which output tiles ALREADY EXIST when
        // this node runs. The latter decides whether the first write to a tile
        // carries c_pf or overwrites, so it must be the predicted set at this point
        // in the program, copied before the loop starts adding to it.
        auto const &pa = tiles_of(a_id, av);
        auto const &pb = tiles_of(b_id, bv);
        auto const  pc = tiles_of(c_id, cv); // by value: the pre-node set

        ParsedEinsumSpec per_tile;
        per_tile.c_indices = cidx;
        per_tile.a_indices = aidx;
        per_tile.b_indices = bidx;
        per_tile.raw       = fmt::format("{} <- {} ; {}", fmt::join(cidx, ","), fmt::join(aidx, ","), fmt::join(bidx, ","));

        Plan                             plan;
        std::map<std::vector<int>, bool> written; // output coord -> already written once
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
            if (!pa.contains(acoord) || !pb.contains(bcoord)) {
                continue; // structural zero
            }
            for (size_t ax = 0; ax < cidx.size(); ++ax) {
                ccoord[ax] = ucoord[c_tab[ax]];
            }

            bool const first = !written[ccoord];
            // First write to a PRE-EXISTING tile carries the real c_pf (the runtime's
            // up-front scale); first write to a tile that does not exist yet carries
            // 0, a pure overwrite, so this does not rely on the new tile being zeroed.
            PrefactorScalar const node_c_pf = first ? (pc.contains(ccoord) ? c_pf : PrefactorScalar{double{0}}) //
                                                    : PrefactorScalar{double{1}};

            plan.a_coords.push_back(acoord);
            plan.b_coords.push_back(bcoord);
            plan.c_coords.push_back(ccoord);
            plan.c_pfs.push_back(node_c_pf);
            written[ccoord] = true;
        }

        // Pre-existing output tiles that received NO contribution are still scaled by
        // c_pf in the runtime. Emitting only the contributing nodes would leave them
        // untouched - a silent numerical difference - so scale them explicitly.
        for (auto const &coord : pc) {
            if (!written.contains(coord)) {
                plan.leftover.push_back(coord);
            }
        }

        if (plan.a_coords.empty() && plan.leftover.empty()) {
            // Every tile pair was a structural zero and nothing pre-existed: the
            // original node is a no-op, but leave it rather than silently changing
            // what the graph contains.
            continue;
        }

        plan.index       = ni;
        plan.kind        = Plan::Kind::Einsum;
        plan.touched     = {a_id, b_id, c_id};
        plan.dtype       = c_h.dtype;
        plan.src_a       = av;
        plan.src_b       = bv;
        plan.dst         = cv;
        plan.spec        = per_tile;
        plan.ab_pf       = ab_pf;
        plan.leftover_pf = c_pf;
        // C now holds everything it held before plus every tile this contraction
        // writes, which is what a later consumer of C must see.
        auto &pc_live = tiles_of(c_id, cv);
        pc_live.insert(plan.c_coords.begin(), plan.c_coords.end());
        plans.push_back(std::move(plan));
    }

    // ── Stranding fixpoint ───────────────────────────────────────────────────
    // Expanding a node replaces its whole-tensor reads and writes with per-tile
    // ones, so the whole-tensor TensorId loses every reference the expansion
    // owned. Any node left behind that still names that id would have its
    // dependency edge silently dropped: with no writer, a reader can be scheduled
    // before the tiles are filled. A candidate may therefore expand only if every
    // node touching its tiled tensors expands too, and rejecting one candidate can
    // strand another, so this iterates to a fixpoint.
    std::vector<bool> alive(plans.size(), true);
    for (bool changed = true; changed;) {
        changed = false;

        std::unordered_set<size_t> expanding;
        for (size_t p = 0; p < plans.size(); ++p) {
            if (alive[p]) {
                expanding.insert(plans[p].index);
            }
        }

        std::unordered_set<TensorId> stranded;
        for (size_t i = 0; i < n; ++i) {
            if (expanding.contains(i)) {
                continue;
            }
            auto note = [&](TensorId tid) {
                if (auto const *h = graph.find_tensor(tid); h != nullptr && h->is_tiled) {
                    stranded.insert(tid);
                }
            };
            for (auto tid : nodes[i].inputs) {
                note(tid);
            }
            for (auto tid : nodes[i].outputs) {
                note(tid);
            }
        }

        for (size_t p = 0; p < plans.size(); ++p) {
            if (!alive[p]) {
                continue;
            }
            bool const hit = std::ranges::any_of(plans[p].touched, [&](TensorId tid) { return stranded.contains(tid); });
            if (hit) {
                alive[p]        = false;
                changed         = true;
                Node const &nd  = nodes[plans[p].index];
                auto const  why = "a tiled operand is also used by a node that cannot expand, so expanding would drop that dependency";
                ++_num_declined;
                report(2, fmt::format("declining '{}': {}", nd.label, why));
                EINSUMS_LOG_DEBUG("TiledExpansion: declining node {} - {}", nd.id, why);
            }
        }
    }

    // ── Emit ─────────────────────────────────────────────────────────────────
    for (size_t p = 0; p < plans.size(); ++p) {
        if (!alive[p]) {
            continue;
        }
        Plan             &pl = plans[p];
        std::vector<Node> emitted;

        switch (pl.kind) {
        case Plan::Kind::Einsum:
            emitted.reserve(pl.a_coords.size() + pl.leftover.size());
            for (size_t i = 0; i < pl.a_coords.size(); ++i) {
                emitted.push_back(graph.make_einsum_node(pl.src_a.tile_id(pl.a_coords[i]), pl.src_b.tile_id(pl.b_coords[i]),
                                                         pl.dst.tile_id(pl.c_coords[i]), pl.spec, pl.c_pfs[i], pl.ab_pf,
                                                         /*conj_a=*/false, /*conj_b=*/false,
                                                         fmt::format("tile_einsum({})", fmt::join(pl.c_coords[i], ","))));
            }
            for (auto const &coord : pl.leftover) {
                emitted.push_back(
                    emit_tile_scale(pl.dst.tile_id(coord), pl.leftover_pf, pl.dtype, fmt::format("tile_scale({})", fmt::join(coord, ","))));
            }
            break;
        case Plan::Kind::Scale:
            emitted.reserve(pl.coords.size());
            for (auto const &coord : pl.coords) {
                emitted.push_back(
                    emit_tile_scale(pl.dst.tile_id(coord), pl.alpha, pl.dtype, fmt::format("tile_scale({})", fmt::join(coord, ","))));
            }
            break;
        case Plan::Kind::Axpy:
            emitted.reserve(pl.coords.size());
            for (auto const &coord : pl.coords) {
                emitted.push_back(emit_tile_axpy(pl.src_a.tile_id(coord), pl.dst.tile_id(coord), pl.alpha, pl.dtype,
                                                 fmt::format("tile_axpy({})", fmt::join(coord, ","))));
            }
            break;
        }

        _num_tile_nodes += emitted.size();
        ++_num_expanded;
        remove[pl.index] = true;
        inserts.emplace_back(pl.index, std::move(emitted));
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
