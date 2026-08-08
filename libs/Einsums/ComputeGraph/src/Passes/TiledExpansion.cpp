//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

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
    /// Frobenius norm of a stored, materialized tile. Empty when there is nothing
    /// to measure -- absent, or not yet materialized, since planning must not
    /// allocate. An unmeasurable tile is never screened.
    std::function<std::optional<double>(std::vector<int> const &)> tile_norm;
};

/// Odometer depth the gather/scatter executors walk on the stack. Densification
/// declines above it rather than allocating scratch per replay; a tiled tensor of
/// rank 12 is far outside anything the library is used for.
constexpr size_t kMaxWindowRank = 12;

/**
 * @brief Where one tile sits inside a densified operand's buffer, resolved at
 *        PASS time.
 *
 * Both sides are dense column-major and share their fastest axis, so the copy is
 * a run of `run` contiguous elements repeated `runs` times, with an odometer over
 * the remaining axes stepping the buffer side. The tile side needs no odometer at
 * all: it is contiguous, so run @p r starts at `r * run`.
 *
 * The point of precomputing it is what the executor then does NOT do. Building a
 * ``std::vector<SliceSpec>`` and an ``at_view`` per tile per replay put the
 * allocator on the hot path - malloc and free were about a tenth of the replay's
 * samples - to rediscover offsets planning already knew.
 */
struct TileWindow {
    TensorId            id{0};   ///< the tile, resolved through the graph at execute time
    size_t              base{0}; ///< element offset of the window's origin in the buffer
    size_t              run{1};  ///< contiguous elements per run
    size_t              runs{1}; ///< how many runs
    std::vector<size_t> extent;  ///< tile extents of axes 1.., outermost last
    std::vector<size_t> step;    ///< what one step along each of those moves in the buffer
};

/// Call `body(buffer_offset, tile_offset)` once per contiguous run.
template <typename Body>
void for_each_run(TileWindow const &w, Body &&body) {
    std::array<size_t, kMaxWindowRank> index{};
    size_t                             buffer_offset = w.base;
    size_t const                       axes          = w.extent.size();
    for (size_t r = 0; r < w.runs; ++r) {
        body(buffer_offset, r * w.run);
        for (size_t k = 0; k < axes; ++k) {
            buffer_offset += w.step[k];
            if (++index[k] < w.extent[k]) {
                break;
            }
            buffer_offset -= w.step[k] * w.extent[k]; // wrapped; carry outward
            index[k] = 0;
        }
    }
}

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

TiledExpansion::TiledExpansion(size_t max_nodes, double zero_tile_tolerance, Densify densify, FuseTiles fuse)
    : TiledExpansion(max_nodes, zero_tile_tolerance, densify, fuse, CostModel::detect_default()) {
}

TiledExpansion::TiledExpansion(size_t max_nodes, double zero_tile_tolerance, Densify densify, FuseTiles fuse, CostModel cost_model)
    : _max_nodes(max_nodes), _zero_tolerance(zero_tile_tolerance), _densify(densify), _fuse(fuse), _cost_model(std::move(cost_model)) {
}

std::vector<std::string> TiledExpansion::explain() const {
    if (_num_expanded == 0 && _num_declined == 0) {
        return {};
    }
    std::vector<std::string> out;
    if (_num_expanded != 0) {
        out.push_back(fmt::format("TiledExpansion: expanded {} tiled op(s) into {} per-tile node(s) ({} screened out as structurally "
                                  "zero, {} densified, {} fused, {} gather(s) reused)",
                                  _num_expanded, _num_tile_nodes, _num_screened, _num_densified, _num_fused, _num_gathers_reused));
    }
    if (_num_declined != 0) {
        out.push_back(
            fmt::format("TiledExpansion: declined {} tiled op(s) (over node budget or tile sparsity undecidable)", _num_declined));
    }
    return out;
}

void TiledExpansion::reset_stats() {
    _num_expanded       = 0;
    _num_tile_nodes     = 0;
    _num_declined       = 0;
    _num_screened       = 0;
    _num_densified      = 0;
    _num_fused          = 0;
    _num_gathers_reused = 0;
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

    // Tensors some node in this graph writes. Their tile CONTENTS are whatever was
    // there before execution, so they must never be screened on value.
    std::unordered_set<TensorId> produced;
    for (auto const &node : nodes) {
        for (auto tid : node.outputs) {
            produced.insert(tid);
        }
    }

    // One TensorId per (tiled tensor, tile coord). Registering a tile twice would
    // give one buffer two ids and defeat every aliasing analysis downstream.
    std::map<std::pair<TensorId, std::vector<int>>, TensorId> tile_ids;

    // Which tiles each tiled tensor holds. Seeded from the stored tiles and
    // advanced across the walk by each op's runtime tile-creation rule (a tensor
    // produced by an earlier node does not have its tiles yet at pass time, so
    // its sparsity cannot be read off the object). After the first sweep below
    // this holds each tensor's FINAL set -- everything one full execution leaves
    // stored -- which is the state every REPLAY starts from, because tiles are
    // only ever added and the expanded graph is the tensors' only mutator.
    //
    // Sound despite being computed before the stranding fixpoint runs: the sets
    // mirror the RUNTIME's tile creation, which is the same whether a node is
    // expanded or left opaque; and if a node ends up rejected it stays in the
    // graph still naming its tensors, so they are stranded and every plan that
    // depended on them is rejected too. Nothing is created from a prediction
    // either, since minting happens only during emission.
    std::unordered_map<TensorId, std::set<std::vector<int>>> predicted;
    auto tiles_of = [&predicted](TensorId tid, TiledView const &v) -> std::set<std::vector<int>> & {
        auto it = predicted.find(tid);
        if (it == predicted.end()) {
            auto const cs = v.coords();
            it            = predicted.emplace(tid, std::set<std::vector<int>>(cs.begin(), cs.end())).first;
        }
        return it->second;
    };

    // Operand tiles that screen out as zero, computed once per tensor. Screening a
    // tensor the graph writes would read values the graph has not computed yet, so
    // those are skipped outright rather than measured.
    std::unordered_map<TensorId, std::set<std::vector<int>>> screened;
    auto screened_tiles = [&](TensorId tid, TiledView const &v) -> std::set<std::vector<int>> const & {
        auto it = screened.find(tid);
        if (it != screened.end()) {
            return it->second;
        }
        std::set<std::vector<int>> s;
        if (_zero_tolerance >= 0.0 && !produced.contains(tid)) {
            for (auto const &co : v.coords()) {
                if (auto const nrm = v.tile_norm(co); nrm && *nrm <= _zero_tolerance) {
                    s.insert(co);
                }
            }
        }
        return screened.emplace(tid, std::move(s)).first->second;
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
            out.tile_norm = [t](std::vector<int> const &co) -> std::optional<double> {
                if (!t->has_tile(co)) {
                    return std::nullopt;
                }
                auto const &tile = std::as_const(*t).tile(co);
                if (!tile.is_materialized()) {
                    return std::nullopt;
                }
                return static_cast<double>(linear_algebra::norm(linear_algebra::Norm::FROBENIUS, tile));
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
                    auto *t = static_cast<GeneralRuntimeTensor<T, std::allocator<T>> *>(g->live_tensor_ptr(tid));
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

    // One dense `y_tile += alpha * x_tile`. Recorded as a real OpKind::Axpby with
    // the destination among its inputs, so the liveness and hoisting passes read
    // it as the accumulation it is.
    auto emit_tile_axpy = [&graph](TensorId xt, TensorId yt, PrefactorScalar alpha, packed_gemm::ScalarType dt, std::string label) {
        Node nd;
        nd.id      = graph.reserve_node_id();
        nd.kind    = OpKind::Axpby;
        nd.label   = std::move(label);
        nd.inputs  = {xt, yt};
        nd.outputs = {yt};

        // Live scalars shared with the executor, same contract as a captured
        // axpby: the descriptor is what downstream passes read AND what the
        // executor uses, so a fold into alpha reaches the replay. A descriptor
        // the executor ignored would be worse than none.
        auto params   = std::make_shared<AxpbyParams>();
        params->alpha = alpha;
        params->beta  = PrefactorScalar{1.0};

        AxpbyDescriptor desc;
        desc.alpha  = params->alpha;
        desc.beta   = params->beta;
        desc.params = params;
        nd.op_data  = desc;

        Graph *g   = &graph;
        nd.execute = [g, xt, yt, params, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense      = GeneralRuntimeTensor<T, std::allocator<T>>;
                auto const *xptr = static_cast<Dense const *>(g->live_tensor_ptr(xt));
                auto       *yptr = static_cast<Dense *>(g->live_tensor_ptr(yt));
                auto const  b    = as<T>(params->beta);
                if (b == T{1}) {
                    linear_algebra::axpy(as<T>(params->alpha), *xptr, yptr);
                } else {
                    linear_algebra::axpby(as<T>(params->alpha), *xptr, b, yptr);
                }
            });
        };
        return nd;
    };

    // One dense `c_tile = beta*c_tile + alpha*P(a_tile)` through string_permute
    // (HPTT). Attaches a real PermuteDescriptor only when the scalars are
    // representable in its plain doubles, the same honesty rule as the scale.
    auto emit_tile_permute = [&graph](TensorId at, TensorId ct, ParsedPermuteSpec const &pspec, PrefactorScalar alpha, PrefactorScalar beta,
                                      packed_gemm::ScalarType dt, std::string label) {
        Node nd;
        nd.id    = graph.reserve_node_id();
        nd.kind  = OpKind::Permute;
        nd.label = std::move(label);
        // Same RMW convention as the dense op: beta != 0 reads the destination.
        nd.inputs  = is_zero(beta) ? std::vector<TensorId>{at} : std::vector<TensorId>{at, ct};
        nd.outputs = {ct};
        Graph *g   = &graph;
        nd.execute = [g, at, ct, pspec, alpha, beta, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense      = GeneralRuntimeTensor<T, std::allocator<T>>;
                auto const *aptr = static_cast<Dense const *>(g->live_tensor_ptr(at));
                auto       *cptr = static_cast<Dense *>(g->live_tensor_ptr(ct));
                dispatch::string_permute(pspec, as<T>(beta), cptr, as<T>(alpha), *aptr);
            });
        };
        if (is_real_valued(alpha) && is_real_valued(beta)) {
            PermuteDescriptor pd;
            pd.alpha     = as_real<double>(alpha);
            pd.beta      = as_real<double>(beta);
            pd.c_indices = pspec.c_indices;
            pd.a_indices = pspec.a_indices;
            nd.op_data   = pd;
        }
        return nd;
    };

    // One dense reduction over per-tile pairs: r[0] = sum_i dot(a_i, b_i),
    // true_dot when conjugated. A single node whose inputs are the PER-TILE
    // ids, which is what frees the whole-tensor tiled ids from the stranding
    // fixpoint - the entire reason a tiled dot used to poison expansion.
    auto emit_tiled_dot = [&graph](std::vector<TensorId> as_, std::vector<TensorId> bs, TensorId r, bool conj, packed_gemm::ScalarType dt,
                                   std::string label) {
        Node nd;
        nd.id = graph.reserve_node_id();
        // Custom, not Dot: OpKind::Dot is in is_gpu_capable_op and GPUPlacement
        // would target this node, but it is an N-pair reduction whose CPU
        // executor cannot run with its operands swapped to device shadows.
        nd.kind  = OpKind::Custom;
        nd.label = std::move(label);
        nd.inputs.reserve(as_.size() * 2);
        nd.inputs.insert(nd.inputs.end(), as_.begin(), as_.end());
        nd.inputs.insert(nd.inputs.end(), bs.begin(), bs.end());
        nd.outputs = {r};
        Graph *g   = &graph;
        nd.execute = [g, as_ = std::move(as_), bs = std::move(bs), r, conj, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;
                T acc{0};
                for (size_t i = 0; i < as_.size(); ++i) {
                    auto const *ap = static_cast<Dense const *>(g->tensor(as_[i]).tensor_ptr);
                    auto const *bp = static_cast<Dense const *>(g->tensor(bs[i]).tensor_ptr);
                    acc += conj ? linear_algebra::true_dot(*ap, *bp) : linear_algebra::dot(*ap, *bp);
                }
                auto *rp      = static_cast<Dense *>(g->tensor(r).tensor_ptr);
                rp->data()[0] = acc;
            });
        };
        return nd;
    };

    // One dense `c_tile = alpha*(a_tile/b_tile) + beta*c_tile`.
    auto emit_tile_divide = [&graph](TensorId at, TensorId bt, TensorId ct, PrefactorScalar alpha, PrefactorScalar beta,
                                     packed_gemm::ScalarType dt, std::string label) {
        Node nd;
        nd.id    = graph.reserve_node_id();
        nd.kind  = OpKind::DirectDivision;
        nd.label = std::move(label);
        // Same RMW convention as the dense op: beta != 0 reads the destination.
        nd.inputs  = is_zero(beta) ? std::vector<TensorId>{at, bt} : std::vector<TensorId>{at, bt, ct};
        nd.outputs = {ct};
        Graph *g   = &graph;
        nd.execute = [g, at, bt, ct, alpha, beta, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense      = GeneralRuntimeTensor<T, std::allocator<T>>;
                auto const *aptr = static_cast<Dense const *>(g->live_tensor_ptr(at));
                auto const *bptr = static_cast<Dense const *>(g->tensor(bt).tensor_ptr);
                auto       *cptr = static_cast<Dense *>(g->live_tensor_ptr(ct));
                linear_algebra::direct_division(as<T>(alpha), *aptr, *bptr, as<T>(beta), cptr);
            });
        };
        return nd;
    };

    // ── Fused elementwise lowering ───────────────────────────────────────────
    // One node that runs the same elementwise operation over a whole list of
    // tiles, emitted instead of one node per tile when the tiles are too small to
    // be worth dispatching separately. The work is identical either way; only the
    // number of times the executor is entered differs.

    // Bytes one tile holds, read off the grid alone -- no tile object is touched,
    // so this is valid for predicted tiles that do not exist yet.
    auto tile_bytes = [](TiledView const &v, std::vector<int> const &co, size_t elem_size) {
        size_t bytes = elem_size;
        for (size_t ax = 0; ax < v.rank; ++ax) {
            bytes *= static_cast<size_t>((*v.sizes)[ax][static_cast<size_t>(co[ax])]);
        }
        return bytes;
    };

    // Should a group of elementwise tile ops become ONE node? Unlike densifying a
    // contraction this is not a choice between two amounts of work -- fusing does
    // the same arithmetic and the same memory traffic, so it is never slower. What
    // per-tile nodes buy is visibility to CSE and InplaceOptimization, which no
    // cost model can price. So the question asked is the one that can be answered:
    // are the dispatches those nodes cost amortized by the traffic they do?
    //
    // @p streams is how many times each tile's bytes cross the bus: 1 to fill with
    // zeros, 2 for an in-place scale, 3 for an axpy or a divide.
    auto should_fuse = [&](TiledView const &v, std::vector<std::vector<int>> const &coords, size_t elem_size, size_t streams) {
        if (coords.size() < 2 || _fuse == FuseTiles::Never) {
            return false;
        }
        if (_fuse == FuseTiles::Always) {
            return true;
        }
        size_t bytes = 0;
        for (auto const &co : coords) {
            bytes += tile_bytes(v, co, elem_size);
        }
        double const traffic_us = _cost_model.estimate_memory_time_us(bytes * streams, Target::CPU);
        return traffic_us < static_cast<double>(coords.size()) * _cost_model.node_overhead_us(Target::CPU);
    };

    auto emit_fused_scale = [&graph](std::vector<TensorId> tids, PrefactorScalar pf, packed_gemm::ScalarType dt, std::string label) {
        Node sc;
        sc.id      = graph.reserve_node_id();
        sc.kind    = OpKind::TileElementwise;
        sc.label   = std::move(label);
        sc.inputs  = tids;
        sc.outputs = tids;
        Graph *g   = &graph;
        sc.execute = [g, tids = std::move(tids), pf, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                bool const zero = is_zero(pf);
                for (TensorId const tid : tids) {
                    auto *t = static_cast<GeneralRuntimeTensor<T, std::allocator<T>> *>(g->live_tensor_ptr(tid));
                    if (zero) {
                        t->zero();
                    } else {
                        *t *= as<T>(pf);
                    }
                }
            });
        };
        return sc;
    };

    auto emit_fused_axpy = [&graph](std::vector<TensorId> xs, std::vector<TensorId> ys, PrefactorScalar alpha, packed_gemm::ScalarType dt,
                                    std::string label) {
        Node nd;
        nd.id     = graph.reserve_node_id();
        nd.kind   = OpKind::TileElementwise;
        nd.label  = std::move(label);
        nd.inputs = xs;
        nd.inputs.insert(nd.inputs.end(), ys.begin(), ys.end());
        nd.outputs = ys;
        Graph *g   = &graph;
        nd.execute = [g, xs = std::move(xs), ys = std::move(ys), alpha, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;
                for (size_t i = 0; i < xs.size(); ++i) {
                    auto const *xptr = static_cast<Dense const *>(g->tensor(xs[i]).tensor_ptr);
                    auto       *yptr = static_cast<Dense *>(g->tensor(ys[i]).tensor_ptr);
                    linear_algebra::axpy(as<T>(alpha), *xptr, yptr);
                }
            });
        };
        return nd;
    };

    auto emit_fused_divide = [&graph](std::vector<TensorId> as_, std::vector<TensorId> bs, std::vector<TensorId> cs, PrefactorScalar alpha,
                                      PrefactorScalar beta, packed_gemm::ScalarType dt, std::string label) {
        Node nd;
        nd.id     = graph.reserve_node_id();
        nd.kind   = OpKind::TileElementwise;
        nd.label  = std::move(label);
        nd.inputs = as_;
        nd.inputs.insert(nd.inputs.end(), bs.begin(), bs.end());
        // Same RMW convention as the dense op: beta != 0 reads the destination.
        if (!is_zero(beta)) {
            nd.inputs.insert(nd.inputs.end(), cs.begin(), cs.end());
        }
        nd.outputs = cs;
        Graph *g   = &graph;
        nd.execute = [g, as_ = std::move(as_), bs = std::move(bs), cs = std::move(cs), alpha, beta, dt]() {
            detail::dispatch_scalar_type(dt, [&]<typename T>(T /*tag*/) {
                using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;
                for (size_t i = 0; i < as_.size(); ++i) {
                    auto const *aptr = static_cast<Dense const *>(g->tensor(as_[i]).tensor_ptr);
                    auto const *bptr = static_cast<Dense const *>(g->tensor(bs[i]).tensor_ptr);
                    auto       *cptr = static_cast<Dense *>(g->tensor(cs[i]).tensor_ptr);
                    linear_algebra::direct_division(as<T>(alpha), *aptr, *bptr, as<T>(beta), cptr);
                }
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
        enum class Kind : std::uint8_t { Einsum, Scale, Axpy, Divide, Permute, Dot };

        size_t                  index{0};
        Kind                    kind{Kind::Scale};
        std::vector<TensorId>   touched; ///< whole-tensor tiled ids this node uses
        packed_gemm::ScalarType dtype{};
        size_t                  elem_size{0}; ///< bytes per element, for the fusion gate
        /// Source views. Einsum and Divide use all three; Scale uses `dst`; Axpy
        /// uses `src_a` for X and `dst` for Y.
        TiledView src_a, src_b, dst;
        // Einsum
        ParsedEinsumSpec              spec;
        PrefactorScalar               ab_pf{double{1}};
        std::vector<std::vector<int>> a_coords, b_coords, c_coords;
        std::vector<PrefactorScalar>  c_pfs;
        std::vector<std::vector<int>> leftover; ///< output tiles that are only scaled
        PrefactorScalar               leftover_pf{double{0}};
        /// Estimated microseconds for each lowering, from the shared CostModel.
        /// The per-tile figure sums one estimate per emitted contraction, so it
        /// carries that many launch and allocation overheads - which is exactly
        /// what makes small tiles lose. The dense figure is one contraction plus
        /// the gather/scatter traffic.
        double est_tiled_us{0.0};
        double est_dense_us{0.0};
        // Scale / Axpy / Divide / Permute
        PrefactorScalar               alpha{double{1}};
        PrefactorScalar               beta{double{0}}; ///< Divide / Permute
        std::vector<std::vector<int>> coords;
        // Permute: the parsed spec for the per-tile dense permutes. Source and
        // target coordinates ride in a_coords / c_coords (bijective pairing).
        ParsedPermuteSpec pspec;
        // Dot: shared coordinates ride in a_coords; the dense scalar result and
        // whether the reduction conjugates its first operand.
        TensorId result_id{0};
        bool     conj{false};
    };
    std::vector<Plan> plans;

    // Two sweeps over the same walk. Sweep 0 only advances the tile sets, so by
    // its end `predicted` holds each tensor's final set. Sweep 1 re-walks and
    // builds the plans against those final sets (its own inserts are no-ops).
    // Planning against the final set instead of the position's set is what makes
    // the expansion correct on replay 2+, where every tensor enters holding what
    // the previous execution left: leftover scales must cover tiles a later node
    // creates, a first write must carry c_pf because the tile exists by then, and
    // a scratch-zeroing scale whose tiles are all created later is load-bearing
    // rather than a no-op. It stays correct on the first execution because a
    // final-set tile not yet created is minted zero at emission, and scaling or
    // contracting a zero tile matches the runtime's absent-tile semantics.
    for (int sweep = 0; sweep < 2; ++sweep) {
        bool const planning = sweep == 1;

        for (size_t ni = 0; ni < n; ++ni) {
            Node const &src = nodes[ni];

            auto decline = [&](std::string_view why) {
                if (!planning) {
                    return; // sweep 0 tracks tile sets only; sweep 1 repeats the walk and reports
                }
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
                    // tiled_scale scales whichever tiles exist AT EXECUTION: the final
                    // set, since even tiles a LATER node creates exist here from the
                    // second execution on (and are zero the first time, when scaling
                    // them is a no-op). A loop body's scratch-zeroing scale is the
                    // load-bearing case: positionally it sees no tiles at all.
                    auto const &pa     = tiles_of(a_id, av);
                    auto const  coords = std::vector<std::vector<int>>(pa.begin(), pa.end());
                    if (coords.empty()) {
                        continue; // no tiles on any execution: the op is a no-op, leave it alone
                    }
                    if (coords.size() > _max_nodes) {
                        decline(fmt::format("projected {} tiles exceeds the {}-node budget", coords.size(), _max_nodes));
                        continue;
                    }
                    if (!planning) {
                        continue;
                    }
                    Plan p;
                    p.index     = ni;
                    p.kind      = Plan::Kind::Scale;
                    p.touched   = {a_id};
                    p.dtype     = a_h.dtype;
                    p.elem_size = a_h.element_size;
                    p.dst       = av;
                    p.alpha     = alpha;
                    p.coords    = coords;
                    plans.push_back(std::move(p));
                    continue;
                }

                if (edesc->op == TiledElementwiseOp::Divide) {
                    // C = alpha*(A/B) + beta*C, one dense divide per tile STORED IN A.
                    // An absent A tile is a rigorous zero, so C keeps beta*C there --
                    // which is the leftover-scale rule again, reused verbatim below.
                    if (src.outputs.size() != 1 || src.inputs.size() < 2) {
                        continue;
                    }
                    TensorId const a_id = src.inputs[0];
                    TensorId const b_id = src.inputs[1];
                    TensorId const c_id = src.outputs[0];
                    auto const    &a_h  = graph.tensor(a_id);
                    auto const    &b_h  = graph.tensor(b_id);
                    auto const    &c_h  = graph.tensor(c_id);
                    if (!a_h.is_tiled || !b_h.is_tiled || !c_h.is_tiled) {
                        continue;
                    }
                    if (a_h.dtype != c_h.dtype || b_h.dtype != c_h.dtype) {
                        continue;
                    }
                    TiledView av, bv, cv;
                    if (!bind_tiled(a_h, a_id, av) || !bind_tiled(b_h, b_id, bv) || !bind_tiled(c_h, c_id, cv)) {
                        decline("an operand has no backing tiled object");
                        continue;
                    }
                    if (*av.sizes != *bv.sizes || *av.sizes != *cv.sizes) {
                        // The runtime throws; declining leaves the opaque node to throw.
                        decline("A, B and C do not share a tile grid");
                        continue;
                    }
                    auto const &pa = tiles_of(a_id, av);
                    auto const &pb = tiles_of(b_id, bv);
                    auto const  pc = tiles_of(c_id, cv); // final set, by value
                    if (!std::ranges::all_of(pa, [&pb](auto const &co) { return pb.contains(co); })) {
                        // A numerator block with no denominator block. The runtime
                        // throws rather than producing infinities; let it.
                        decline("a numerator tile has no denominator tile");
                        continue;
                    }
                    auto const coords = std::vector<std::vector<int>>(pa.begin(), pa.end());
                    if (coords.empty() && pc.empty()) {
                        continue;
                    }
                    if (coords.size() + pc.size() > _max_nodes) {
                        decline(fmt::format("projected {} tiles exceeds the {}-node budget", coords.size() + pc.size(), _max_nodes));
                        continue;
                    }
                    Plan p;
                    p.index     = ni;
                    p.kind      = Plan::Kind::Divide;
                    p.touched   = {a_id, b_id, c_id};
                    p.dtype     = c_h.dtype;
                    p.elem_size = c_h.element_size;
                    p.src_a     = av;
                    p.src_b     = bv;
                    p.dst       = cv;
                    p.alpha     = alpha;
                    p.beta      = edesc->params->beta;
                    p.coords    = coords;
                    // C tiles the numerator never reaches are still scaled by beta.
                    for (auto const &co : pc) {
                        if (!pa.contains(co)) {
                            p.leftover.push_back(co);
                        }
                    }
                    p.leftover_pf = edesc->params->beta;
                    if (planning) {
                        plans.push_back(std::move(p));
                    }
                    tiles_of(c_id, cv).insert(coords.begin(), coords.end());
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
                p.index     = ni;
                p.kind      = Plan::Kind::Axpy;
                p.touched   = {x_id, y_id};
                p.dtype     = y_h.dtype;
                p.elem_size = y_h.element_size;
                p.src_a     = xv;
                p.dst       = yv;
                p.alpha     = alpha;
                p.coords    = coords;
                if (planning) {
                    plans.push_back(std::move(p));
                }
                // Y gains every tile X has: tiled_axpy creates the matching Y tile when
                // it is absent.
                tiles_of(y_id, yv).insert(coords.begin(), coords.end());
                continue;
            }

            // ── Tiled permute: C = beta*C + alpha*P(A) ───────────────────────────
            // The permutation acts on the tile grid and within each tile alike:
            // each stored A tile contributes to exactly one C tile (coordinates
            // permuted the same way the axes are), so the lowering is one dense
            // per-tile permute per stored A tile, each carrying (alpha, beta), plus
            // the leftover-scale rule for stored C tiles the permutation never
            // reaches - mirroring detail::tiled_permute's runtime semantics.
            if (auto const *pdesc = std::get_if<TiledPermuteDescriptor>(&src.op_data)) {
                if (src.outputs.size() != 1 || src.inputs.empty()) {
                    continue;
                }
                TensorId const a_id = src.inputs[0];
                TensorId const c_id = src.outputs[0];
                auto const    &a_h  = graph.tensor(a_id);
                auto const    &c_h  = graph.tensor(c_id);
                if (!a_h.is_tiled || !c_h.is_tiled || a_h.dtype != c_h.dtype) {
                    continue;
                }
                TiledView av, cv;
                if (!bind_tiled(a_h, a_id, av) || !bind_tiled(c_h, c_id, cv)) {
                    decline("an operand has no backing tiled object");
                    continue;
                }
                size_t const rank = pdesc->c_indices.size();
                if (rank != pdesc->a_indices.size() || rank != av.rank || rank != cv.rank) {
                    decline("spec rank does not match the operand ranks");
                    continue;
                }
                std::vector<size_t> perm(rank);
                bool                perm_ok = true;
                for (size_t i = 0; i < rank && perm_ok; ++i) {
                    auto it = std::ranges::find(pdesc->a_indices, pdesc->c_indices[i]);
                    perm_ok = it != pdesc->a_indices.end();
                    if (perm_ok) {
                        perm[i] = static_cast<size_t>(it - pdesc->a_indices.begin());
                    }
                }
                if (!perm_ok) {
                    decline("output index missing from the input indices");
                    continue;
                }
                bool grids_ok = true;
                for (size_t i = 0; i < rank; ++i) {
                    if ((*cv.sizes)[i] != (*av.sizes)[perm[i]]) {
                        grids_ok = false;
                        break;
                    }
                }
                if (!grids_ok) {
                    // The runtime throws on a grid mismatch; declining preserves that.
                    decline("C's tile grid is not A's grid permuted like the axes");
                    continue;
                }

                auto const                   &pa = tiles_of(a_id, av);
                auto const                    pc = tiles_of(c_id, cv); // final set, by value
                std::vector<std::vector<int>> sources(pa.begin(), pa.end());
                std::vector<std::vector<int>> targets;
                std::set<std::vector<int>>    targeted;
                targets.reserve(sources.size());
                for (auto const &co : sources) {
                    std::vector<int> tk(rank);
                    for (size_t i = 0; i < rank; ++i) {
                        tk[i] = co[perm[i]];
                    }
                    targeted.insert(tk);
                    targets.push_back(std::move(tk));
                }

                Plan p;
                p.index           = ni;
                p.kind            = Plan::Kind::Permute;
                p.touched         = {a_id, c_id};
                p.dtype           = c_h.dtype;
                p.elem_size       = c_h.element_size;
                p.src_a           = av;
                p.dst             = cv;
                p.alpha           = pdesc->alpha;
                p.beta            = pdesc->beta;
                p.pspec.c_indices = pdesc->c_indices;
                p.pspec.a_indices = pdesc->a_indices;
                p.pspec.raw       = fmt::format("{} <- {}", fmt::join(pdesc->c_indices, ","), fmt::join(pdesc->a_indices, ","));
                p.a_coords        = std::move(sources);
                p.c_coords        = std::move(targets);
                // Stored C tiles the permutation never reaches still take beta.
                for (auto const &co : pc) {
                    if (!targeted.contains(co)) {
                        p.leftover.push_back(co);
                    }
                }
                p.leftover_pf = pdesc->beta;
                if (p.a_coords.empty() && p.leftover.empty()) {
                    continue; // nothing stored anywhere: a no-op, leave it alone
                }
                if (p.a_coords.size() + p.leftover.size() > _max_nodes) {
                    decline(
                        fmt::format("projected {} tiles exceeds the {}-node budget", p.a_coords.size() + p.leftover.size(), _max_nodes));
                    continue;
                }

                // Price both lowerings for the emit-site gate. A permute is pure
                // memory traffic, so per-tile is a per-dispatch overhead on a tiny
                // move -- the cost that made a blocked rank-4 permute explode into
                // hundreds of near-free nodes -- while densified pays gather + one
                // whole-tensor permute + scatter.
                {
                    double const overhead = _cost_model.node_overhead_us(Target::CPU);
                    size_t const elem     = c_h.element_size;
                    double       tiled    = 0.0;
                    for (size_t i = 0; i < p.a_coords.size(); ++i) {
                        size_t const bytes = tile_bytes(av, p.a_coords[i], elem) //
                                             + (is_zero(p.beta) ? 1 : 2) * tile_bytes(cv, p.c_coords[i], elem);
                        tiled += _cost_model.estimate_memory_time_us(bytes, Target::CPU) + overhead;
                    }
                    for (auto const &co : p.leftover) {
                        tiled += _cost_model.estimate_memory_time_us(2 * tile_bytes(cv, co, elem), Target::CPU) + overhead;
                    }
                    p.est_tiled_us = tiled;

                    size_t dense_elems = 1;
                    for (auto const &axis : *cv.sizes) {
                        size_t ext = 0;
                        for (int const s : axis) {
                            ext += static_cast<size_t>(s);
                        }
                        dense_elems *= ext;
                    }
                    size_t const dense_bytes = dense_elems * elem;
                    // Gather reads the A tiles and writes the buffer; the permute
                    // reads and writes whole buffers; the scatter reads the result
                    // and writes (for beta != 0 also reads) the C tiles.
                    p.est_dense_us = _cost_model.estimate_memory_time_us(2 * dense_bytes, Target::CPU)                           //
                                     + _cost_model.estimate_memory_time_us(2 * dense_bytes, Target::CPU)                         //
                                     + _cost_model.estimate_memory_time_us((is_zero(p.beta) ? 2 : 3) * dense_bytes, Target::CPU) //
                                     + 4 * overhead;
                }

                tiles_of(c_id, cv).insert(p.c_coords.begin(), p.c_coords.end());
                if (planning) {
                    plans.push_back(std::move(p));
                }
                continue;
            }

            // ── Tiled dot / dotc: dense scalar = sum over shared tiles ───────────
            if (auto const *ddesc = std::get_if<TiledDotDescriptor>(&src.op_data)) {
                if (src.inputs.size() != 2 || src.outputs.size() != 1) {
                    continue;
                }
                TensorId const a_id = src.inputs[0];
                TensorId const b_id = src.inputs[1];
                TensorId const r_id = src.outputs[0];
                auto const    &a_h  = graph.tensor(a_id);
                auto const    &b_h  = graph.tensor(b_id);
                auto const    &r_h  = graph.tensor(r_id);
                if (!a_h.is_tiled || !b_h.is_tiled || r_h.is_tiled || a_h.dtype != b_h.dtype) {
                    continue;
                }
                TiledView av, bv;
                if (!bind_tiled(a_h, a_id, av) || !bind_tiled(b_h, b_id, bv)) {
                    decline("an operand has no backing tiled object");
                    continue;
                }
                if (*av.sizes != *bv.sizes) {
                    // The runtime throws on a grid mismatch; declining preserves that.
                    decline("operands do not share a tile grid");
                    continue;
                }
                auto const &pa = tiles_of(a_id, av);
                auto const &pb = tiles_of(b_id, bv);
                Plan        p;
                p.index     = ni;
                p.kind      = Plan::Kind::Dot;
                p.touched   = {a_id, b_id};
                p.dtype     = a_h.dtype;
                p.elem_size = a_h.element_size;
                p.src_a     = av;
                p.src_b     = bv;
                p.result_id = r_id;
                p.conj      = ddesc->conjugated;
                for (auto const &co : pa) {
                    if (pb.contains(co)) {
                        p.a_coords.push_back(co);
                    }
                }
                // No shared tiles is a valid reduction over nothing: the emitted
                // node writes exactly the 0 the runtime would.
                if (planning) {
                    plans.push_back(std::move(p));
                }
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

            auto const &cidx = tdesc->indices->spec.c_indices;
            auto const &aidx = tdesc->indices->spec.a_indices;
            auto const &bidx = tdesc->indices->spec.b_indices;
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
            size_t const nu = unique.size();
            auto pos = [&unique](std::string const &s) { return static_cast<size_t>(std::ranges::find(unique, s) - unique.begin()); };

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

            // Enumeration order: CONTRACTED indices slowest, output indices fastest, so
            // every tile GEMM at the same accumulation step is emitted as one contiguous
            // run. GEMMBatching only batches a group whose span contains no outside node
            // touching the same buffers, and letting the contracted index vary fastest
            // drops each output tile's later accumulations in between the first writes,
            // which disqualifies every group. Ordering it this way is what makes the
            // tile GEMMs batchable at all.
            //
            // Per output tile the contracted steps stay in ascending order, so each tile
            // accumulates in exactly the order the runtime uses and the result is
            // bit-identical; only independent tiles move relative to each other.
            //
            // `unique` holds the C letters first (add_unique(cidx) ran first), so the
            // positions from cidx.size() up are exactly the contracted ones.
            std::vector<size_t> perm;
            perm.reserve(nu);
            for (size_t u = cidx.size(); u < nu; ++u) {
                perm.push_back(u);
            }
            for (size_t u = 0; u < cidx.size(); ++u) {
                perm.push_back(u);
            }
            std::vector<int> pgrid(nu);
            for (size_t t = 0; t < nu; ++t) {
                pgrid[t] = grid[perm[t]];
            }

            auto const   stride = grid_strides(pgrid);
            size_t const total  = stride.empty() ? 0 : stride[0] * static_cast<size_t>(pgrid[0]);
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

            // Operand sparsity and which output tiles exist when this node runs --
            // all FINAL sets in this sweep. For inputs that is required, not merely
            // safe: a tile some later node creates already holds data here on every
            // execution after the first, so its combinations must be emitted (they
            // contract zeros the first time). For the output it decides c_pf below.
            auto const &pa = tiles_of(a_id, av);
            auto const &pb = tiles_of(b_id, bv);
            auto const  pc = tiles_of(c_id, cv); // by value: the loop below adds to the live set
            auto const &sa = screened_tiles(a_id, av);
            auto const &sb = screened_tiles(b_id, bv);

            ParsedEinsumSpec per_tile;
            per_tile.c_indices = cidx;
            per_tile.a_indices = aidx;
            per_tile.b_indices = bidx;
            per_tile.raw       = fmt::format("{} <- {} ; {}", fmt::join(cidx, ","), fmt::join(aidx, ","), fmt::join(bidx, ","));

            // Index roles, so each tile contraction can be priced as a GEMM: an index
            // in C came from A or from B (M or N); one absent from C is contracted (K).
            std::vector<bool> in_c(nu, false), in_a(nu, false);
            for (auto const &nm : cidx) {
                in_c[pos(nm)] = true;
            }
            for (auto const &nm : aidx) {
                in_a[pos(nm)] = true;
            }
            size_t const elem_size = c_h.element_size;

            Plan                             plan;
            std::map<std::vector<int>, bool> written; // output coord -> already written once
            for (size_t s = 0; s < total; ++s) {
                size_t           rem = s;
                std::vector<int> ucoord(nu);
                for (size_t t = 0; t < nu; ++t) {
                    ucoord[perm[t]] = static_cast<int>(rem / stride[t]);
                    rem %= stride[t];
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
                if (sa.contains(acoord) || sb.contains(bcoord)) {
                    // Numerically zero operand: contributes nothing. Treated exactly
                    // like an absent tile, so if this leaves the output tile with no
                    // contribution at all it is simply never created, and the next
                    // contraction sees it as absent.
                    if (planning) {
                        ++_num_screened;
                    }
                    continue;
                }
                for (size_t ax = 0; ax < cidx.size(); ++ax) {
                    ccoord[ax] = ucoord[c_tab[ax]];
                }

                bool const first = !written[ccoord];
                // The first write to a tile carries the real c_pf (the runtime's
                // up-front scale of an existing tile). Against the final set every
                // written tile "pre-exists": minted zero for the first execution,
                // where c_pf*0 degenerates to the overwrite, and holding the previous
                // execution's value on every replay, where c_pf is load-bearing --
                // an overwrite there would drop the accumulation semantics.
                PrefactorScalar const node_c_pf = first ? (pc.contains(ccoord) ? c_pf : PrefactorScalar{double{0}}) //
                                                        : PrefactorScalar{double{1}};

                // This tile contraction's GEMM shape, so the cost model can price it:
                // M over output indices from A, N over output indices from B, K over the
                // contracted ones.
                size_t tM = 1, tN = 1, tK = 1;
                for (size_t u = 0; u < nu; ++u) {
                    auto const ext = static_cast<size_t>(part[u][static_cast<size_t>(ucoord[u])]);
                    if (in_c[u]) {
                        (in_a[u] ? tM : tN) *= ext;
                    } else {
                        tK *= ext;
                    }
                }
                plan.est_tiled_us += _cost_model.estimate_total_gemm_time_us(tM, tN, tK, elem_size, Target::CPU);

                plan.a_coords.push_back(acoord);
                plan.b_coords.push_back(bcoord);
                plan.c_coords.push_back(ccoord);
                plan.c_pfs.push_back(node_c_pf);
                written[ccoord] = true;
            }

            // The densified alternative: one contraction over the FULL extents, plus the
            // cost of marshalling every operand in and the result back out. The gather
            // and scatter are pure memory traffic, so they are priced at bandwidth.
            {
                size_t dM = 1, dN = 1, dK = 1;
                for (size_t u = 0; u < nu; ++u) {
                    size_t extent = 0;
                    for (int const sz : part[u]) {
                        extent += static_cast<size_t>(sz);
                    }
                    if (in_c[u]) {
                        (in_a[u] ? dM : dN) *= extent;
                    } else {
                        dK *= extent;
                    }
                }
                // A is M x K, B is K x N, C is M x N; C is touched twice, read out of the
                // dense buffer and accumulated into its tiles.
                size_t const bytes = elem_size * (dM * dK + dK * dN + 2 * dM * dN);
                plan.est_dense_us  = _cost_model.estimate_total_gemm_time_us(dM, dN, dK, elem_size, Target::CPU) +
                                    _cost_model.estimate_memory_time_us(bytes, Target::CPU);
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
            plan.elem_size   = elem_size;
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
            if (planning) {
                plans.push_back(std::move(plan));
            }
        }
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

    // ── Densified lowering ───────────────────────────────────────────────────
    // Gather each tiled operand into a dense buffer, contract once, scatter the
    // result back. Chosen when the tiles are too small to be worth a dispatch
    // each; see the gate at the emit site.
    //
    // The buffers are plain heap tensors whose lifetime is handed to the graph via
    // adopt(), because they must outlive this pass and be alive for every replay.
    // They are registered as ordinary tensors so the dense contraction is a normal
    // Einsum node that later passes can reason about, rather than an opaque blob.
    // A dense gather already emitted and still valid at this point in the emission.
    // One tiled operand is typically contracted many times in a row -- the CCSD
    // residual gathers t1 into a dense buffer 33 times and t2 eleven times -- and
    // re-copying it for each is pure waste while nothing has written it.
    //
    // The key is the exact list of tile ids gathered, which is what makes this safe
    // in both directions: a tensor that has since GAINED a tile produces a
    // different list and misses, and an entry is dropped outright as soon as any
    // emitted node writes a tile it covers. Entries can only be invalidated by
    // nodes this pass emits, because a candidate whose tiled operands are touched
    // by a node that does not expand is rejected by the stranding fixpoint above.
    struct GatheredBuffer {
        std::vector<size_t>          dims;
        std::vector<TensorId>        sources;
        std::unordered_set<TensorId> source_set;
        TensorId                     buffer{0};
    };
    std::vector<GatheredBuffer> gathered;

    // Dense extent of an operand axis is the sum of its tile sizes; a tile's
    // offset along that axis is the prefix sum below its grid coordinate.
    auto dense_dims = [](TiledView const &v) {
        std::vector<size_t> d(v.rank, 0);
        for (size_t ax = 0; ax < v.rank; ++ax) {
            for (int const sz : (*v.sizes)[ax]) {
                d[ax] += static_cast<size_t>(sz);
            }
        }
        return d;
    };

    // Where one tile lands in the dense buffer, as runs the executor can copy
    // without consulting anything. Both sides are column-major, so the buffer's
    // axis 0 and the tile's are equally contiguous and a run is the tile's
    // extent along it.
    auto window_of = [](TiledView const &v, std::vector<int> const &co, std::vector<size_t> const &dims) {
        std::vector<size_t> stride(v.rank, 1);
        for (size_t ax = 1; ax < v.rank; ++ax) {
            stride[ax] = stride[ax - 1] * dims[ax - 1];
        }
        auto extent_at = [&](size_t ax) { return static_cast<size_t>((*v.sizes)[ax][static_cast<size_t>(co[ax])]); };

        TileWindow w;
        w.id = v.tile_id(co);
        for (size_t ax = 0; ax < v.rank; ++ax) {
            size_t offset = 0;
            for (int k = 0; k < co[ax]; ++k) {
                offset += static_cast<size_t>((*v.sizes)[ax][static_cast<size_t>(k)]);
            }
            w.base += offset * stride[ax];
        }
        w.run = extent_at(0);
        for (size_t ax = 1; ax < v.rank; ++ax) {
            w.extent.push_back(extent_at(ax));
            w.step.push_back(stride[ax]);
            w.runs *= extent_at(ax);
        }
        return w;
    };

    // Gather one tiled operand into a fresh dense buffer: zero, then copy every
    // stored tile (an absent tile is a structural zero, which is exactly what the
    // zeroed buffer already holds). Reuses a buffer already holding this exact
    // tile set if one is still valid, in which case no node is emitted at all.
    // Shared by the densified einsum and densified permute lowerings.
    auto make_dense_gather = [&]<typename T>(T /*tag*/, TiledView const &v, std::vector<size_t> const &dims, char const *what,
                                             std::vector<Node> &out) -> TensorId {
        using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;

        std::vector<TileWindow> windows;
        std::vector<TensorId>   ins;
        for (auto const &co : v.coords()) {
            windows.push_back(window_of(v, co, dims));
            ins.push_back(windows.back().id);
        }
        for (auto const &have : gathered) {
            if (have.dims == dims && have.sources == ins) {
                ++_num_gathers_reused;
                return have.buffer;
            }
        }

        auto *buf = new Dense(fmt::format("densify_{}", what), dims);
        graph.adopt([buf]() { delete buf; });
        TensorId const buf_id = graph.register_tensor(make_handle(*buf, 0));

        Node g;
        g.id      = graph.reserve_node_id();
        g.kind    = OpKind::TileGather;
        g.label   = fmt::format("tile_gather({})", what);
        g.inputs  = ins;
        g.outputs = {buf_id};
        Graph *gp = &graph;
        g.execute = [gp, buf, windows = std::move(windows)]() {
            buf->zero();
            T *dest = buf->data();
            for (auto const &w : windows) {
                T const *tile = static_cast<Dense const *>(gp->tensor(w.id).tensor_ptr)->data();
                for_each_run(w, [&](size_t buffer_offset, size_t tile_offset) {
                    T       *d = dest + buffer_offset;
                    T const *s = tile + tile_offset;
                    for (size_t i = 0; i < w.run; ++i) {
                        d[i] = s[i];
                    }
                });
            }
        };
        out.push_back(std::move(g));
        gathered.push_back(
            {.dims = dims, .sources = ins, .source_set = std::unordered_set<TensorId>(ins.begin(), ins.end()), .buffer = buf_id});
        return buf_id;
    };

    // Scatter a dense result buffer back into the given output tiles, each as
    // C_tile = pf * C_tile + dense_slice (pf == 0 is a pure overwrite). Shared by
    // the densified einsum and densified permute lowerings.
    auto make_dense_scatter = [&]<typename T>(T /*tag*/, GeneralRuntimeTensor<T, std::allocator<T>> *bc, TensorId idc, TiledView const &dst,
                                              std::vector<size_t> const &dc, std::vector<std::vector<int>> const &c_tiles,
                                              std::vector<PrefactorScalar> const &c_tile_pf, std::vector<Node> &out) {
        using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;

        std::vector<TileWindow>      dsts;
        std::vector<PrefactorScalar> pfs;
        std::vector<TensorId>        outs;
        for (size_t i = 0; i < c_tiles.size(); ++i) {
            dsts.push_back(window_of(dst, c_tiles[i], dc));
            pfs.push_back(c_tile_pf[i]);
            outs.push_back(dsts.back().id);
        }
        Node sc;
        sc.id     = graph.reserve_node_id();
        sc.kind   = OpKind::TileScatter;
        sc.label  = "tile_scatter(C)";
        sc.inputs = {idc};
        // A nonzero prefactor reads the destination, so declare it an input too
        // (bug-1009's RMW convention) or a later pass may reorder past the read.
        for (size_t i = 0; i < outs.size(); ++i) {
            if (!is_zero(pfs[i])) {
                sc.inputs.push_back(outs[i]);
            }
        }
        sc.outputs = outs;
        Graph *gp  = &graph;
        sc.execute = [gp, bc, dsts = std::move(dsts), pfs = std::move(pfs)]() {
            T const *source = bc->data();
            for (size_t i = 0; i < dsts.size(); ++i) {
                auto const &w    = dsts[i];
                T          *tile = static_cast<Dense *>(gp->tensor(w.id).tensor_ptr)->data();
                if (is_zero(pfs[i])) {
                    for_each_run(w, [&](size_t buffer_offset, size_t tile_offset) {
                        T const *s = source + buffer_offset;
                        T       *d = tile + tile_offset;
                        for (size_t k = 0; k < w.run; ++k) {
                            d[k] = s[k];
                        }
                    });
                } else {
                    T const pf = as<T>(pfs[i]);
                    for_each_run(w, [&](size_t buffer_offset, size_t tile_offset) {
                        T const *s = source + buffer_offset;
                        T       *d = tile + tile_offset;
                        for (size_t k = 0; k < w.run; ++k) {
                            d[k] = pf * d[k] + s[k];
                        }
                    });
                }
            }
        };
        out.push_back(std::move(sc));
    };

    auto emit_densified = [&](Plan const &pl, std::vector<Node> &out) -> bool {
        size_t const ra = pl.src_a.rank, rb = pl.src_b.rank, rc = pl.dst.rank;

        // The executors walk their odometer on the stack; decline rather than
        // grow it per replay. Checked before anything is emitted, so declining
        // leaves the per-tile lowering a clean slate.
        if (ra > kMaxWindowRank || rb > kMaxWindowRank || rc > kMaxWindowRank) {
            return false;
        }

        std::vector<size_t> const da = dense_dims(pl.src_a), db = dense_dims(pl.src_b), dc = dense_dims(pl.dst);

        // Distinct output tiles, each with the prefactor its FIRST write carried.
        // The dense contraction produces the whole sum, so the scatter applies that
        // prefactor once: C_tile = c_pf * C_tile + dense_slice.
        std::vector<std::vector<int>> c_tiles;
        std::vector<PrefactorScalar>  c_tile_pf;
        {
            std::map<std::vector<int>, size_t> seen;
            for (size_t i = 0; i < pl.c_coords.size(); ++i) {
                if (auto const it = seen.find(pl.c_coords[i]); it == seen.end()) {
                    seen.emplace(pl.c_coords[i], c_tiles.size());
                    c_tiles.push_back(pl.c_coords[i]);
                    c_tile_pf.push_back(pl.c_pfs[i]);
                }
            }
        }

        bool ok = true;
        detail::dispatch_scalar_type(pl.dtype, [&]<typename T>(T tag) {
            using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;

            auto *bc = new Dense("densify_c", dc);
            graph.adopt([bc]() { delete bc; });
            TensorId const idc = graph.register_tensor(make_handle(*bc, 0));

            TensorId const ida = make_dense_gather(tag, pl.src_a, da, "a", out);
            TensorId const idb = make_dense_gather(tag, pl.src_b, db, "b", out);

            // One dense contraction. c_pf is 0 because the scatter applies the real
            // one per output tile; ab_pf rides along here as it would per tile.
            out.push_back(graph.make_einsum_node(ida, idb, idc, pl.spec, PrefactorScalar{double{0}}, pl.ab_pf,
                                                 /*conj_a=*/false, /*conj_b=*/false, "densified_einsum"));

            // Scatter into exactly the tiles the per-tile path would have written,
            // so symmetry-forbidden output blocks are still never created.
            make_dense_scatter(tag, bc, idc, pl.dst, dc, c_tiles, c_tile_pf, out);
        });
        return ok;
    };

    // Densified permute: gather A into a dense buffer, permute ONCE, scatter into
    // C. The gathered buffer holds zeros wherever A has no tile, so the permuted
    // buffer is zero over every C tile the permutation never reaches -- scattering
    // C_tile = beta*C_tile + slice over targets AND leftovers alike therefore
    // reproduces the leftover-scale rule with no separate scale nodes.
    auto emit_densified_permute = [&](Plan const &pl, std::vector<Node> &out) -> bool {
        if (pl.src_a.rank > kMaxWindowRank || pl.dst.rank > kMaxWindowRank) {
            return false;
        }

        std::vector<size_t> const da = dense_dims(pl.src_a), dc = dense_dims(pl.dst);

        std::vector<std::vector<int>> c_tiles = pl.c_coords;
        c_tiles.insert(c_tiles.end(), pl.leftover.begin(), pl.leftover.end());
        std::vector<PrefactorScalar> const c_tile_pf(c_tiles.size(), pl.beta);

        detail::dispatch_scalar_type(pl.dtype, [&]<typename T>(T tag) {
            using Dense = GeneralRuntimeTensor<T, std::allocator<T>>;

            auto *bc = new Dense("densify_c", dc);
            graph.adopt([bc]() { delete bc; });
            TensorId const idc = graph.register_tensor(make_handle(*bc, 0));

            TensorId const ida = make_dense_gather(tag, pl.src_a, da, "a", out);

            // One dense permute carrying alpha; beta is 0 here because the scatter
            // applies the real beta per output tile.
            out.push_back(emit_tile_permute(ida, idc, pl.pspec, pl.alpha, PrefactorScalar{double{0}}, pl.dtype, "densified_permute"));

            make_dense_scatter(tag, bc, idc, pl.dst, dc, c_tiles, c_tile_pf, out);
        });
        return true;
    };

    // Scales of a set of output tiles, fused or one per tile. Every leftover set --
    // a contraction's, a divide's -- goes through here too, since scaling a tile
    // the operation never reached is the same elementwise op at the same size.
    auto append_scales = [&](TiledView const &v, std::vector<std::vector<int>> const &coords, PrefactorScalar pf,
                             packed_gemm::ScalarType dt, size_t elem_size, std::vector<Node> &out) {
        if (coords.empty()) {
            return;
        }
        if (should_fuse(v, coords, elem_size, /*streams=*/is_zero(pf) ? 1 : 2)) {
            std::vector<TensorId> tids;
            tids.reserve(coords.size());
            for (auto const &co : coords) {
                tids.push_back(v.tile_id(co));
            }
            out.push_back(emit_fused_scale(std::move(tids), pf, dt, fmt::format("tile_scale(x{})", coords.size())));
            ++_num_fused;
            return;
        }
        for (auto const &co : coords) {
            out.push_back(emit_tile_scale(v.tile_id(co), pf, dt, fmt::format("tile_scale({})", fmt::join(co, ","))));
        }
    };

    // ── Emit ─────────────────────────────────────────────────────────────────
    for (size_t p = 0; p < plans.size(); ++p) {
        if (!alive[p]) {
            continue;
        }
        Plan             &pl = plans[p];
        std::vector<Node> emitted;

        switch (pl.kind) {
        case Plan::Kind::Einsum: {
            // Whichever lowering the cost model prices lower. Densifying does more
            // arithmetic but pays one launch instead of thousands, so the comparison
            // has to be in time; the class documentation records why a gate on the
            // flop RATIO cannot decide it.
            bool const densify =
                !pl.a_coords.empty() && (_densify == Densify::Always || (_densify == Densify::Auto && pl.est_dense_us < pl.est_tiled_us));
            if (densify && emit_densified(pl, emitted)) {
                ++_num_densified;
                report(2, fmt::format("densified '{}': {} tile contractions -> gather+einsum+scatter (est {:.1f} us vs {:.1f} us)",
                                      nodes[pl.index].label, pl.a_coords.size(), pl.est_dense_us, pl.est_tiled_us));
                append_scales(pl.dst, pl.leftover, pl.leftover_pf, pl.dtype, pl.elem_size, emitted);
                break;
            }
            emitted.reserve(pl.a_coords.size() + pl.leftover.size());
            for (size_t i = 0; i < pl.a_coords.size(); ++i) {
                emitted.push_back(graph.make_einsum_node(pl.src_a.tile_id(pl.a_coords[i]), pl.src_b.tile_id(pl.b_coords[i]),
                                                         pl.dst.tile_id(pl.c_coords[i]), pl.spec, pl.c_pfs[i], pl.ab_pf,
                                                         /*conj_a=*/false, /*conj_b=*/false,
                                                         fmt::format("tile_einsum({})", fmt::join(pl.c_coords[i], ","))));
            }
            append_scales(pl.dst, pl.leftover, pl.leftover_pf, pl.dtype, pl.elem_size, emitted);
            break;
        }
        case Plan::Kind::Scale:
            emitted.reserve(pl.coords.size());
            append_scales(pl.dst, pl.coords, pl.alpha, pl.dtype, pl.elem_size, emitted);
            break;
        case Plan::Kind::Axpy:
            emitted.reserve(pl.coords.size());
            if (should_fuse(pl.dst, pl.coords, pl.elem_size, /*streams=*/3)) {
                std::vector<TensorId> xs, ys;
                xs.reserve(pl.coords.size());
                ys.reserve(pl.coords.size());
                for (auto const &coord : pl.coords) {
                    xs.push_back(pl.src_a.tile_id(coord));
                    ys.push_back(pl.dst.tile_id(coord));
                }
                emitted.push_back(
                    emit_fused_axpy(std::move(xs), std::move(ys), pl.alpha, pl.dtype, fmt::format("tile_axpy(x{})", pl.coords.size())));
                ++_num_fused;
                break;
            }
            for (auto const &coord : pl.coords) {
                emitted.push_back(emit_tile_axpy(pl.src_a.tile_id(coord), pl.dst.tile_id(coord), pl.alpha, pl.dtype,
                                                 fmt::format("tile_axpy({})", fmt::join(coord, ","))));
            }
            break;
        case Plan::Kind::Permute: {
            // Same time-based gate as the contraction: per-tile pays a dispatch
            // per tiny move, densified pays gather + one permute + scatter.
            bool const densify =
                !pl.a_coords.empty() && (_densify == Densify::Always || (_densify == Densify::Auto && pl.est_dense_us < pl.est_tiled_us));
            if (densify && emit_densified_permute(pl, emitted)) {
                ++_num_densified;
                report(2, fmt::format("densified '{}': {} tile permutes -> gather+permute+scatter (est {:.1f} us vs {:.1f} us)",
                                      nodes[pl.index].label, pl.a_coords.size(), pl.est_dense_us, pl.est_tiled_us));
                break;
            }
            emitted.reserve(pl.a_coords.size() + pl.leftover.size());
            for (size_t i = 0; i < pl.a_coords.size(); ++i) {
                emitted.push_back(emit_tile_permute(pl.src_a.tile_id(pl.a_coords[i]), pl.dst.tile_id(pl.c_coords[i]), pl.pspec, pl.alpha,
                                                    pl.beta, pl.dtype, fmt::format("tile_permute({})", fmt::join(pl.c_coords[i], ","))));
            }
            append_scales(pl.dst, pl.leftover, pl.leftover_pf, pl.dtype, pl.elem_size, emitted);
            break;
        }
        case Plan::Kind::Dot: {
            std::vector<TensorId> as_, bs;
            as_.reserve(pl.a_coords.size());
            bs.reserve(pl.a_coords.size());
            for (auto const &co : pl.a_coords) {
                as_.push_back(pl.src_a.tile_id(co));
                bs.push_back(pl.src_b.tile_id(co));
            }
            emitted.push_back(emit_tiled_dot(std::move(as_), std::move(bs), pl.result_id, pl.conj, pl.dtype,
                                             fmt::format("tile_dot(x{})", pl.a_coords.size())));
            break;
        }
        case Plan::Kind::Divide:
            emitted.reserve(pl.coords.size() + pl.leftover.size());
            if (should_fuse(pl.dst, pl.coords, pl.elem_size, /*streams=*/is_zero(pl.beta) ? 3 : 4)) {
                std::vector<TensorId> as_, bs, cs;
                as_.reserve(pl.coords.size());
                bs.reserve(pl.coords.size());
                cs.reserve(pl.coords.size());
                for (auto const &coord : pl.coords) {
                    as_.push_back(pl.src_a.tile_id(coord));
                    bs.push_back(pl.src_b.tile_id(coord));
                    cs.push_back(pl.dst.tile_id(coord));
                }
                emitted.push_back(emit_fused_divide(std::move(as_), std::move(bs), std::move(cs), pl.alpha, pl.beta, pl.dtype,
                                                    fmt::format("tile_divide(x{})", pl.coords.size())));
                ++_num_fused;
            } else {
                for (auto const &coord : pl.coords) {
                    emitted.push_back(emit_tile_divide(pl.src_a.tile_id(coord), pl.src_b.tile_id(coord), pl.dst.tile_id(coord), pl.alpha,
                                                       pl.beta, pl.dtype, fmt::format("tile_divide({})", fmt::join(coord, ","))));
                }
            }
            append_scales(pl.dst, pl.leftover, pl.leftover_pf, pl.dtype, pl.elem_size, emitted);
            break;
        }

        // Anything this group writes invalidates every gathered copy covering it.
        // Done after the whole group so a contraction that reads a tensor it also
        // writes still gathers the pre-write contents, exactly as it did before.
        for (auto const &nd : emitted) {
            for (TensorId const written : nd.outputs) {
                std::erase_if(gathered, [written](GatheredBuffer const &g) { return g.source_set.contains(written); });
            }
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
    report(1, fmt::format("expanded {} tiled op(s) into {} node(s) ({} densified, {} elementwise group(s) fused, {} gather(s) reused), "
                          "declined {}",
                          _num_expanded, _num_tile_nodes, _num_densified, _num_fused, _num_gathers_reused, _num_declined));
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
