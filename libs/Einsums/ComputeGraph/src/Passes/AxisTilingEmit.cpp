//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// AxisTiling: writing a plan into the graph as loops, and the symmetry a plan can exploit.

#include <Einsums/Config/Namespace.hpp>

#include "AxisTilingDetail.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::axis_tiling)

namespace {

/// Scale @p handle's tensor by @p factor in the PARENT, before the loop: the destination prefactor a
/// contraction applied once, which the loop's additions must not apply again.
template <typename T>
void scale_in_parent(Graph &graph, TensorHandle const &handle, T factor) {
    CaptureGuard const guard(graph);
    if (handle.is_tensor_view) {
        scale(factor, static_cast<RuntimeTensorView<T> *>(handle.live_ptr()));
    } else {
        scale(factor, static_cast<RuntimeTensor<T> *>(handle.live_ptr()));
    }
}

/// Zero the accumulator in the PARENT, before the loop.
///
/// The reduction the region wrote OVERWROTE its destination, and the loop accumulates into it,
/// so the two agree only if the destination starts at zero on every replay. A scale by zero
/// assigns rather than multiplies, which is what keeps a destination holding a non-finite value
/// from carrying it into the sum.
template <typename T>
void zero_accumulator(Graph &graph, TensorHandle const &handle) {
    scale_in_parent(graph, handle, T{0});
}

/// The index letters a slot keeps once its sliced axes have become loop variables.
///
/// An axis taken a point at a time is DROPPED from the operand rather than kept as an axis of
/// extent one. Both spell the same arithmetic, and the drop is what makes the body's
/// contractions the shapes a kernel recognises: at one occupied pair the four-index contraction
/// is an ordinary matrix product over the auxiliary index, which is what the hand-written pair
/// loop performs, where a rank-four contraction with two unit axes would reach the generic
/// algorithm instead. An axis cut into blocks keeps its letter, at the block's extent.
std::vector<std::string> surviving(std::vector<std::string> const &letters, std::vector<int> const &labels,
                                   std::vector<AxisWalk> const &walks) {
    std::vector<std::string> out;
    out.reserve(letters.size());
    for (std::size_t p = 0; p < letters.size(); ++p) {
        if (p < labels.size() && labels[p] != kFree && !walks[static_cast<std::size_t>(labels[p])].range) {
            continue;
        }
        out.push_back(letters[p]);
    }
    return out;
}

/// The transposes a contraction over two matrices and one summed letter needs, when it is one.
struct GemmShape {
    bool trans_a{false};
    bool trans_b{false};
};

/// Whether a reduced contraction is a plain matrix product, and with which transposes.
std::optional<GemmShape> gemm_shape(std::vector<std::string> const &c, std::vector<std::string> const &a, std::vector<std::string> const &b,
                                    std::vector<std::string> const &link) {
    if (c.size() != 2 || a.size() != 2 || b.size() != 2 || link.size() != 1) {
        return std::nullopt;
    }
    std::string const &k = link[0];
    GemmShape          shape;
    if (a[0] == k && a[1] == c[0]) {
        shape.trans_a = true;
    } else if (a[1] == k && a[0] == c[0]) {
        shape.trans_a = false;
    } else {
        return std::nullopt;
    }
    if (b[0] == k && b[1] == c[1]) {
        shape.trans_b = false;
    } else if (b[1] == k && b[0] == c[1]) {
        shape.trans_b = true;
    } else {
        return std::nullopt;
    }
    return shape;
}

/// Whether an operand is one a vendor GEMM can read: a column-major matrix, whose first axis
/// steps by one element.
///
/// The other grouped kernels do not ask, because they call the same per-member routine the
/// ungrouped form calls and that routine reads strides. A batched GEMM does not: it hands BLAS
/// a leading dimension and a base pointer, so an operand whose first axis steps by more than one
/// is not a matrix it can describe. Dropping a MIDDLE axis of a three-index tensor leaves
/// exactly that, and the region fuzz found it as a wrong answer rather than as a refusal.
template <typename T>
bool gemm_readable(RuntimeTensorView<T> const &view) {
    return view.rank() < 2 || view.stride(0) == 1;
}

/// A prefactor a grouped kernel can take, which is a real number however the tensor is typed.
std::optional<double> real_prefactor(PrefactorScalar const &value) {
    if (!is_real_valued(value)) {
        return std::nullopt;
    }
    return as_real<double>(value);
}

/// Build the loop body: the captured algebra at one slice, a chunk of slices at a time.
///
/// Every operand the body reads or writes is a VIEW. A caller's tensor is viewed at the slice
/// the member is at, or in full when it carries no sliced axis, so what the body touches is the
/// caller's own buffer and a rebind at a new geometry follows it. An intermediate carrying a
/// sliced axis is re-declared on the body at slice extents and viewed in full, which is what
/// makes every operand one type and the emission one function rather than a dispatch over which
/// of them happen to be views.
///
/// A chunk of more than one slice is emitted as the GROUPED form of the same body: the members
/// of a chunk are independent, so one node per family per chunk replaces one node per family per
/// member wherever a grouped kernel exists for that family. The accumulation is the exception
/// and is grouped anyway, because a grouped accumulation runs its entries sequentially against a
/// repeated destination, which is the same summation order the members were captured in.
///
/// A range plan emits one body per loop of @ref loop_walks, each at depth one: an axis cut into
/// blocks is read through a RANGE view and kept in every spec at the block's extent, so one
/// contraction covers the block, and @p tag keeps the parameters of one loop apart from another's.
///
/// A triangle's loops pass @p table, the block positions each iteration visits, in place of the
/// grid, and a loop over tiles off the diagonal sets @p mirror: its contraction writes a scratch
/// tile, which is added to the tile's own block of the output and, exchanged, to the mirrored one.
template <typename T>
void emit_body(Graph &parent, Graph &body, Plan const &plan, std::vector<AxisWalk> const &walks, std::string const &tag,
               std::vector<std::vector<std::size_t>> const *table = nullptr, bool mirror = false) {
    std::size_t const depth = plan.range ? 1 : std::max<std::size_t>(1, plan.depth);

    // ── Declarations, before the guard ──────────────────────────────────────
    // A declaration is not a capture, and a buffer the resource phase is to place has to exist
    // before the nodes that read it do.
    std::map<std::pair<TensorId, std::size_t>, RuntimeTensor<T> *> owned;
    for (auto const &[tid, labels] : plan.body_owned) {
        auto const              &handle = parent.tensor(tid);
        std::vector<std::size_t> dims;
        dims.reserve(handle.dims.size());
        for (std::size_t p = 0; p < handle.dims.size(); ++p) {
            if (p < labels.size() && labels[p] != kFree) {
                if (auto const &walk = walks[static_cast<std::size_t>(labels[p])]; walk.range) {
                    dims.push_back(walk.length);
                }
                continue;
            }
            dims.push_back(handle.dims[p]);
        }
        for (std::size_t member = 0; member < depth; ++member) {
            owned[{tid, member}] = &body.declare_runtime_tensor<T>(fmt::format("{}#{}", handle.name, member), dims, /*intermediate=*/true);
        }
    }
    // A mirrored tile's contribution, at the shape of the tile's block of the output.
    RuntimeTensor<T> *scratch = nullptr;
    if (mirror) {
        auto const              &out    = plan.ops[plan.triangle->writer].slots[0];
        auto const              &handle = parent.tensor(out.tid);
        std::vector<std::size_t> dims;
        dims.reserve(handle.dims.size());
        for (std::size_t p = 0; p < handle.dims.size(); ++p) {
            bool const sliced = p < out.labels.size() && out.labels[p] != kFree;
            dims.push_back(sliced ? walks[static_cast<std::size_t>(out.labels[p])].length : handle.dims[p]);
        }
        scratch = &body.declare_runtime_tensor<T>("axtile_mirror", dims, /*intermediate=*/true);
    }
    std::vector<RuntimeTensor<T> *> partials(depth, nullptr);
    if (plan.accumulator_id != 0) {
        for (std::size_t member = 0; member < depth; ++member) {
            partials[member] = &body.declare_runtime_tensor<T>(fmt::format("axtile_partial#{}", member), std::vector<std::size_t>{1},
                                                               /*intermediate=*/true);
        }
    }

    CaptureGuard const guard(body);

    // ── The slice index, one parameter per axis per member ──────────────────
    auto const  bounds   = table != nullptr ? declare_table_bounds(body, walks, *table, "axtile", tag)
                                            : declare_walk_bounds(body, depth, walks, "axtile", tag);
    auto const &index_of = bounds.start;
    auto const &end_of   = bounds.end;

    // ── The operands ────────────────────────────────────────────────────────
    std::map<std::tuple<TensorId, std::vector<int>, std::size_t>, RuntimeTensorView<T> *> views;

    auto axes_for = [&](std::vector<int> const &labels, std::size_t rank, std::size_t member) {
        std::vector<ViewAxis> out;
        out.reserve(rank);
        for (std::size_t p = 0; p < rank; ++p) {
            if (p < labels.size() && labels[p] != kFree) {
                auto const k = static_cast<std::size_t>(labels[p]);
                out.push_back(walks[k].range ? ViewAxis::range(BoundExpr(index_of[member][k]), BoundExpr(end_of[member][k]))
                                             : ViewAxis::drop(BoundExpr(index_of[member][k])));
            } else {
                out.push_back(ViewAxis::full());
            }
        }
        return out;
    };

    auto view_of = [&](TensorId tid, std::vector<int> const &labels, std::size_t member) -> RuntimeTensorView<T> & {
        auto const key = std::tuple{tid, labels, member};
        if (auto const hit = views.find(key); hit != views.end()) {
            return *hit->second;
        }
        RuntimeTensorView<T> *made = nullptr;
        if (auto const local = owned.find({tid, member}); local != owned.end()) {
            made = &view_runtime(*local->second, std::vector<ViewAxis>(local->second->rank(), ViewAxis::full()));
        } else {
            // The live object, not the caller's: the caller may have dropped its tensor once it
            // was captured, and capturing over the stand-in still names the same operand.
            auto const &handle = parent.tensor(tid);
            auto        recipe = axes_for(labels, handle.rank, member);
            made = handle.is_tensor_view ? &view_runtime(*static_cast<RuntimeTensorView<T> *>(handle.live_ptr()), std::move(recipe))
                                         : &view_runtime(*static_cast<RuntimeTensor<T> *>(handle.live_ptr()), std::move(recipe));
        }
        views.emplace(key, made);
        return *made;
    };

    // ── The algebra, one family at a time ───────────────────────────────────
    bool const chunked = depth > 1;
    for (auto const &op : plan.ops) {
        bool const accumulating = plan.accumulator_id != 0 && op.slots[0].tid == plan.accumulator_id;
        // A contraction summing a sliced axis adds its block's share to a destination the parent
        // scaled once before the loop.
        bool const summing = std::ranges::any_of(plan.summed_into, [&](auto const &entry) { return entry.first == op.slots[0].tid; });

        // [slot][member]. Collected before anything is emitted, because a grouped node takes the
        // whole chunk at once and an ungrouped one takes it a member at a time.
        std::vector<std::vector<RuntimeTensorView<T> *>> operand(op.slots.size());
        for (std::size_t s = 0; s < op.slots.size(); ++s) {
            operand[s].reserve(depth);
            for (std::size_t member = 0; member < depth; ++member) {
                if (s == 0 && accumulating) {
                    operand[s].push_back(
                        &view_runtime(*partials[member], std::vector<ViewAxis>(partials[member]->rank(), ViewAxis::full())));
                } else {
                    operand[s].push_back(&view_of(op.slots[s].tid, op.slots[s].labels, member));
                }
            }
        }
        auto sources = [&](std::size_t slot) {
            std::vector<RuntimeTensorView<T> const *> out;
            out.reserve(depth);
            for (auto *view : operand[slot]) {
                out.push_back(view);
            }
            return out;
        };

        switch (op.kind) {
        case OpKind::Einsum: {
            auto const &desc  = op.op_data.get<EinsumDescriptor>();
            auto const  lists = live_index_lists(desc);
            auto const  c     = surviving(lists.c, op.slots[0].labels, walks);
            auto const  a     = surviving(lists.a, op.slots[1].labels, walks);
            auto const  b     = surviving(lists.b, op.slots[2].labels, walks);
            auto const  alpha = real_prefactor(live_ab_prefactor(desc));
            auto const  beta  = real_prefactor(live_c_prefactor(desc));
            if (mirror && &op == &plan.ops[plan.triangle->writer]) {
                // The tile's contribution once, then its two destinations. Both adds are permutes
                // with the contraction's own destination prefactor, so a prefactor of zero
                // overwrites each block exactly as the captured contraction did, and every element
                // of the output is written by exactly one of the two.
                auto &tile = view_runtime(*scratch, std::vector<ViewAxis>(scratch->rank(), ViewAxis::full()));
                compute_graph::detail::einsum_parsed(einsum_spec(a, b, c, live_conj_a(desc), live_conj_b(desc), lists.operators), T{0},
                                                     &tile, as<T>(live_ab_prefactor(desc)), *operand[1][0], *operand[2][0]);

                T const     beta  = as<T>(live_c_prefactor(desc));
                auto const &sigma = plan.triangle->sigma;
                compute_graph::detail::permute_parsed(permute_spec(c, c), beta, operand[0][0], T{1}, tile);

                // The mirrored block at q holds the contribution at the exchanged index, which is
                // sign times the tile at apply(sigma, q).
                std::vector<std::string> exchanged;
                exchanged.reserve(c.size());
                for (std::size_t p = 0; p < c.size(); ++p) {
                    exchanged.push_back(c[static_cast<std::size_t>(static_cast<unsigned char>(sigma.permutation[p]))]);
                }
                auto &mirrored = view_of(op.slots[0].tid, plan.triangle->mirror_labels, 0);
                compute_graph::detail::permute_parsed(permute_spec(c, exchanged), beta, &mirrored, T(static_cast<double>(sigma.sign)),
                                                      tile);
                break;
            }
            auto const as_gemm  = gemm_shape(c, a, b, lists.link);
            bool       readable = true;
            for (auto const &slot : operand) {
                for (auto const *view : slot) {
                    readable = readable && gemm_readable(*view);
                }
            }
            // A GEMM computes the one term; an operator's other terms need the per-member path. A
            // summing contraction's members all add into one destination, one after another.
            if (chunked && !summing && readable && as_gemm.has_value() && alpha.has_value() && beta.has_value() && !live_conj_a(desc) &&
                !live_conj_b(desc) && lists.operators.empty()) {
                grouped_batched_gemm(*alpha, sources(1), sources(2), *beta, operand[0], as_gemm->trans_a, as_gemm->trans_b);
                break;
            }
            auto const spec = einsum_spec(a, b, c, live_conj_a(desc), live_conj_b(desc), lists.operators);
            for (std::size_t member = 0; member < depth; ++member) {
                compute_graph::detail::einsum_parsed(spec, summing ? T{1} : as<T>(live_c_prefactor(desc)), operand[0][member],
                                                     as<T>(live_ab_prefactor(desc)), *operand[1][member], *operand[2][member]);
            }
            break;
        }
        case OpKind::Permute: {
            auto const &desc = op.op_data.get<PermuteDescriptor>();
            // A permutation keeps its scalars in the params block where every other kind does,
            // and its own snapshots are plain complex doubles, so the live block is read first
            // and the snapshot only stands in for a node that has none.
            PrefactorScalar const alpha = desc.params ? desc.params->alpha : PrefactorScalar{desc.alpha};
            PrefactorScalar const beta  = desc.params ? desc.params->beta : PrefactorScalar{desc.beta};
            auto const            spec  = permute_spec(surviving(desc.c_indices, op.slots[0].labels, walks),
                                                       surviving(desc.a_indices, op.slots[1].labels, walks), desc.operators);
            auto const            re_a  = real_prefactor(alpha);
            auto const            re_c  = real_prefactor(beta);
            if (chunked && re_a.has_value() && re_c.has_value() && desc.operators.empty()) {
                compute_graph::detail::grouped_permute_parsed(spec, operand[0], sources(1), std::vector<double>(depth, *re_c),
                                                              std::vector<double>(depth, *re_a));
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                compute_graph::detail::permute_parsed(spec, as<T>(beta), operand[0][member], as<T>(alpha), *operand[1][member]);
            }
            break;
        }
        case OpKind::Axpby: {
            auto const &desc  = op.op_data.get<AxpbyDescriptor>();
            auto const  alpha = real_prefactor(live_alpha(desc));
            auto const  beta  = real_prefactor(live_beta(desc));
            if (chunked && alpha.has_value() && beta.has_value()) {
                grouped_axpby(std::vector<double>(depth, *alpha), sources(1), std::vector<double>(depth, *beta), operand[0]);
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                axpby(as<T>(live_alpha(desc)), *operand[1][member], as<T>(live_beta(desc)), operand[0][member]);
            }
            break;
        }
        case OpKind::Scale: {
            auto const &desc = op.op_data.get<ScaleDescriptor>();
            for (std::size_t member = 0; member < depth; ++member) {
                scale(as<T>(live_factor(desc)), operand[0][member]);
            }
            break;
        }
        case OpKind::DirectProduct:
        case OpKind::DirectDivision: {
            auto const &desc  = op.op_data.get<ElementwiseBinaryDescriptor>();
            T const     alpha = as<T>(live_alpha(desc));
            T const     beta  = as<T>(live_beta(desc));
            if (chunked) {
                if (op.kind == OpKind::DirectProduct) {
                    grouped_direct_product(std::vector<T>(depth, alpha), sources(1), sources(2), std::vector<T>(depth, beta), operand[0]);
                } else {
                    grouped_direct_division(std::vector<T>(depth, alpha), sources(1), sources(2), std::vector<T>(depth, beta), operand[0]);
                }
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                if (op.kind == OpKind::DirectProduct) {
                    direct_product(alpha, *operand[1][member], *operand[2][member], beta, operand[0][member]);
                } else {
                    direct_division(alpha, *operand[1][member], *operand[2][member], beta, operand[0][member]);
                }
            }
            break;
        }
        case OpKind::Dot: {
            if (chunked) {
                grouped_dot(operand[0], sources(1), sources(2));
            } else {
                for (std::size_t member = 0; member < depth; ++member) {
                    dot_python(operand[0][member], *operand[1][member], *operand[2][member]);
                }
            }
            if (accumulating) {
                // The loop-carried accumulation. The destination is repeated, which is what
                // makes a grouped accumulation run its entries sequentially and sum the chunk
                // in the order its members were captured in.
                auto const           &handle = parent.tensor(plan.accumulator_id);
                RuntimeTensorView<T> &into   = view_of(plan.accumulator_id, std::vector<int>(handle.rank, kFree), 0);
                if (chunked) {
                    grouped_axpby(std::vector<double>(depth, 1.0), sources(0), std::vector<double>(depth, 1.0),
                                  std::vector<RuntimeTensorView<T> *>(depth, &into));
                } else {
                    axpby(T{1}, *operand[0][0], T{1}, &into);
                }
            }
            break;
        }
        case OpKind::Custom: {
            // A registered node, re-emitted per member through its slice hook: its descriptor at
            // this member's slice, over the members' views, read and written as any other kind's.
            DescriptorHooks const &hooks = find_descriptor_codec(op.op_data.name())->hooks;
            for (std::size_t member = 0; member < depth; ++member) {
                auto  sliced = hooks.slice(op.op_data,
                                           slice_request(op, [&](int label) { return index_of[member][static_cast<std::size_t>(label)]; }));
                auto &ctx    = CaptureContext::current();
                std::vector<TensorId> inputs;
                for (std::size_t s = 1; s < operand.size(); ++s) {
                    inputs.push_back(ctx.get_slot(*operand[s][member]).first);
                }
                std::vector<TensorId> const outputs{ctx.get_slot(*operand[0][member]).first};
                ctx.record_built(OpKind::Custom, chunked ? fmt::format("{}#{}", op.label, member) : op.label, plan.dtype,
                                 operand[0][member]->rank(), std::move(*sliced), std::span<TensorId const>{inputs},
                                 std::span<TensorId const>{outputs}, inputs, outputs);
            }
            break;
        }
        default:
            break;
        }
    }
}

/// Check, in the PARENT before the loop and on every replay, that @p handle's tensor still has the
/// symmetry @p desc it was declared with. A triangle computes half its tiles from that promise, so
/// a broken one is an error here rather than a wrong result later.
template <typename T>
void check_declared_symmetry(Graph &graph, TensorHandle const &handle, SymmetryDescriptor const &desc) {
    CaptureGuard const guard(graph);
    auto const         label = fmt::format("AxisTiling: check the declared symmetry of '{}'", handle.name);
    auto const         fail  = [name = handle.name]() {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "AxisTiling: '{}' was declared symmetric, and a tiling that computes half of a contraction from that "
                                "declaration relies on it, but its values no longer have the declared symmetry",
                                name);
    };
    if (handle.is_tensor_view) {
        auto *tensor = static_cast<RuntimeTensorView<T> *>(handle.live_ptr());
        custom(label, std::tie(*tensor), std::tuple<>{}, [tensor, desc, fail]() {
            if (!check_symmetry(*tensor, desc)) {
                fail();
            }
        });
    } else {
        auto *tensor = static_cast<RuntimeTensor<T> *>(handle.live_ptr());
        custom(label, std::tie(*tensor), std::tuple<>{}, [tensor, desc, fail]() {
            if (!check_symmetry(*tensor, desc)) {
                fail();
            }
        });
    }
}

} // namespace

bool emit_tiled_loop(Graph &graph, Plan const &plan) {
    auto emit = [&]<typename T>() {
        std::size_t prologue = 0;
        if (plan.accumulator_id != 0) {
            zero_accumulator<T>(graph, graph.tensor(plan.accumulator_id));
            ++prologue;
        }
        // A destination a contraction sums a sliced axis into takes the contraction's prefactor
        // here, once; the loop then only adds. A scale by zero assigns, so a non-finite value the
        // destination held is not carried into the sum.
        for (auto const &[tid, prefactor] : plan.summed_into) {
            if (T const factor = as<T>(prefactor); factor != T{1}) {
                scale_in_parent(graph, graph.tensor(tid), factor);
                ++prologue;
            }
        }
        if (plan.triangle.has_value()) {
            for (auto const &[tid, desc] : plan.triangle->checks) {
                check_declared_symmetry<T>(graph, graph.tensor(tid), desc);
                ++prologue;
            }
        }
        // The zeroing was appended; it belongs where the region was, ahead of the loop. One
        // splice moves it there and erases the region in the same edit, so no writer can end up
        // behind a reader that survived.
        std::vector<bool> remove(graph.nodes().size(), false);
        std::vector<Node> moved;
        for (std::size_t k = remove.size() - prologue; k < remove.size(); ++k) {
            remove[k] = true;
            moved.push_back(graph.nodes()[k]);
        }
        for (std::size_t const position : plan.replaced) {
            remove[position] = true;
        }
        // At the site the analysis chose, which every statement in between allows: the last
        // replaced position when nothing between depends on an earlier statement, the first when
        // no later statement depends on anything between.
        std::size_t const site = plan.site;
        graph.replace_nodes(remove, {{site, std::move(moved)}});

        // Where that landed: the splice shifts an insert down by the erased nodes BELOW it. The
        // loops follow the zeroing, one after another, in the order @ref loop_walks lists them.
        auto const below = static_cast<std::size_t>(std::ranges::count_if(plan.replaced, [site](std::size_t p) { return p < site; }));
        std::size_t const position = (site - below) + prologue;
        if (plan.triangle.has_value()) {
            auto const loops = triangle_loops(plan.extents[0], plan.blocks[0]);
            for (std::size_t l = 0; l < loops.size(); ++l) {
                std::size_t const iterations = loops[l].table.size();
                Graph            &body =
                    graph.add_loop_at(fmt::format("AxisTiling#{}", l), iterations,
                                      PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(iterations) - 1)), position + l);
                emit_body<T>(graph, body, plan, loops[l].walks, fmt::format("{}", l), &loops[l].table, loops[l].mirror);
            }
            return;
        }
        auto const loops = loop_walks(plan);
        for (std::size_t l = 0; l < loops.size(); ++l) {
            std::size_t const iterations = plan.range ? walk_iterations(loops[l]) : plan.iterations;
            Graph            &body =
                graph.add_loop_at(plan.range ? fmt::format("AxisTiling#{}", l) : std::string{"AxisTiling"}, iterations,
                                  PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(iterations) - 1)), position + l);
            emit_body<T>(graph, body, plan, loops[l], plan.range ? fmt::format("{}", l) : std::string{});
        }
    };

    switch (plan.dtype) {
    case packed_gemm::ScalarType::Float32:
        emit.template operator()<float>();
        return true;
    case packed_gemm::ScalarType::Float64:
        emit.template operator()<double>();
        return true;
    case packed_gemm::ScalarType::Complex64:
        emit.template operator()<std::complex<float>>();
        return true;
    case packed_gemm::ScalarType::Complex128:
        emit.template operator()<std::complex<double>>();
        return true;
    default:
        return false;
    }
}

bool preferred(Plan const &lhs, Plan const &rhs) {
    if (lhs.triangle.has_value() != rhs.triangle.has_value()) {
        return lhs.triangle.has_value();
    }
    if (!same_price(lhs.price_us, rhs.price_us)) {
        return lhs.price_us < rhs.price_us;
    }
    return lhs.traffic < rhs.traffic;
}

namespace {

/// Every way to pair up the positions in @p rest so that paired positions have equal extents, each
/// written as swaps into a copy of @p current, appended to @p out with the empty pairing first.
void exchanges(std::vector<std::size_t> const &rest, std::vector<std::size_t> const &dims, std::size_t from, SymmetryOp current,
               std::vector<bool> &taken, std::vector<SymmetryOp> &out) {
    while (from < rest.size() && taken[from]) {
        ++from;
    }
    if (from == rest.size()) {
        out.push_back(current);
        return;
    }
    taken[from] = true;
    exchanges(rest, dims, from + 1, current, taken, out); // This position stays where it is.
    for (std::size_t other = from + 1; other < rest.size(); ++other) {
        if (taken[other] || dims[rest[from]] != dims[rest[other]]) {
            continue;
        }
        taken[other]                    = true;
        auto paired                     = current;
        paired.permutation[rest[from]]  = static_cast<int8_t>(rest[other]);
        paired.permutation[rest[other]] = static_cast<int8_t>(rest[from]);
        exchanges(rest, dims, from + 1, paired, taken, out);
        taken[other] = false;
    }
    taken[from] = false;
}

} // namespace

TriangleVerdict find_triangle(Graph const &graph, Plan const &plan, std::function<bool(TensorId)> const &backed,
                              EscapeAnalysis const &escape) {
    // Axes of different extents cannot be exchanged at all; that is not a decision worth a line.
    if (plan.extents.size() != 2 || plan.extents[0] != plan.extents[1]) {
        return {};
    }
    if (plan.accumulator_id != 0 || !plan.summed_into.empty()) {
        return {.reason = &AxisTiling::TriangleShape};
    }
    std::set<TensorId> owned;
    for (auto const &[tid, labels] : plan.body_owned) {
        owned.insert(tid);
    }
    std::optional<std::size_t> writer;
    for (std::size_t o = 0; o < plan.ops.size(); ++o) {
        if (owned.contains(plan.ops[o].slots[0].tid)) {
            continue;
        }
        if (writer.has_value()) {
            return {.reason = &AxisTiling::TriangleShape}; // Two caller's tensors written.
        }
        writer = o;
    }
    if (!writer.has_value()) {
        return {.reason = &AxisTiling::TriangleShape};
    }
    auto const &op = plan.ops[*writer];
    if (op.kind != OpKind::Einsum) {
        return {.reason = &AxisTiling::NotExchangeable};
    }
    auto const &labels = op.slots[0].labels;
    auto const  row    = std::ranges::find(labels, 0);
    auto const  col    = std::ranges::find(labels, 1);
    if (row == labels.end() || col == labels.end() || std::ranges::count(labels, 0) != 1 || std::ranges::count(labels, 1) != 1) {
        return {.reason = &AxisTiling::TriangleShape};
    }
    // The output is the writer's alone in the region: a reader would see half-mirrored blocks.
    TensorId const out = graph.buffer_of(op.slots[0].tid);
    for (std::size_t o = 0; o < plan.ops.size(); ++o) {
        for (std::size_t slot = (o == *writer ? 1 : 0); slot < plan.ops[o].slots.size(); ++slot) {
            if (graph.buffer_of(plan.ops[o].slots[slot].tid) == out) {
                return {.reason = &AxisTiling::TriangleShape};
            }
        }
    }
    if (backed(op.slots[0].tid)) {
        return {.reason = &AxisTiling::TriangleBacked};
    }

    auto const &dims = graph.tensor(op.slots[0].tid).dims;
    if (dims.size() > static_cast<std::size_t>(kMaxSymmetryRank)) {
        return {.reason = &AxisTiling::NotExchangeable};
    }
    auto const p_row        = static_cast<std::size_t>(row - labels.begin());
    auto const p_col        = static_cast<std::size_t>(col - labels.begin());
    auto       base         = SymmetryOp::identity();
    base.permutation[p_row] = static_cast<int8_t>(p_col);
    base.permutation[p_col] = static_cast<int8_t>(p_row);
    std::vector<std::size_t> rest;
    for (std::size_t p = 0; p < dims.size(); ++p) {
        if (p != p_row && p != p_col) {
            rest.push_back(p);
        }
    }
    std::vector<SymmetryOp> candidates;
    std::vector<bool>       taken(rest.size(), false);
    exchanges(rest, dims, 0, base, taken, candidates);

    for (auto const &sigma : candidates) {
        std::vector<std::pair<TensorId, SymmetryDescriptor>> relied_on;
        auto const                                           proven = contribution_symmetry(escape, op.id, sigma, &relied_on);
        if (!proven.has_value()) {
            continue;
        }
        // Every promise the proof rests on gets a check the emitted loop can run.
        bool const checkable = std::ranges::all_of(relied_on, [&](auto const &entry) {
            auto const &handle = graph.tensor(entry.first);
            return handle.is_runtime && handle.dtype == plan.dtype;
        });
        if (!checkable) {
            return {.reason = &AxisTiling::NotExchangeable};
        }
        Plan::Triangle triangle;
        triangle.writer               = *writer;
        triangle.sigma                = *proven;
        triangle.mirror_labels        = labels;
        triangle.mirror_labels[p_row] = 1;
        triangle.mirror_labels[p_col] = 0;
        triangle.checks               = std::move(relied_on);
        return {.triangle = std::move(triangle)};
    }
    return {.reason = &AxisTiling::NotExchangeable};
}

std::vector<std::pair<std::size_t, TensorId>> symmetric_seeds(Graph const &graph, Scope const &scope, std::set<TensorId> const &declined) {
    std::set<TensorId> sliceably_written;
    for (auto const &node : graph.nodes()) {
        if (tileable_kind(node.kind) || registered_sliceable(node)) {
            sliceably_written.insert(node.outputs.begin(), node.outputs.end());
        }
    }

    std::vector<std::pair<std::size_t, TensorId>> out;
    std::optional<EscapeAnalysis>                 escape;
    for (auto const &node : graph.nodes()) {
        if (node.kind != OpKind::Einsum || node.outputs.size() != 1) {
            continue;
        }
        std::vector<std::pair<std::size_t, TensorId>> operands;
        for (TensorId const tid : node.inputs) {
            auto const *handle = graph.find_tensor(tid);
            if (handle == nullptr || handle->is_tiled || handle->is_distributed || !scope.intermediate(tid) || declined.contains(tid) ||
                !sliceably_written.contains(tid)) {
                continue;
            }
            operands.emplace_back(handle->total_bytes(), tid);
        }
        auto const *output = graph.find_tensor(node.outputs[0]);
        if (operands.empty() || output == nullptr || output->dims.size() > static_cast<std::size_t>(kMaxSymmetryRank)) {
            continue;
        }

        auto const &dims   = output->dims;
        bool        proven = false;
        for (std::size_t p = 0; p < dims.size() && !proven; ++p) {
            for (std::size_t q = p + 1; q < dims.size() && !proven; ++q) {
                if (dims[p] != dims[q] || dims[p] < 3) {
                    continue;
                }
                auto base           = SymmetryOp::identity();
                base.permutation[p] = static_cast<int8_t>(q);
                base.permutation[q] = static_cast<int8_t>(p);
                std::vector<std::size_t> rest;
                for (std::size_t r = 0; r < dims.size(); ++r) {
                    if (r != p && r != q) {
                        rest.push_back(r);
                    }
                }
                std::vector<SymmetryOp> candidates;
                std::vector<bool>       taken(rest.size(), false);
                exchanges(rest, dims, 0, base, taken, candidates);
                if (!escape.has_value()) {
                    escape.emplace(EscapeAnalysis::over(graph));
                }
                proven = std::ranges::any_of(
                    candidates, [&](SymmetryOp const &sigma) { return contribution_symmetry(*escape, node.id, sigma).has_value(); });
            }
        }
        if (!proven) {
            continue;
        }
        for (auto const &entry : operands) {
            if (std::ranges::find(out, entry) == out.end()) {
                out.push_back(entry);
            }
        }
    }
    std::ranges::sort(
        out, [](auto const &lhs, auto const &rhs) { return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second; });
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::axis_tiling)
