//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/DescriptorConformance.hpp>
#include <Einsums/ComputeGraph/DescriptorHooks.hpp>
#include <Einsums/ComputeGraph/DescriptorRegistry.hpp>
#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <omp.h>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

using Bytes = std::vector<std::byte>;

/// What a tensor holds, as bytes keyed by tile: a dense tensor is the one entry under the empty
/// coordinate, and a tiled tensor is its stored tiles that are not all zero, since an absent tile
/// and a zero one are the same value.
using Snapshot = std::map<std::vector<int>, Bytes>;

/// A one-node graph built from a sample, and where its node and tensors are.
struct Built {
    std::unique_ptr<Graph> graph;
    ConformanceOperands    operands;
    std::size_t            node{0};

    [[nodiscard]] Node const &the_node() const { return graph->nodes()[node]; }
};

/// Call @p visit with every stored block of @p id and its coordinate: the whole buffer of a dense
/// tensor, under the empty coordinate, or each tile of a tiled one.
template <typename Visit>
void for_each_block(Graph &graph, TensorId id, Visit &&visit) {
    TensorHandle const &handle = graph.tensor(id);
    detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
        if (handle.is_tiled) {
            auto *tiled = static_cast<TiledRuntimeTensor<T> *>(graph.live_tensor_ptr(id));
            for (auto &[coord, tile] : tiled->tiles()) {
                visit(coord, std::as_writable_bytes(std::span<T>{tile.data(), tile.size()}));
            }
            return;
        }
        OperandAccessor const accessor = resolve_operand(graph, id, "check_descriptor", "operand");
        auto                 *impl     = accessor.impl<T>();
        visit(std::vector<int>{}, std::as_writable_bytes(std::span<T>{impl->data(), impl->size()}));
    });
}

bool all_zero(std::span<std::byte const> bytes) {
    return std::ranges::all_of(bytes, [](std::byte b) { return b == std::byte{0}; });
}

Snapshot bytes_of(Graph &graph, TensorId id) {
    bool const tiled = graph.tensor(id).is_tiled;
    Snapshot   out;
    for_each_block(graph, id, [&](std::vector<int> const &coord, std::span<std::byte> bytes) {
        if (!tiled || !all_zero(bytes)) {
            out.emplace(coord, Bytes(bytes.begin(), bytes.end()));
        }
    });
    return out;
}

std::vector<Snapshot> bytes_of(Graph &graph, std::vector<TensorId> const &ids) {
    std::vector<Snapshot> all;
    all.reserve(ids.size());
    for (TensorId const id : ids) {
        all.push_back(bytes_of(graph, id));
    }
    return all;
}

/// Make @p ids hold @p values again: every stored block takes its snapshot or zero, and a tile the
/// snapshot holds that is not stored is created first.
void write_bytes(Graph &graph, std::vector<TensorId> const &ids, std::vector<Snapshot> const &values) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
        TensorHandle const &handle = graph.tensor(ids[i]);
        if (handle.is_tiled) {
            detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
                auto *tiled = static_cast<TiledRuntimeTensor<T> *>(graph.live_tensor_ptr(ids[i]));
                for (auto const &[coord, bytes] : values[i]) {
                    (void)tiled->tile(coord); // infer-and-create
                }
            });
        }
        for_each_block(graph, ids[i], [&](std::vector<int> const &coord, std::span<std::byte> bytes) {
            if (auto const it = values[i].find(coord); it != values[i].end()) {
                std::copy(it->second.begin(), it->second.end(), bytes.begin());
            } else {
                std::ranges::fill(bytes, std::byte{0});
            }
        });
    }
}

/// The largest elementwise difference between @p id's blocks and @p reference, relative to the
/// larger magnitude of the two (absolute below 1); a block one side lacks counts as zero.
double relative_difference(Graph &graph, TensorId id, Snapshot const &reference) {
    Snapshot const got = bytes_of(graph, id);
    return detail::dispatch_scalar_type(graph.tensor(id).dtype, [&]<typename T>(T /*tag*/) {
        double worst   = 0.0;
        auto   compare = [&](Bytes const &lhs, Bytes const *rhs) {
            std::size_t const count = lhs.size() / sizeof(T);
            for (std::size_t i = 0; i < count; ++i) {
                T a{};
                T b{};
                std::memcpy(&a, lhs.data() + (i * sizeof(T)), sizeof(T));
                if (rhs != nullptr) {
                    std::memcpy(&b, rhs->data() + (i * sizeof(T)), sizeof(T));
                }
                double const scale = std::max({1.0, static_cast<double>(std::abs(a)), static_cast<double>(std::abs(b))});
                worst              = std::max(worst, static_cast<double>(std::abs(a - b)) / scale);
            }
        };
        for (auto const &[coord, bytes] : got) {
            auto const it = reference.find(coord);
            compare(bytes, it != reference.end() ? &it->second : nullptr);
        }
        for (auto const &[coord, bytes] : reference) {
            if (!got.contains(coord)) {
                compare(bytes, nullptr);
            }
        }
        return worst;
    });
}

/// All bits set: a NaN in every floating-point type, so a read of it shows in the result.
void poison(Graph &graph, std::vector<TensorId> const &ids) {
    for (TensorId const id : ids) {
        for_each_block(graph, id,
                       [](std::vector<int> const & /*coord*/, std::span<std::byte> bytes) { std::ranges::fill(bytes, std::byte{0xff}); });
    }
}

/// Capture @p sample's node carrying @p descriptor into a fresh graph holding @p params.
Built build(ConformanceSample const &sample, OpData descriptor, std::map<std::string, std::int64_t> const &params) {
    Built built;
    built.graph = std::make_unique<Graph>(fmt::format("check_descriptor:{}", sample.name));
    for (auto const &[name, value] : params) {
        built.graph->params_ptr()->set(name, value);
    }
    {
        CaptureGuard const guard(*built.graph);
        built.operands = sample.operands(*built.graph);
        CaptureContext::current().record_built(
            OpKind::Custom, sample.name, sample.dtype, sample.rank, std::move(descriptor), std::span<TensorId const>{built.operands.inputs},
            std::span<TensorId const>{built.operands.outputs}, built.operands.inputs, built.operands.outputs);
    }
    // Lifecycle nodes for graph-owned tensors come first; the sample's node is the only Custom one.
    auto const &nodes = built.graph->nodes();
    auto const  it    = std::ranges::find_if(nodes, [](Node const &node) { return node.kind == OpKind::Custom; });
    built.node        = static_cast<std::size_t>(it - nodes.begin());
    return built;
}

/// Run @p built with its outputs set to @p prior first, and return what they hold afterwards.
std::vector<Snapshot> run(Built &built, std::vector<Snapshot> const &prior) {
    write_bytes(*built.graph, built.operands.outputs, prior);
    built.graph->execute();
    return bytes_of(*built.graph, built.operands.outputs);
}

/// The checks for one sample, each appending what it finds to @p problems.
class SampleCheck {
  public:
    SampleCheck(DescriptorCodec const &codec, ConformanceSample const &sample, std::vector<std::string> &problems)
        : _codec(codec), _sample(sample), _problems(problems) {}

    /// What the node computed on the sample's own descriptor, when it could be run.
    [[nodiscard]] std::optional<std::vector<Snapshot>> const &result() const { return _result; }

    void run_all() {
        if (!attempt("capture and run it", [this] { reference(); })) {
            return;
        }
        _result = _reference;
        attempt("check its output extents", [this] { extents(); });
        attempt("check its destination", [this] { destination(); });
        attempt("check its effects", [this] { effects(); });
        attempt("check its accesses", [this] { accesses(); });
        attempt("check its saved form", [this] { saved_form(); });
        attempt("check its threading", [this] { threading(); });
        attempt("check its tiles", [this] { tiles(); });
        attempt("check its slices", [this] { slices(); });
    }

  private:
    DescriptorCodec const    &_codec;
    ConformanceSample const  &_sample;
    std::vector<std::string> &_problems;

    Built                 _built;
    std::vector<Snapshot> _inputs;
    std::vector<Snapshot> _prior;
    std::vector<Snapshot> _reference;

    std::optional<std::vector<Snapshot>> _result;

    void report(std::string const &what) { _problems.push_back(fmt::format("sample '{}': {}", _sample.name, what)); }

    template <typename F>
    bool attempt(std::string_view doing, F &&step) {
        try {
            step();
            return true;
        } catch (std::exception const &e) {
            report(fmt::format("threw while trying to {}: {}", doing, e.what()));
            return false;
        }
    }

    /// A fresh graph from the sample, which must reproduce the reference's inputs.
    Built rebuild(OpData descriptor, std::map<std::string, std::int64_t> const &params) {
        Built built = build(_sample, std::move(descriptor), params);
        if (bytes_of(*built.graph, built.operands.inputs) != _inputs) {
            report("its operands function filled different inputs on a second call, so nothing below can be compared");
        }
        return built;
    }

    void reference() {
        _built     = build(_sample, _sample.descriptor, _sample.params);
        _inputs    = bytes_of(*_built.graph, _built.operands.inputs);
        _prior     = bytes_of(*_built.graph, _built.operands.outputs);
        _reference = run(_built, _prior);
    }

    void extents() {
        if (auto const mismatch = hooked_extent_mismatch(*_built.graph, _built.the_node())) {
            report(*mismatch);
        }
    }

    void destination() {
        Node const &node = _built.the_node();
        if (auto const violation = destination_rule_violation(*_built.graph, node)) {
            report(*violation);
        }
        if (!passes::pure_overwrite(node)) {
            return;
        }
        Built fresh = rebuild(_sample.descriptor, _sample.params);
        poison(*fresh.graph, fresh.operands.outputs);
        fresh.graph->execute();
        if (bytes_of(*fresh.graph, fresh.operands.outputs) != _reference) {
            report("is said to overwrite its destination without reading it, but a poisoned destination changes its result");
        }
    }

    void effects() {
        if (!effects_of(_built.the_node()).deterministic) {
            return;
        }
        if (run(_built, _prior) != _reference) {
            report("is said to be deterministic, but a second run on the same inputs computes different bytes");
        }
    }

    void accesses() {
        std::map<std::string, std::int64_t> declared;
        for (auto const &name : param_reads(_built.the_node())) {
            auto const it = _sample.params.find(name);
            if (it == _sample.params.end()) {
                report(fmt::format("declares a read of parameter '{}', which the sample does not set", name));
                continue;
            }
            declared.emplace(it->first, it->second);
        }

        Built fresh = rebuild(_sample.descriptor, declared);
        try {
            fresh.graph->execute();
        } catch (std::exception const &e) {
            report(fmt::format("fails with only the parameters it declares ({}), so it reads one it does not declare: {}",
                               declared.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(std::views::keys(declared), ", ")),
                               e.what()));
            return;
        }
        if (bytes_of(*fresh.graph, fresh.operands.outputs) != _reference) {
            report("computes something else with only the parameters it declares, so it reads one it does not declare");
        }
    }

    void threading() {
        if (!kernel_moldability(_built.the_node())) {
            return;
        }
        unsigned const width = _sample.width != 0 ? _sample.width : static_cast<unsigned>(std::max(1, omp_get_max_threads()));
        if (width < 2) {
            return;
        }
        Built fresh = rebuild(_sample.descriptor, _sample.params);
        write_bytes(*fresh.graph, fresh.operands.outputs, _prior);
        fresh.graph->nodes()[fresh.node].thread_width = static_cast<std::uint16_t>(width);
        DataflowExecutor executor;
        fresh.graph->execute(executor);

        double worst = 0.0;
        for (std::size_t i = 0; i < fresh.operands.outputs.size(); ++i) {
            worst = std::max(worst, relative_difference(*fresh.graph, fresh.operands.outputs[i], _reference[i]));
        }
        bool const differs =
            _sample.width_tolerance == 0.0 ? bytes_of(*fresh.graph, fresh.operands.outputs) != _reference : worst > _sample.width_tolerance;
        if (differs) {
            report(fmt::format("is moldable, but at width {} it computes something else than at width 1 (largest relative "
                               "difference {:.3g}, allowed {:.3g})",
                               width, worst, _sample.width_tolerance));
        }
    }

    void tiles() {
        Node const            &node  = _built.the_node();
        DescriptorHooks const *hooks = descriptor_hooks(node);
        if (hooks == nullptr || !hooks->tiles || !hooks->tile || node.outputs.size() != 1) {
            return;
        }
        TensorId const      out    = node.outputs[0];
        TensorHandle const &handle = _built.graph->tensor(out);
        if (!handle.is_tiled) {
            return;
        }
        std::vector<std::vector<int>> sizes;
        detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
            sizes = static_cast<TiledRuntimeTensor<T> *>(_built.graph->live_tensor_ptr(out))->tile_sizes();
        });
        auto const written = hooks->tiles(node.op_data, node, TileQuery{.tile_sizes = &sizes, .params = _built.graph->params_ptr().get()});
        if (!written) {
            report("its tiles hook cannot say which tiles it writes");
            return;
        }
        std::set<std::vector<int>> const named(written->begin(), written->end());
        for (auto const &[coord, bytes] : _reference[0]) {
            if (!named.contains(coord)) {
                report(fmt::format("writes tile ({}) nonzero, but its tiles hook does not name it", fmt::join(coord, ",")));
            }
        }

        for (auto const &coord : *written) {
            auto descriptor = hooks->tile(_sample.descriptor, coord);
            if (!descriptor) {
                report(fmt::format("its tile hook cannot describe tile ({}), which its tiles hook names", fmt::join(coord, ",")));
                continue;
            }
            std::vector<std::size_t> dims;
            for (std::size_t ax = 0; ax < coord.size(); ++ax) {
                dims.push_back(static_cast<std::size_t>(sizes[ax][static_cast<std::size_t>(coord[ax])]));
            }

            Graph graph(fmt::format("check_descriptor:{}:tile", _sample.name));
            for (auto const &[name, value] : _sample.params) {
                graph.params_ptr()->set(name, value);
            }
            TensorId tile_id{};
            {
                CaptureGuard const        guard(graph);
                ConformanceOperands const operands = _sample.operands(graph);
                detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
                    auto &tile = graph.create_runtime_tensor<T>("tile", dims, false);
                    tile_id    = CaptureContext::current().get_slot(tile).first;
                });
                std::vector<TensorId> const outputs{tile_id};
                CaptureContext::current().record_built(OpKind::Custom, fmt::format("{}:tile", _sample.name), _sample.dtype, coord.size(),
                                                       std::move(*descriptor), std::span<TensorId const>{operands.inputs},
                                                       std::span<TensorId const>{outputs}, operands.inputs, outputs);
            }
            graph.execute();
            Snapshot const got   = bytes_of(graph, tile_id);
            Bytes const   &bytes = got.begin()->second;
            auto const     whole = _reference[0].find(coord);
            bool const     same  = whole != _reference[0].end() ? bytes == whole->second : all_zero(bytes);
            if (!same) {
                report(fmt::format("its tile hook's node for tile ({}) computes something else than the whole node's tile",
                                   fmt::join(coord, ",")));
            }
        }
    }

    /// Every slice of every axis the node generates, run through its slice hook, against the same
    /// slice of the whole node's output.
    void slices() {
        Node const            &node  = _built.the_node();
        DescriptorHooks const *hooks = descriptor_hooks(node);
        if (hooks == nullptr || !hooks->axes || !hooks->slice || node.outputs.size() != 1) {
            return;
        }
        TensorId const      out    = node.outputs[0];
        TensorHandle const &handle = _built.graph->tensor(out);
        if (handle.is_tiled) {
            return;
        }
        auto const letters = hooks->axes(node.op_data, node);
        if (!letters || letters->empty() || letters->front().size() != handle.dims.size()) {
            report("its axes hook does not give one letter per axis of its output");
            return;
        }
        auto const generated = [&](std::size_t axis) {
            auto const &letter = letters->front()[axis];
            for (std::size_t s = 1; s < letters->size(); ++s) {
                if (std::ranges::find((*letters)[s], letter) != (*letters)[s].end()) {
                    return false;
                }
            }
            return true;
        };
        Bytes const      &whole = _reference[0].begin()->second;
        std::size_t const width = handle.element_size;

        for (std::size_t axis = 0; axis < handle.dims.size(); ++axis) {
            if (!generated(axis)) {
                continue;
            }
            for (std::size_t index = 0; index < handle.dims[axis]; ++index) {
                SliceRequest request;
                request.dropped.assign(letters->size(), {});
                for (std::size_t s = 0; s < letters->size(); ++s) {
                    request.dropped[s].assign((*letters)[s].size(), std::nullopt);
                }
                request.dropped[0][axis] = std::string{"check_descriptor:slice"};
                auto descriptor          = hooks->slice(_sample.descriptor, request);
                if (!descriptor) {
                    report(fmt::format("its slice hook cannot slice output axis {}, which no input names", axis));
                    break;
                }

                std::vector<std::size_t> dims;
                for (std::size_t a = 0; a < handle.dims.size(); ++a) {
                    if (a != axis) {
                        dims.push_back(handle.dims[a]);
                    }
                }
                Graph graph(fmt::format("check_descriptor:{}:slice", _sample.name));
                for (auto const &[name, value] : _sample.params) {
                    graph.params_ptr()->set(name, value);
                }
                graph.params_ptr()->set("check_descriptor:slice", static_cast<std::int64_t>(index));
                TensorId slice_id{};
                {
                    CaptureGuard const        guard(graph);
                    ConformanceOperands const operands = _sample.operands(graph);
                    detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
                        auto &slice = graph.create_runtime_tensor<T>("slice", dims, false);
                        slice_id    = CaptureContext::current().get_slot(slice).first;
                    });
                    std::vector<TensorId> const outputs{slice_id};
                    CaptureContext::current().record_built(OpKind::Custom, fmt::format("{}:slice", _sample.name), _sample.dtype,
                                                           dims.size(), std::move(*descriptor), std::span<TensorId const>{operands.inputs},
                                                           std::span<TensorId const>{outputs}, operands.inputs, outputs);
                }
                graph.execute();
                Bytes const got = bytes_of(graph, slice_id).begin()->second;

                // The same slice of the whole output, walked in the slice's own (column-major) order.
                Bytes                    want;
                std::vector<std::size_t> coord(handle.dims.size(), 0);
                coord[axis]             = index;
                std::size_t const count = got.size() / width;
                for (std::size_t n = 0; n < count; ++n) {
                    std::size_t rest = n;
                    for (std::size_t a = 0; a < handle.dims.size(); ++a) {
                        if (a == axis) {
                            continue;
                        }
                        coord[a] = rest % handle.dims[a];
                        rest /= handle.dims[a];
                    }
                    std::size_t offset = 0;
                    for (std::size_t a = 0; a < handle.dims.size(); ++a) {
                        offset += coord[a] * handle.strides[a];
                    }
                    want.insert(want.end(), whole.begin() + static_cast<std::ptrdiff_t>(offset * width),
                                whole.begin() + static_cast<std::ptrdiff_t>((offset + 1) * width));
                }
                if (got != want) {
                    report(fmt::format("its slice hook's node for index {} of output axis {} computes something else than that slice "
                                       "of the whole node",
                                       index, axis));
                    break;
                }
            }
        }
    }

    void saved_form() {
        std::string const saved  = json::emit(_codec.write(_sample.descriptor));
        auto              parsed = json::parse(saved);
        if (!parsed) {
            report(fmt::format("write produced JSON that does not parse back: {}", saved));
            return;
        }
        if (!parsed->is_object()) {
            report(fmt::format("write produced {}, not a JSON object", saved));
            return;
        }
        json::Object const &fields   = parsed->as_object();
        OpData              restored = _codec.read(fields);
        if (auto const unread = fields.unconsumed_keys(); !unread.empty()) {
            report(fmt::format("read leaves key(s) {} unread, which the graph loader refuses", fmt::join(unread, ", ")));
        }
        if (restored.name() != _sample.descriptor.name()) {
            report(fmt::format("read returned a '{}', not a '{}'", restored.name(), _sample.descriptor.name()));
            return;
        }
        if (std::string const again = json::emit(_codec.write(restored)); again != saved) {
            report(fmt::format("does not survive write then read: {} came back as {}", saved, again));
        }

        Built fresh = rebuild(std::move(restored), _sample.params);
        if (run(fresh, _prior) != _reference) {
            report("rebuilt from its saved form, computes something else");
        }
    }
};

/// Hold the codec's @c equal to the samples: it must call each descriptor equal to itself, answer
/// the same both ways round, and call two descriptors equal only when the second computes what the
/// first does on the first's inputs.
void check_equal(DescriptorCodec const &codec, std::span<ConformanceSample const> samples,
                 std::vector<std::optional<std::vector<Snapshot>>> const &results, std::vector<std::string> &problems) {
    auto const &equal = codec.hooks.equal;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        ConformanceSample const &first = samples[i];
        try {
            if (!equal(first.descriptor, first.descriptor)) {
                problems.push_back(fmt::format("sample '{}': equal says its descriptor differs from itself", first.name));
            }
            for (std::size_t j = i + 1; j < samples.size(); ++j) {
                ConformanceSample const &second = samples[j];
                bool const               forth  = equal(first.descriptor, second.descriptor);
                if (forth != equal(second.descriptor, first.descriptor)) {
                    problems.push_back(
                        fmt::format("samples '{}' and '{}': equal gives a different answer each way round", first.name, second.name));
                    continue;
                }
                if (!forth || !results[i]) {
                    continue;
                }
                Built swapped = build(first, second.descriptor, first.params);
                swapped.graph->execute();
                if (bytes_of(*swapped.graph, swapped.operands.outputs) != *results[i]) {
                    problems.push_back(fmt::format("samples '{}' and '{}': equal calls their descriptors the same computation, but on "
                                                   "'{}''s inputs the second computes something else",
                                                   first.name, second.name, first.name));
                }
            }
        } catch (std::exception const &e) {
            problems.push_back(fmt::format("sample '{}': threw while checking equal: {}", first.name, e.what()));
        }
    }
}

} // namespace

std::vector<std::string> check_descriptor(std::string_view name, std::span<ConformanceSample const> samples) {
    std::vector<std::string> problems;
    DescriptorCodec const   *codec = find_descriptor_codec(name);
    if (codec == nullptr) {
        problems.push_back(fmt::format("no codec is registered as '{}'", name));
        return problems;
    }
    std::vector<std::optional<std::vector<Snapshot>>> results(samples.size());
    bool                                              usable = true;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        ConformanceSample const &sample = samples[i];
        if (sample.descriptor.name() != name) {
            problems.push_back(fmt::format("sample '{}': carries a '{}', not a '{}'", sample.name, sample.descriptor.name(), name));
            usable = false;
            continue;
        }
        if (!sample.operands) {
            problems.push_back(fmt::format("sample '{}': has no operands function", sample.name));
            usable = false;
            continue;
        }
        SampleCheck check(*codec, sample, problems);
        check.run_all();
        results[i] = check.result();
    }
    if (usable && codec->hooks.equal) {
        check_equal(*codec, samples, results, problems);
    }
    return problems;
}

EINSUMS_NAMESPACE_END(compute_graph)
