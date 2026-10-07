//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DescriptorHooks.cpp
/// @brief A descriptor registered outside the library answers the passes' questions through its
///        hooks, and the passes act on the answers.
///
/// Each descriptor here stands for what an outside library would register: a producer that reads
/// a loop parameter, a node that logs, one that accumulates into its destination, one that fills
/// its destination on its own thread. Each case runs the real passes over a captured graph, so a
/// hook that a pass stopped consulting fails here rather than in that library's tests.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/DescriptorConformance.hpp>
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <omp.h>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The log every LogDescriptor node appends its tag to when it runs.
std::vector<int> &log_sink() {
    static std::vector<int> sink;
    return sink;
}

/// ``row(j) := A(r, j)``, with ``r`` read from the graph's ParamTable at run time.
struct ReadRowDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.ReadRow";
};

/// Appends its tag to @ref log_sink, and fills its one output with the tag when it has one.
struct LogDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Log";
    std::int64_t                      tag{0};
};

/// ``y += x``, element by element.
struct AddIntoDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.AddInto";
};

/// ``y := 1`` on the calling thread.
struct SerialFillDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.SerialFill";
};

/// ``y := x x``, x repeated twice, so y is twice as long as x.
struct TwiceDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Twice";
};

/// ``y := x``, whose hook claims y has five elements whatever x has.
struct MisreportDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Misreport";
};

/// ``y := x + n``, n counting its runs, while its codec claims a pure node.
struct FlakyDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Flaky";
};

/// ``y += x``, while its hook claims y is overwritten without being read.
struct FakeOverwriteDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.FakeOverwrite";
};

/// ``y := x + k``, k a parameter it does not declare.
struct HiddenParamDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.HiddenParam";
};

/// ``y := x + k``, k a parameter it declares.
struct DeclaredParamDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.DeclaredParam";
};

/// ``y := scale * x``, whose codec forgets to save ``scale``.
struct ForgetfulDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Forgetful";
    double                            scale{1.0};
};

/// A node whose cost hook answers from its fields; the executor does nothing.
struct CostedDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Costed";
    double                            serial_us{-1.0}; ///< Negative: not given
    double                            flops{0.0};
    std::size_t                       bytes{0};
    bool                              linear_speedup{false}; ///< Give a speedup curve of exactly the width
};

/// How many times a CostedDescriptor's speedup curve has been asked.
int &speedup_calls() {
    static int calls = 0;
    return calls;
}

/// ``y := x + t``, t the OpenMP thread count it runs with, while it claims to be moldable.
struct WidthDependentDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.WidthDependent";
};

/// ``y := factor * x``, overwriting y, with equal comparing the factors.
struct ScaledDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Scaled";
    double                            factor{1.0};
};

/// As ScaledDescriptor, but its effects are opaque.
struct ScaledOpaqueDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.ScaledOpaque";
    double                            factor{1.0};
};

/// As ScaledDescriptor, but it says nothing about its destination.
struct ScaledNoDestinationDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.ScaledNoDestination";
    double                            factor{1.0};
};

/// As ScaledDescriptor, but its equal calls every pair the same.
struct ScaledLyingEqualDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.ScaledLyingEqual";
    double                            factor{1.0};
};

/// Writes the diagonal tiles of a 2-D tiled output, each filled with ``v(0) * (1 + 10 I + J)``,
/// and zeroes any other tile it finds stored. @c lie makes its hooks misreport: 1 names only the
/// first diagonal tile, 2 describes tiles that compute one more than the whole node does.
struct BlockFillDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.BlockFill";
    int                               lie{0};
};

/// As BlockFillDescriptor, but it says nothing about its destination, so it is not expanded.
struct BlockFillUnsaidDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.BlockFillUnsaid";
    int                               lie{0};
};

/// Writes tile (I, J) of a BlockFill node alone, as a dense tensor.
struct BlockFillTileDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.BlockFillTile";
    int                               row{0};
    int                               col{0};
    double                            offset{0.0};
};

/// The value BlockFill writes everywhere in tile (row, col).
double block_value(double v, int row, int col) {
    return v * (1.0 + (10.0 * row) + col);
}

/// The diagonal tiles of a grid of @p sizes.
std::vector<cg::TileCoord> diagonal(std::vector<std::vector<int>> const &sizes) {
    std::vector<cg::TileCoord> out;
    for (int i = 0; i < static_cast<int>(std::min(sizes[0].size(), sizes[1].size())); ++i) {
        out.push_back({i, i});
    }
    return out;
}

/// Writes ``X[i, a] = (i + 1) (a + 2)`` from nothing. A sliced one names, per output axis, the
/// parameter holding that axis's index, and writes the rest; @c lie makes its slices one off.
struct GridDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.Grid";
    std::vector<std::string>          index_params{"", ""};
    int                               lie{0};
};

/// As GridDescriptor, with opaque effects, so AxisTiling leaves it out of a loop.
struct GridOpaqueDescriptor {
    static constexpr std::string_view descriptor_name = "hooks_test.GridOpaque";
    std::vector<std::string>          index_params{"", ""};
    int                               lie{0};
};

/// The value a Grid node writes at (i, a).
double grid_value(std::size_t i, std::size_t a) {
    return static_cast<double>(i + 1) * static_cast<double>(a + 2);
}

/// An executor over one input x and one output y, both rank 1 doubles.
template <typename Body>
std::function<void()> unary(cg::Graph &graph, std::span<cg::TensorId const> inputs, std::span<cg::TensorId const> outputs, Body body) {
    cg::OperandAccessor const x = cg::resolve_operand(graph, inputs[0], "hooks_test", "x");
    cg::OperandAccessor const y = cg::resolve_operand(graph, outputs[0], "hooks_test", "y");
    return [x, y, body]() {
        auto const *src = x.impl<double>();
        auto       *dst = y.impl<double>();
        for (std::size_t i = 0; i < dst->size(); ++i) {
            dst->data()[i] = body(src->data()[i], dst->data()[i]);
        }
    };
}

/// Register a ScaledDescriptor-shaped codec for @p D: saves its factor, scales x into y.
template <typename D>
void register_scaled(cg::DescriptorHooksFor<D> hooks) {
    cg::register_descriptor<D>(
        [](D const &desc) {
            cg::json::Object fields;
            fields.set("factor", cg::json::Value{desc.factor});
            return cg::json::Value{std::move(fields)};
        },
        [](cg::json::Object const &fields) {
            cg::json::Value const *factor = fields.take("factor");
            return D{.factor = factor != nullptr ? factor->as_double() : 1.0};
        },
        [](D const &desc, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
           std::span<cg::TensorId const> outputs) -> std::function<void()> {
            return unary(graph, inputs, outputs, [factor = desc.factor](double x, double) { return factor * x; });
        },
        std::move(hooks));
}

/// The destination of a ScaledDescriptor-shaped node: it writes all of y and reads only x.
cg::DestinationUse overwrites_y(cg::Node const & /*node*/) {
    return cg::DestinationUse{.reads = false, .overwrites_all = true, .operand_count = 1};
}

/// Register a BlockFill-shaped codec for @p D with @p hooks.
template <typename D>
void register_block_fill(cg::DescriptorHooksFor<D> hooks) {
    cg::register_descriptor<D>(
        [](D const &desc) {
            cg::json::Object fields;
            fields.set("lie", cg::json::Value{static_cast<double>(desc.lie)});
            return cg::json::Value{std::move(fields)};
        },
        [](cg::json::Object const &fields) {
            cg::json::Value const *lie = fields.take("lie");
            return D{.lie = lie != nullptr ? static_cast<int>(lie->as_double()) : 0};
        },
        [](D const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
           std::span<cg::TensorId const> outputs) -> std::function<void()> {
            cg::OperandAccessor const v      = cg::resolve_operand(graph, inputs[0], "hooks_test.BlockFill", "v");
            cg::TensorId const        out_id = outputs[0];
            return [v, out_id, &graph]() {
                double const value = v.impl<double>()->data()[0];
                auto        *out   = static_cast<TiledRuntimeTensor<double> *>(graph.live_tensor_ptr(out_id));
                auto const   wrote = diagonal(out->tile_sizes());
                for (auto &[coord, tile] : out->tiles()) {
                    std::fill_n(tile.data(), tile.size(), 0.0);
                }
                for (auto const &coord : wrote) {
                    auto &tile = out->tile(coord);
                    tile.materialize();
                    std::fill_n(tile.data(), tile.size(), block_value(value, coord[0], coord[1]));
                }
            };
        },
        std::move(hooks));
}

/// Register a Grid-shaped codec for @p D, with @p effects.
template <typename D>
void register_grid(cg::NodeEffects effects) {
    cg::register_descriptor<D>(
        [](D const &desc) {
            cg::json::Array params;
            for (auto const &name : desc.index_params) {
                params.emplace_back(cg::json::Value{name});
            }
            cg::json::Object fields;
            fields.set("index_params", cg::json::Value{std::move(params)});
            fields.set("lie", cg::json::Value{static_cast<double>(desc.lie)});
            return cg::json::Value{std::move(fields)};
        },
        [](cg::json::Object const &fields) {
            D desc;
            if (auto const *params = fields.take("index_params"); params != nullptr && params->is_array()) {
                desc.index_params.clear();
                for (auto const &name : params->as_array()) {
                    desc.index_params.push_back(name.as_string());
                }
            }
            if (auto const *lie = fields.take("lie"); lie != nullptr) {
                desc.lie = static_cast<int>(lie->as_double());
            }
            return desc;
        },
        [](D const &desc, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const>,
           std::span<cg::TensorId const> outputs) -> std::function<void()> {
            cg::OperandAccessor const out = cg::resolve_operand(graph, outputs[0], "hooks_test.Grid", "X");
            return [out, desc, params = graph.params_ptr()]() {
                auto                    *x = out.impl<double>();
                std::vector<std::size_t> kept;
                std::size_t              fixed[2]{0, 0};
                for (std::size_t axis = 0; axis < 2; ++axis) {
                    if (desc.index_params[axis].empty()) {
                        kept.push_back(axis);
                    } else {
                        fixed[axis] = static_cast<std::size_t>(params->get(desc.index_params[axis]) + desc.lie);
                    }
                }
                std::size_t const n0 = !kept.empty() ? x->dim(0) : 1;
                std::size_t const n1 = kept.size() > 1 ? x->dim(1) : 1;
                for (std::size_t p0 = 0; p0 < n0; ++p0) {
                    for (std::size_t p1 = 0; p1 < n1; ++p1) {
                        std::size_t coord[2]{fixed[0], fixed[1]};
                        if (!kept.empty()) {
                            coord[kept[0]] = p0;
                        }
                        if (kept.size() > 1) {
                            coord[kept[1]] = p1;
                        }
                        std::size_t const offset = (p0 * (!kept.empty() ? x->stride(0) : 0)) + (p1 * (kept.size() > 1 ? x->stride(1) : 0));
                        x->data()[offset]        = grid_value(coord[0], coord[1]);
                    }
                }
            };
        },
        cg::DescriptorHooksFor<D>{
            .accesses =
                [](D const &desc, cg::Node const &) {
                    cg::NamedAccesses accesses;
                    for (auto const &name : desc.index_params) {
                        if (!name.empty()) {
                            accesses.param_reads.push_back(name);
                        }
                    }
                    return accesses;
                },
            .effects     = [effects](D const &) { return effects; },
            .destination = [](D const &,
                              cg::Node const &) { return cg::DestinationUse{.reads = false, .overwrites_all = true, .operand_count = 0}; },
            .axes        = [](D const &, cg::Node const &) { return std::optional<std::vector<std::vector<std::string>>>{{{"i", "a"}}}; },
            .slice       = [](D const &desc, cg::SliceRequest const &request) -> std::optional<cg::OpData> {
                D sliced = desc;
                for (std::size_t axis = 0; axis < 2; ++axis) {
                    sliced.index_params[axis] = request.dropped[0][axis].value_or("");
                }
                return cg::OpData{sliced};
            }});
}

/// A codec's write and read for a descriptor with no fields.
template <typename D>
cg::json::Value write_nothing(D const & /*descriptor*/) {
    return cg::json::Value{cg::json::Object{}};
}

template <typename D>
D read_nothing(cg::json::Object const & /*fields*/) {
    return D{};
}

/// Register every descriptor here once per process, whatever order the cases run in.
void register_hook_descriptors() {
    static bool const registered = [] {
        cg::register_descriptor<ReadRowDescriptor>(
            write_nothing<ReadRowDescriptor>, read_nothing<ReadRowDescriptor>,
            [](ReadRowDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                cg::OperandAccessor const a = cg::resolve_operand(graph, inputs[0], "hooks_test.ReadRow", "A");
                cg::OperandAccessor const r = cg::resolve_operand(graph, outputs[0], "hooks_test.ReadRow", "row");
                return [a, r, params = graph.params_ptr()]() {
                    auto const *src = a.impl<double>();
                    auto       *dst = r.impl<double>();
                    auto const  row = static_cast<std::size_t>(params->get("r"));
                    for (std::size_t j = 0; j < dst->size(); ++j) {
                        dst->data()[j] = src->data()[(row * src->stride(0)) + (j * src->stride(1))];
                    }
                };
            },
            cg::DescriptorHooksFor<ReadRowDescriptor>{
                .accesses = [](ReadRowDescriptor const &, cg::Node const &) { return cg::NamedAccesses{.param_reads = {"r"}}; }});

        cg::register_descriptor<LogDescriptor>(
            [](LogDescriptor const &desc) {
                cg::json::Object fields;
                fields.set("tag", cg::json::Value{static_cast<double>(desc.tag)});
                return cg::json::Value{std::move(fields)};
            },
            [](cg::json::Object const &fields) {
                cg::json::Value const *tag = fields.take("tag");
                return LogDescriptor{.tag = tag != nullptr ? static_cast<std::int64_t>(tag->as_double()) : 0};
            },
            [](LogDescriptor const &desc, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const>,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                if (outputs.empty()) {
                    return [tag = desc.tag]() { log_sink().push_back(static_cast<int>(tag)); };
                }
                cg::OperandAccessor const out = cg::resolve_operand(graph, outputs[0], "hooks_test.Log", "out");
                return [out, tag = desc.tag]() {
                    auto *dst = out.impl<double>();
                    for (std::size_t i = 0; i < dst->size(); ++i) {
                        dst->data()[i] = static_cast<double>(tag);
                    }
                    log_sink().push_back(static_cast<int>(tag));
                };
            },
            cg::DescriptorHooksFor<LogDescriptor>{
                .effects = [](LogDescriptor const &) { return cg::NodeEffects{.deterministic = true, .external_effects = true}; }});

        cg::register_descriptor<AddIntoDescriptor>(
            write_nothing<AddIntoDescriptor>, read_nothing<AddIntoDescriptor>,
            [](AddIntoDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                cg::OperandAccessor const x = cg::resolve_operand(graph, inputs[0], "hooks_test.AddInto", "x");
                cg::OperandAccessor const y = cg::resolve_operand(graph, outputs[0], "hooks_test.AddInto", "y");
                return [x, y]() {
                    auto const *src = x.impl<double>();
                    auto       *dst = y.impl<double>();
                    for (std::size_t i = 0; i < dst->size(); ++i) {
                        dst->data()[i] += src->data()[i];
                    }
                };
            },
            cg::DescriptorHooksFor<AddIntoDescriptor>{.destination = [](AddIntoDescriptor const &, cg::Node const &) {
                return cg::DestinationUse{.reads = true, .overwrites_all = false, .operand_count = 1};
            }});

        cg::register_descriptor<SerialFillDescriptor>(
            write_nothing<SerialFillDescriptor>, read_nothing<SerialFillDescriptor>,
            [](SerialFillDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const>,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                cg::OperandAccessor const y = cg::resolve_operand(graph, outputs[0], "hooks_test.SerialFill", "y");
                return [y]() {
                    auto *dst = y.impl<double>();
                    std::fill_n(dst->data(), dst->size(), 1.0);
                };
            },
            cg::DescriptorHooksFor<SerialFillDescriptor>{.destination =
                                                             [](SerialFillDescriptor const &, cg::Node const &) {
                                                                 return cg::DestinationUse{
                                                                     .reads = false, .overwrites_all = true, .operand_count = 0};
                                                             },
                                                         .threading = [](SerialFillDescriptor const &) { return cg::Threading::Serial; }});

        cg::register_descriptor<TwiceDescriptor>(
            write_nothing<TwiceDescriptor>, read_nothing<TwiceDescriptor>,
            [](TwiceDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                cg::OperandAccessor const x = cg::resolve_operand(graph, inputs[0], "hooks_test.Twice", "x");
                cg::OperandAccessor const y = cg::resolve_operand(graph, outputs[0], "hooks_test.Twice", "y");
                return [x, y]() {
                    auto const *src = x.impl<double>();
                    auto       *dst = y.impl<double>();
                    for (std::size_t i = 0; i < dst->size(); ++i) {
                        dst->data()[i] = src->data()[i % src->size()];
                    }
                };
            },
            cg::DescriptorHooksFor<TwiceDescriptor>{.output_extents = [](TwiceDescriptor const &, cg::Node const &,
                                                                         cg::ExtentQuery const &query) -> std::optional<cg::ExtentList> {
                if (query.input_extents.empty() || query.input_extents[0].size() != 1) {
                    return std::nullopt;
                }
                return cg::ExtentList{{2 * query.input_extents[0][0]}};
            }});

        cg::register_descriptor<MisreportDescriptor>(
            write_nothing<MisreportDescriptor>, read_nothing<MisreportDescriptor>,
            [](MisreportDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                cg::OperandAccessor const x = cg::resolve_operand(graph, inputs[0], "hooks_test.Misreport", "x");
                cg::OperandAccessor const y = cg::resolve_operand(graph, outputs[0], "hooks_test.Misreport", "y");
                return [x, y]() { std::copy_n(x.impl<double>()->data(), y.impl<double>()->size(), y.impl<double>()->data()); };
            },
            cg::DescriptorHooksFor<MisreportDescriptor>{
                .output_extents = [](MisreportDescriptor const &, cg::Node const &,
                                     cg::ExtentQuery const &) -> std::optional<cg::ExtentList> { return cg::ExtentList{{5}}; }});

        cg::register_descriptor<FlakyDescriptor>(
            write_nothing<FlakyDescriptor>, read_nothing<FlakyDescriptor>,
            [](FlakyDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                auto runs = std::make_shared<int>(0);
                auto step = unary(graph, inputs, outputs, [runs](double x, double) { return x + static_cast<double>(*runs); });
                return [runs, step]() {
                    ++*runs;
                    step();
                };
            });

        cg::register_descriptor<FakeOverwriteDescriptor>(
            write_nothing<FakeOverwriteDescriptor>, read_nothing<FakeOverwriteDescriptor>,
            [](FakeOverwriteDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t,
               std::span<cg::TensorId const> inputs, std::span<cg::TensorId const> outputs) -> std::function<void()> {
                return unary(graph, inputs, outputs, [](double x, double y) { return y + x; });
            },
            cg::DescriptorHooksFor<FakeOverwriteDescriptor>{.destination = [](FakeOverwriteDescriptor const &, cg::Node const &) {
                return cg::DestinationUse{.reads = false, .overwrites_all = true, .operand_count = 1};
            }});

        cg::register_descriptor<HiddenParamDescriptor>(
            write_nothing<HiddenParamDescriptor>, read_nothing<HiddenParamDescriptor>,
            [](HiddenParamDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const> inputs,
               std::span<cg::TensorId const> outputs) -> std::function<void()> {
                return unary(graph, inputs, outputs,
                             [params = graph.params_ptr()](double x, double) { return x + static_cast<double>(params->get("k")); });
            });

        cg::register_descriptor<DeclaredParamDescriptor>(
            write_nothing<DeclaredParamDescriptor>, read_nothing<DeclaredParamDescriptor>,
            [](DeclaredParamDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t,
               std::span<cg::TensorId const> inputs, std::span<cg::TensorId const> outputs) -> std::function<void()> {
                return unary(graph, inputs, outputs,
                             [params = graph.params_ptr()](double x, double) { return x + static_cast<double>(params->get("k")); });
            },
            cg::DescriptorHooksFor<DeclaredParamDescriptor>{
                .accesses = [](DeclaredParamDescriptor const &, cg::Node const &) { return cg::NamedAccesses{.param_reads = {"k"}}; }});

        cg::register_descriptor<CostedDescriptor>(
            [](CostedDescriptor const &) { return cg::json::Value{cg::json::Object{}}; },
            [](cg::json::Object const &) { return CostedDescriptor{}; },
            [](CostedDescriptor const &, cg::Graph &, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const>,
               std::span<cg::TensorId const>) -> std::function<void()> { return [] {}; },
            cg::DescriptorHooksFor<CostedDescriptor>{.cost = [](CostedDescriptor const &desc, cg::CostQuery const &) {
                cg::CostEstimate estimate{.flops = desc.flops, .bytes = desc.bytes};
                if (desc.serial_us >= 0.0) {
                    estimate.serial_us = desc.serial_us;
                }
                if (desc.linear_speedup) {
                    estimate.speedup = [](unsigned width) {
                        ++speedup_calls();
                        return static_cast<double>(width);
                    };
                }
                return std::optional<cg::CostEstimate>{estimate};
            }});

        cg::register_descriptor<WidthDependentDescriptor>(
            write_nothing<WidthDependentDescriptor>, read_nothing<WidthDependentDescriptor>,
            [](WidthDependentDescriptor const &, cg::Graph &graph, packed_gemm::ScalarType, std::size_t,
               std::span<cg::TensorId const> inputs, std::span<cg::TensorId const> outputs) -> std::function<void()> {
                return unary(graph, inputs, outputs, [](double x, double) { return x + static_cast<double>(omp_get_max_threads()); });
            });

        register_scaled<ScaledDescriptor>(
            {.destination = [](ScaledDescriptor const &, cg::Node const &node) { return overwrites_y(node); },
             .equal       = [](ScaledDescriptor const &a, ScaledDescriptor const &b) { return a.factor == b.factor; }});
        register_scaled<ScaledOpaqueDescriptor>(
            {.effects     = [](ScaledOpaqueDescriptor const &) { return cg::opaque_effects; },
             .destination = [](ScaledOpaqueDescriptor const &, cg::Node const &node) { return overwrites_y(node); },
             .equal       = [](ScaledOpaqueDescriptor const &a, ScaledOpaqueDescriptor const &b) { return a.factor == b.factor; }});
        register_scaled<ScaledNoDestinationDescriptor>(
            {.equal = [](ScaledNoDestinationDescriptor const &a, ScaledNoDestinationDescriptor const &b) { return a.factor == b.factor; }});
        register_scaled<ScaledLyingEqualDescriptor>(
            {.destination = [](ScaledLyingEqualDescriptor const &, cg::Node const &node) { return overwrites_y(node); },
             .equal       = [](ScaledLyingEqualDescriptor const &, ScaledLyingEqualDescriptor const &) { return true; }});

        auto block_tiles = [](auto const &desc, cg::Node const &, cg::TileQuery const &query) {
            auto all = diagonal(*query.tile_sizes);
            if (desc.lie == 1) {
                all.resize(1);
            }
            return std::optional<std::vector<cg::TileCoord>>{all};
        };
        auto block_tile = [](auto const &desc, cg::TileCoord const &coord) {
            return std::optional<cg::OpData>{
                cg::OpData{BlockFillTileDescriptor{.row = coord[0], .col = coord[1], .offset = desc.lie == 2 ? 1.0 : 0.0}}};
        };
        register_block_fill<BlockFillDescriptor>(
            {.destination = [](BlockFillDescriptor const &,
                               cg::Node const &) { return cg::DestinationUse{.reads = false, .overwrites_all = true, .operand_count = 1}; },
             .tiles       = block_tiles,
             .tile        = block_tile});
        register_block_fill<BlockFillUnsaidDescriptor>({.tiles = block_tiles, .tile = block_tile});

        cg::register_descriptor<BlockFillTileDescriptor>(
            [](BlockFillTileDescriptor const &desc) {
                cg::json::Object fields;
                fields.set("row", cg::json::Value{static_cast<double>(desc.row)});
                fields.set("col", cg::json::Value{static_cast<double>(desc.col)});
                fields.set("offset", cg::json::Value{desc.offset});
                return cg::json::Value{std::move(fields)};
            },
            [](cg::json::Object const &fields) {
                auto number = [&fields](char const *key) {
                    cg::json::Value const *value = fields.take(key);
                    return value != nullptr ? value->as_double() : 0.0;
                };
                return BlockFillTileDescriptor{
                    .row = static_cast<int>(number("row")), .col = static_cast<int>(number("col")), .offset = number("offset")};
            },
            [](BlockFillTileDescriptor const &desc, cg::Graph &graph, packed_gemm::ScalarType, std::size_t,
               std::span<cg::TensorId const> inputs, std::span<cg::TensorId const> outputs) -> std::function<void()> {
                cg::OperandAccessor const v   = cg::resolve_operand(graph, inputs[0], "hooks_test.BlockFillTile", "v");
                cg::OperandAccessor const out = cg::resolve_operand(graph, outputs[0], "hooks_test.BlockFillTile", "tile");
                return [v, out, desc]() {
                    double const value = block_value(v.impl<double>()->data()[0], desc.row, desc.col) + desc.offset;
                    auto        *tile  = out.impl<double>();
                    std::fill_n(tile->data(), tile->size(), value);
                };
            },
            cg::DescriptorHooksFor<BlockFillTileDescriptor>{.destination = [](BlockFillTileDescriptor const &, cg::Node const &) {
                return cg::DestinationUse{.reads = false, .overwrites_all = true, .operand_count = 1};
            }});

        register_grid<GridDescriptor>(cg::NodeEffects{});
        register_grid<GridOpaqueDescriptor>(cg::opaque_effects);

        cg::register_descriptor<ForgetfulDescriptor>(
            [](ForgetfulDescriptor const &) { return cg::json::Value{cg::json::Object{}}; },
            [](cg::json::Object const &) { return ForgetfulDescriptor{}; },
            [](ForgetfulDescriptor const &desc, cg::Graph &graph, packed_gemm::ScalarType, std::size_t,
               std::span<cg::TensorId const> inputs, std::span<cg::TensorId const> outputs) -> std::function<void()> {
                return unary(graph, inputs, outputs, [scale = desc.scale](double x, double) { return scale * x; });
            });
        return true;
    }();
    (void)registered;
}

/// A conformance sample over x = (1, 2, 3) and y = (10, 20, 30), both owned by the graph.
template <typename D>
cg::ConformanceSample sample_of(std::string name, D descriptor, std::map<std::string, std::int64_t> params = {}) {
    return cg::ConformanceSample{.name       = std::move(name),
                                 .descriptor = cg::OpData{std::move(descriptor)},
                                 .dtype      = packed_gemm::ScalarType::Float64,
                                 .rank       = 1,
                                 .operands =
                                     [](cg::Graph &graph) {
                                         auto &x = graph.create_runtime_tensor<double>("x", {3UL}, false);
                                         auto &y = graph.create_runtime_tensor<double>("y", {3UL}, false);
                                         for (std::size_t i = 0; i < 3; ++i) {
                                             x(i) = static_cast<double>(i + 1);
                                             y(i) = 10.0 * static_cast<double>(i + 1);
                                         }
                                         auto &ctx = cg::CaptureContext::current();
                                         return cg::ConformanceOperands{.inputs  = {ctx.get_slot(x).first},
                                                                        .outputs = {ctx.get_slot(y).first}};
                                     },
                                 .params = std::move(params)};
}

/// The problems check_descriptor finds in one sample, joined for a failure message.
std::vector<std::string> check_one(cg::ConformanceSample const &sample) {
    return cg::check_descriptor(sample.descriptor.name(), std::span<cg::ConformanceSample const>{&sample, 1});
}

/// Record a node of descriptor @p desc that reads @p inputs and writes @p outputs.
template <typename D>
void record(std::string label, D desc, std::vector<cg::TensorId> inputs, std::vector<cg::TensorId> outputs) {
    auto &ctx = cg::CaptureContext::current();
    ctx.record_built(cg::OpKind::Custom, std::move(label), packed_gemm::ScalarType::Float64, 1, std::move(desc),
                     std::span<cg::TensorId const>{inputs}, std::span<cg::TensorId const>{outputs}, inputs, outputs);
}

/// The position of the node labelled @p label in @p graph.
std::size_t position_of(cg::Graph const &graph, std::string_view label) {
    auto const &nodes = graph.nodes();
    auto const  it    = std::ranges::find_if(nodes, [label](cg::Node const &node) { return node.label == label; });
    REQUIRE(it != nodes.end());
    return static_cast<std::size_t>(it - nodes.begin());
}

/// Whether the hazard scan gives node @p to an edge from node @p from.
bool has_edge(cg::Graph &graph, std::size_t from, std::size_t to) {
    graph.topological_sort();
    auto const &preds = graph.dependencies().predecessors[to];
    return std::ranges::find(preds, from) != preds.end();
}

} // namespace

// accesses: without the parameter read the hook declares, the producer is invariant by its
// tensors, so the loop would hoist it ahead of the WriteParam that positions it, and Reorder would
// be free to do the same inside the body.
TEST_CASE("Descriptor hooks - a declared parameter read orders the node after the write", "[ComputeGraph][DescriptorHooks]") {
    register_hook_descriptors();
    RuntimeTensor<double> A{"A", {3UL, 3UL}};
    RuntimeTensor<double> row{"row", {3UL}};
    RuntimeTensor<double> total{"total", {3UL}};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(1 + (3 * i) + j); // 1..9
        }
    }
    row.zero();
    std::size_t iter = 0;

    cg::Graph graph("hooks_read_row");
    auto     &body = graph.add_loop("rows", 3, [&iter](std::size_t) {
        ++iter;
        return iter < 3;
    });
    {
        cg::CaptureGuard const capture(body);
        cg::write_param("r", std::function<std::int64_t()>([&iter] { return static_cast<std::int64_t>(iter); }));
        auto &ctx = cg::CaptureContext::current();
        record("read_row", ReadRowDescriptor{}, {ctx.get_slot(A).first}, {ctx.get_slot(row).first});
        cg::axpby(1.0, row, 1.0, &total);
    }

    auto const &reader = body.nodes()[1];
    CHECK(cg::param_reads(reader) == std::vector<std::string>{"r"});
    CHECK(has_edge(body, 0, 1));

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    INFO(pm.explain());

    iter = 0;
    total.zero();
    REQUIRE_NOTHROW(graph.execute());
    CHECK(total(0) + total(1) + total(2) == 45.0); // 1 + 2 + ... + 9
}

// effects: a registered descriptor is pure unless its hook says otherwise; this one logs, so it
// stays in its loop, survives with an unread output, and keeps its order against another.
TEST_CASE("Descriptor hooks - a declared effect keeps the node in its loop and alive", "[ComputeGraph][DescriptorHooks]") {
    register_hook_descriptors();
    log_sink().clear();

    cg::Graph graph("hooks_log");
    auto     &body = graph.add_loop("loop", 3, [](std::size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const capture(body);
        auto                  &scratch = body.create_runtime_tensor<double>("scratch", {2UL});
        auto                  &ctx     = cg::CaptureContext::current();
        record("log_with_output", LogDescriptor{.tag = 1}, {}, {ctx.get_slot(scratch).first});
        record("log", LogDescriptor{.tag = 2}, {}, {});
    }

    // Found by label: the scratch tensor's Alloc comes first.
    std::size_t const with_output = position_of(body, "log_with_output");
    std::size_t const without     = position_of(body, "log");
    CHECK(cg::effects_of(body.nodes()[with_output]).external_effects);
    CHECK(has_edge(body, with_output, without));

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    INFO(pm.explain());

    graph.execute();
    CHECK(log_sink() == std::vector<int>{1, 2, 1, 2, 1, 2});
}

// destination: the node lists only x, and says it reads y, so capture appends y as the trailing
// destination, the rule holds, and the loop keeps the accumulation in place.
TEST_CASE("Descriptor hooks - a declared destination read follows the destination rule", "[ComputeGraph][DescriptorHooks]") {
    register_hook_descriptors();
    RuntimeTensor<double> x{"x", {3UL}};
    RuntimeTensor<double> y{"y", {3UL}};
    for (std::size_t i = 0; i < 3; ++i) {
        x(i) = static_cast<double>(i + 1);
    }
    y.zero();

    cg::Graph    graph("hooks_add_into");
    auto        &body = graph.add_loop("loop", 3, [](std::size_t iter) { return iter < 2; });
    cg::TensorId x_id{};
    cg::TensorId y_id{};
    {
        cg::CaptureGuard const capture(body);
        auto                  &ctx = cg::CaptureContext::current();
        x_id                       = ctx.get_slot(x).first;
        y_id                       = ctx.get_slot(y).first;
        record("add_into", AddIntoDescriptor{}, {x_id}, {y_id});
    }

    auto const &node = body.nodes()[0];
    CHECK(cg::reads_destination(node));
    CHECK(cg::destination_operand_count(node) == std::optional<std::size_t>{1});
    CHECK(node.inputs == std::vector<cg::TensorId>{x_id, y_id});
    CHECK_FALSE(cg::destination_rule_violation(body, node).has_value());
    CHECK_FALSE(cg::passes::pure_overwrite(node));

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    INFO(pm.explain());

    graph.execute();
    CHECK(y(0) + y(1) + y(2) == 18.0); // 3 iterations of 1 + 2 + 3
}

// threading and a full overwrite: the planner may not widen a node that runs its own thread
// discipline, and a node that writes all of its destination makes an earlier write dead.
TEST_CASE("Descriptor hooks - declared threading and overwrite reach the passes' predicates", "[ComputeGraph][DescriptorHooks]") {
    register_hook_descriptors();
    RuntimeTensor<double> x{"x", {3UL}};
    RuntimeTensor<double> y{"y", {3UL}};
    x.zero();
    y.zero();

    cg::Graph graph("hooks_serial_fill");
    {
        cg::CaptureGuard const capture(graph);
        auto                  &ctx = cg::CaptureContext::current();
        record("fill", SerialFillDescriptor{}, {}, {ctx.get_slot(y).first});
        record("add_into", AddIntoDescriptor{}, {ctx.get_slot(x).first}, {ctx.get_slot(y).first});
    }

    auto const &fill = graph.nodes()[0];
    CHECK_FALSE(cg::kernel_moldability(fill));
    CHECK(cg::passes::pure_overwrite(fill));
    CHECK_FALSE(cg::reads_destination(fill));

    // Without a threading hook a registered node keeps the Custom default.
    CHECK(cg::kernel_moldability(graph.nodes()[1]));
}

// output_extents: a graph-owned output follows its producer's own account of its size when a
// bind changes the inputs. Without the hook it kept its captured extents, and the consumer read
// elements the producer never wrote.
TEST_CASE("Descriptor hooks - declared output extents follow a bind", "[ComputeGraph][DescriptorHooks][Bind]") {
    register_hook_descriptors();
    auto   x      = create_random_tensor<double>("x", 3);
    double result = 0.0;

    cg::Graph graph("hooks_twice");
    auto     &twice = graph.scratch<double, 1>("twice", 6);
    {
        cg::CaptureGuard const capture(graph);
        auto                  &ctx = cg::CaptureContext::current();
        record("twice", TwiceDescriptor{}, {ctx.get_slot(x).first}, {ctx.get_slot(twice).first});
        cg::dot(&result, twice, twice);
    }
    graph.apply<cg::passes::Materialization>();
    graph.annotate_dims(x, {"n"});
    graph.execute();
    CHECK(twice.dim(0) == 6);

    auto small = create_random_tensor<double>("x_small", 2);
    REQUIRE_NOTHROW(graph.bind("x", small));
    CHECK(twice.dim(0) == 4);

    graph.execute();
    double const expected = 2.0 * ((small(0) * small(0)) + (small(1) * small(1)));
    CHECK_THAT(result, Catch::Matchers::WithinRel(expected, 1e-14));
}

// An output whose declared extents differ from the hook's account is reported by verify.
TEST_CASE("Descriptor hooks - verify reports output extents the hook disagrees with", "[ComputeGraph][DescriptorHooks]") {
    register_hook_descriptors();
    RuntimeTensor<double> x{"x", {3UL}};
    RuntimeTensor<double> y{"y", {3UL}};
    x.zero();
    y.zero();

    cg::Graph graph("hooks_misreport");
    {
        cg::CaptureGuard const capture(graph);
        auto                  &ctx = cg::CaptureContext::current();
        record("misreport", MisreportDescriptor{}, {ctx.get_slot(x).first}, {ctx.get_slot(y).first});
    }

    auto const problems = graph.verify();
    REQUIRE(problems.size() == 1);
    CHECK(problems[0].find("output_extents hook gives [5]") != std::string::npos);
}

// ── Conformance kit ───────────────────────────────────────────────────────

// Codecs whose claims hold come back clean, including one that accumulates into its destination
// and one whose output extents follow its input.
TEST_CASE("Descriptor conformance - a codec that tells the truth has no problems", "[ComputeGraph][DescriptorHooks][Conformance]") {
    register_hook_descriptors();
    for (auto const &sample : {sample_of("add_into", AddIntoDescriptor{}), sample_of("forgetful_at_default", ForgetfulDescriptor{}),
                               sample_of("declared_param", DeclaredParamDescriptor{}, {{"k", 4}})}) {
        auto const problems = check_one(sample);
        INFO(fmt::format("{}", fmt::join(problems, "\n")));
        CHECK(problems.empty());
    }
}

// Each false claim is caught by the check that holds the codec to it, and named.
TEST_CASE("Descriptor conformance - each false claim is reported", "[ComputeGraph][DescriptorHooks][Conformance]") {
    register_hook_descriptors();
    auto const reported = [](cg::ConformanceSample const &sample, std::string_view expected) {
        auto const problems = check_one(sample);
        INFO(fmt::format("{}", fmt::join(problems, "\n")));
        CHECK(
            std::ranges::any_of(problems, [expected](std::string const &problem) { return problem.find(expected) != std::string::npos; }));
    };

    reported(sample_of("flaky", FlakyDescriptor{}), "is said to be deterministic");
    reported(sample_of("fake_overwrite", FakeOverwriteDescriptor{}), "a poisoned destination changes its result");
    reported(sample_of("hidden_param", HiddenParamDescriptor{}, {{"k", 4}}), "fails with only the parameters it declares");
    reported(sample_of("forgetful", ForgetfulDescriptor{.scale = 2.0}), "rebuilt from its saved form, computes something else");
    reported(sample_of("misreport", MisreportDescriptor{}), "output_extents hook gives [5]");
}

// ── Cost ──────────────────────────────────────────────────────────────────

namespace {

/// The serial time ThreadPlanning models for a graph holding one CostedDescriptor node, which is
/// the makespan before widening: one node, never run, so nothing measured can stand in for it.
double modelled_serial_us(CostedDescriptor const &desc, cg::CostModel const &model) {
    RuntimeTensor<double> x{"x", {4UL}};
    RuntimeTensor<double> y{"y", {4UL}};
    x.zero();
    y.zero();
    cg::Graph graph("hooks_costed");
    {
        cg::CaptureGuard const capture(graph);
        auto                  &ctx = cg::CaptureContext::current();
        record("costed", desc, {ctx.get_slot(x).first}, {ctx.get_slot(y).first});
    }
    cg::passes::ThreadPlanning planner(model, 4);
    planner.run(graph);
    return planner.makespan_before_us();
}

} // namespace

// cost: the planner prices a registered node from its hook instead of as the bytes it moves.
TEST_CASE("Descriptor hooks - a declared cost prices the node for the planner", "[ComputeGraph][DescriptorHooks][Cost]") {
    register_hook_descriptors();
    cg::CostModel const model = cg::CostModel::detect_default();

    // A serial time the descriptor knows is taken as given.
    CHECK(modelled_serial_us(CostedDescriptor{.serial_us = 1234.5}, model) == 1234.5);

    // Flops are priced as the square GEMM of that many flops: 2e9 flops is a 1000-cube.
    double const gemm_us = model.cpu.estimate_gemm_time_us(1000, 1000, 1000, 1);
    CHECK_THAT(modelled_serial_us(CostedDescriptor{.flops = 2.0e9, .bytes = 64}, model), Catch::Matchers::WithinRel(gemm_us, 1e-12));

    // Bytes alone are priced as traffic, as the planner prices any node without a hook.
    double const memory_us = model.cpu.estimate_memory_time_us(std::size_t{1} << 26, 1);
    CHECK_THAT(modelled_serial_us(CostedDescriptor{.bytes = std::size_t{1} << 26}, model), Catch::Matchers::WithinRel(memory_us, 1e-12));

    // A speedup curve of its own is what the widening search consults.
    speedup_calls() = 0;
    (void)modelled_serial_us(CostedDescriptor{.serial_us = 1.0e6, .linear_speedup = true}, model);
    CHECK(speedup_calls() > 0);
}

// threading: a node that claims to be moldable but computes with the thread count it is given is
// caught by the conformance kit's width check.
TEST_CASE("Descriptor conformance - a moldable claim is held to the result at another width",
          "[ComputeGraph][DescriptorHooks][Conformance]") {
    register_hook_descriptors();
    if (omp_get_max_threads() < 3) {
        SKIP("needs a default OpenMP thread count above 2, so width 2 differs from it");
    }
    auto sample         = sample_of("width_dependent", WidthDependentDescriptor{});
    sample.width        = 2;
    auto const problems = check_one(sample);
    INFO(fmt::format("{}", fmt::join(problems, "\n")));
    CHECK(std::ranges::any_of(problems,
                              [](std::string const &problem) { return problem.find("is moldable, but at width 2") != std::string::npos; }));

    // Within a tolerance wide enough to cover the difference, the same claim passes.
    sample.width_tolerance = 1.0;
    auto const tolerated   = check_one(sample);
    INFO(fmt::format("{}", fmt::join(tolerated, "\n")));
    CHECK(std::ranges::none_of(tolerated, [](std::string const &problem) { return problem.find("is moldable") != std::string::npos; }));
}

// ── Merging ───────────────────────────────────────────────────────────────

namespace {

/// Two nodes of @p first's and @p second's descriptors over the same x, each read by its own
/// consumer, run through CSE; returns how many of the two survive, after checking both consumers
/// still read what their own node computes.
template <typename D>
std::size_t survivors_of_cse(D first, D second) {
    RuntimeTensor<double> x{"x", {3UL}};
    RuntimeTensor<double> out1{"out1", {3UL}};
    RuntimeTensor<double> out2{"out2", {3UL}};
    for (std::size_t i = 0; i < 3; ++i) {
        x(i) = static_cast<double>(i + 1);
    }
    out1.zero();
    out2.zero();

    cg::Graph graph("hooks_cse");
    {
        cg::CaptureGuard const capture(graph);
        auto                  &t1  = graph.create_runtime_tensor<double>("t1", {3UL});
        auto                  &t2  = graph.create_runtime_tensor<double>("t2", {3UL});
        auto                  &ctx = cg::CaptureContext::current();
        record("first", first, {ctx.get_slot(x).first}, {ctx.get_slot(t1).first});
        record("second", second, {ctx.get_slot(x).first}, {ctx.get_slot(t2).first});
        cg::axpby(1.0, t1, 0.0, &out1);
        cg::axpby(1.0, t2, 0.0, &out2);
    }
    (void)graph.apply<cg::passes::CSE>();
    graph.execute();
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(out1(i) == first.factor * x(i));
        CHECK(out2(i) == second.factor * x(i));
    }
    return static_cast<std::size_t>(
        std::ranges::count_if(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::Custom; }));
}

} // namespace

// equal: two registered nodes the hook calls the same computation over the same inputs merge, and
// only when the node is pure and overwrites what it writes.
TEST_CASE("Descriptor hooks - a declared equality lets CSE merge two nodes", "[ComputeGraph][DescriptorHooks][CSE]") {
    register_hook_descriptors();
    CHECK(survivors_of_cse(ScaledDescriptor{.factor = 2.0}, ScaledDescriptor{.factor = 2.0}) == 1);
    CHECK(survivors_of_cse(ScaledDescriptor{.factor = 2.0}, ScaledDescriptor{.factor = 3.0}) == 2);

    // Not when its effects are opaque, nor when it says nothing about its destination.
    CHECK(survivors_of_cse(ScaledOpaqueDescriptor{.factor = 2.0}, ScaledOpaqueDescriptor{.factor = 2.0}) == 2);
    CHECK(survivors_of_cse(ScaledNoDestinationDescriptor{.factor = 2.0}, ScaledNoDestinationDescriptor{.factor = 2.0}) == 2);
}

// The conformance kit holds equal to the samples: a truthful one is clean, and one that calls
// different computations the same is caught.
TEST_CASE("Descriptor conformance - an equality claim is held to the results", "[ComputeGraph][DescriptorHooks][Conformance]") {
    register_hook_descriptors();
    std::vector<cg::ConformanceSample> truthful{sample_of("two", ScaledDescriptor{.factor = 2.0}),
                                                sample_of("two_again", ScaledDescriptor{.factor = 2.0}),
                                                sample_of("three", ScaledDescriptor{.factor = 3.0})};
    auto const                         clean = cg::check_descriptor("hooks_test.Scaled", truthful);
    INFO(fmt::format("{}", fmt::join(clean, "\n")));
    CHECK(clean.empty());

    std::vector<cg::ConformanceSample> lying{sample_of("two", ScaledLyingEqualDescriptor{.factor = 2.0}),
                                             sample_of("three", ScaledLyingEqualDescriptor{.factor = 3.0})};
    auto const                         problems = cg::check_descriptor("hooks_test.ScaledLyingEqual", lying);
    INFO(fmt::format("{}", fmt::join(problems, "\n")));
    CHECK(std::ranges::any_of(
        problems, [](std::string const &problem) { return problem.find("the second computes something else") != std::string::npos; }));
}

// ── Tiles ─────────────────────────────────────────────────────────────────

namespace {

using Grid = std::vector<std::vector<int>>;

/// Every element of a 2-D tiled tensor of extents @p rows x @p cols, absent tiles reading zero.
std::vector<double> gather(TiledRuntimeTensor<double> const &tensor, std::size_t rows, std::size_t cols) {
    std::vector<double> out(rows * cols, 0.0);
    auto const         &offsets = tensor.tile_offsets();
    auto const         &sizes   = tensor.tile_sizes();
    for (int ti = 0; ti < static_cast<int>(sizes[0].size()); ++ti) {
        for (int tj = 0; tj < static_cast<int>(sizes[1].size()); ++tj) {
            if (!tensor.has_tile({ti, tj})) {
                continue;
            }
            auto const &tile = tensor.tile({ti, tj});
            for (int r = 0; r < sizes[0][ti]; ++r) {
                for (int c = 0; c < sizes[1][tj]; ++c) {
                    out[(static_cast<std::size_t>(offsets[0][ti] + r) * cols) + static_cast<std::size_t>(offsets[1][tj] + c)] =
                        tile(std::vector<std::size_t>{static_cast<std::size_t>(r), static_cast<std::size_t>(c)});
                }
            }
        }
    }
    return out;
}

/// A BlockFill-shaped producer of @p desc writing C, and ``D = C B`` reading it; returns the
/// graph after @p expand (when set) has run, executed, with D's elements.
template <typename D>
struct ProducerChain {
    Grid const                 c_grid{{2, 3}, {3, 2}};
    Grid const                 b_grid{{3, 2}, {2, 2}};
    RuntimeTensor<double>      v{"v", {1UL}};
    TiledRuntimeTensor<double> C{"C", c_grid};
    TiledRuntimeTensor<double> B{"B", b_grid};
    TiledRuntimeTensor<double> Dt{"D", Grid{{2, 3}, {2, 2}}};
    cg::Graph                  graph{"hooks_producer_chain"};

    explicit ProducerChain(D desc) {
        v(0) = 2.0;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                auto &tile = B.tile({i, j});
                tile.materialize();
                for (std::size_t e = 0; e < tile.size(); ++e) {
                    tile.data()[e] = 0.5 + static_cast<double>((7 * i) + (3 * j) + static_cast<int>(e));
                }
            }
        }
        cg::CaptureGuard const capture(graph);
        auto                  &ctx = cg::CaptureContext::current();
        record("fill", desc, {ctx.get_slot(v).first}, {ctx.get_slot(C).first});
        cg::einsum("ij <- ik ; kj", &Dt, C, B);
    }

    std::size_t count(cg::OpKind kind) const {
        return static_cast<std::size_t>(std::ranges::count_if(graph.nodes(), [kind](cg::Node const &node) { return node.kind == kind; }));
    }
};

} // namespace

// tiles and tile: a registered producer writing a tiled tensor becomes one node per tile it writes,
// and the tiled contraction reading it expands too instead of being stranded.
TEST_CASE("Descriptor hooks - a declared tiling expands the producer and frees its readers", "[ComputeGraph][DescriptorHooks][Tiled]") {
    register_hook_descriptors();

    ProducerChain reference(BlockFillDescriptor{});
    reference.graph.execute();
    auto const want = gather(reference.Dt, 5, 4);

    ProducerChain chain(BlockFillDescriptor{});
    REQUIRE(chain.count(cg::OpKind::Einsum) == 0);
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, cg::passes::Densify::Never, cg::passes::FuseTiles::Never));
    chain.graph.apply(pm);
    INFO(pm.explain());

    // One producer node per diagonal tile, and the contraction expanded over the two tiles of C
    // that exist: C(0,0) and C(1,1), each against the two tiles of B's row.
    auto const tile_nodes = std::ranges::count_if(
        chain.graph.nodes(), [](cg::Node const &node) { return node.op_data.name() == BlockFillTileDescriptor::descriptor_name; });
    CHECK(tile_nodes == 2);
    CHECK(std::ranges::none_of(chain.graph.nodes(), [](cg::Node const &node) { return node.label == "fill"; }));
    CHECK(chain.count(cg::OpKind::Einsum) == 4);

    chain.graph.execute();
    auto const got = gather(chain.Dt, 5, 4);
    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("element " << i);
        CHECK_THAT(got[i], Catch::Matchers::WithinRel(want[i], 1e-14));
    }
}

// A producer that does not say it overwrites its output is left whole, and strands its readers as
// any node TiledExpansion cannot expand does.
TEST_CASE("Descriptor hooks - a producer that does not overwrite its tiles is not expanded", "[ComputeGraph][DescriptorHooks][Tiled]") {
    register_hook_descriptors();
    ProducerChain   chain(BlockFillUnsaidDescriptor{});
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, cg::passes::Densify::Never, cg::passes::FuseTiles::Never));
    chain.graph.apply(pm);
    CHECK(chain.count(cg::OpKind::Einsum) == 0);
    CHECK(std::ranges::any_of(chain.graph.nodes(), [](cg::Node const &node) { return node.label == "fill"; }));
}

namespace {

/// A conformance sample of a BlockFill node writing a 2 x 2 grid of tiles.
cg::ConformanceSample block_fill_sample(std::string name, BlockFillDescriptor desc) {
    return cg::ConformanceSample{.name       = std::move(name),
                                 .descriptor = cg::OpData{desc},
                                 .dtype      = packed_gemm::ScalarType::Float64,
                                 .rank       = 2,
                                 .operands   = [](cg::Graph &graph) {
                                     auto &v   = graph.create_runtime_tensor<double>("v", {1UL}, false);
                                     v(0)      = 2.0;
                                     auto *C   = graph.own(std::make_unique<TiledRuntimeTensor<double>>("C", Grid{{2, 3}, {3, 2}}));
                                     auto &ctx = cg::CaptureContext::current();
                                     return cg::ConformanceOperands{.inputs = {ctx.get_slot(v).first}, .outputs = {ctx.get_slot(*C).first}};
                                 }};
}

} // namespace

// The conformance kit holds tiles and tile to the whole node: a truthful producer is clean, and
// one whose tiles hook misses a tile, or whose tile hook computes another value, is caught.
TEST_CASE("Descriptor conformance - a tiling claim is held to the whole node", "[ComputeGraph][DescriptorHooks][Conformance][Tiled]") {
    register_hook_descriptors();
    auto check = [](BlockFillDescriptor desc) {
        auto const sample = block_fill_sample("block_fill", desc);
        return cg::check_descriptor("hooks_test.BlockFill", std::span<cg::ConformanceSample const>{&sample, 1});
    };

    auto const clean = check(BlockFillDescriptor{});
    INFO(fmt::format("{}", fmt::join(clean, "\n")));
    CHECK(clean.empty());

    auto const missed = check(BlockFillDescriptor{.lie = 1});
    INFO(fmt::format("{}", fmt::join(missed, "\n")));
    CHECK(std::ranges::any_of(
        missed, [](std::string const &p) { return p.find("writes tile (1,1) nonzero, but its tiles hook") != std::string::npos; }));

    auto const wrong = check(BlockFillDescriptor{.lie = 2});
    INFO(fmt::format("{}", fmt::join(wrong, "\n")));
    CHECK(std::ranges::any_of(
        wrong, [](std::string const &p) { return p.find("computes something else than the whole node's tile") != std::string::npos; }));
}

// ── Slices ────────────────────────────────────────────────────────────────

namespace {

/// ``E = sum X X`` with X written by a Grid-shaped producer of @p desc, through AxisTiling under a
/// cap smaller than X; returns the pass, with E in @p energy after a run.
template <typename D>
std::shared_ptr<cg::passes::AxisTiling> tile_grid_energy(D desc, double &energy) {
    constexpr std::size_t nocc = 4;
    constexpr std::size_t nvir = 6;
    RuntimeTensor<double> E{"E", std::vector<std::size_t>{1}};
    E.zero();
    cg::Graph graph("hooks_grid_energy");
    auto     &X = graph.scratch_runtime<double>("X", std::vector<std::size_t>{nocc, nvir});
    {
        cg::CaptureGuard const capture(graph);
        auto                  &ctx = cg::CaptureContext::current();
        record("grid", desc, {}, {ctx.get_slot(X).first});
        cg::dot_python(&E, X, X);
    }
    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(static_cast<std::int64_t>(nvir * sizeof(double)));
    cg::apply_single_pass(*tiling, graph);
    // The default pipeline after the tiling, as a caller would run it: it materializes the body's
    // slices, and its hoisting and reordering must keep the sliced producer after the parameter
    // write it declares it reads.
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();
    energy = E(0);
    // A replay restarts the sweep rather than continuing it.
    E.zero();
    graph.execute();
    CHECK(E(0) == energy);
    return tiling;
}

} // namespace

// axes and slice: AxisTiling takes a registered producer into its loop, writing one slice at a time,
// and the reduction it feeds comes out the same.
TEST_CASE("Descriptor hooks - a declared slicing lets AxisTiling stream a producer", "[ComputeGraph][DescriptorHooks][AxisTiling]") {
    register_hook_descriptors();
    double want = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t a = 0; a < 6; ++a) {
            want += grid_value(i, a) * grid_value(i, a);
        }
    }

    double     energy = 0.0;
    auto const tiling = tile_grid_energy(GridDescriptor{}, energy);
    CHECK(tiling->num_tiled() == 1);
    CHECK(tiling->largest_after() < tiling->largest_before());
    CHECK_THAT(energy, Catch::Matchers::WithinRel(want, 1e-12));

    // A producer whose effects are opaque is not the same computation at every slice.
    double     opaque_energy = 0.0;
    auto const opaque        = tile_grid_energy(GridOpaqueDescriptor{}, opaque_energy);
    CHECK(opaque->num_tiled() == 0);
    CHECK_THAT(opaque_energy, Catch::Matchers::WithinRel(want, 1e-12));
}

// The conformance kit holds slice to the whole node: a truthful one is clean, one that writes the
// next slice is caught.
TEST_CASE("Descriptor conformance - a slicing claim is held to the whole node",
          "[ComputeGraph][DescriptorHooks][Conformance][AxisTiling]") {
    register_hook_descriptors();
    auto check = [](GridDescriptor desc) {
        cg::ConformanceSample const sample{.name       = "grid",
                                           .descriptor = cg::OpData{desc},
                                           .dtype      = packed_gemm::ScalarType::Float64,
                                           .rank       = 2,
                                           .operands   = [](cg::Graph &graph) {
                                               auto &X   = graph.create_runtime_tensor<double>("X", {4UL, 6UL}, false);
                                               auto &ctx = cg::CaptureContext::current();
                                               return cg::ConformanceOperands{.inputs = {}, .outputs = {ctx.get_slot(X).first}};
                                           }};
        return cg::check_descriptor("hooks_test.Grid", std::span<cg::ConformanceSample const>{&sample, 1});
    };
    auto const clean = check(GridDescriptor{});
    INFO(fmt::format("{}", fmt::join(clean, "\n")));
    CHECK(clean.empty());

    auto const wrong = check(GridDescriptor{.lie = 1});
    INFO(fmt::format("{}", fmt::join(wrong, "\n")));
    CHECK(std::ranges::any_of(wrong,
                              [](std::string const &p) { return p.find("computes something else than that slice") != std::string::npos; }));
}
