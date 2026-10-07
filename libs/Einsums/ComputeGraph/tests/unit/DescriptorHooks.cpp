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
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
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
        return true;
    }();
    (void)registered;
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
