//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file DescriptorRegistry.hpp
/// @brief Descriptors declared outside this library that a graph can still save, load and rebuild.
///
/// A @ref NodeDescriptor declared anywhere can ride on an @ref OpKind::Custom node, but such a node
/// has only the closure it was captured with: nothing can write it to a file, read it back, or
/// build its executor again after a pass rewrites it. Registering a @ref DescriptorCodec under the
/// descriptor's name supplies all three, and from then on a Custom node carrying that descriptor
/// is reconstructible - @ref reconstruction_blocker accepts it, @ref build_executor builds it, and
/// the graph IR saves and loads it.

#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief How to save, load and rebuild one descriptor type.
 *
 * @ref name is the descriptor's @c descriptor_name. It must be QUALIFIED - contain a ``.``, as in
 * ``"tiledarray.Contraction"`` - so a registered name can never collide with one of this library's
 * own descriptors, none of which has one.
 */
struct DescriptorCodec {
    /// The descriptor's @c descriptor_name.
    std::string name;

    /// The descriptor's fields as a JSON object. Called only with an @ref OpData holding this
    /// descriptor.
    std::function<einsums::compute_graph::json::Value(OpData const &descriptor)> write;

    /// The descriptor rebuilt from what @ref write produced. Read each key with
    /// @c json::Object::take: the loader reports any key left unread, as it does for this
    /// library's own descriptors. Throw @c std::invalid_argument on a malformed object; the loader
    /// reports the message against the node.
    std::function<OpData(einsums::compute_graph::json::Object const &fields)> read;

    /// The node's executor, built from data alone: the descriptor, the node's destination dtype and
    /// rank, and its operand ids in @p graph. Resolve each operand once with @ref resolve_operand
    /// and read it through the returned accessor on every run, so the node follows rebind() and
    /// the memory planner.
    std::function<std::function<void()>(OpData const &descriptor, Graph &graph, packed_gemm::ScalarType dtype, std::size_t rank,
                                        std::span<TensorId const> inputs, std::span<TensorId const> outputs)>
        build;
};

/**
 * @brief Register @p codec for the descriptor named @p codec.name.
 *
 * Do it before capturing, saving or loading a graph that uses the descriptor, typically from a
 * namespace-scope initializer in the code that declares it. Registration is thread-safe.
 *
 * @throws std::invalid_argument when the name is not qualified, is already registered, or when
 *         any of the three functions is empty.
 */
EINSUMS_EXPORT void register_descriptor(DescriptorCodec codec);

/// @brief The codec registered for the descriptor named @p name, or null. Thread-safe; the codec
///        lives for the rest of the process.
[[nodiscard]] EINSUMS_EXPORT DescriptorCodec const *find_descriptor_codec(std::string_view name) noexcept;

/**
 * @brief Register a codec for @p D from functions over @p D itself.
 *
 * @p write takes a ``D const &`` and returns a @c json::Value; @p read takes a
 * ``json::Object const &`` and returns a @p D; @p build takes a ``D const &`` followed by the
 * arguments of @ref DescriptorCodec::build.
 */
template <NodeDescriptor D, typename Write, typename Read, typename Build>
void register_descriptor(Write write, Read read, Build build) {
    register_descriptor(DescriptorCodec{
        .name  = std::string(D::descriptor_name),
        .write = [write = std::move(write)](OpData const &descriptor) -> json::Value { return write(descriptor.get<D>()); },
        .read  = [read = std::move(read)](json::Object const &fields) -> OpData { return OpData{read(fields)}; },
        .build = [build = std::move(build)](
                     OpData const &descriptor, Graph &graph, packed_gemm::ScalarType dtype, std::size_t rank,
                     std::span<TensorId const> inputs,
                     std::span<TensorId const> outputs) { return build(descriptor.get<D>(), graph, dtype, rank, inputs, outputs); }});
}

EINSUMS_NAMESPACE_END(compute_graph)
