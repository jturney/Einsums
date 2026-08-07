//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Runtime.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Build a host<->device transfer node for @p handle.
 *
 * Shared by TransferInsertion (H2D / D2H staging) and TransferElimination
 * (D2H eviction); those three call sites differ only in @p kind, the
 * @p label_prefix, and @p size_bytes. The executor is a device_synchronize
 * placeholder on the mock backend; the graph executor swaps in a real
 * gpu::memcpy once device shadow allocations exist.
 *
 * @param kind         OpKind::HostToDevice or OpKind::DeviceToHost.
 * @param handle       The tensor being staged/evicted (supplies id and name).
 * @param label_prefix Human-readable prefix, e.g. "H2D", "D2H", "D2H_evict".
 * @param size_bytes   Bytes moved (handle.total_bytes(), or an eviction size).
 */
inline Node make_transfer_node(OpKind kind, TensorHandle const &handle, std::string_view label_prefix, std::size_t size_bytes) {
    Node n;
    n.kind   = kind;
    n.target = Target::CPU; // the transfer itself is a host-initiated operation
    n.label  = fmt::format("{}({})", label_prefix, handle.name);

    TransferDescriptor desc;
    desc.tensor_id  = handle.id;
    desc.size_bytes = size_bytes;
    n.op_data       = desc;

    n.inputs          = {handle.id};
    n.outputs         = {handle.id};
    n.estimated_bytes = size_bytes;
    n.execute         = []() { gpu::device_synchronize(); };

    return n;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
