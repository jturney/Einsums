//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Workspace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/Checkpoint.hpp>

#include <complex>
#include <unordered_set>

namespace einsums::tensor_io::checkpoint {

namespace {

/// Runtime dispatch: write a type-erased tensor to a TensorFile.
void write_handle(TensorFile &file, compute_graph::TensorHandle const &handle) {
    if (!handle.tensor_ptr || handle.rank == 0)
        return;

    auto go = [&]<typename T>(T /*tag*/) {
        switch (handle.rank) {
        case 1:
            file.write(handle.name, *static_cast<Tensor<T, 1> const *>(handle.tensor_ptr));
            break;
        case 2:
            file.write(handle.name, *static_cast<Tensor<T, 2> const *>(handle.tensor_ptr));
            break;
        case 3:
            file.write(handle.name, *static_cast<Tensor<T, 3> const *>(handle.tensor_ptr));
            break;
        case 4:
            file.write(handle.name, *static_cast<Tensor<T, 4> const *>(handle.tensor_ptr));
            break;
        default:
            break;
        }
    };

    switch (handle.dtype) {
    case packed_gemm::ScalarType::Float32:
        go(float{});
        break;
    case packed_gemm::ScalarType::Float64:
        go(double{});
        break;
    case packed_gemm::ScalarType::Complex64:
        go(std::complex<float>{});
        break;
    case packed_gemm::ScalarType::Complex128:
        go(std::complex<double>{});
        break;
    default:
        break;
    }
}

/// Runtime dispatch: read a type-erased tensor from a TensorFile.
void read_handle(TensorFile &file, compute_graph::TensorHandle const &handle) {
    if (!handle.tensor_ptr || handle.rank == 0)
        return;
    if (!file.contains(handle.name))
        return;

    auto go = [&]<typename T>(T /*tag*/) {
        switch (handle.rank) {
        case 1:
            file.read(handle.name, *static_cast<Tensor<T, 1> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 2:
            file.read(handle.name, *static_cast<Tensor<T, 2> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 3:
            file.read(handle.name, *static_cast<Tensor<T, 3> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 4:
            file.read(handle.name, *static_cast<Tensor<T, 4> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        default:
            break;
        }
    };

    switch (handle.dtype) {
    case packed_gemm::ScalarType::Float32:
        go(float{});
        break;
    case packed_gemm::ScalarType::Float64:
        go(double{});
        break;
    case packed_gemm::ScalarType::Complex64:
        go(std::complex<float>{});
        break;
    case packed_gemm::ScalarType::Complex128:
        go(std::complex<double>{});
        break;
    default:
        break;
    }
}

/// Distributed write dispatch.
void write_handle_distributed(DistributedTensorFile &file, compute_graph::TensorHandle const &handle) {
    if (!handle.tensor_ptr || handle.rank == 0)
        return;

    auto go_repl = [&]<typename T>(T /*tag*/) {
        switch (handle.rank) {
        case 1:
            file.write(handle.name, *static_cast<Tensor<T, 1> const *>(handle.tensor_ptr));
            break;
        case 2:
            file.write(handle.name, *static_cast<Tensor<T, 2> const *>(handle.tensor_ptr));
            break;
        case 3:
            file.write(handle.name, *static_cast<Tensor<T, 3> const *>(handle.tensor_ptr));
            break;
        case 4:
            file.write(handle.name, *static_cast<Tensor<T, 4> const *>(handle.tensor_ptr));
            break;
        default:
            break;
        }
    };

    auto go_local = [&]<typename T>(T /*tag*/) {
        switch (handle.rank) {
        case 1:
            file.write_local(handle.name, *static_cast<Tensor<T, 1> const *>(handle.tensor_ptr));
            break;
        case 2:
            file.write_local(handle.name, *static_cast<Tensor<T, 2> const *>(handle.tensor_ptr));
            break;
        case 3:
            file.write_local(handle.name, *static_cast<Tensor<T, 3> const *>(handle.tensor_ptr));
            break;
        case 4:
            file.write_local(handle.name, *static_cast<Tensor<T, 4> const *>(handle.tensor_ptr));
            break;
        default:
            break;
        }
    };

    auto dispatch = [&](auto go) {
        switch (handle.dtype) {
        case packed_gemm::ScalarType::Float32:
            go(float{});
            break;
        case packed_gemm::ScalarType::Float64:
            go(double{});
            break;
        case packed_gemm::ScalarType::Complex64:
            go(std::complex<float>{});
            break;
        case packed_gemm::ScalarType::Complex128:
            go(std::complex<double>{});
            break;
        default:
            break;
        }
    };

    if (handle.is_distributed && !handle.is_replicated)
        dispatch(go_local);
    else
        dispatch(go_repl);
}

/// Distributed read dispatch.
void read_handle_distributed(DistributedTensorFile &file, compute_graph::TensorHandle const &handle) {
    if (!handle.tensor_ptr || handle.rank == 0)
        return;
    if (!file.contains(handle.name))
        return;

    auto go_repl = [&]<typename T>(T /*tag*/) {
        switch (handle.rank) {
        case 1:
            file.read(handle.name, *static_cast<Tensor<T, 1> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 2:
            file.read(handle.name, *static_cast<Tensor<T, 2> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 3:
            file.read(handle.name, *static_cast<Tensor<T, 3> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 4:
            file.read(handle.name, *static_cast<Tensor<T, 4> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        default:
            break;
        }
    };

    auto go_local = [&]<typename T>(T /*tag*/) {
        switch (handle.rank) {
        case 1:
            file.read_local(handle.name, *static_cast<Tensor<T, 1> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 2:
            file.read_local(handle.name, *static_cast<Tensor<T, 2> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 3:
            file.read_local(handle.name, *static_cast<Tensor<T, 3> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        case 4:
            file.read_local(handle.name, *static_cast<Tensor<T, 4> *>(const_cast<void *>(handle.tensor_ptr)));
            break;
        default:
            break;
        }
    };

    auto dispatch = [&](auto go) {
        switch (handle.dtype) {
        case packed_gemm::ScalarType::Float32:
            go(float{});
            break;
        case packed_gemm::ScalarType::Float64:
            go(double{});
            break;
        case packed_gemm::ScalarType::Complex64:
            go(std::complex<float>{});
            break;
        case packed_gemm::ScalarType::Complex128:
            go(std::complex<double>{});
            break;
        default:
            break;
        }
    };

    if (handle.is_distributed && !handle.is_replicated)
        dispatch(go_local);
    else
        dispatch(go_repl);
}

bool passes_filter(std::string const &name, std::unordered_set<std::string> const &filter) {
    return filter.empty() || filter.count(name) > 0;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Graph
// ═══════════════════════════════════════════════════════════════════════════════

void save(std::string const &path, compute_graph::Graph const &graph, std::vector<std::string> const &tensor_names) {
    TensorFile                            file(path, TensorFile::Mode::Write);
    std::unordered_set<std::string> const filter(tensor_names.begin(), tensor_names.end());

    size_t count = 0;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (passes_filter(handle.name, filter)) {
            write_handle(file, handle);
            count++;
        }
    }
    EINSUMS_LOG_INFO("checkpoint::save: wrote {} tensors to '{}'", count, path);
}

void restore(std::string const &path, compute_graph::Graph &graph) {
    TensorFile file(path, TensorFile::Mode::Read);

    size_t count = 0;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        read_handle(file, handle);
        count++;
    }
    EINSUMS_LOG_INFO("checkpoint::restore: read {} tensors from '{}'", count, path);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace
// ═══════════════════════════════════════════════════════════════════════════════

void save(std::string const &path, compute_graph::Workspace const &workspace, std::vector<std::string> const &tensor_names) {
    TensorFile                            file(path, TensorFile::Mode::Write);
    std::unordered_set<std::string> const filter(tensor_names.begin(), tensor_names.end());

    size_t count = 0;
    for (auto const &handle : workspace.tensor_handles()) {
        if (passes_filter(handle.name, filter)) {
            write_handle(file, handle);
            count++;
        }
    }
    EINSUMS_LOG_INFO("checkpoint::save(workspace): wrote {} tensors to '{}'", count, path);
}

void restore(std::string const &path, compute_graph::Workspace &workspace) {
    TensorFile file(path, TensorFile::Mode::Read);

    size_t count = 0;
    for (auto const &handle : workspace.tensor_handles()) {
        read_handle(file, handle);
        count++;
    }
    EINSUMS_LOG_INFO("checkpoint::restore(workspace): read {} tensors from '{}'", count, path);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Distributed
// ═══════════════════════════════════════════════════════════════════════════════

void save_distributed(std::string const &path, compute_graph::Graph const &graph, std::vector<std::string> const &tensor_names) {
    DistributedTensorFile                 file(path, DistributedTensorFile::Mode::Write);
    std::unordered_set<std::string> const filter(tensor_names.begin(), tensor_names.end());

    size_t count = 0;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (passes_filter(handle.name, filter)) {
            write_handle_distributed(file, handle);
            count++;
        }
    }
    EINSUMS_LOG_INFO("checkpoint::save_distributed: wrote {} tensors to '{}' (rank {}/{})", count, path, comm::world_rank(),
                     comm::world_size());
}

void restore_distributed(std::string const &path, compute_graph::Graph &graph) {
    DistributedTensorFile file(path, DistributedTensorFile::Mode::Read);

    size_t count = 0;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        read_handle_distributed(file, handle);
        count++;
    }
    EINSUMS_LOG_INFO("checkpoint::restore_distributed: read {} tensors from '{}' (rank {}/{})", count, path, comm::world_rank(),
                     comm::world_size());
}

} // namespace einsums::tensor_io::checkpoint
