//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file GraphIO.hpp
/// @brief ComputeGraph-aware .etn file I/O operations.
///
/// These functions record DiskRead/DiskWrite nodes into the graph during
/// capture. They work with IOPrefetch for async I/O overlap when using
/// the DataflowExecutor.
///
/// @code
/// #include <Einsums/TensorIO/GraphIO.hpp>
///
/// {
///     cg::CaptureGuard guard(graph);
///     tensor_io::read_etn("integrals.etn", "ERI", &eri);
///     cg::einsum("ik;kj->ij", &C, eri, B);
///     tensor_io::write_etn("result.etn", "C", &C);
/// }
/// @endcode

#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/TensorIO/Checkpoint.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <string>

namespace einsums {
namespace APIARY_MODULE("io") tensor_io {

/**
 * @brief Read a tensor from a .etn file (graph-aware).
 *
 * During graph capture, records a DiskRead node. Outside capture, reads immediately.
 * Works with IOPrefetch for async overlap when using DataflowExecutor.
 */
// clang-format off
template <CoreBasicTensorConcept TensorType>
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("read", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("read", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("read", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("read", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void read_etn(std::string file_path, std::string tensor_name, TensorType *output) {
    auto executor = [file_path, tensor_name, output]() {
        TensorFile file(file_path, TensorFile::Mode::Read);
        file.read(tensor_name, *output);
    };

    if (!compute_graph::CaptureContext::current().is_capturing()) {
        executor();
        return;
    }

    compute_graph::read(fmt::format("read_etn({}/{})", file_path, tensor_name), file_path, tensor_name, output, std::move(executor));
}

/**
 * @brief Write a tensor to a .etn file (graph-aware).
 *
 * During graph capture, records a DiskWrite node that executes after the
 * tensor is computed. Outside capture, writes immediately.
 */
// clang-format off
template <CoreBasicTensorConcept TensorType>
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("write", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("write", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("write", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("write", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void write_etn(std::string file_path, std::string tensor_name, TensorType const *input) {
    auto executor = [file_path, tensor_name, input]() {
        TensorFile file(file_path, TensorFile::Mode::ReadWrite);
        file.write(tensor_name, *input);
    };

    if (!compute_graph::CaptureContext::current().is_capturing()) {
        executor();
        return;
    }

    compute_graph::write(fmt::format("write_etn({}/{})", file_path, tensor_name), file_path, tensor_name, input, std::move(executor));
}

/**
 * @brief A slab descriptor — per-dimension half-open ranges into a stored tensor.
 *
 * Used by @ref read_slice_etn and @ref write_slice_etn to address a
 * hyperslab of a `.etn` entry. `ranges[d]` is `{start, end}` (end is
 * exclusive). The struct is captured by reference into the graph
 * executor lambda, so the *same* graph node can drive a loop by
 * mutating `ranges` between executor invocations — useful for
 * read-transform-write-back patterns over batches of an ERI.
 */
struct APIARY_EXPOSE Slab {
    APIARY_EXPOSE Slab() = default;
    APIARY_EXPOSE explicit Slab(std::vector<std::pair<size_t, size_t>> ranges) : ranges(std::move(ranges)) {}

    APIARY_EXPOSE std::vector<std::pair<size_t, size_t>> ranges;
};

namespace detail {
/// Dispatcher: route a Slab onto the right TensorFile slice method.
/// Static-rank `Tensor<T, N>` exposes `::Rank` as a constant — use the
/// std::array overload. Runtime-rank `GeneralRuntimeTensor<T, A>` has
/// no such constant and goes through the std::vector overload.
template <typename TensorType>
void file_read_slice_dispatch(TensorFile &file, std::string_view name, TensorType &t, std::vector<std::pair<size_t, size_t>> const &rng) {
    if constexpr (compute_graph::HasCompileTimeRank<TensorType>) {
        constexpr size_t                         R = TensorType::Rank;
        std::array<std::pair<size_t, size_t>, R> arr{};
        for (size_t d = 0; d < R; ++d) {
            arr[d] = rng[d];
        }
        file.read_slice(name, t, arr);
    } else {
        file.read_slice(name, t, rng);
    }
}

template <typename TensorType>
void file_write_slice_dispatch(TensorFile &file, std::string_view name, TensorType const &t,
                               std::vector<std::pair<size_t, size_t>> const &rng) {
    if constexpr (compute_graph::HasCompileTimeRank<TensorType>) {
        constexpr size_t                         R = TensorType::Rank;
        std::array<std::pair<size_t, size_t>, R> arr{};
        for (size_t d = 0; d < R; ++d) {
            arr[d] = rng[d];
        }
        file.write_slice(name, t, arr);
    } else {
        file.write_slice(name, t, rng);
    }
}
} // namespace detail

/**
 * @brief Read a slab of a tensor from a .etn file (graph-aware).
 *
 * During graph capture, records a DiskRead node whose executor reads
 * the slab described by @p slab into @p output. The Slab is captured
 * by reference, so a loop can drive the same graph node over different
 * slab ranges by mutating `slab.ranges` between executor invocations.
 *
 * @p output must be pre-sized to the slab shape. The bound TensorFile
 * methods will throw at execute time if dimensions disagree.
 *
 * @code
 * Slab slab{{ {0, 4}, {0, 4} }};
 * tensor_io::read_slice_etn("integrals.etn", "ERI", slab, &block);
 * @endcode
 */
// clang-format off
template <CoreBasicTensorConcept TensorType>
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("read_slice", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("read_slice", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("read_slice", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("read_slice", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void read_slice_etn(std::string file_path, std::string tensor_name, Slab const &slab, TensorType *output) {
    auto executor = [file_path, tensor_name, slab_ptr = &slab, output]() {
        TensorFile file(file_path, TensorFile::Mode::Read);
        detail::file_read_slice_dispatch(file, tensor_name, *output, slab_ptr->ranges);
    };

    if (!compute_graph::CaptureContext::current().is_capturing()) {
        executor();
        return;
    }

    compute_graph::read(fmt::format("read_slice_etn({}/{})", file_path, tensor_name), file_path, tensor_name, output, std::move(executor));
}

/**
 * @brief Write a slab of a tensor to a .etn file (graph-aware).
 *
 * During capture, records a DiskWrite node that, when executed, opens
 * the file in ReadWrite mode and patches the slab described by
 * @p slab with the contents of @p input. The target entry must
 * already exist (e.g. via a prior @ref write_etn or
 * @ref TensorFile::reserve). The Slab is captured by reference for
 * loop-driven write-back patterns.
 *
 * @code
 * Slab slab{{ {0, 4}, {0, 4} }};
 * tensor_io::write_slice_etn("scratch.etn", "T", slab, &block);
 * @endcode
 */
// clang-format off
template <CoreBasicTensorConcept TensorType>
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("write_slice", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("write_slice", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("write_slice", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("write_slice", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void write_slice_etn(std::string file_path, std::string tensor_name, Slab const &slab, TensorType const *input) {
    auto executor = [file_path, tensor_name, slab_ptr = &slab, input]() {
        TensorFile file(file_path, TensorFile::Mode::ReadWrite);
        detail::file_write_slice_dispatch(file, tensor_name, *input, slab_ptr->ranges);
    };

    if (!compute_graph::CaptureContext::current().is_capturing()) {
        executor();
        return;
    }

    compute_graph::write(fmt::format("write_slice_etn({}/{})", file_path, tensor_name), file_path, tensor_name, input, std::move(executor));
}

/**
 * @brief Checkpoint all graph tensors to a .etn file (graph-aware).
 *
 * During capture, records a DiskWrite node. Outside capture, saves immediately.
 */
inline void checkpoint_etn(std::string file_path, compute_graph::Graph &graph) {
    auto &ctx = compute_graph::CaptureContext::current();
    if (!ctx.is_capturing()) {
        checkpoint::save(file_path, graph);
        return;
    }

    auto *graph_ptr = &graph;
    auto  executor  = [file_path, graph_ptr]() { checkpoint::save(file_path, *graph_ptr); };

    ctx.record(compute_graph::OpKind::DiskWrite, fmt::format("checkpoint_etn({})", file_path), {}, {}, std::move(executor));
}

} // namespace APIARY_MODULE("io")tensor_io
} // namespace einsums
