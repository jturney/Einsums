//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_io)

namespace {

/// The graph's name for an entry's element type; an integer entry has none.
packed_gemm::ScalarType scalar_type_of(DType dtype) {
    switch (dtype) {
    case DType::Float32:
        return packed_gemm::ScalarType::Float32;
    case DType::Float64:
        return packed_gemm::ScalarType::Float64;
    case DType::Complex64:
        return packed_gemm::ScalarType::Complex64;
    case DType::Complex128:
        return packed_gemm::ScalarType::Complex128;
    default:
        return packed_gemm::ScalarType::Unknown;
    }
}

/// The extents and graph element type of @p entry in @p file, refusing an integer entry.
std::pair<std::vector<size_t>, packed_gemm::ScalarType> describe_entry(TensorFile const &file, std::string const &entry) {
    auto       dims  = file.dims(entry);
    auto const dtype = scalar_type_of(file.dtype(entry));
    if (dtype == packed_gemm::ScalarType::Unknown) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "entry '{}' in '{}' holds integers, and a graph tensor holds floating-point values",
                                entry, file.path());
    }
    return {std::move(dims), dtype};
}

} // namespace

compute_graph::TensorBacking etn_backing(std::string path, std::string entry) {
    std::vector<size_t>     dims;
    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown};
    {
        TensorFile const file(path, TensorFile::Mode::Read);
        std::tie(dims, dtype) = describe_entry(file, entry);
    }

    compute_graph::TensorBacking out;
    out.kind     = std::string{etn_backing_kind};
    out.location = {{"entry", entry}, {"path", path}};
    out.dims     = dims;
    out.dtype    = dtype;
    // Opens the file at every read rather than holding it open: a file replaced between replays is
    // then the one read, and the entry is checked again so one rewritten with another shape is
    // refused rather than read as the wrong elements.
    out.read_slab = [path = std::move(path), entry = std::move(entry), dims = std::move(dims),
                     dtype](void *destination, std::vector<std::pair<size_t, size_t>> const &ranges) {
        TensorFile const file(path, TensorFile::Mode::Read);
        auto const [now_dims, now_dtype] = describe_entry(file, entry);
        if (now_dims != dims || now_dtype != dtype) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "entry '{}' in '{}' is now [{}], not the [{}] the graph was built against", entry,
                                    path, fmt::join(now_dims, ","), fmt::join(dims, ","));
        }
        file.read_slab(entry, destination, ranges);
    };
    return out;
}

namespace {

/// Opens a saved ``.etn`` location, which names exactly a path and an entry.
compute_graph::TensorBacking open_saved(std::vector<std::pair<std::string, std::string>> const &location) {
    std::string path;
    std::string entry;
    for (auto const &[key, value] : location) {
        if (key == "path") {
            path = value;
        } else if (key == "entry") {
            entry = value;
        } else {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "an {} location has no key '{}'; it names a 'path' and an 'entry'",
                                    etn_backing_kind, key);
        }
    }
    if (path.empty() || entry.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "an {} location needs both a 'path' and an 'entry'", etn_backing_kind);
    }
    try {
        return etn_backing(std::move(path), std::move(entry));
    } catch (std::invalid_argument const &) {
        throw;
    } catch (std::exception const &error) {
        // The loader reports invalid_argument against the tensor; a missing file is that too.
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", error.what());
    }
}

// NOLINTNEXTLINE(bugprone-throwing-static-initialization): registering a fresh, qualified name cannot throw.
[[maybe_unused]] bool const registered = [] {
    compute_graph::register_backing_kind({.name = std::string{etn_backing_kind}, .open = open_saved});
    return true;
}();

} // namespace

EINSUMS_NAMESPACE_END(tensor_io)
