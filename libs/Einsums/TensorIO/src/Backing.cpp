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

#include <complex>
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

namespace {

/// The backing of an entry whose shape is known: its reader and, for a write backing, its writer.
///
/// Both open the file at every call rather than holding it open: a file replaced between
/// replays is then the one used, and the entry is checked again, so one rewritten with another
/// shape is refused rather than read or written as the wrong elements.
compute_graph::TensorBacking complete(std::string const &path, std::string const &entry, std::vector<size_t> const &dims,
                                      packed_gemm::ScalarType dtype, bool write) {
    compute_graph::TensorBacking out;
    out.kind     = std::string{etn_backing_kind};
    out.location = {{"entry", entry}, {"path", path}};
    out.dims     = dims;
    out.dtype    = dtype;
    out.write    = write;

    auto const check = [path, entry, dims, dtype](TensorFile const &file) {
        auto const [now_dims, now_dtype] = describe_entry(file, entry);
        if (now_dims != dims || now_dtype != dtype) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "entry '{}' in '{}' is now [{}], not the [{}] the graph was built against", entry,
                                    path, fmt::join(now_dims, ","), fmt::join(dims, ","));
        }
    };
    out.read_slab = [path, entry, check](void *destination, std::vector<std::pair<size_t, size_t>> const &ranges) {
        TensorFile const file(path, TensorFile::Mode::Read);
        check(file);
        file.read_slab(entry, destination, ranges);
    };
    if (write) {
        out.write_slab = [path, entry, check](void const *source, std::vector<std::pair<size_t, size_t>> const &ranges) {
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            check(file);
            file.write_slab(entry, source, ranges);
        };
    }
    return out;
}

/// Make @p file hold an entry of @p dims and @p dtype under @p entry: reuse one that already
/// does, refuse one that holds another shape or type, and reserve a new one otherwise.
void hold(TensorFile &file, std::string const &entry, std::vector<size_t> const &dims, packed_gemm::ScalarType dtype) {
    if (file.contains(entry)) {
        auto const [now_dims, now_dtype] = describe_entry(file, entry);
        if (now_dims != dims || now_dtype != dtype) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "entry '{}' in '{}' already holds [{}] of another shape or type than the tensor's [{}]; remove it or name "
                "another entry",
                entry, file.path(), fmt::join(now_dims, ","), fmt::join(dims, ","));
        }
        return;
    }
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        file.reserve<float>(entry, dims);
        break;
    case packed_gemm::ScalarType::Float64:
        file.reserve<double>(entry, dims);
        break;
    case packed_gemm::ScalarType::Complex64:
        file.reserve<std::complex<float>>(entry, dims);
        break;
    case packed_gemm::ScalarType::Complex128:
        file.reserve<std::complex<double>>(entry, dims);
        break;
    default:
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "entry '{}' in '{}': a tensor of unknown element type cannot be stored", entry,
                                file.path());
    }
}

} // namespace

compute_graph::TensorBacking etn_backing(std::string path, std::string entry, bool write) {
    if (!write) {
        std::vector<size_t>     dims;
        packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown};
        {
            TensorFile const file(path, TensorFile::Mode::Read);
            std::tie(dims, dtype) = describe_entry(file, entry);
        }
        return complete(path, entry, dims, dtype, false);
    }

    // A write backing names storage the graph fills, which may not exist yet: it is completed by
    // the tensor it is annotated on, whose shape and type the entry then takes.
    compute_graph::TensorBacking out;
    out.kind     = std::string{etn_backing_kind};
    out.location = {{"entry", entry}, {"path", path}};
    out.write    = true;
    out.prepare  = [path = std::move(path), entry = std::move(entry)](std::vector<size_t> const &dims, packed_gemm::ScalarType dtype) {
        {
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            hold(file, entry, dims, dtype);
        }
        return complete(path, entry, dims, dtype, true);
    };
    return out;
}

namespace {

/// Opens a saved ``.etn`` location, which names exactly a path and an entry.
compute_graph::TensorBacking open_saved(std::vector<std::pair<std::string, std::string>> const &location, bool write) {
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
        return etn_backing(std::move(path), std::move(entry), write);
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
