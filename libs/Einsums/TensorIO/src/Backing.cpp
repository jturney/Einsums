//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <atomic>
#include <complex>
#include <memory>
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
/// What a write backing knows about the entry's order: whether it may still choose it, which it may
/// only for an entry it created, whose order its caller left open, before any slab is written.
struct OrderState {
    bool              may_choose{false};
    std::atomic<bool> written{false};
};

compute_graph::TensorBacking complete(std::string const &path, std::string const &entry, std::vector<size_t> const &dims,
                                      packed_gemm::ScalarType dtype, bool write, std::shared_ptr<OrderState> const &order_state = {}) {
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
    out.stored_order = [path, entry]() {
        TensorFile const file(path, TensorFile::Mode::Read);
        return file.stored_order(entry);
    };
    if (write) {
        out.write_slab = [path, entry, check, order_state](void const *source, std::vector<std::pair<size_t, size_t>> const &ranges) {
            if (order_state) {
                order_state->written = true;
            }
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            check(file);
            file.write_slab(entry, source, ranges);
        };
        out.order_open = [order_state]() { return order_state && order_state->may_choose && !order_state->written; };
        out.reorder    = [path, entry, order_state](std::vector<size_t> const &order) {
            if (!order_state || !order_state->may_choose || order_state->written) {
                return false;
            }
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            file.set_stored_order(entry, order);
            return true;
        };
    }
    return out;
}

/// Make @p file hold an entry of @p dims and @p dtype under @p entry: reuse one that already
/// does, refuse one that holds another shape or type, and reserve a new one otherwise, stored in
/// @p order when it is given. An order given for an entry that already exists has to be the one it
/// is stored in, since the elements another run wrote are laid out by it.
/// @return True when the entry was reserved here, false when an existing one was reused.
bool hold(TensorFile &file, std::string const &entry, std::vector<size_t> const &dims, packed_gemm::ScalarType dtype,
          std::vector<size_t> const &order) {
    if (file.contains(entry)) {
        auto const [now_dims, now_dtype] = describe_entry(file, entry);
        if (now_dims != dims || now_dtype != dtype) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "entry '{}' in '{}' already holds [{}] of another shape or type than the tensor's [{}]; remove it or name "
                "another entry",
                entry, file.path(), fmt::join(now_dims, ","), fmt::join(dims, ","));
        }
        if (!order.empty() && file.stored_order(entry) != order) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "entry '{}' in '{}' is already stored with its axes in the order [{}], not the [{}] asked for; remove "
                                    "it or name another entry",
                                    entry, file.path(), fmt::join(file.stored_order(entry), ","), fmt::join(order, ","));
        }
        return false;
    }
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        file.reserve<float>(entry, dims, order);
        break;
    case packed_gemm::ScalarType::Float64:
        file.reserve<double>(entry, dims, order);
        break;
    case packed_gemm::ScalarType::Complex64:
        file.reserve<std::complex<float>>(entry, dims, order);
        break;
    case packed_gemm::ScalarType::Complex128:
        file.reserve<std::complex<double>>(entry, dims, order);
        break;
    default:
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "entry '{}' in '{}': a tensor of unknown element type cannot be stored", entry,
                                file.path());
    }
    return true;
}

/// The scratch entry an update backing writes the next value of @p entry into.
std::string next_entry_of(std::string const &entry) {
    return entry + ".next";
}

/// The backing of a tensor the graph reads from @p entry and whose next value it writes into
/// @p next, which @ref compute_graph::TensorBacking::publish then makes @p entry.
compute_graph::TensorBacking update_backing(std::string const &path, std::string const &entry, std::string const &next,
                                            std::vector<size_t> const &dims, packed_gemm::ScalarType dtype,
                                            std::vector<size_t> const &order) {
    // The reader and the order are the entry's, as for a read backing.
    compute_graph::TensorBacking out = complete(path, entry, dims, dtype, false);
    out.location                     = {{"entry", entry}, {"next", next}, {"path", path}};
    out.write                        = true;
    out.update                       = true;
    out.prepare = [path, entry, next, dims, dtype, order](std::vector<size_t> const &tensor_dims, packed_gemm::ScalarType tensor_dtype) {
        // Refused before the scratch entry is touched, with the words annotate_backing would use.
        if (tensor_dims != dims) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "the tensor is [{}] and the storage holds [{}]", fmt::join(tensor_dims, ","),
                                    fmt::join(dims, ","));
        }
        if (tensor_dtype != dtype) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "the tensor holds {} and the storage holds {}",
                                    compute_graph::scalar_type_name(tensor_dtype), compute_graph::scalar_type_name(dtype));
        }
        {
            // The scratch entry is the graph's own: one a crash left behind, or one of another
            // shape, type or order from an earlier program, is replaced rather than refused.
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            if (file.contains(next) &&
                (file.dims(next) != dims || scalar_type_of(file.dtype(next)) != dtype || file.stored_order(next) != order)) {
                file.remove(next);
            }
            hold(file, next, dims, dtype, order);
        }
        compute_graph::TensorBacking prepared = update_backing(path, entry, next, dims, dtype, order);
        prepared.prepare                      = nullptr;
        prepared.write_slab = [path, next, dims, dtype, order](void const *source, std::vector<std::pair<size_t, size_t>> const &ranges) {
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            // The publish at the end of each replay turns the scratch entry into the entry, so the
            // first write of the next replay reserves it again, in the space the old value freed.
            if (!file.contains(next)) {
                hold(file, next, dims, dtype, order);
            }
            if (auto const [now_dims, now_dtype] = describe_entry(file, next); now_dims != dims || now_dtype != dtype) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "entry '{}' in '{}' is now [{}], not the [{}] the graph was built against",
                                        next, path, fmt::join(now_dims, ","), fmt::join(dims, ","));
            }
            file.write_slab(next, source, ranges);
        };
        // Every update backing of one file publishes in one call, as one commit of the file: a crash
        // leaves all of the graph's entries at the values the replay started from, or all at the new.
        prepared.publish_scope = fmt::format("{}:{}", etn_backing_kind, path);
        prepared.publish       = [path](std::vector<compute_graph::TensorBacking const *> const &group) {
            std::vector<std::pair<std::string, std::string>> moves;
            moves.reserve(group.size());
            for (compute_graph::TensorBacking const *member : group) {
                std::string member_entry;
                std::string member_next;
                for (auto const &[key, value] : member->location) {
                    if (key == "entry") {
                        member_entry = value;
                    } else if (key == "next") {
                        member_next = value;
                    }
                }
                moves.emplace_back(std::move(member_next), std::move(member_entry));
            }
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            file.replace(moves);
        };
        return prepared;
    };
    return out;
}

} // namespace

compute_graph::TensorBacking etn_backing(std::string path, std::string entry, bool write, std::vector<std::size_t> order, bool update) {
    if (update) {
        if (write || !order.empty()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "etn: an update backing of entry '{}' reads and writes it in the order it is stored, so it takes "
                                    "neither write nor order",
                                    entry);
        }
        std::string const next = next_entry_of(entry);
        detail::check_entry_name(path, "etn", next);
        std::vector<size_t>     dims;
        packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown};
        std::vector<size_t>     stored;
        {
            TensorFile const file(path, TensorFile::Mode::Read);
            std::tie(dims, dtype) = describe_entry(file, entry);
            stored                = file.stored_order(entry);
        }
        return update_backing(path, entry, next, dims, dtype, stored);
    }
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
    out.prepare  = [path = std::move(path), entry = std::move(entry), order = std::move(order)](std::vector<size_t> const &dims,
                                                                                                packed_gemm::ScalarType    dtype) {
        // The graph may choose how a new entry is laid out, unless its caller already has.
        auto state = std::make_shared<OrderState>();
        {
            TensorFile file(path, TensorFile::Mode::ReadWrite);
            state->may_choose = hold(file, entry, dims, dtype, order) && order.empty();
        }
        return complete(path, entry, dims, dtype, true, state);
    };
    return out;
}

namespace {

/// Opens a saved ``.etn`` location, which names exactly a path and an entry.
compute_graph::TensorBacking open_saved(std::vector<std::pair<std::string, std::string>> const &location, bool write) {
    std::string path;
    std::string entry;
    std::string next;
    for (auto const &[key, value] : location) {
        if (key == "path") {
            path = value;
        } else if (key == "entry") {
            entry = value;
        } else if (key == "next") {
            next = value;
        } else {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "an {} location has no key '{}'; it names a 'path' and an 'entry', and an update backing a 'next'",
                                    etn_backing_kind, key);
        }
    }
    if (path.empty() || entry.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "an {} location needs both a 'path' and an 'entry'", etn_backing_kind);
    }
    // An update backing names its scratch entry, which is always the entry's own.
    bool const update = !next.empty();
    if (update && (!write || next != next_entry_of(entry))) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "an {} location names the scratch entry '{}', and an update of '{}' writes '{}'",
                                etn_backing_kind, next, entry, next_entry_of(entry));
    }
    try {
        return update ? etn_backing(std::move(path), std::move(entry), false, {}, true)
                      : etn_backing(std::move(path), std::move(entry), write);
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
