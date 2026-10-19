//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/DescriptorRegistry.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Json/Json.hpp>
#include <Einsums/TensorIO/Entries.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_io)

namespace {

namespace cg = compute_graph;

void copy_now(std::string const &src_path, std::string const &src_entry, std::string const &dst_path, std::string const &dst_entry) {
    // One handle when both name one file: a second writable handle would find the first's commit
    // and refuse its own.
    std::error_code ec;
    if (std::filesystem::exists(dst_path, ec) && std::filesystem::equivalent(src_path, dst_path, ec)) {
        TensorFile file(src_path, TensorFile::Mode::ReadWrite);
        file.copy(file, src_entry, dst_entry);
        return;
    }
    TensorFile const source(src_path, TensorFile::Mode::Read);
    TensorFile       destination(dst_path, TensorFile::Mode::ReadWrite);
    destination.copy(source, src_entry, dst_entry);
}

void rename_now(std::string const &path, std::string const &old_name, std::string const &new_name) {
    TensorFile file(path, TensorFile::Mode::ReadWrite);
    file.rename(old_name, new_name);
}

void swap_now(std::string const &path, std::string const &first, std::string const &second) {
    TensorFile file(path, TensorFile::Mode::ReadWrite);
    file.swap(first, second);
}

void remove_now(std::string const &path, std::string const &name) {
    TensorFile file(path, TensorFile::Mode::ReadWrite);
    file.remove(name);
}

void restage_now(std::string const &path, std::string const &name, std::vector<size_t> const &order) {
    TensorFile file(path, TensorFile::Mode::ReadWrite);
    file.restage(name, order);
}

void compact_now(std::string const &path) {
    TensorFile file(path, TensorFile::Mode::ReadWrite);
    file.compact();
}

/// A captured @ref copy_entry.
struct CopyEntryDescriptor {
    static constexpr std::string_view descriptor_name = "tensor_io.CopyEntry";
    std::string                       src_path;
    std::string                       src_entry;
    std::string                       dst_path;
    std::string                       dst_entry;

    static constexpr auto fields() {
        return std::array{std::pair{"src_path", &CopyEntryDescriptor::src_path}, std::pair{"src_entry", &CopyEntryDescriptor::src_entry},
                          std::pair{"dst_path", &CopyEntryDescriptor::dst_path}, std::pair{"dst_entry", &CopyEntryDescriptor::dst_entry}};
    }

    void run() const { copy_now(src_path, src_entry, dst_path, dst_entry); }
};

/// A captured @ref rename_entry.
struct RenameEntryDescriptor {
    static constexpr std::string_view descriptor_name = "tensor_io.RenameEntry";
    std::string                       path;
    std::string                       old_name;
    std::string                       new_name;

    static constexpr auto fields() {
        return std::array{std::pair{"path", &RenameEntryDescriptor::path}, std::pair{"old_name", &RenameEntryDescriptor::old_name},
                          std::pair{"new_name", &RenameEntryDescriptor::new_name}};
    }

    void run() const { rename_now(path, old_name, new_name); }
};

/// A captured @ref swap_entries.
struct SwapEntriesDescriptor {
    static constexpr std::string_view descriptor_name = "tensor_io.SwapEntries";
    std::string                       path;
    std::string                       first;
    std::string                       second;

    static constexpr auto fields() {
        return std::array{std::pair{"path", &SwapEntriesDescriptor::path}, std::pair{"first", &SwapEntriesDescriptor::first},
                          std::pair{"second", &SwapEntriesDescriptor::second}};
    }

    void run() const { swap_now(path, first, second); }
};

/// A captured @ref remove_entry.
struct RemoveEntryDescriptor {
    static constexpr std::string_view descriptor_name = "tensor_io.RemoveEntry";
    std::string                       path;
    std::string                       name;

    static constexpr auto fields() {
        return std::array{std::pair{"path", &RemoveEntryDescriptor::path}, std::pair{"name", &RemoveEntryDescriptor::name}};
    }

    void run() const { remove_now(path, name); }
};

/// A captured @ref restage_entry. Its order is not a string, so it is written and read beside the fields.
struct RestageEntryDescriptor {
    static constexpr std::string_view descriptor_name = "tensor_io.RestageEntry";
    std::string                       path;
    std::string                       name;
    std::vector<size_t>               order;

    static constexpr auto fields() {
        return std::array{std::pair{"path", &RestageEntryDescriptor::path}, std::pair{"name", &RestageEntryDescriptor::name}};
    }

    void run() const { restage_now(path, name, order); }
};

/// A captured @ref compact_file.
struct CompactFileDescriptor {
    static constexpr std::string_view descriptor_name = "tensor_io.CompactFile";
    std::string                       path;

    static constexpr auto fields() { return std::array{std::pair{"path", &CompactFileDescriptor::path}}; }

    void run() const { compact_now(path); }
};

/// Register @p D, whose node runs @c D::run.
template <typename D>
void register_entry_descriptor() {
    cg::register_descriptor<D>(
        [](D const &desc) {
            json::Object object;
            for (auto const &[key, member] : D::fields()) {
                object.set(key, json::Value{desc.*member});
            }
            if constexpr (std::is_same_v<D, RestageEntryDescriptor>) {
                json::Array order;
                for (size_t const axis : desc.order) {
                    order.emplace_back(axis);
                }
                object.set("order", json::Value{std::move(order)});
            }
            return json::Value{std::move(object)};
        },
        [](json::Object const &object) {
            D desc;
            for (auto const &[key, member] : D::fields()) {
                json::Value const *value = object.take(key);
                if (value == nullptr || !value->is_string()) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "a {} node needs the string '{}'", D::descriptor_name, key);
                }
                desc.*member = value->as_string();
            }
            if constexpr (std::is_same_v<D, RestageEntryDescriptor>) {
                json::Value const *order = object.take("order");
                if (order == nullptr || !order->is_array()) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "a {} node needs the array 'order'", D::descriptor_name);
                }
                for (json::Value const &axis : order->as_array()) {
                    if (!axis.is_int() || axis.as_int() < 0) {
                        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "a {} node's 'order' holds axes, which are counts",
                                                D::descriptor_name);
                    }
                    desc.order.push_back(static_cast<size_t>(axis.as_int()));
                }
            }
            return desc;
        },
        [](D const &desc, cg::Graph &, packed_gemm::ScalarType, std::size_t, std::span<cg::TensorId const>,
           std::span<cg::TensorId const>) -> std::function<void()> { return [desc]() { desc.run(); }; },
        cg::DescriptorHooksFor<D>{
            // It changes a file the graph's other disk reads and writes may touch, and has no output
            // to keep it alive or in its loop: it is ordered against every one of them, runs at every
            // iteration, and is never removed.
            .effects   = [](D const &) { return cg::NodeEffects{.deterministic = true, .external_effects = true}; },
            .threading = [](D const &) { return cg::Threading::Serial; },
        });
}

// NOLINTNEXTLINE(bugprone-throwing-static-initialization): registering fresh, qualified names cannot throw.
[[maybe_unused]] bool const registered = [] {
    register_entry_descriptor<CopyEntryDescriptor>();
    register_entry_descriptor<RenameEntryDescriptor>();
    register_entry_descriptor<SwapEntriesDescriptor>();
    register_entry_descriptor<RemoveEntryDescriptor>();
    register_entry_descriptor<RestageEntryDescriptor>();
    register_entry_descriptor<CompactFileDescriptor>();
    return true;
}();

/// Run @p desc now, or record it into the graph being captured.
template <typename D>
void run_or_record(D desc, std::string label) {
    auto &ctx = cg::CaptureContext::current();
    if (!ctx.is_capturing()) {
        desc.run();
        return;
    }
    ctx.record_built(cg::OpKind::Custom, std::move(label), packed_gemm::ScalarType::Float64, 0, cg::OpData{std::move(desc)}, {}, {}, {},
                     {});
}

} // namespace

void copy_entry(std::string src_path, std::string src_entry, std::string dst_path, std::string dst_entry) {
    auto label = fmt::format("copy_etn({}#{} -> {}#{})", src_path, src_entry, dst_path, dst_entry);
    run_or_record(CopyEntryDescriptor{.src_path  = std::move(src_path),
                                      .src_entry = std::move(src_entry),
                                      .dst_path  = std::move(dst_path),
                                      .dst_entry = std::move(dst_entry)},
                  std::move(label));
}

void rename_entry(std::string path, std::string old_name, std::string new_name) {
    auto label = fmt::format("rename_etn({}#{} -> {})", path, old_name, new_name);
    run_or_record(RenameEntryDescriptor{.path = std::move(path), .old_name = std::move(old_name), .new_name = std::move(new_name)},
                  std::move(label));
}

void swap_entries(std::string path, std::string first, std::string second) {
    auto label = fmt::format("swap_etn({}#{} <-> {})", path, first, second);
    run_or_record(SwapEntriesDescriptor{.path = std::move(path), .first = std::move(first), .second = std::move(second)}, std::move(label));
}

void remove_entry(std::string path, std::string name) {
    auto label = fmt::format("remove_etn({}#{})", path, name);
    run_or_record(RemoveEntryDescriptor{.path = std::move(path), .name = std::move(name)}, std::move(label));
}

void restage_entry(std::string path, std::string name, std::vector<std::size_t> order) {
    auto label = fmt::format("restage_etn({}#{} -> [{}])", path, name, fmt::join(order, ","));
    run_or_record(RestageEntryDescriptor{.path = std::move(path), .name = std::move(name), .order = std::move(order)}, std::move(label));
}

void compact_file(std::string path) {
    auto label = fmt::format("compact_etn({})", path);
    run_or_record(CompactFileDescriptor{.path = std::move(path)}, std::move(label));
}

EINSUMS_NAMESPACE_END(tensor_io)
