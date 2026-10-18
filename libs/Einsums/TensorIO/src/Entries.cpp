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

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_io)

namespace {

namespace cg = compute_graph;

void copy_now(std::string const &src_path, std::string const &src_entry, std::string const &dst_path, std::string const &dst_entry) {
    // One handle when both name one file: a second writable handle would write its own entry table
    // over the first's when it closed.
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

/// Register @p D, whose node runs @c D::run.
template <typename D>
void register_entry_descriptor() {
    cg::register_descriptor<D>(
        [](D const &desc) {
            json::Object object;
            for (auto const &[key, member] : D::fields()) {
                object.set(key, json::Value{desc.*member});
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

EINSUMS_NAMESPACE_END(tensor_io)
