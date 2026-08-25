//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/ComputeGraph/UsageAnalysis.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

std::string_view manifest_direction_name(ManifestDirection direction) noexcept {
    switch (direction) {
    case ManifestDirection::Input:
        return "input";
    case ManifestDirection::Output:
        return "output";
    case ManifestDirection::InOut:
        return "inout";
    }
    return "unknown";
}

std::string_view tensor_ownership_name(TensorOwnership scope) noexcept {
    switch (scope) {
    case TensorOwnership::Graph:
        return "graph";
    case TensorOwnership::Pipeline:
        return "pipeline";
    case TensorOwnership::Workspace:
        return "workspace";
    }
    return "unknown";
}

std::string_view scalar_type_name(packed_gemm::ScalarType dtype) noexcept {
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return "float32";
    case packed_gemm::ScalarType::Float64:
        return "float64";
    case packed_gemm::ScalarType::Complex64:
        return "complex64";
    case packed_gemm::ScalarType::Complex128:
        return "complex128";
    case packed_gemm::ScalarType::Unknown:
        return "unknown";
    }
    return "unknown";
}

std::optional<ManifestDirection> manifest_direction_from_name(std::string_view name) noexcept {
    for (auto const direction : {ManifestDirection::Input, ManifestDirection::Output, ManifestDirection::InOut}) {
        if (manifest_direction_name(direction) == name) {
            return direction;
        }
    }
    return std::nullopt;
}

std::optional<TensorOwnership> tensor_ownership_from_name(std::string_view name) noexcept {
    for (auto const scope : {TensorOwnership::Graph, TensorOwnership::Pipeline, TensorOwnership::Workspace}) {
        if (tensor_ownership_name(scope) == name) {
            return scope;
        }
    }
    return std::nullopt;
}

std::optional<packed_gemm::ScalarType> scalar_type_from_name(std::string_view name) noexcept {
    for (auto const dtype : {packed_gemm::ScalarType::Float32, packed_gemm::ScalarType::Float64, packed_gemm::ScalarType::Complex64,
                             packed_gemm::ScalarType::Complex128, packed_gemm::ScalarType::Unknown}) {
        if (scalar_type_name(dtype) == name) {
            return dtype;
        }
    }
    return std::nullopt;
}

// ── InterfaceManifest ──────────────────────────────────────────────────────

ManifestEntry const *InterfaceManifest::find(std::string_view name) const noexcept {
    auto const by_name = [name](ManifestEntry const &entry) { return entry.name == name; };
    if (auto const it = std::ranges::find_if(inputs, by_name); it != inputs.end()) {
        return &*it;
    }
    if (auto const it = std::ranges::find_if(outputs, by_name); it != outputs.end()) {
        return &*it;
    }
    return nullptr;
}

ManifestEntry const *InterfaceManifest::find_by_id(TensorId id) const noexcept {
    auto const by_id = [id](ManifestEntry const &entry) { return entry.id == id; };
    if (auto const it = std::ranges::find_if(inputs, by_id); it != inputs.end()) {
        return &*it;
    }
    if (auto const it = std::ranges::find_if(outputs, by_id); it != outputs.end()) {
        return &*it;
    }
    return nullptr;
}

std::vector<ManifestEntry> InterfaceManifest::entries() const {
    std::vector<ManifestEntry> all = inputs;
    for (auto const &entry : outputs) {
        // An InOut slot is held in both vectors; keep it once.
        if (entry.direction != ManifestDirection::InOut) {
            all.push_back(entry);
        }
    }
    std::ranges::sort(all, [](ManifestEntry const &lhs, ManifestEntry const &rhs) {
        return lhs.name != rhs.name ? lhs.name < rhs.name : lhs.id < rhs.id;
    });
    return all;
}

std::vector<std::string> InterfaceManifest::names() const {
    std::vector<std::string> out;
    for (auto const &entry : entries()) {
        out.push_back(entry.name);
    }
    return out;
}

std::size_t InterfaceManifest::size() const noexcept {
    auto const inout = std::ranges::count_if(inputs, [](ManifestEntry const &e) { return e.direction == ManifestDirection::InOut; });
    return inputs.size() + outputs.size() - static_cast<std::size_t>(inout);
}

// ── Ownership scope plumbing ───────────────────────────────────────────────

void Graph::add_scope_map(TensorScopeMapPtr map) {
    if (!map) {
        return;
    }
    std::scoped_lock const lock(*_content_mutex);
    if (std::ranges::find(_scope_maps, map) != _scope_maps.end()) {
        return;
    }
    _scope_maps.push_back(std::move(map));

    // Handles registered before the table arrived would otherwise keep the
    // default scope forever; capture order is not something a caller controls.
    for (auto &[id, handle] : _tensors) {
        if (handle.tensor_ptr != nullptr && !handle.is_intermediate) {
            handle.ownership = scope_for_ptr(handle.tensor_ptr);
        }
    }
}

TensorOwnership Graph::scope_for_ptr(void const *ptr) const noexcept {
    if (ptr == nullptr) {
        return TensorOwnership::Graph;
    }
    // Last attached table wins, which is the narrower scope in the plumbing
    // Pipeline sets up (the workspace table is attached first).
    TensorOwnership scope = TensorOwnership::Graph;
    for (auto const &map : _scope_maps) {
        if (map == nullptr) {
            continue;
        }
        if (auto const it = map->find(ptr); it != map->end()) {
            scope = it->second;
        }
    }
    return scope;
}

// ── Graph::manifest ────────────────────────────────────────────────────────

InterfaceManifest Graph::manifest() {
    // A view registered outside capture only learns its parent here, and
    // ManifestEntry::aliases_input is derived from that link. Idempotent.
    link_alias_storage();

    UsageAnalysis const &uses = usage();

    std::vector<ManifestEntry> collected;
    for (auto const &[id, handle] : _tensors) {
        if (handle.is_intermediate) {
            continue;
        }
        TensorUsage const *use = uses.find(*this, id);
        if (use == nullptr) {
            // Referenced by no node, not even through a control-flow subtree.
            // Not part of the interface: registering a handle is not using it.
            continue;
        }

        std::size_t const reads  = use->reads();
        std::size_t const writes = use->writes();
        if (reads == 0 && writes == 0) {
            continue;
        }

        ManifestEntry entry;
        entry.id = id;
        // A slot bound once keeps the name the contract was written with; see
        // Graph::_interface_names for why the handle's own name cannot serve.
        auto const pinned = _interface_names.find(id);
        entry.name        = pinned != _interface_names.end() ? pinned->second : handle.name;
        entry.direction   = writes == 0 ? ManifestDirection::Input : reads == 0 ? ManifestDirection::Output : ManifestDirection::InOut;
        entry.dtype       = handle.dtype;
        entry.rank        = handle.rank;
        entry.dims        = handle.dims;
        entry.dim_symbols = handle.dim_symbols;
        entry.scope       = handle.ownership;

        entry.spaces.reserve(handle.spaces.size());
        for (SpaceId const space : handle.spaces) {
            if (!space.valid()) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': manifest of tensor '{}' (id {}) carries an invalid index-space id; the "
                                        "annotation is corrupt, not merely absent",
                                        _name, handle.name, id);
            }
            try {
                entry.spaces.emplace_back(space_registry().space(space).name);
            } catch (std::exception const &e) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': manifest of tensor '{}' (id {}) names index space {}, which this graph's "
                                        "registry cannot resolve ({}); a SpaceId is registry-local and means nothing anywhere else",
                                        _name, handle.name, id, space.value(), e.what());
            }
        }
        entry.spaces_inferred = handle.spaces_inferred;

        collected.push_back(std::move(entry));
    }

    // Alias declarations, once every entry is known: an alias of an
    // intermediate is the graph's own business and is not declared here.
    std::unordered_map<TensorId, std::size_t> index_of;
    for (std::size_t i = 0; i < collected.size(); ++i) {
        index_of.emplace(collected[i].id, i);
    }
    for (auto &entry : collected) {
        TensorHandle const *handle = find_tensor(entry.id);
        if (handle == nullptr || handle->aliases == 0) {
            continue;
        }
        if (index_of.contains(handle->aliases)) {
            entry.aliases_input = handle->aliases;
        }
    }

    std::ranges::sort(collected, [](ManifestEntry const &lhs, ManifestEntry const &rhs) {
        return lhs.name != rhs.name ? lhs.name < rhs.name : lhs.id < rhs.id;
    });

    // Binding is by name, so two entries sharing one is an ambiguity a caller
    // cannot resolve. Report it here rather than at bind time, where only one
    // of the two would ever be reachable.
    for (std::size_t i = 1; i < collected.size(); ++i) {
        if (collected[i].name == collected[i - 1].name) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': two interface tensors are both named '{}' (ids {} and {}); a manifest binds by "
                                    "name, so give one of them a distinct name",
                                    _name, collected[i].name, collected[i - 1].id, collected[i].id);
        }
    }

    InterfaceManifest contract;
    for (auto const &entry : collected) {
        if (entry.direction != ManifestDirection::Output) {
            contract.inputs.push_back(entry);
        }
        if (entry.direction != ManifestDirection::Input) {
            contract.outputs.push_back(entry);
        }
    }
    return contract;
}

std::vector<std::string> Graph::unbound_manifest_entries() {
    std::vector<std::string> out;
    for (auto const &entry : manifest().entries()) {
        if (!_bound_operands.contains(entry.id)) {
            out.push_back(entry.name);
        }
    }
    return out;
}

// ── Named gate-flag arrays ─────────────────────────────────────────────────

void Graph::name_gate_flags(std::string name, std::shared_ptr<std::vector<std::uint8_t>> buffer) {
    if (name.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': name_gate_flags: a gate-flag array's name must not be empty", _name);
    }
    if (buffer == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': name_gate_flags('{}'): the array is null", _name, name);
    }
    for (auto const &[existing_name, existing_buffer] : _named_gate_flags) {
        if (existing_name == name) {
            if (existing_buffer == buffer) {
                return; // the same registration, made twice
            }
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': name_gate_flags('{}'): that name already names a different gate-flag array", _name, name);
        }
        if (existing_buffer == buffer) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': name_gate_flags('{}'): this array is already named '{}'; one array carries one name, so "
                                    "that a saved file's name identifies it",
                                    _name, name, existing_name);
        }
    }
    _named_gate_flags.emplace_back(std::move(name), std::move(buffer));
    std::ranges::sort(_named_gate_flags, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
}

std::string Graph::gate_flag_name(std::shared_ptr<std::vector<std::uint8_t>> const &buffer) const {
    for (auto const &[name, candidate] : _named_gate_flags) {
        if (candidate == buffer) {
            return name;
        }
    }
    return {};
}

GateFlags Graph::gate_flags(std::string_view name) const {
    for (auto const &[candidate, buffer] : _named_gate_flags) {
        if (candidate == name) {
            return GateFlags::adopt(buffer);
        }
    }
    std::vector<std::string> known;
    known.reserve(_named_gate_flags.size());
    for (auto const &[candidate, buffer] : _named_gate_flags) {
        known.push_back(candidate);
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': no gate-flag array is named '{}'. Known names: [{}]", _name, name,
                            fmt::join(known, ", "));
}

// ── Graph::bind internals ──────────────────────────────────────────────────

void Graph::bind_scalar_impl(std::string const &name, void *storage, packed_gemm::ScalarType dtype, std::size_t element_size) {
    InterfaceManifest const contract = manifest();
    ManifestEntry const    &entry    = lookup_manifest_entry(contract, name);
    if (entry.rank != 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': bind_scalar('{}'): that slot is rank {}, not a scalar; bind a tensor to it with bind()", _name,
                                name, entry.rank);
    }
    if (dtype != entry.dtype) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind_scalar('{}'): dtype mismatch (given {}, interface declares {})",
                                _name, name, scalar_type_name(dtype), scalar_type_name(entry.dtype));
    }
    if (storage == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind_scalar('{}'): the storage pointer is null", _name, name);
    }

    TensorHandle &handle = tensor(entry.id);
    handle.tensor_ptr    = storage;
    handle.data_ptr      = storage;
    handle.element_size  = element_size;
    // The interface name is what the contract is written in; pin it exactly as
    // rebind() does, so a second bind_scalar of the same slot still finds it.
    _interface_names.emplace(entry.id, entry.name);
    // A rank-0 handle has no strided span to reason about, so it is recorded as
    // bound with an empty one: it counts as supplied and cannot participate in
    // the undeclared-aliasing check, which is the same treatment a deferred
    // shell gets.
    _bound_operands[entry.id] = BoundSpan{};
}

ManifestEntry const &Graph::lookup_manifest_entry(InterfaceManifest const &contract, std::string const &name) const {
    if (ManifestEntry const *entry = contract.find(name); entry != nullptr) {
        return *entry;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind('{}'): no such interface tensor. Known names: [{}]", _name, name,
                            fmt::join(contract.names(), ", "));
}

void Graph::validate_bind_shape(ManifestEntry const &entry, packed_gemm::ScalarType dtype, std::size_t rank,
                                std::vector<std::size_t> const &dims) const {
    if (dtype != entry.dtype) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind('{}'): dtype mismatch (given {}, interface declares {})", _name,
                                entry.name, scalar_type_name(dtype), scalar_type_name(entry.dtype));
    }
    if (rank != entry.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind('{}'): rank mismatch (given {}, interface declares {})", _name,
                                entry.name, rank, entry.rank);
    }
    for (std::size_t d = 0; d < rank; ++d) {
        // An axis the manifest gave a symbol is not checked against a number here: the
        // bind solver reconciles it against every other slot naming the same symbol, which
        // is the whole point of declaring one. An axis with no symbol is literal and its
        // extent is part of the contract, exactly as it always was.
        if (d < entry.dim_symbols.size() && !entry.dim_symbols[d].empty()) {
            continue;
        }
        if (dims[d] != entry.dims[d]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': bind('{}'): dim {} mismatch (given {}, interface declares {}). That axis is literal; "
                                    "declare it with annotate_dims if it is meant to vary between problems",
                                    _name, entry.name, d, dims[d], entry.dims[d]);
        }
    }
}

void Graph::validate_bind_spaces(ManifestEntry const &entry, std::vector<SpaceId> const &incoming) const {
    if (incoming.empty()) {
        return; // Unannotated: nothing to check, and silence is the contract.
    }

    std::vector<std::string> incoming_names;
    incoming_names.reserve(incoming.size());
    for (SpaceId const space : incoming) {
        if (!space.valid()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind('{}'): the given tensor carries an invalid index-space id",
                                    _name, entry.name);
        }
        incoming_names.emplace_back(space_registry().space(space).name);
    }

    if (incoming_names != entry.spaces) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': bind('{}'): index-space mismatch (given [{}], interface declares [{}])",
                                _name, entry.name, fmt::join(incoming_names, ", "), fmt::join(entry.spaces, ", "));
    }
}

void Graph::note_bound_operand(InterfaceManifest const &contract, ManifestEntry const &entry, BoundSpan const &span) {
    // Byte-span OVERLAP, not pointer equality. Two overlapping slices of one
    // buffer share no base address unless one happens to start where the other
    // does, so an identity test calls them unrelated and the hazard edges
    // between the two slots are silently lost - the same failure shape as the
    // full-cover alias bug, arriving through a different door.
    for (auto const &[other_id, other_span] : _bound_operands) {
        if (other_id == entry.id || !span.overlaps(other_span)) {
            continue;
        }
        ManifestEntry const *other = contract.find_by_id(other_id);
        if (other == nullptr) {
            continue;
        }
        bool const child_is_entry = entry.aliases_input == other->id;
        bool const child_is_other = other->aliases_input == entry.id;
        if (!child_is_entry && !child_is_other) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': bind('{}'): the given tensor overlaps the storage of the tensor already bound to "
                                    "'{}' by {} bytes, and the manifest declares no alias between them. Silently accepting an "
                                    "undeclared aliasing bind is what loses the hazard edges between the two slots; declare the "
                                    "relation on the manifest entry if it is intended",
                                    _name, entry.name, other->name, span.overlap_bytes(other_span));
        }
        // Declared, so install the link the declaration promises. A loaded graph
        // has no addresses and this pair has no View node, so this record is the
        // ONLY thing that can make the hazard scan order the two slots against
        // each other; leaving the declaration as documentation would accept the
        // bind and still race.
        declare_alias(child_is_entry ? entry.id : other->id, child_is_entry ? other->id : entry.id);
    }

    // Recorded whether or not it has an address to offer: an operand with no
    // address is still bound, it simply cannot participate in the check above.
    _bound_operands.insert_or_assign(entry.id, span);
    _interface_names.insert_or_assign(entry.id, entry.name);
}

EINSUMS_NAMESPACE_END(compute_graph)
