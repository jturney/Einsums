//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/DescriptorRegistry.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {
struct Registry {
    std::shared_mutex                                   mutex;
    std::map<std::string, DescriptorCodec, std::less<>> codecs; // node-based, so a returned pointer stays valid
};

Registry &registry() {
    static Registry instance;
    return instance;
}
} // namespace

void register_descriptor(DescriptorCodec codec) {
    if (codec.name.find('.') == std::string::npos) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "register_descriptor: '{}' is not qualified; a registered descriptor's name must contain a '.', as in "
                                "'library.Name', so it cannot collide with one of Einsums' own descriptors",
                                codec.name);
    }
    if (!codec.write || !codec.read || !codec.build) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "register_descriptor: '{}' needs all three of write, read and build", codec.name);
    }
    auto                               &reg = registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    auto [it, inserted] = reg.codecs.try_emplace(codec.name, std::move(codec));
    if (!inserted) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "register_descriptor: '{}' is already registered", it->first);
    }
}

DescriptorHooks const *descriptor_hooks(Node const &node) noexcept {
    if (node.kind != OpKind::Custom) {
        return nullptr;
    }
    // Every registered name is qualified and no library descriptor's is, so an unqualified name
    // is answered without taking the lock.
    std::string_view const name = node.op_data.name();
    if (name.find('.') == std::string_view::npos) {
        return nullptr;
    }
    DescriptorCodec const *codec = find_descriptor_codec(name);
    return codec != nullptr ? &codec->hooks : nullptr;
}

NodeEffects effects_of(Node const &node) {
    if (DescriptorHooks const *hooks = descriptor_hooks(node); hooks != nullptr && hooks->effects) {
        return hooks->effects(node.op_data);
    }
    return node.effects;
}

DescriptorCodec const *find_descriptor_codec(std::string_view name) noexcept {
    auto                               &reg = registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    auto const                          it = reg.codecs.find(name);
    return it != reg.codecs.end() ? &it->second : nullptr;
}

EINSUMS_NAMESPACE_END(compute_graph)
