//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

void FactorizationRegistry::add(std::shared_ptr<FactorizationProvider> provider) {
    if (provider == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "FactorizationRegistry::add: the provider is null");
    }
    std::string const name = provider->name();
    auto const        clash =
        std::ranges::find_if(_providers, [&name](std::shared_ptr<FactorizationProvider> const &held) { return held->name() == name; });
    if (clash != _providers.end()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "FactorizationRegistry: a provider named '{}' is already registered (it claims tag '{}'). Two under one "
                                "name would make which of them ran depend on registration order",
                                name, (*clash)->tag());
    }
    _providers.push_back(std::move(provider));
}

std::vector<std::shared_ptr<FactorizationProvider>> FactorizationRegistry::for_tag(std::string_view tag) const {
    std::vector<std::shared_ptr<FactorizationProvider>> out;
    for (auto const &provider : _providers) {
        if (provider->tag() == tag) {
            out.push_back(provider);
        }
    }
    return out;
}

bool FactorizationRegistry::claims(std::string_view tag) const {
    return std::ranges::any_of(_providers, [tag](std::shared_ptr<FactorizationProvider> const &held) { return held->tag() == tag; });
}

bool FactorizationRegistry::remove(std::string_view name) {
    auto const found =
        std::ranges::find_if(_providers, [name](std::shared_ptr<FactorizationProvider> const &held) { return held->name() == name; });
    if (found == _providers.end()) {
        return false;
    }
    _providers.erase(found);
    return true;
}

void FactorizationRegistry::clear() {
    _providers.clear();
}

std::size_t FactorizationRegistry::size() const {
    return _providers.size();
}

FactorizationRegistry &global_factorization_registry() {
    static FactorizationRegistry registry;
    return registry;
}

EINSUMS_NAMESPACE_END(compute_graph)
