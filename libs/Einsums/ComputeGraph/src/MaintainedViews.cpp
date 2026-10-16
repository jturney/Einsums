//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/MaintainedViews.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {
struct Registry {
    std::mutex                                            mutex;
    std::unordered_map<void const *, std::weak_ptr<void>> views;
};

Registry &registry() {
    static Registry instance;
    return instance;
}
} // namespace

void note_maintained_view(void const *view, std::weak_ptr<void> life) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.views.insert_or_assign(view, std::move(life));
}

void forget_maintained_view(void const *view) noexcept {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    reg.views.erase(view);
}

bool is_maintained_view(void const *view) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    return reg.views.contains(view);
}

std::optional<std::weak_ptr<void>> maintained_view_life(void const *view) {
    auto                  &reg = registry();
    std::scoped_lock const lock(reg.mutex);
    auto const             hit = reg.views.find(view);
    return hit != reg.views.end() ? std::optional<std::weak_ptr<void>>{hit->second} : std::nullopt;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
