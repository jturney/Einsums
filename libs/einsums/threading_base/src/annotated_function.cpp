//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
#    include <einsums/threading_base/annotated_function.hpp>

#    include <string>
#    include <unordered_set>
#    include <utility>

namespace einsums::detail {
char const *store_function_annotation(std::string name) {
    static thread_local std::unordered_set<std::string> names;
    auto                                                r = names.emplace(std::move(name));
    return (*std::get<0>(r)).c_str();
}
} // namespace einsums::detail

#else

#    include <string>

namespace einsums::detail {
char const *store_function_annotation(std::string) {
    return "<unknown>";
}
} // namespace einsums::detail

#endif
