//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Config/Version.hpp>
#include <Einsums/Preprocessor/Stringify.hpp>

#include <string_view>

// The only TU that sees the git identity, so only it recompiles when HEAD moves.
#include "GitInfo.hpp"

EINSUMS_NAMESPACE_BEGIN()

char const EINSUMS_CHECK_VERSION[] = EINSUMS_PP_STRINGIFY(EINSUMS_CHECK_VERSION);

std::string_view git_commit() {
    return EINSUMS_GIT_COMMIT_STRING;
}

std::string_view git_branch() {
    return EINSUMS_GIT_BRANCH_STRING;
}

bool git_dirty() {
    return EINSUMS_GIT_DIRTY_FLAG != 0;
}

EINSUMS_NAMESPACE_END()
