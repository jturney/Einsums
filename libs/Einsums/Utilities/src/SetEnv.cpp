//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Utilities/SetEnv.hpp>

#include <cstdlib>

namespace einsums {

void set_env_var(std::string const &name, std::string const &value) {
#if defined(EINSUMS_WINDOWS)
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), /*overwrite=*/1);
#endif
}

void unset_env_var(std::string const &name) {
#if defined(EINSUMS_WINDOWS)
    _putenv_s(name.c_str(), "");
#else
    unsetenv(name.c_str());
#endif
}

} // namespace einsums
