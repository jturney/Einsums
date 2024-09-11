#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(SPDLOG_INSTALL TRUE)
set(SPDLOG_FMT_EXTERNAL TRUE)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.x
)

FetchContent_MakeAvailable(spdlog)

target_link_libraries(einsums_base_libraries INTERFACE spdlog::spdlog)