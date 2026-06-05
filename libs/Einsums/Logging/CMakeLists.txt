#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LoggingHeaders Einsums/Logging.hpp)

set(LoggingSources Logging.cpp)

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)

if(BUILD_TYPE_LOWER STREQUAL "debug")
  set(EINSUMS_ACTIVE_LOG_LEVEL 0)
elseif(BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
  set(EINSUMS_ACTIVE_LOG_LEVEL 1)
else()
  set(EINSUMS_ACTIVE_LOG_LEVEL 2)
endif()

einsums_add_config_cond_define_namespace(
  DEFINE EINSUMS_ACTIVE_LOG_LEVEL VALUE ${EINSUMS_ACTIVE_LOG_LEVEL} NAMESPACE LOGGING
)

set(LoggingDeps spdlog::spdlog)
if (TARGET Tracy::TracyClient)
  list(APPEND LoggingDeps Tracy::TracyClient)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Logging
  PYBIND
  SOURCES ${LoggingSources}
  HEADERS ${LoggingHeaders}
  DEPENDENCIES ${LoggingDeps}
  MODULE_DEPENDENCIES Einsums_Assertion Einsums_Config Einsums_StringUtil
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
target_compile_definitions(
  Einsums_Logging PUBLIC -DEINSUMS_ACTIVE_LOG_LEVEL=${EINSUMS_ACTIVE_LOG_LEVEL}
)
