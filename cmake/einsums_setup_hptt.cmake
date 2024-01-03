#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

add_subdirectory(${PROJECT_SOURCE_DIR}/external/hptt)

target_link_libraries(einsums_base_libraries INTERFACE hptt::hptt)

einsums_export_internal_targets(hptt)
