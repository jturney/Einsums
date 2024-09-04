#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(concepts_headers
        einsums/concepts/complex.hpp
        einsums/concepts/file.hpp
        einsums/concepts/smart_pointer.hpp
        einsums/concepts/tensor.hpp
)

set(concepts_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums concepts
        GLOBAL_HEADER_GEN ON
        SOURCES ${concepts_sources}
        HEADERS ${concepts_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_tensor_base
        CMAKE_SUBDIRS examples tests
)
