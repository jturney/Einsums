#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(PackedGemmHeaders
    Einsums/PackedGemm/ContractionKey.hpp
    Einsums/PackedGemm/EinsumPackedGemm.hpp
    Einsums/PackedGemm/InitModule.hpp
    Einsums/PackedGemm/Packing.hpp
)

set(PackedGemmSources
    ContractionKey.cpp
    InitModule.cpp
    Packing.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums PackedGemm
  SOURCES ${PackedGemmSources}
  HEADERS ${PackedGemmHeaders}
  MODULE_DEPENDENCIES
    Einsums_BLAS
    Einsums_Config
    Einsums_Concepts
    Einsums_Logging
    Einsums_Profile
    Einsums_Runtime
    Einsums_HPTT
  CMAKE_SUBDIRS tests
)
