#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PreprocessorHeaders Einsums/Preprocessor/Cat.hpp Einsums/Preprocessor/Expand.hpp
                        Einsums/Preprocessor/NArgs.hpp Einsums/Preprocessor/Stringify.hpp
                        Einsums/Preprocessor/Unused.hpp
)

set(PreprocessorSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Preprocessor
  SOURCES ${PreprocessorSources}
  HEADERS ${PreprocessorHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
