#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ConceptsHeaders
    Einsums/Concepts/Complex.hpp Einsums/Concepts/File.hpp Einsums/Concepts/SmartPointer.hpp
    Einsums/Concepts/TensorConcepts.hpp Einsums/Concepts/SubscriptChooser.hpp
    Einsums/Concepts/NamedRequirements.hpp Einsums/Concepts/Scalar.hpp
)

set(ConceptsSources CheckRequirements.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Concepts
  SOURCES ${ConceptsSources}
  HEADERS ${ConceptsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Preprocessor Einsums_TensorBase
  CMAKE_SUBDIRS examples tests
)
