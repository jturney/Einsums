#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(ComputeGraphTypesHeaders
    Einsums/ComputeGraphTypes.hpp
    Einsums/ComputeGraphTypes/Enums.hpp
    Einsums/ComputeGraphTypes/Ids.hpp
    Einsums/ComputeGraphTypes/Descriptors.hpp
    Einsums/ComputeGraphTypes/GraphData.hpp
    Einsums/ComputeGraphTypes/Spaces.hpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums ComputeGraphTypes
  PYBIND
  HEADERS ${ComputeGraphTypesHeaders}
  # Config carries <Einsums/Config/Namespace.hpp>, which every namespace block
  # in the tree now opens with. This module had no dependencies at all, so its
  # apiary codegen ran without Config on the include path.
  MODULE_DEPENDENCIES Einsums_Config
)
