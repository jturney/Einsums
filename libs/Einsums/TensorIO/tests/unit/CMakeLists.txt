#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TensorIOTests
    TensorFileBasic.cpp
    TensorFileSliceWrite.cpp
    GraphSlab.cpp
)

foreach(test ${TensorIOTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${test}
    NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.TensorIO" ${TestName}
  )
endforeach()

einsums_add_python_unit_test(
  "Modules.TensorIO" TensorFilePython
  SCRIPT test_tensor_file_python.py
)

einsums_add_python_unit_test(
  "Modules.TensorIO" GraphSlabPython
  SCRIPT test_graph_slab_python.py
)
