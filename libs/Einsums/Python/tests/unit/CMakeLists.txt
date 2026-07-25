#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Python-only module: the package shell and the rc -> argv translation that
# starts the runtime. No C++ unit tests here.
if(EINSUMS_BUILD_PYTHON)
  einsums_add_python_unit_test("Modules.Python" CliOptionsPython SCRIPT test_cli_options_python.py)
endif()
