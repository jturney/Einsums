#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# * v1.10.4-6-EAT6 on EAT branch is upstream v1.10.4-6 tag +2, so last upstream tag
#   plus extended array dimensions to support higher rank tensors plus deleter stuff.
#   * v1.10.4-6+3 Oct 2023 redirect aligned_alloc to omp_aligned_alloc
# * find_package() is disabled since we need patched source
# * upstream CMakeLists.txt isn't usable and project is header-only, so to keep code
#   changes and build changes separate, we won't let FetchContent build (`SOURCE_SUBDIR
#   fake`) and will create the interface Einsums_h5cpp target after download.
# * MakeAvailable called here so that install (of vendored headers into einsums namespace)
#   can be localized into this file.

find_package(ZLIB REQUIRED)
find_package(TargetHDF5 REQUIRED)
