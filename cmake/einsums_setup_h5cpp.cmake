#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Find h5cpp dependencies
find_package(ZLIB REQUIRED)
find_package(TargetHDF5 REQUIRED)

# add_subdirectory(${PROJECT_SOURCE_DIR}/external/h5cpp) * v1.10.4-6-EAT6 on EAT branch is upstream v1.10.4-6 tag +2, so
# last upstream tag plus extended array dimensions to support higher rank tensors plus deleter stuff. * v1.10.4-6+3 Oct
# 2023 redirect aligned_alloc to omp_aligned_alloc * find_package() is disabled since we need patched source * upstream
# CMakeLists.txt isn't useable and project is header-only, so to keep code changes and build changes separate, we won't
# let FetchContent build (`SOURCE_SUBDIR fake`) and will create the interface Einsums_h5cpp target after download. *
# MakeAvailable called here so that install (of vendored headers into einsums namespace) can be localized into this
# file.

fetchcontent_declare(
    h5cpp
    URL https://github.com/Einsums/h5cpp/archive/v1.10.4-6+3.tar.gz
    URL_HASH SHA256=c790e38b3251fc2114c452858c85c86659aacb6037f24901bed55b72f6e2af03
    SOURCE_SUBDIR fake SYSTEM OVERRIDE_FIND_PACKAGE
)

fetchcontent_makeavailable(h5cpp)

include(einsums_add_module)
einsums_add_module(einsums h5cpp)
# add_library("${ein}::h5cpp" ALIAS Einsums_h5cpp)

if(EINSUMS_H5CPP_USE_OMP_ALIGNED_ALLOC)
    target_compile_definitions(einsums_h5cpp INTERFACE $<BUILD_INTERFACE:H5CPP_USE_OMP_ALIGNED_ALLOC>)
endif()
set_target_properties(einsums_h5cpp PROPERTIES EXPORT_NAME h5cpp)
target_include_directories(
    einsums_h5cpp
    # SYSTEM suppresses "error: non-constant-expression cannot be narrowed" for some compilers
    SYSTEM
    INTERFACE $<BUILD_INTERFACE:${h5cpp_SOURCE_DIR}>
              # TODO return to this when build headers adjusted   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/einsums>
)
target_link_libraries(einsums_h5cpp INTERFACE tgt::hdf5 ZLIB::ZLIB)

# install( DIRECTORY ${h5cpp_SOURCE_DIR}/h5cpp COMPONENT ${ein}_Development DESTINATION
# ${CMAKE_INSTALL_INCLUDEDIR}/einsums )

target_link_libraries(einsums_base_libraries INTERFACE einsums_h5cpp)
