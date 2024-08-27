# * v1.10.4-6-EAT6 on EAT branch is upstream v1.10.4-6 tag +2, so last upstream tag
#   plus extended array dimensions to support higher rank tensors plus deleter stuff.
#   * v1.10.4-6+3 Oct 2023 redirect aligned_alloc to omp_aligned_alloc
# * find_package() is disabled since we need patched source
# * upstream CMakeLists.txt isn't useable and project is header-only, so to keep code
#   changes and build changes separate, we won't let FetchContent build (`SOURCE_SUBDIR
#   fake`) and will create the interface Einsums_h5cpp target after download.
# * MakeAvailable called here so that install (of vendored headers into einsums namespace)
#   can be localized into this file.

find_package(ZLIB REQUIRED)
find_package(TargetHDF5 REQUIRED)

include(FetchContent)
FetchContent_Declare(
        h5cpp
        URL https://github.com/Einsums/h5cpp/archive/v1.10.4-6+4.tar.gz
        URL_HASH SHA256=f9acf8d35ac1584d0c0f36471abc3faaa3e181d2713950b1af21e4f4bc0f9991
)

FetchContent_GetProperties(h5cpp)
if (NOT h5cpp_POPULATED)
    FetchContent_Populate(h5cpp)
endif()
set(h5cpp_ROOT ${h5cpp_SOURCE_DIR})


add_library(Einsums_h5cpp INTERFACE)
add_library("Einsums::h5cpp" ALIAS Einsums_h5cpp)

if (EINSUMS_H5CPP_USE_OMP_ALIGNED_ALLOC)
    target_compile_definitions(
            Einsums_h5cpp
            INTERFACE
            $<BUILD_INTERFACE:H5CPP_USE_OMP_ALIGNED_ALLOC>
    )
endif()
set_target_properties(
        Einsums_h5cpp
        PROPERTIES
        EXPORT_NAME h5cpp
)
target_include_directories(
        Einsums_h5cpp
        # SYSTEM suppresses "error: non-constant-expression cannot be narrowed" for some compilers
        SYSTEM
        INTERFACE
        $<BUILD_INTERFACE:${h5cpp_SOURCE_DIR}>
        # TODO return to this when build headers adjusted   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/einsums>
)
target_link_libraries(
        Einsums_h5cpp
        INTERFACE
        tgt::hdf5
        ZLIB::ZLIB
)

install(
        TARGETS Einsums_h5cpp
        EXPORT EinsumsH5cppTarget
        COMPONENT core
)

install(
        DIRECTORY
        ${h5cpp_SOURCE_DIR}/h5cpp
        COMPONENT core
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsums
)

export(
        TARGETS Einsums_h5cpp
        NAMESPACE Einsums::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/Einsums/EinsumsH5cppTarget.cmake"
)

install(
        EXPORT EinsumsH5cppTarget
        NAMESPACE Einsums::
        FILE EinsumsH5cppTarget.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Einsums
        COMPONENT cmake
)


#install(
#        TARGETS Einsums_h5cpp
#        EXPORT Einsums_h5cpp
#        COMPONENT Einsums_Development
#)
#
#install(
#        DIRECTORY
#        ${h5cpp_SOURCE_DIR}/h5cpp
#        COMPONENT ${ein}_Development
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsums
#)
#
#export(
#        TARGETS Einsums_h5cpp
#        NAMESPACE Einsums::
#        FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/Einsums/EinsumsH5cppTarget.cmake"
#)
