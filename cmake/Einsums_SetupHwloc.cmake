find_package(Hwloc REQUIRED)
if (NOT HWLOC_FOUND)
    einsums_error("HWLOC could not be found, please specify HWLOC_ROOT to point to the correct location.")
endif ()
