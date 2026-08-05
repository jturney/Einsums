#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# FindTargetLAPACK.cmake
# ----------------------
#
# LAPACK cmake module to wrap FindLAPACK.cmake in a target.
#
# This module sets the following variables in your project: ::
#
# TargetLAPACK_FOUND - true if BLAS/LAPACK found on the system TargetLAPACK_MESSAGE - status message
# with BLAS/LAPACK library path list
#
# This module *unsets* the following conventional LAPACK variables so as to force using the target:
# ::
#
# LAPACK_FOUND LAPACK_LIBRARIES
#
# In order of decreasing precedence, this module returns in a target ``tgt::lapack`` (1) the
# libraries passed through CMake variable LAPACK_LIBRARIES, (2) the libraries defined in a
# detectable TargetLAPACKConfig.cmake file (skip via CMAKE_DISABLE_FIND_PACKAGE_TargetLAPACK), or
# (3) the libraries detected by the usual FindLAPACK.cmake module.
#
# Einsums specialization * sets property VENDOR for MKL-ness * sets property INT_INTERFACE for
# lp64/ilp64 * uses targets, not variables, from FindLAPACK (CMake 3.18)

set(PN TargetLAPACK)

# 1st precedence - libraries passed in through -DLAPACK_LIBRARIES
if(LAPACK_LIBRARIES)
  if(NOT ${PN}_FIND_QUIETLY)
    message(STATUS "LAPACK detection suppressed.")
  endif()

  set(_VENDOR "All")
  foreach(_l IN LISTS LAPACK_LIBRARIES)
    get_filename_component(_lname ${_l} NAME)
    if(${_lname} MATCHES "mkl")
      set(_VENDOR "MKL")
      break()
    elseif(${_lname} MATCHES "openblas")
      set(_VENDOR "OpenBLAS")
      break()
    endif()
  endforeach()

  add_library(tgt::lapack INTERFACE IMPORTED)
  set_property(TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
  set_property(TARGET tgt::lapack PROPERTY VENDOR ${_VENDOR})
  set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE lp64) # TODO assumption!
else()
  # 2nd precedence - target already prepared and findable in TargetLAPACKConfig.cmake
  if(NOT "${CMAKE_DISABLE_FIND_PACKAGE_${PN}}")
    find_package(TargetLAPACK QUIET CONFIG)
  endif()
  if(TARGET tgt::lapack)
    if(NOT ${PN}_FIND_QUIETLY)
      message(STATUS "TargetLAPACKConfig detected.")
    endif()
  else()
    # 3rd precedence - usual variables from FindLAPACK.cmake
    if(NOT MSVC)
      find_package(MKL CONFIG)
    endif()
    if(TARGET MKL::MKL)
      add_library(tgt::lapack INTERFACE IMPORTED)
      set_property(TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES MKL::MKL)
      set_property(TARGET tgt::lapack PROPERTY VENDOR "MKL")
      set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE ${MKL_INTERFACE})

    else()
      set(BLA_VENDOR OpenBLAS)
      set(BLA_SIZEOF_INTEGER 4)
      find_package(BLAS MODULE)
      find_package(LAPACK MODULE)

      if((TARGET BLAS::BLAS) AND (TARGET LAPACK::LAPACK))
        add_library(tgt::lapack INTERFACE IMPORTED)
        set_property(TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES LAPACK::LAPACK BLAS::BLAS)
        set_property(TARGET tgt::lapack PROPERTY VENDOR ${BLA_VENDOR})
        set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE lp64)

      else()
        set(BLA_VENDOR OpenBLAS)
        set(BLA_SIZEOF_INTEGER 8)
        find_package(BLAS MODULE)
        find_package(LAPACK MODULE)

        if((TARGET BLAS::BLAS) AND (TARGET LAPACK::LAPACK))
          add_library(tgt::lapack INTERFACE IMPORTED)
          set_property(
            TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES LAPACK::LAPACK BLAS::BLAS
          )
          set_property(TARGET tgt::lapack PROPERTY VENDOR ${BLA_VENDOR})
          set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE ilp64)

        else()
          set(BLA_VENDOR Generic)
          set(BLA_SIZEOF_INTEGER ANY)
          find_package(BLAS MODULE)
          find_package(LAPACK MODULE)

          if((TARGET BLAS::BLAS) AND (TARGET LAPACK::LAPACK))
            add_library(tgt::lapack INTERFACE IMPORTED)
            set_property(
              TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES LAPACK::LAPACK BLAS::BLAS
            )
            set_property(TARGET tgt::lapack PROPERTY VENDOR ${BLA_VENDOR})
            set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE lp64) # TODO assumption!

          else()
            set(BLA_VENDOR All)
            find_package(BLAS REQUIRED MODULE)
            find_package(LAPACK REQUIRED MODULE)

            add_library(tgt::lapack INTERFACE IMPORTED)
            set_property(
              TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES LAPACK::LAPACK BLAS::BLAS
            )
            set_property(TARGET tgt::lapack PROPERTY VENDOR ${BLA_VENDOR})
            set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE lp64) # TODO assumption!
          endif()
        endif()
      endif()

    endif()

    # The VENDOR set above is BLA_VENDOR, which records what the search asked
    # for rather than what it found. They diverge on Windows: find_package(MKL
    # CONFIG) is skipped under MSVC, the OpenBLAS and Generic passes then fail,
    # and the "All" pass succeeds against MKL while reporting "All". Consumers
    # keyed on the vendor silently take the wrong branch - EINSUMS_HAVE_MKL is
    # never defined, so MKL's xerbla is not routed into our logger, and the FFT
    # default falls back to fftw3 on a build that links MKL. Re-derive it from
    # the libraries actually resolved, matching how the LAPACK_LIBRARIES branch
    # at the top of this file names the vendor.
    if(TARGET tgt::lapack)
      foreach(_l IN LISTS LAPACK_LIBRARIES BLAS_LIBRARIES)
        get_filename_component(_lname "${_l}" NAME)
        if(_lname MATCHES "[mM][kK][lL]")
          set_property(TARGET tgt::lapack PROPERTY VENDOR "MKL")
          break()
        elseif(_lname MATCHES "[oO][pP][eE][nN][bB][lL][aA][sS]")
          set_property(TARGET tgt::lapack PROPERTY VENDOR "OpenBLAS")
          break()
        endif()
      endforeach()
    endif()

    if(NOT ${PN}_FIND_QUIETLY)
      message(STATUS "LAPACK detected.")
    endif()

    unset(BLA_VENDOR)
    unset(BLA_SIZEOF_INTEGER)
    unset(LAPACK_FOUND)
    unset(LAPACK_LIBRARIES)
  endif()
endif()

get_property(
  _ill
  TARGET tgt::lapack
  PROPERTY INTERFACE_LINK_LIBRARIES
)
get_property(
  _ven
  TARGET tgt::lapack
  PROPERTY VENDOR
)
get_property(
  _int
  TARGET tgt::lapack
  PROPERTY INT_INTERFACE
)
set(${PN}_MESSAGE "Found LAPACK ${_ven}w/${_int}: ${_ill}")
if((TARGET tgt::blas) AND (TARGET tgt::lapk))
  get_property(
    _illb
    TARGET tgt::blas
    PROPERTY INTERFACE_LINK_LIBRARIES
  )
  get_property(
    _illl
    TARGET tgt::lapk
    PROPERTY INTERFACE_LINK_LIBRARIES
  )
  set(${PN}_MESSAGE "Found LAPACK ${_ven}w/${_int}: ${_illl};${_illb}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PN} DEFAULT_MSG ${PN}_MESSAGE)
