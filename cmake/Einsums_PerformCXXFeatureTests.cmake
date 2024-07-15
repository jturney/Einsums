#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# ##################################################################################################
# C++ feature tests
# ##################################################################################################
function(einsums_perform_cxx_feature_tests)
  einsums_check_for_cxx11_std_atomic(REQUIRED "einsums needs support for C++11 std::atomic")

  # Separately check for 128 bit atomics
  einsums_check_for_cxx11_std_atomic_128bit(DEFINITIONS EINSUMS_HAVE_CXX11_STD_ATOMIC_128BIT)

  einsums_check_for_cxx11_std_quick_exit(DEFINITIONS EINSUMS_HAVE_CXX11_STD_QUICK_EXIT)

  einsums_check_for_cxx11_std_shared_ptr_lwg3018(
    DEFINITIONS EINSUMS_HAVE_CXX11_STD_SHARED_PTR_LWG3018
  )

  einsums_check_for_c11_aligned_alloc(DEFINITIONS EINSUMS_HAVE_C11_ALIGNED_ALLOC)

  einsums_check_for_cxx17_std_aligned_alloc(DEFINITIONS EINSUMS_HAVE_CXX17_STD_ALIGNED_ALLOC)

  einsums_check_for_cxx17_aligned_new(DEFINITIONS EINSUMS_HAVE_CXX17_ALIGNED_NEW)

  einsums_check_for_cxx17_copy_elision(DEFINITIONS EINSUMS_HAVE_CXX17_COPY_ELISION)

  einsums_check_for_cxx17_memory_resource(DEFINITIONS EINSUMS_HAVE_CXX17_MEMORY_RESOURCE)

  # C++20 feature tests
  einsums_check_for_cxx20_no_unique_address_attribute(
    DEFINITIONS EINSUMS_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
  )

  einsums_check_for_cxx20_std_disable_sized_sentinel_for(
    DEFINITIONS EINSUMS_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
  )

  einsums_check_for_cxx20_trivial_virtual_destructor(
    DEFINITIONS EINSUMS_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR
  )

  einsums_check_for_cxx23_static_call_operator(DEFINITIONS EINSUMS_HAVE_CXX23_STATIC_CALL_OPERATOR)

  einsums_check_for_cxx23_static_call_operator_gpu(
    DEFINITIONS EINSUMS_HAVE_CXX23_STATIC_CALL_OPERATOR_GPU
  )

  einsums_check_for_cxx_lambda_capture_decltype(
    DEFINITIONS EINSUMS_HAVE_CXX_LAMBDA_CAPTURE_DECLTYPE
  )
endfunction()
