#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# PythonDemo — minimal real Einsums module annotated for the einsums-pybind
# autogen path. Phase 6's job is to prove the pipeline works end-to-end:
# einsums_add_module(... PYBIND ...) drives the codegen, the generated
# pybind11 binding TU compiles, and the resulting Python extension is
# importable.

set(PythonDemoHeaders
    Einsums/PythonDemo/Counter.hpp
    Einsums/PythonDemo/Coverage.hpp
)

set(PythonDemoSources
    Counter.cpp
    Coverage.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums PythonDemo
  PYBIND
  SOURCES ${PythonDemoSources}
  HEADERS ${PythonDemoHeaders}
  MODULE_DEPENDENCIES
    Einsums_Config
)

# End-to-end Python smoke test: import the generated extension and
# exercise enough of the binding to catch regressions in the autogen
# pipeline.
# End-to-end Python smoke test: import the aggregated `einsums` extension
# (built by einsums_finalize_pybind at the end of root configure) and
# exercise the PythonDemo bindings through it.
if(EINSUMS_BUILD_PYTHON AND EINSUMS_WITH_TESTS)
  add_test(NAME einsums_pybind_python_smoke
      COMMAND ${Python_EXECUTABLE} -c "
import einsums as m

# Counter — basic ctor/methods/getter+setter merged into property
c = m.Counter()
assert c.value == 0
assert c.bump() == 1
assert c.bump_by(5) == 6
c.reset()
assert c.value == 0
c2 = m.Counter(42, 'demo')
assert c2.value == 42 and c2.label == 'demo'
assert m.sum_of_squares(5) == 55

# Vec3 — fields + operator overloads (__add__, __mul__, __eq__)
v = m.Vec3(1.0, 2.0, 3.0)
assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0
v.x = 10.0; assert v.x == 10.0
sum_ = m.Vec3(1, 0, 0) + m.Vec3(0, 1, 0)
assert sum_ == m.Vec3(1, 1, 0)
scaled = m.Vec3(1, 2, 3) * 2.0
assert scaled == m.Vec3(2, 4, 6)
# z is read-only via @readonly
try:
    v.z = 99.0
    assert False, 'z should be read-only'
except (AttributeError, TypeError):
    pass

# Resource — shared_ptr holder + reference_internal RVP
r = m.Resource('main', 4)
assert r.label == 'main'
slot2 = r.slot(2)
assert slot2.value == 2
del r       # parent gone, but slot2 should still be alive via reference_internal
assert slot2.value == 2

# Worker — release_gil call
w = m.Worker()
assert w.crunch(100) == 5050

# Tensor — the real Einsums tensor binding. Only RuntimeTensor is bound now;
# typed Tensor<T, Rank> bindings were removed in favor of a single runtime-rank
# surface so the (T) x (Rank) cross-product collapses to (T) x (Alloc).
t = m.RuntimeTensorD('weights', [3, 4])
assert t.dim(0) == 3 and t.dim(1) == 4
assert t.size == 12 and t.rank() == 2

# Submodule + exception (Phase 9c)
assert hasattr(m, 'demo'), 'expected einsums.demo submodule'
assert hasattr(m.demo, 'CounterError'), 'expected einsums.demo.CounterError'
assert hasattr(m.demo, 'raise_counter_error'), 'expected einsums.demo.raise_counter_error'
try:
    m.demo.raise_counter_error('boom')
    assert False, 'expected CounterError'
except m.demo.CounterError as e:
    assert 'boom' in str(e)

print('OK')
"
  )
  # Under a sanitizer build, preload the runtime so the dlopened _core's
  # interceptors install in time (see einsums_add_python_unit_test and the root
  # CMakeLists for EINSUMS_PYTEST_SANITIZER_PRELOAD).
  set(_smoke_env "PYTHONPATH=${CMAKE_BINARY_DIR}/lib")
  if(EINSUMS_PYTEST_SANITIZER_PRELOAD)
    set(_smoke_env "${_smoke_env};${EINSUMS_PYTEST_SANITIZER_PRELOAD}")
    if(EINSUMS_WITH_SANITIZERS MATCHES "address")
      set(_smoke_env "${_smoke_env};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1")
    endif()
  endif()
  set_tests_properties(einsums_pybind_python_smoke PROPERTIES
      ENVIRONMENT "${_smoke_env}"
  )
endif()
