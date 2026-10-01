..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_SIMD:

****
SIMD
****

The ``SIMD`` module provides a portable, header-only SIMD abstraction for
vectorized operations. It wraps platform-specific intrinsics behind a clean
C++20 interface that works across:

- x86_64: SSE2, SSSE3, SSE4.1/4.2, AVX, AVX2, and AVX-512.
- ARM: NEON on Apple Silicon and other aarch64 targets.

The module is used internally by the HPTT transpose library and can be used
directly for performance-critical inner loops.

Platform Detection
==================

.. code-block:: cpp

    #include <Einsums/SIMD/Platform.hpp>

    using namespace einsums::simd;

    // Compile-time constants
    static_assert(native_bits == 128 || native_bits == 256 || native_bits == 512);
    static_assert(has_neon || has_sse2);  // At least one must be true

    // Number of elements that fit in a native register
    constexpr size_t float_lanes = native_lanes<float>;   // 4 (SSE), 8 (AVX), 16 (AVX-512)
    constexpr size_t double_lanes = native_lanes<double>;  // 2, 4, or 8

Core Vec Type
=============

``Vec<T>`` wraps a platform SIMD register for type ``T``:

.. code-block:: cpp

    #include <Einsums/SIMD/Vec.hpp>
    #include <Einsums/SIMD/Operations.hpp>

    using namespace einsums::simd;

    // Broadcast a scalar to all lanes
    Vec<float> a = broadcast<float>(3.14f);

    // Load from memory (aligned or unaligned)
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Vec<float> b = loadu<float>(data);

    // Arithmetic
    Vec<float> c = add(a, b);       // or: a + b
    Vec<float> d = mul(a, b);       // or: a * b
    Vec<float> e = fmadd(a, b, c);  // a*b + c (fused multiply-add)

    // Store back to memory
    storeu(data, c);

Arithmetic, Comparisons and Masks
=================================

Beyond ``add``, ``sub``, ``mul`` and ``fmadd``, ``Operations.hpp`` has ``div``,
``sqrt``, ``min``, ``max``, ``abs`` and ``neg`` for ``float`` and ``double``,
and the operators ``/`` and unary ``-``. ``min(a, b)`` is exactly
``a < b ? a : b``, so a NaN on either side, or two zeros of either sign, give
``b`` on every backend. ``abs`` and ``neg`` change only the sign bit.

``fmsub(a, b, c)``, ``fnmadd(a, b, c)`` and ``fnmsub(a, b, c)`` are
``a*b - c``, ``-(a*b) + c`` and ``-(a*b) - c``, with the x86 names. Like
``fmadd`` they round once on aarch64 and wherever ``has_fma`` is true, and
multiply and add separately elsewhere, so a kernel that depends on the single
rounding can check ``has_fma`` at compile time.

``floor``, ``ceil``, ``trunc`` and ``round`` match their ``std::`` namesakes
lane for lane, signed zeros included; ``round`` takes ties away from zero, as
``std::round`` does, and ``round_even`` takes them to the even neighbour. The
result is still floating point; ``convert`` turns it into an integer. The
SSE2 baseline has no rounding instruction and emulates one with an add and a
subtract of 2^23 or 2^52, which is wrong under ``-ffast-math`` or any other
reassociating mode.

A comparison returns a ``Mask<T>``, one flag per lane, held as the hardware
holds it: a k-register on AVX-512, a vector of all-ones or zero lanes on AVX,
SSE and NEON. The comparisons follow IEEE 754, so every one involving a NaN is
false except ``cmp_ne``.

.. code-block:: cpp

    Mask<float> const too_big = cmp_gt(v, limit);         // lanes where v > limit
    Vec<float> const  clamped = select(too_big, limit, v); // limit there, v elsewhere
    if (any(cmp_ne(v, v))) {                               // is any lane a NaN?
        ...
    }

``select(mask, a, b)`` takes ``a`` where the mask is set, for ``float``,
``double`` and the 32- and 64-bit integers. Masks combine with ``&``, ``|``,
``^`` and ``!`` (or ``bitwise_and``, ``bitwise_or``, ``bitwise_xor`` and
``bitwise_andnot``), and ``any``, ``all``, ``none`` and ``count`` reduce one to
a ``bool`` or a lane count. Write ``!m``, not ``~m``: generic kernels also run
with a ``bool`` mask, and ``~true`` is ``-2``, which is still true.

``first_n<T>(n)`` sets the first ``n`` lanes, a loop tail's mask, and
``mask_all<T>()`` and ``mask_none<T>()`` set every lane or none.
``to_bits(m)`` and ``mask_from_bits<T>(bits)`` convert to and from an integer
whose bit ``i`` is lane ``i``, and ``mask_cast<U>(m)`` reuses a mask for an
element type of the same width, so a ``float`` comparison can steer an
``int32_t`` index vector. Code that used a mask as bits, as the vector of
all-ones or zero lanes masks used to be, gets that vector from ``to_vec(m)``;
the numeric form, one or zero, is ``select(m, broadcast(T(1)), broadcast(T(0)))``.

A masked operation is ``select`` over the unmasked one:
``select(m, fmadd(a, b, c), c)`` is a masked FMA, and AVX-512 compiles it to
the masked instruction.

The 32- and 64-bit integers have the same six comparisons, ordered by their
own signedness, so ``cmp_lt`` on ``Vec<uint32_t>`` treats ``0xFFFFFFFF`` as
the largest value. Besides the immediate shifts ``shift_left<N>(v)`` and
``shift_right<N>(v)``, ``shift_left(v, count)`` and ``shift_right(v, count)``
shift each lane by the matching lane of ``count``, which must be below the
element's width in bits. Every right shift is logical.

Reductions, Partial Loads and Conversions
=========================================

``Reduce.hpp`` folds a Vec to a scalar with ``reduce_add``, ``reduce_min``
and ``reduce_max``, for ``float``, ``double`` and ``int32_t``. The lanes are
combined in a tree whose shape depends on the width, so a floating-point sum
can differ in its last bits between ISAs and from a sequential loop.

``Partial.hpp`` handles the tail of a loop whose length is not a multiple of
the lane count. ``loadu_partial(p, n)`` reads the first ``n`` elements and
zeroes the rest of the Vec; ``storeu_partial(p, v, n)`` writes the first
``n`` lanes. Neither touches memory past those elements, so a tail that ends
at the end of an allocation is safe:

.. code-block:: cpp

    std::size_t i = 0;
    for (; i + L <= n; i += L) {
        storeu(y + i, fmadd(a, loadu(x + i), loadu(y + i)));
    }
    if (i < n) {
        storeu_partial(y + i, fmadd(a, loadu_partial(x + i, n - i), loadu_partial(y + i, n - i)), n - i);
    }

The masked forms behind them are public too: ``loadu(p, m)`` reads the lanes
``m`` sets and gives zero in the rest, ``storeu(p, v, m)`` writes only the lanes
``m`` sets, and ``gather(base, idx, m)`` reads ``base[idx[i]]`` only where set.
None of them touches an inactive lane's memory, so an out-of-range index in an
inactive lane is safe. ``loadu_partial(p, n)`` is ``loadu(p, first_n<T>(n))``.

A masked access is a single instruction on AVX-512, and on AVX and AVX2 for
``float``, ``double`` and the 32- and 64-bit integers. SSE and NEON have no
masked load, so there it goes through a stack buffer, which a tail of two or
three elements does not repay. ``native_masked_memory<T>`` (also spelled
``native_partial<T>``) is ``true`` exactly where the masked instruction
exists, so a kernel compiled per rung can keep a scalar tail on the others:

.. code-block:: cpp

    if constexpr (native_masked_memory<T>) {
        // one masked vector step, as above
    } else {
        for (; i < n; ++i) { /* scalar tail */ }
    }

``Convert.hpp`` converts between ``int32_t`` and ``float``, which have the
same lane count: ``convert<float>(vi)`` rounds to nearest and
``convert<int32_t>(vf)`` truncates toward zero. A NaN or an out-of-range
value gives an unspecified lane.

Where the build has the 16-bit float types, it also widens ``half_t`` and
``bfloat16_t`` to ``float`` and narrows back. A 16-bit Vec holds exactly
twice ``Vec<float>``'s lanes, so it widens to two float vectors:

.. code-block:: cpp

    Vec<float> lo = convert_low<float>(h);    // lanes 0 .. L/2 - 1 of h
    Vec<float> hi = convert_high<float>(h);   // lanes L/2 .. L - 1
    Vec<bfloat16_t> b = convert<bfloat16_t>(lo, hi);

Widening is exact; narrowing rounds to nearest, ties to even, as a C++
conversion does. NEON uses its conversion instructions and every other build
converts lane by lane through memory.

``float`` and ``double``, and ``int32_t`` and ``double``, convert in the same
shape, because ``Vec<double>`` has half the lanes of the 32-bit vectors:

.. code-block:: cpp

    Vec<double> lo = convert_low<double>(vf);     // lanes 0 .. L/2 - 1 of a Vec<float>
    Vec<double> hi = convert_high<double>(vf);    // lanes L/2 .. L - 1
    Vec<float>  f  = convert<float>(lo, hi);      // rounded to nearest even
    Vec<int32_t> i = convert<int32_t>(lo, hi);    // truncated toward zero

These use conversion instructions on every ISA. The scalar fallback has one
lane of each type, so it converts one ``Vec`` to another instead, as in
``convert<double>(vf)``.

``int64_t`` and ``double`` have the same lane count and convert like
``int32_t`` and ``float``: ``convert<int64_t>(vd)`` truncates toward zero and
``convert<double>(vl)`` rounds to nearest. AVX-512DQ and NEON have
instructions for both. Other x86 builds have none, so they take a short
exact sequence when every lane is below 2^51 in magnitude, which covers any
table index, and convert lane by lane otherwise.

``bitcast<To>(v)`` reads the same bits as another element type of the same
width, ``Vec<float>`` as ``Vec<int32_t>`` or ``Vec<double>`` as
``Vec<uint64_t>`` and back. It compiles to nothing, and with the integer
operations it reaches the exponent and mantissa bits of a float.

One Kernel for Vectors and Scalars
==================================

``Generic.hpp`` lets a kernel be written once as a template over its value
type and instantiated with ``Vec<double>``, a problem per lane, or with
``double``, one problem at a time: on a GPU thread, or on the CPU as a
reference.

.. code-block:: cpp

    #include <Einsums/SIMD/Generic.hpp>

    namespace simd = einsums::simd;

    template <typename V>
    V step(V x, V y, simd::scalar_t<V> const *table) {
        using S = simd::scalar_t<V>;
        V t = simd::fmadd(x, y, simd::splat<V>(S(0.5)));
        t *= 2;
        auto const k = simd::convert<simd::gather_index_t<S>>(simd::floor(t));
        return simd::select(simd::cmp_lt(t, x), simd::lookup(table, k), simd::sqrt(t));
    }

The header adds scalar overloads of the operations above (the fused forms,
``div``, ``sqrt``, ``min``, ``max``, ``abs``, ``neg``, the rounding functions,
the comparisons, which return ``bool`` where the vector ones return a
``Mask<T>``, ``select``, ``any``, ``all``, ``none``, ``count``, the mask
combinations, the masked ``loadu`` and ``storeu``, and ``convert``), and the generic operations a scalar has no other spelling for:
``scalar_t<V>``, ``lanes_v<V>`` and ``is_vec_v<V>``; ``splat<V>(x)``;
``load<V>(p)`` and ``store(p, v)``; and ``lookup(base, idx)``, an index
gather that reads ``base[idx]`` for a scalar index. ``lookup`` exists because
``gather(base, n)`` with an integer is a strided load and would silently
accept a scalar index.

Each scalar overload matches one lane of the vector operation bit for bit, so
the scalar instantiation is an exact reference for the vector one, provided the
compiler does not fuse a kernel's own separate multiply and add into an FMA.
GCC does so by default (``-ffp-contract=fast``), even in intrinsic code and
differently in each instantiation, so compile a bit-for-bit comparison with
``-ffp-contract=off``. In particular, ``min`` is
``a < b ? a : b``, the fused forms round once exactly where the vector forms
do, and a comparison returns ``bool``. Call them qualified, as
``simd::fmadd``: argument-dependent lookup finds nothing for ``double``. They
are constrained templates, so under ``using namespace einsums::simd`` an
unqualified ``sqrt(2.0)`` still calls the C library. ``std::min`` and
``std::max`` are templates too, so code with both ``using namespace std`` and
``using namespace einsums::simd`` must qualify ``min`` and ``max`` on scalars.

A ``Vec<T>`` mixes with a scalar in ``+ - * /`` and in ``+= -= *= /=`` when
the scalar is ``T`` or an integer: ``v * 2`` compiles, ``Vec<float> * 2.0``
does not. In the scalar instantiation ``float * 2.0`` would compute in
``double``, so the two instantiations would round differently; write
``S(2.0)``. The scalar fallback build, where ``Vec<T>`` is a single ``T``
that converts implicitly, cannot enforce this.

Mixed Precision at Equal Width
==============================

``Vec<T, N>`` holds ``N`` lanes of ``T``. ``N`` defaults to the native lane
count, so ``Vec<float>`` is one register as always; a larger ``N``, a whole
multiple of the native count, is that many registers in lane order. The use
is a kernel run in two precisions over the same batch: the FP32 tier is
``Vec<float>`` and the FP64 tier ``Vec<double, lanes<float>>``, with as many
doubles as the float vector has floats. ``Mask<T, N>`` is laid out the same
way.

.. code-block:: cpp

    #include <Einsums/SIMD/Generic.hpp>

    using F = simd::Vec<float>;                       // FP32 tier
    using D = simd::Vec<double, simd::lanes<float>>;  // FP64 tier, same lanes

    F const x   = simd::load<F>(xs);
    auto const idx = simd::convert<int32_t>(simd::floor(x * inv_dx)); // computed once
    D const f   = simd::lookup(table64, idx);         // the FP64 tier reuses it
    D const xd  = simd::convert<double>(x);           // lane i stays lane i
    auto const m = simd::mask_cast<double>(simd::cmp_lt(x, limit));

Every operation of the sections above, and every operation generic kernels
use, takes ``Vec<T, N>`` and ``Mask<T, N>``: it is the native operation on
each register, so the results are bit for bit the native ones side by side.
``convert`` moves the same lanes between any two element types with the same
lane count, widening or narrowing ``float`` and ``double`` and converting
between ``double`` and the 32- and 64-bit integers, and ``mask_cast`` does the
same for masks. ``lookup`` on a wide double vector takes an index vector of
either integer width, so the two tiers can share the index the FP32 tier
computes; AVX2 and AVX-512 gather doubles at 32-bit indices directly.
``first_n<T, N>(n)`` and ``mask_from_bits<T, N>`` build wide masks. A wide
vector of doubles at FP32's width uses twice the registers, so a long fused
chain of them may spill where the same chain in ``Vec<double>`` did not.

Examples
========

Each program in ``libs/Einsums/SIMD/examples`` checks its result against a
scalar loop and runs as a test:

- ``PlatformInfo``: the width this build compiled for, and the rungs the CPU
  supports.
- ``Saxpy``: the basic loop shape, with a partial tail.
- ``DotProduct``: independent accumulators and one ``reduce_add``.
- ``ComplexAxpy``: complex arithmetic with ``CVec``.
- ``ColumnSum``: strided ``gather`` against a contiguous traversal.
- ``BlockTranspose``: a matrix transpose in register tiles.
- ``ClampAndSanitize``: comparisons, ``select`` and ``any`` in place of
  branches.
- ``IntegerBits``: shifts, bitwise logic and ``cmp_eq`` counting.
- ``QuantizedGemv``: int8 dot products and ``convert``.
- ``StreamingScale``: ``prefetch``, ``stream_store`` and ``stream_fence``.
- ``RuntimeDispatch``: one kernel compiled per rung and chosen at run time.
- ``GemmMicroKernel``: a register-blocked matrix-multiply kernel.
- ``OperatorsAndFunctions``: the two spellings, and what ``fmadd`` changes.

Shuffle and Transpose
=====================

The ``Shuffle.hpp`` header provides in-register matrix transposes for
micro-kernels:

.. code-block:: cpp

    #include <Einsums/SIMD/Shuffle.hpp>

    // Transpose a 4x4 float matrix stored in 4 Vec<float> registers (SSE/NEON)
    Vec<float> rows[4];
    // ... load rows ...
    transpose_inplace(rows);  // Now rows[i] holds column i

    // On AVX: 8x8 float transpose
    // On AVX-512: 16x16 float transpose

``storeu_interleaved<R>(dst, rows)`` stores ``R`` rows lane by lane, row index
fastest: ``dst[k * R + r] = rows[r][k]``, exactly ``R * L`` elements, for any
``R`` up to the lane count. It is the store a transpose into a narrow panel
needs, such as a GEMM B panel of six rows on a machine with eight or sixteen
float lanes:

.. code-block:: cpp

    Vec<float> rows[6];               // six rows of L elements each
    storeu_interleaved<6>(panel, rows);  // panel: 6 * L contiguous elements

The rows are transposed in registers and written with full-width stores that
overlap, each later store rewriting the lanes the one before it spilled; only
the last one or two columns need a partial store.

Gather and Scatter
==================

Strided memory access, with hardware gathers where the ISA has them:

.. code-block:: cpp

    #include <Einsums/SIMD/Gather.hpp>

    float data[64] = { /* ... */ };

    // Every third element: data[0], data[3], data[6], ...
    Vec<float> strided = gather(data, 3);

    // The same with the stride known at compile time
    Vec<float> fixed = gather_fixed<3>(data);

    // Write the lanes back to the same strided locations
    scatter(data, 3, strided);

A gather can also take one index per lane, in elements. The index vector has
the element's width, so the lane counts match: ``Vec<int32_t>`` for
``float`` and ``Vec<int64_t>`` for ``double``, named by
``gather_index_t<T>``. With ``floor`` and ``convert`` it looks up a
tabulated function at the grid point below each lane's argument:

.. code-block:: cpp

    // table[k] holds f(k * dx)
    Vec<int64_t> k = convert<int64_t>(floor(x * broadcast(1.0 / dx)));
    Vec<double>  f = gather(table, k);

AVX2 and AVX-512 use their gather instructions; every other build, NEON
included, reads lane by lane. Every index must address a valid element.

Complex Numbers
===============

``ComplexVec.hpp`` provides SIMD operations on interleaved complex data:

.. code-block:: cpp

    #include <Einsums/SIMD/ComplexVec.hpp>

    // Load interleaved complex data: [re0, im0, re1, im1, ...]
    std::complex<float> z[4] = {{1,2}, {3,4}, {5,6}, {7,8}};
    CVec<float> a = complex_loadu(z);

    // Complex multiply
    CVec<float> b = complex_broadcast(std::complex<float>(1.0f, -1.0f));  // (1 - i)
    CVec<float> c = complex_mul(a, b);

    // Conjugate
    CVec<float> conj = conjugate(a);

    // The first n complex values, for a loop tail; n counts complex values
    CVec<float> t = complex_loadu_partial(z, n);
    complex_storeu_partial(z, t, n);

    // The complex values of a CVec added together
    std::complex<float> sum = complex_reduce_add(a);

``reduce_add`` on a CVec would add real and imaginary lanes together;
``complex_reduce_add`` keeps them apart, folding the upper half of the lanes
onto the lower in the same order on every backend.

Prefetch and Streaming
======================

.. code-block:: cpp

    #include <Einsums/SIMD/Prefetch.hpp>

    // Prefetch for read
    prefetch<PrefetchHint::T0>(data);       // Into L1 cache
    prefetch<PrefetchHint::NTA>(data);      // Non-temporal (streaming)

    // Prefetch multiple rows of a matrix
    prefetch_rows<4>(matrix_ptr, stride);

    // Non-temporal store (bypasses cache, useful for write-only patterns)
    stream_store(dst, vec);

Runtime Feature Detection
=========================

Everything above is *compile-time*: the ISA baked into a translation unit by
its compiler flags. ``RuntimeFeatures.hpp`` adds the *runtime* half - what
the CPU executing the process actually supports - so that a single portable
binary can carry kernels for several ISA levels and pick the best one at
startup.

.. code-block:: cpp

    #include <Einsums/SIMD/RuntimeFeatures.hpp>

    using namespace einsums::simd;

    // Cached, thread-safe, detected once per process.
    CpuFeatures const &f = cpu_features();
    if (f.avx512f) { /* ... */ }

    // The dispatch rung for this process: Baseline, V2, V3, V4, or Sme.
    InstructionSet arch = selected_arch();

The x86 rungs follow the psABI micro-architecture levels, which map directly
onto compiler flags:

============  =====================================  =========================
Rung          ISA content                            Compiler flag
============  =====================================  =========================
``Baseline``  x86-64 (SSE2) / aarch64 (NEON)         none (toolchain default)
``V2``        SSE3...SSE4.2, POPCNT, CMPXCHG16B      ``-march=x86-64-v2``
``V3``        adds AVX2, FMA, BMI1/2, F16C           ``-march=x86-64-v3``
``V4``        adds AVX-512 F/BW/CD/DQ/VL             ``-march=x86-64-v4``
============  =====================================  =========================

All AVX-family reports are gated on operating-system state enablement
(OSXSAVE + XCR0, queried with ``xgetbv``), not just CPUID bits: a feature is
reported only if using it will not fault. On aarch64, optional features
(FEAT_FP16, FEAT_BF16, FEAT_I8MM, FEAT_DotProd, FEAT_SVE, FEAT_SVE2, and the
SME family) are detected via ``sysctl`` on macOS and ``getauxval`` on Linux;
NEON itself is the aarch64 baseline.

aarch64 has one optional rung, ``Sme`` (SME2 with FP64 outer products,
compiled with ``-march=armv8.6-a+sme2+sme-f64f64``). Its features do not
nest the way the x86 levels do: Apple M4 has SME but no non-streaming SVE.
So the ladder is not a ranking of enumerator values. ``supports()`` says
whether a machine can run a rung, and ``preference_order()`` lists each
architecture's rungs from most to least preferred (``V4, V3, V2, Baseline``
on x86, ``Sme, Baseline`` on aarch64). The ``Sme`` gate also requires
whatever else the rung's flags switch on for the compiler in use: GCC
before 15 makes ``+sme`` imply ``+sve2``, and a build with such a compiler
only selects ``Sme`` on a core that has SVE2 as well.

Overriding the rung
-------------------

Set :option:`--einsums:simd:arch`, or its environment variable
``EINSUMS_SIMD_ARCH``, to ``baseline``, ``v2``, ``v3``, ``v4``, ``sme``, or
one of the aliases ``sse2``/``sse4.2``/``avx2``/``avx512``/``sme2`` to force
another rung. This is the primary tool for
testing every rung of a dispatch ladder on one machine. An override can only
choose a rung the machine supports. Asking for one it cannot run logs a
warning and takes the next supported rung in the architecture's preference
order, so ``v4`` on an AVX2 machine gives ``v3``. A rung of another
architecture, such as ``v3`` on aarch64, is ignored with a warning. For test
suites, prefer ``einsums_add_simd_rung_tests()``: it wraps each per-rung
registration in the ``simd_rung_guard`` launcher, which turns an unsupported
rung into an honest ctest "Skipped" (exit 77) instead of a silent rerun at
another rung. The value is read once and cached; tests that need to exercise
the resolution logic itself should call ``resolve_arch()`` with explicit
arguments instead of setting the option.

Building a dispatch ladder
--------------------------

A kernel that should use the widest registers the CPU has is written once and
compiled several times, once per rung. ``einsums_add_simd_dispatch_sources``
(``cmake/Einsums_AddSIMDDispatch.cmake``) generates the per-rung translation
units: each includes the implementation file, compiles with that rung's
``-march`` flags, and defines ``EINSUMS_SIMD_ARCH_NS`` to a namespace of its
own. Because the SIMD headers key off compiler-defined macros, the same
source widens ``Vec<T>`` and every operation to each rung's register width.

.. code-block:: cmake

    einsums_add_simd_dispatch_sources(MyKernelRungs IMPL src/KernelImpl.cpp RUNGS baseline v2 v3 v4)
    target_sources(my_target PRIVATE ${MyKernelRungs})
    set_source_files_properties(src/KernelDispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${MyKernelRungs_DEFINITIONS}")

The implementation wraps its entry points in the rung namespace:

.. code-block:: cpp

    // KernelImpl.cpp: compiled once per rung.
    namespace mylib::EINSUMS_SIMD_ARCH_NS {
    void kernel(float const *x, float *y, std::size_t n) { /* Vec<float> code */ }
    }

and one arch-neutral file declares every copy that was built and picks one.
``EINSUMS_SIMD_FOR_EACH_BUILT_RUNG`` names each built namespace,
``EINSUMS_SIMD_LADDER`` expands to the five slots ``select()`` takes (with
``nullptr`` for rungs not built), and ``select()`` returns the slot of the rung
``selected_arch()`` chose:

.. code-block:: cpp

    // KernelDispatch.cpp: compiled once, at the ambient flags.
    namespace mylib {
    #define DECLARE(ns) namespace ns { void kernel(float const *, float *, std::size_t); }
    EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(DECLARE)
    #undef DECLARE

    void kernel(float const *x, float *y, std::size_t n) {
        using Fn = void (*)(float const *, float *, std::size_t);
        static Fn const fn = einsums::simd::select<Fn>(EINSUMS_SIMD_LADDER(kernel));
        fn(x, y, n);
    }
    }

Resolve the pointer once, as the function-local static does: the choice
cannot change while the program runs. The ``RuntimeDispatch`` example is a
complete, buildable version of this, and HPTT and PackedGemm are the library's
own consumers.

Each copy of the implementation is a different program under the same source,
so nothing it defines may share a name with another copy: the linker keeps one
definition per name and hands it to every caller. Your own code is kept apart
by ``EINSUMS_SIMD_ARCH_NS``. The SIMD headers keep theirs apart themselves:
they declare everything inside an inline namespace named for the features the
translation unit was compiled with, such as
``einsums::simd::isa_avx_avx2_fma_sse3_ssse3_sse41_sse42``. It is transparent,
so ``einsums::simd::Vec<float>`` names it, but ``Vec<float>`` at v3 and at v4
are different types to the linker, and a helper that does not inline is a
different function in each copy. ``EINSUMS_SIMD_ISA_NS`` expands to the name.

Neither namespace covers templates from other libraries. A ``std::vector``
member or an fmt formatter instantiated in the implementation file is emitted
by every copy under one name, compiled at that copy's flags, and the program
calls whichever copy the linker kept. Keep such code in an arch-neutral file
and pass the kernel plain pointers and sizes, as HPTT does: its planner is
compiled once, and only its kernels (``TransposeKernels.cpp``) once per rung.
On Linux, ``einsums_add_simd_rung_objects_test(<subcategory> <target>)``
registers a test that fails when a per-rung object of ``<target>`` defines a
weak symbol outside its rung's namespaces; HPTT and PackedGemm use it.

The whole mechanism sits behind ``EINSUMS_WITH_SIMD_DISPATCH`` (default ON).
When it is OFF, or when a compile-time pin is in effect (below), the helper
emits a single ``native`` rung compiled at the ambient flags. On aarch64 the
x86 rungs do not exist, so the ladder is that ``native`` rung plus ``sme``
when the caller asks for it.

Interaction with the compile-time pinning options: building with
``EINSUMS_SIMD_NATIVE_ARCH=ON`` or ``EINSUMS_SIMD_TARGET_CPU=<cpu>`` raises
the baseline of *every* SIMD consumer to that target, which makes the binary
non-portable and runtime dispatch pointless; use one approach or the other.

See the :ref:`API reference <modules_Einsums_SIMD_api>` of this module for more details.
