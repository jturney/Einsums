# einsums-pybind

A libtooling-based code generator that walks Einsums headers, finds
declarations marked with `EINSUMS_PYBIND_*` macros, and emits two
artifacts per annotated module:

1. **A pybind11 binding TU** that gets linked into a single `einsums`
   Python extension, so users get one `import einsums` instead of one
   import per module.
2. **A `.pyi` type-stub fragment** for that module. After every module
   has been processed, a small Python aggregator merges fragments into
   per-submodule stubs (`einsums/_core.pyi`, `einsums/linalg.pyi`,
   `einsums/graph.pyi`, …) that pyright / mypy consume.

The tool is gated on the `EINSUMS_PYBIND_AUTOGEN` CMake option (default
`OFF`). When the option is `ON`, modules opt in by passing `PYBIND` to
`einsums_add_module()`, and the rest — bindings + stubs — happens
automatically.

## Quick start

Annotate the C++ class you want to bind:

```cpp
// libs/Einsums/MyModule/include/Einsums/MyModule/Greeter.hpp
#include <Einsums/Python/Annotations.hpp>

namespace einsums::mymodule {

class EINSUMS_PYBIND_EXPOSE Greeter {
public:
    EINSUMS_PYBIND_EXPOSE
    Greeter();

    EINSUMS_PYBIND_EXPOSE
    explicit Greeter(std::string greeting);

    EINSUMS_PYBIND_EXPOSE
    std::string say(std::string const &name) const;
};

} // namespace einsums::mymodule
```

Opt the module into autogen:

```cmake
# libs/Einsums/MyModule/CMakeLists.txt
einsums_add_module(
  Einsums MyModule
  PYBIND                              # <-- this line
  SOURCES Greeter.cpp
  HEADERS Einsums/MyModule/Greeter.hpp
  MODULE_DEPENDENCIES Einsums_Config
)
```

Configure with autogen, build, import:

```bash
cmake -S . -B build -DEINSUMS_BUILD_PYTHON=ON -DEINSUMS_PYBIND_AUTOGEN=ON
cmake --build build --target PyEinsums

PYTHONPATH=build/lib python3 -c "
import einsums
g = einsums.Greeter('hi')
print(g.say('world'))
"
```

That's it. The codegen runs as a build edge (re-fires on header
changes), and the resulting bindings end up alongside every other
annotated module under one `import einsums`.

## Annotation reference

Every macro is a C++11 attribute that places between the class-key and
the class name (or before a function return type). Multiple macros
stack. Under non-Clang compilers all macros expand to nothing, so
production builds carry no overhead.

### Exposure

| Macro | Purpose |
|---|---|
| `EINSUMS_PYBIND_EXPOSE` | Mark a declaration for binding. Without this, the codegen ignores it. |
| `EINSUMS_PYBIND_HIDE` | Suppress binding for an otherwise-exposed declaration (e.g. an inherited member). |

### Naming

| Macro | Purpose |
|---|---|
| `EINSUMS_PYBIND_RENAME("py_name")` | Override the Python identifier used for the binding. |
| `EINSUMS_PYBIND_MODULE("submodule")` | Place the binding inside a Python submodule. Dotted names (`"tensor.algebra"`) request nested submodules. |
| `EINSUMS_PYBIND_EXCEPTION` | Bind the class as a Python exception via `py::register_exception<T>` instead of `py::class_<>`. C++ class must derive from `std::exception` (or compatible). pybind11-only. |

### Class options

| Macro | Purpose |
|---|---|
| `EINSUMS_PYBIND_HOLDER(std::shared_ptr)` | Override the pybind11 holder type. Default is `std::unique_ptr`. |
| `EINSUMS_PYBIND_BUFFER_PROTOCOL` | Flip on pybind11's buffer protocol. Pair with `BUFFER_FROM`. |
| `EINSUMS_PYBIND_BUFFER_FROM(helper)` | Free function `helper(T&)` returning `py::buffer_info`; codegen wraps it in a `.def_buffer()` lambda. |
| `EINSUMS_PYBIND_IMPLICIT_FROM(Source)` | Emit `py::implicitly_convertible<Source, Class>()` after the binding. |
| `EINSUMS_PYBIND_DYNAMIC_ATTR` | Allow Python instances to carry arbitrary attributes. |
| `EINSUMS_PYBIND_NOCOPY` | Skip generation of the copy-ctor binding. |
| `EINSUMS_PYBIND_NOMOVE` | Skip generation of the move-ctor binding. |
| `EINSUMS_PYBIND_NO_BASES` | Force-skip emission of base-class arguments. Usually unnecessary — the emitter auto-skips bases that aren't themselves bound. |
| `EINSUMS_PYBIND_READONLY` | On a field — bind as `def_readonly` instead of `def_readwrite`. |

### Method / free-function options

| Macro | Purpose |
|---|---|
| `EINSUMS_PYBIND_RVP(reference_internal)` | Set `return_value_policy`. Argument is the unqualified policy name. |
| `EINSUMS_PYBIND_KEEP_ALIVE(0, 1)` | Emit `py::keep_alive<nurse, patient>()`. |
| `EINSUMS_PYBIND_RELEASE_GIL` | Wrap the call in `py::call_guard<py::gil_scoped_release>()`. |
| `EINSUMS_PYBIND_OPERATOR("__add__")` | Bind the method as a Python operator instead of a named function. |

### Properties

`EINSUMS_PYBIND_GETTER("name")` and `EINSUMS_PYBIND_SETTER("name")` get
merged into one `.def_property("name", &get, &set)` when the codegen
sees a matching name on a getter/setter pair. A `@getter` with no
matching `@setter` becomes a `.def_property_readonly`.

### Documentation

Doxygen comments (`///` or `/** */`) above an exposed declaration become
the Python docstring automatically. Override explicitly with
`EINSUMS_PYBIND_DOC("text")`.

### Template instantiation

Templated classes need an explicit instantiation directive — pybind11
binds concrete types, not templates.

**Cross-product** (`EINSUMS_PYBIND_INSTANTIATE`): each parameter list
is keyed by the *exact* C++ template-parameter name. The codegen matches
by name, not position, so the order in the macro is free. Python names
are auto-derived from the values.

```cpp
template <typename T, int rank>
class EINSUMS_PYBIND_EXPOSE
    EINSUMS_PYBIND_INSTANTIATE(Tensor,
        T(float, double),
        rank(1, 2))
Tensor { ... };
// Produces: Tensor_float_1, Tensor_float_2, Tensor_double_1, Tensor_double_2
```

**Single instantiation** (`EINSUMS_PYBIND_INSTANTIATE_AS`): pin one
concrete type to a chosen Python name. Use this when one template
parameter depends on another (e.g. `Alloc = std::allocator<T>`), which
a flat cross-product can't express.

```cpp
EINSUMS_PYBIND_INSTANTIATE_AS("Tensor2d_double",
                              GeneralTensor<double, 2, std::allocator<double>>)
```

**Cross-product with name template** (`EINSUMS_PYBIND_INSTANTIATE_TEMPLATE`):
same matching rules; placeholders in the name template use the C++
template-parameter names too.

```cpp
template <typename Element, int Rank>
class EINSUMS_PYBIND_EXPOSE
    EINSUMS_PYBIND_INSTANTIATE_TEMPLATE("Block_{Element}_{Rank}",
        Block,
        Element(float, double),
        Rank(1, 2))
Block { ... };
// Produces: Block_float_1, Block_float_2, Block_double_1, Block_double_2
```

Placeholder values are sanitized to valid Python identifiers
(`std::complex<double>` → `std_complex_double`).

### Free-function template instantiation

`EINSUMS_PYBIND_INSTANTIATE_AS` also works on templated free functions —
each directive defines one instantiation. Multiple directives sharing a
Python name turn into a pybind11 overload set; the codegen picks the
right one at call site via Python's argument types.

```cpp
template <typename T>
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_INSTANTIATE_AS("scale", einsums::RuntimeTensor<float>)
EINSUMS_PYBIND_INSTANTIATE_AS("scale", einsums::RuntimeTensor<double>)
void scale(typename T::ValueType factor, T *A);
```

#### Same-signature overloads → dtype dispatcher

When two or more `INSTANTIATE_AS` lines share a Python name AND their
argument signatures are identical (only the return type or value-type
differs), the codegen automatically collapses them into a single Python
entry that takes a `dtype="..."` kwarg and dispatches at runtime:

```cpp
template <typename T>
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_INSTANTIATE_AS("create_zero_tensor", float)
EINSUMS_PYBIND_INSTANTIATE_AS("create_zero_tensor", double)
EINSUMS_PYBIND_INSTANTIATE_AS("create_zero_tensor", std::complex<float>)
EINSUMS_PYBIND_INSTANTIATE_AS("create_zero_tensor", std::complex<double>)
RuntimeTensor<T> create_zero_tensor(std::string name, std::vector<size_t> dims);
// Python: create_zero_tensor("X", [4, 4], dtype="float64")
```

Recognized dtype aliases (numpy convention): `float32`/`f4`/`f`/`single`
(float), `float64`/`f8`/`d` (double), `complex64`/`c8`/`F`
(complex<float>), `complex128`/`complex`/`c16`/`D` (complex<double>).
The default dtype is `float64` if `double` is in the group, otherwise
the first instantiation's first alias.

#### Bool template parameters → kwargs

For functions templated on leading `bool` parameters (e.g. `template
<bool TransA, bool TransB, typename T>`), pair
`EINSUMS_PYBIND_TEMPLATE_KWARGS` with `EINSUMS_PYBIND_INSTANTIATE_BOOLS`.
The codegen expands `2^N` combinations internally and emits one Python
entry per dtype taking each bool as a kw-only argument:

```cpp
template <bool TransA, bool TransB, typename T>
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_TEMPLATE_KWARGS("trans_a", "trans_b")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensor<float>, float)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensor<double>, double)
void gemm(U alpha, T const &A, T const &B, U beta, T *C);
// Python: gemm(1.0, A, B, 0.0, C, trans_a=True, trans_b=False)
```

The first `INSTANTIATE_BOOLS` argument is the Python name (shared
across the bool fan-out); the rest are the *non-bool* template args.
The 2^N bool combinations are generated automatically; the codegen
then emits a single Python `def` per dtype with an internal `if`-chain
dispatcher.

### Member-template instantiation

Use `EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS` to bind a templated method
with its own template parameters (independent of the enclosing class's
parameters). Multiple directives stack; same-signature ones with
recognized dtypes auto-merge into a `dtype=` dispatcher exactly like
free-function `INSTANTIATE_AS`.

```cpp
template <typename T>
class EINSUMS_PYBIND_EXPOSE Workspace { ... };

class Workspace {
    template <typename U>
    EINSUMS_PYBIND_EXPOSE
    EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS("declare_runtime_tensor",
                                          U=einsums::RuntimeTensor<float>)
    EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS("declare_runtime_tensor",
                                          U=einsums::RuntimeTensor<double>)
    U &declare_runtime_tensor(std::string name, std::vector<size_t> dims);
};
```

`EINSUMS_PYBIND_INSTANTIATE_MEMBER` (no `_AS`) is the same idea for
members whose own parameters depend on the class's. Argument is a
``Name=Type`` pair like ``Dim=std::vector<size_t>``.

### Variadic constructors

Templated classes often have parameter-pack constructors whose arity
depends on a template parameter. The codegen needs to know how many
arguments to bind per instantiation.

```cpp
template <typename T, size_t rank>
struct EINSUMS_PYBIND_EXPOSE
    EINSUMS_PYBIND_INSTANTIATE_AS("Tensor_double_2",
                                  GeneralTensor<double, 2, std::allocator<double>>)
GeneralTensor {
    template <typename... Dims>
    EINSUMS_PYBIND_EXPOSE
    EINSUMS_PYBIND_VARIADIC_FROM(rank, size_t)   // pack -> rank-many size_t args
    GeneralTensor(std::string name, Dims... dims);
};
```

For `Tensor_double_2`, this binds a ctor with signature
`(std::string, size_t, size_t)`. For `Tensor_double_3` it would be
`(std::string, size_t, size_t, size_t)`.

The first arg names the template parameter that gives the count; the
second is the concrete C++ type each expanded slot should take. The
last function parameter is assumed to be the pack.

## Validation

The codegen emits Clang-style file:line errors and exits non-zero on
problems. Common cases:

| Error | Cause |
|---|---|
| `unknown parameter keyword 'X' (template parameters are: ...)` | Either a typo, or an upstream `#define` mangled the keyword before stringification. |
| `class name '<X>' in directive payload does not match` | Same cause: macro expansion changed the class name token. |
| `expected N parameter list(s), got M` | Number of `Param(...)` groups doesn't match the template signature. |
| `parameter keyword '<X>' specified more than once` | Duplicate group. |
| `missing parameter list for template parameter '<X>'` | A template parameter has no matching group. |

The strict name-match for `INSTANTIATE` / `INSTANTIATE_TEMPLATE` is the
load-bearing guard against random macro expansion. If some upstream
header has `#define Element WHATEVER`, the codegen sees `WHATEVER(...)`,
fails the match against the real `Element` parameter, and emits a
diagnostic instead of producing wrong bindings.

## How it builds

1. **Configure** — `einsums_finalize_pybind()` runs at the end of root
   configure. It writes the aggregator main (`PyEinsumsMain.cpp`,
   prototypes + `PYBIND11_MODULE(einsums, m)` calling each module's
   register function), creates the `PyEinsums` `pybind11_add_module`
   target, and emits one `add_custom_command` per opted-in module.

2. **Build** — ninja resolves the dependency chain:
   - `einsums-pybind` tool builds first.
   - Each `Einsums_<Module>` library builds.
   - For every annotated module, `einsums-pybind` runs over its headers
     and emits `${BUILD}/generated/pybind/Einsums_<Module>_pybind.cpp`,
     which contains a `void einsums_pybind_register_<Module>(py::module_ &m)`
     function.
   - The aggregator main and every generated TU compile.
   - `PyEinsums` links them all into `${BUILD}/lib/einsums.cpython-*.so`.

Touching an annotated header re-fires only that module's codegen edge
(via the `add_custom_command`'s `DEPENDS ${headers}`) and re-links the
single shared library.

## Known limitations

- **Default constructors with conditional `requires` clauses** that are
  compile-time `delete`d for some instantiations may emit spurious
  bindings. Use `EINSUMS_PYBIND_HIDE` to suppress per-method.
- **Cross-product with dependent parameters** can't be expressed
  (`Alloc = std::allocator<T>`). Fall back to one
  `EINSUMS_PYBIND_INSTANTIATE_AS` per concrete type.
- **System header detection** assumes Clang's `-print-resource-dir` is
  available and (on macOS) `xcrun --show-sdk-path`. The conda
  `einsums-dev` env satisfies both. Other setups may need to set
  `EINSUMS_PYBIND_CLANG_RESOURCE_DIR` / `EINSUMS_PYBIND_SYSROOT`
  manually before the first configure.
- **`requires requires { … }` clauses block doxygen attachment** —
  clang's `getRawCommentForDecl` doesn't associate `///` comments with
  a function template that has a nested requires-expression. Flatten
  to a single `requires (A && B && …)` clause and the comment (and
  thus the Python docstring) will attach.
- **Stub-side metafunction expansion** isn't supported — return types
  involving `RemoveComplexT<T>` and friends fall back to `Any` in the
  generated `.pyi`. The runtime binding is correct; only the static
  type information is reduced.

## Backend target

einsums-pybind can emit code against either pybind11 (default) or
nanobind. Pass `--target {pybind11,nanobind}` on the command line:

```bash
einsums-pybind --target nanobind --module myext header.hpp -- ...
```

The output differs in:

- **Headers** — `<pybind11/...>` vs `<nanobind/...>`, with nanobind's
  STL bindings split per-type (`<nanobind/stl/string.h>`,
  `<nanobind/stl/vector.h>`, etc.)
- **Module macro** — `PYBIND11_MODULE` vs `NB_MODULE`
- **Namespace** — `py::` vs `nb::`
- **Return value policy** — `py::return_value_policy::reference_internal`
  vs `nb::rv_policy::reference_internal`
- **Buffer protocol** — pybind11 emits `.def_buffer()` lambdas; nanobind
  doesn't have an equivalent directive (use `nb::ndarray<>` for tensor
  protocol instead). `EINSUMS_PYBIND_BUFFER_FROM` directives are
  silently dropped under the nanobind target.

The `einsums_finalize_pybind` CMake integration uses pybind11 today.
Switching the autogen pipeline to nanobind requires also swapping
`pybind11_add_module` for `nanobind_add_module` and the matching
`find_package(nanobind)`. The `--target` flag is what makes the rest of
that switch a one-line change in the cmake hook.

## Stub generation (`.pyi`)

Every codegen invocation also produces a Python type-stub fragment,
emitted alongside the generated `.cpp`:

```
build/generated/pybind/Einsums_LinearAlgebra_pybind.cpp   # bindings
build/generated/pybind/Einsums_LinearAlgebra.pyi          # stub fragment
```

A finalize step (`tools/einsums-pybind/scripts/aggregate_stubs.py`)
runs as a `PyEinsumsStubs` custom target after `PyEinsums` is linked.
It splits each fragment by the `# %%submodule: <name>` sentinels the
emitter inserts and merges them into per-submodule files in the
package directory:

```
build/lib/einsums/
├── _core.cpython-…so      # the C extension
├── _core.pyi              # top-level entities
├── linalg.pyi             # entities tagged @module("linalg")
├── graph.pyi              # entities tagged @module("graph")
├── __init__.py / .pyi     # runtime + stub re-exporting _core
└── py.typed               # PEP 561 marker
```

### What pyright sees

Type translation runs per-instantiation:

```python
# scale (free function with INSTANTIATE_AS for four dtypes)
@overload
def scale(factor: float, A: RuntimeTensorF) -> None: ...
@overload
def scale(factor: float, A: RuntimeTensorD) -> None: ...
@overload
def scale(factor: complex, A: RuntimeTensorC) -> None: ...
@overload
def scale(factor: complex, A: RuntimeTensorZ) -> None: ...

# create_zero_tensor (auto-detected dtype dispatcher)
def create_zero_tensor(name: str, dims: list[int], dtype: str = "float64") \
    -> RuntimeTensorF | RuntimeTensorD | RuntimeTensorC | RuntimeTensorZ: ...

# gemm (TEMPLATE_KWARGS bool fan-out)
@overload
def gemm(alpha: float, A: RuntimeTensorF, B: RuntimeTensorF, beta: float,
         C: RuntimeTensorF, *, trans_a: bool = False, trans_b: bool = False) -> None: ...
@overload
def gemm(alpha: complex, A: RuntimeTensorC, B: RuntimeTensorC, beta: complex,
         C: RuntimeTensorC, *, trans_a: bool = False, trans_b: bool = False) -> None: ...
```

Doxygen comments above an exposed declaration become Python docstrings.
`@getter` / `@setter` pairs become `@property` declarations.
Rich-comparison dunders (`__eq__`, `__lt__`, …) are widened to take
`object` to satisfy LSP — a stub typed `__eq__(self, other: Vec3)`
would otherwise trip pyright's `reportIncompatibleMethodOverride`.

### Cross-module name resolution

When a function in one module takes a tensor from another module
(`cg::scale` taking `RuntimeTensor<T>` defined in the Tensor module),
the visitor records the external annotated class with `is_external=true`
purely for name resolution. The C++ emitter ignores externals (their
binding lives in the owning module's TU); the `.pyi` emitter uses
them to map `GeneralRuntimeTensor<float, std::allocator<float>>` →
`RuntimeTensorF` so cross-module signatures resolve without needing a
shared registry across codegen invocations.

### Type-resolution pipeline

For each per-instantiation parameter / return type, the `.pyi` emitter:

1. Substitutes template names on the **raw C++ type** (preserving forms
   like `typename T::ValueType` for re-resolution).
2. Tries the canonical (typedef-expanded) form via clang's
   `getCanonicalType()` if the as-written form fails — catches
   alias templates like `RuntimeTensor<T>` ↔
   `GeneralRuntimeTensor<T, std::allocator<T>>`.
3. Inlines `typename Class<args>::ValueType` references with
   `args.first` (Einsums tensor convention).
4. Substitutes any known cpp_to_py-mapped class instantiation in
   nested types (so `std::tuple<RuntimeTensor<float>, ...>` reduces
   to `tuple[RuntimeTensorF, ...]`).
5. Falls back to `Any` when none of the above produces a Python-valid
   identifier — pyright will surface the gap rather than the stub
   silently mistyping.

### Python shell modules

Each Python submodule needs a tiny `.py` shell next to `_core.so`. The
recommended pattern uses PEP 562 `__getattr__` so the C extension
isn't loaded until first attribute access:

```python
# einsums/graph.py
import importlib as _importlib

def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    core = _importlib.import_module("._core.graph", "einsums")
    attr = getattr(core, name)
    globals()[name] = attr  # cache for subsequent lookups
    return attr
```

The generated `<sub>.pyi` describes the static surface; the `.py` shell
is just a runtime trampoline.

## Configure-time conditional bindings

Annotated headers can use `#if`/`#else`/`#endif` against any
configure-time define, including everything that
`einsums_add_config_define()` writes into `<Einsums/Config.hpp>`. The
codegen tool runs Clang's full preprocessor and only sees the active
branch:

```cpp
#include <Einsums/Config.hpp>
#include <Einsums/GPU/DeviceVector.hpp>

template <typename T, size_t rank, typename Alloc>
struct EINSUMS_PYBIND_EXPOSE
    EINSUMS_PYBIND_INSTANTIATE_AS("Tensor_double_2",
                                  GeneralTensor<double, 2, std::allocator<double>>)
#if defined(EINSUMS_HAVE_GPU)
    EINSUMS_PYBIND_INSTANTIATE_AS("Tensor_double_2_gpu",
                                  GeneralTensor<double, 2, gpu::DeviceAllocator<double>>)
#endif
GeneralTensor { ... };
```

When `EINSUMS_WITH_GPU=ON`, the GPU instantiation is added; toggling it
off and reconfiguring drops it. The generated `Defines.hpp` files are
in every codegen edge's `DEPENDS`, so re-configure → re-fire codegen
automatically. Also forwarded: every `INTERFACE_COMPILE_DEFINITIONS`
reachable from the module's MODULE_DEPENDENCIES (gets `-D` flags on the
codegen invocation).
- **Visibility warnings** when linking PyEinsums (weak symbols across
  the per-module library and the generated TU) are cosmetic on macOS;
  symbols still resolve correctly.

## Tool architecture (for contributors)

```
src/
  main.cpp              CLI driver: ClangTool + per-TU IR accumulation +
                        emit pass. Drives the post-IR passes
                        (compute_python_overloads, compute_properties)
                        before invoking the C++ and .pyi emitters.
                        Tracks total error count, exits non-zero.
  Visitor.hpp/.cpp      RecursiveASTVisitor that walks declarations,
                        filters by einsums_pybind: annotation, builds
                        the Module IR. Captures annotated classes from
                        outside the current module's headers as
                        ``is_external`` for cross-module name resolution.
  IR.hpp/.cpp           BoundClass / BoundMethod / BoundField / BoundEnum /
                        BoundFunction / BoundParam / BoundInstantiation /
                        BoundProperty / PythonOverload. Plus a
                        deterministic textual dump (``--dump-ir``).
  AnnotationParser.hpp/.cpp
                        Splits raw "einsums_pybind:<directive>:<args>"
                        payloads into structured Directive records.
                        Knows about free-form-tail directives (doc,
                        instantiate, holder) where the tail may contain
                        ':'.
  InstantiateParser.hpp/.cpp
                        Parses INSTANTIATE / INSTANTIATE_TEMPLATE
                        payloads into ParamGroup lists, respects nested
                        ``<>`` and ``()``. Provides cross_product and
                        sanitize_python_name helpers.
  TypeTranslator.hpp/.cpp
                        Wraps Clang's PrintingPolicy for fully-qualified
                        pretty C++ types. Also provides
                        ``translate_python_type`` (and the string-only
                        variant ``translate_python_type_string``) that
                        maps fundamentals + std containers to their
                        Python equivalents.
  DocExtractor.hpp/.cpp
                        Pulls doxygen text from
                        ASTContext::getRawCommentForDeclNoCache, strips
                        leading ``///``/``*`` markers and decoration
                        banner lines (``//////``, ``=====``, ``-----``).
  PythonOverloads.hpp/.cpp
                        Post-IR pass that decides how each free
                        function's raw instantiation list collapses
                        into Python entries: NonTemplate,
                        SingleInstantiation, OverloadSet,
                        DtypeDispatcher, TemplateKwargsDispatcher.
                        Both emitters consume the precomputed view.
  Properties.hpp/.cpp   Post-IR pass that walks each class's methods
                        and collapses @getter/@setter pairs into
                        BoundClass.properties.
  Emitter.hpp/.cpp      IR -> pybind11 C++ (or nanobind, via
                        ``--target``). Two output modes:
                        PYBIND11_MODULE(name, m) (standalone fixtures /
                        goldens) or void register_<Module>(py::module_ &m)
                        (autogen aggregator path). In-process
                        clang::format::reformat() picks up the project's
                        .clang-format.
  PyiEmitter.hpp/.cpp   IR -> Python .pyi stubs. Walks the same IR plus
                        the post-pass views, emits per-submodule blocks
                        delimited by ``# %%submodule: <name>`` sentinels
                        for the aggregator to split.

scripts/
  aggregate_stubs.py    Reads every ``*.pyi`` fragment in
                        ``--frag-dir`` and merges by submodule sentinel
                        into per-submodule files in ``--pkg-dir``.
                        Writes a shared header per output and the
                        PEP-561 ``py.typed`` marker.

tests/
  fixtures/             Annotated headers used by the emitter tests.
  golden/               Expected emitter output (regen with REGEN=1).
  run_smoke.sh          IR-dump substring assertions covering
                        BoundClass/BoundMethod/BoundFunction shapes,
                        property merge, submodule routing, Python
                        type translation, and default-value sanitization.
  run_golden.sh         pybind11 emitter golden-file diff.
```

## Adding a new directive

1. Define the macro in `libs/Einsums/Python/include/Einsums/Python/Annotations.hpp`.
2. If its tail can contain `:`, add it to `directive_takes_free_form_tail`
   in `AnnotationParser.cpp`.
3. Handle it in the emitter (most directives are read via `DirectiveView`
   in `Emitter.cpp`'s class-body / method-emission helpers).
4. Add a fixture under `tests/fixtures/` and regenerate goldens
   (`REGEN=1 tests/run_golden.sh ...`).
5. Exercise it end-to-end in
   `libs/Einsums/PythonDemo/CMakeLists.txt`'s `einsums_pybind_python_smoke`
   ctest entry so the Python side is verified.

## Examples

### Property pair → Python `@property`

```cpp
class EINSUMS_PYBIND_EXPOSE Resource {
public:
    /// Read-only-from-Python access to the underlying name.
    EINSUMS_PYBIND_GETTER("name")
    std::string const &get_name() const;

    /// Pythonic name setter.
    EINSUMS_PYBIND_SETTER("name")
    void set_name(std::string const &n);
};
```

Generated stub:

```python
class Resource:
    @property
    def name(self) -> str:
        """Read-only-from-Python access to the underlying name."""
        ...
    @name.setter
    def name(self, value: str) -> None: ...
```

### Operator overload

```cpp
class EINSUMS_PYBIND_EXPOSE Vec3 {
public:
    /// Component-wise equality.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_OPERATOR("__eq__")
    bool operator==(Vec3 const &other) const;
};
```

Generated stub (note the `object` widening for LSP compliance):

```python
class Vec3:
    def __eq__(self, other: object) -> bool:
        """Component-wise equality."""
        ...
```

### Submodule routing

```cpp
namespace EINSUMS_PYBIND_MODULE("graph") cg {

EINSUMS_PYBIND_EXPOSE
class Graph { ... };

EINSUMS_PYBIND_EXPOSE
void execute(Graph &g);

} // namespace cg
```

Both `Graph` and `execute` end up in `einsums.graph`. The aggregator
writes them into `build/lib/einsums/graph.pyi`. Anything outside the
namespace block (or any entity tagged with its own
`EINSUMS_PYBIND_MODULE("…")`) routes to the chosen submodule.

### Conditional binding gated on a config define

```cpp
#include <Einsums/Config.hpp>

EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_INSTANTIATE_AS("Tensor_double_2",
                              GeneralTensor<double, 2, std::allocator<double>>)
#if defined(EINSUMS_HAVE_GPU)
EINSUMS_PYBIND_INSTANTIATE_AS("Tensor_double_2_gpu",
                              GeneralTensor<double, 2, gpu::DeviceAllocator<double>>)
#endif
template <typename T, size_t rank, typename Alloc>
class GeneralTensor { ... };
```

Toggle `EINSUMS_WITH_GPU` and reconfigure — the codegen picks up the
`Defines.hpp` mtime change and re-fires automatically; the GPU
instantiation appears (or disappears) in the generated bindings + stubs.
