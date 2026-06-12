# Einsums TODO

## Completed (2026-03-31 to 2026-04-02)

### Distributed Computing
- [x] **Comm module**: MPI communication with mock backend. Collectives (allreduce, broadcast, scatter, allgather), blocking + non-blocking, `expected<T, CommError>` return types.
- [x] **ProcessGrid**: 2D Pr×Pc process grid with auto near-square factorization, row/col sub-communicators.
- [x] **DistributionDescriptor**: Per-dimension GridAxis (None/Row/Col) with balanced blocking.
- [x] **DistributionPlanning**: Contraction-aware index classification (target_a→Row, target_b→Col, shared→balanced, link→None). Chain conflict resolution. SUMMA mode for square grids.
- [x] **Materialization**: Resizes deferred tensors to local partitions using DistributionDescriptor.
- [x] **InputSlicing**: Automatic views of pre-allocated inputs. Multi-dimension, cross-axis. Handles Permute nodes.
- [x] **SUMMAExpansion**: Broadcast+GEMM loop on square grids. Uses `tensor_algebra::einsum` for PackedGemm dispatch.
- [x] **CommunicationInsertion**: Inserts allreduce for replicated outputs from distributed inputs. Evaluates all compute nodes.
- [x] **CommunicationElimination**: Removes redundant back-to-back allreduces.
- [x] **CommunicationScheduling**: Splits allreduce into async iallreduce + wait for DataflowExecutor overlap.
- [x] **Batch index distribution**: Shared indices distributed via load-balancing heuristic.
- [x] **Distributed permute**: Cross-axis redistribution via index-aware InputSlicing.
- [x] **Capture-aware dot/norm**: `dot(&result, A, B)` and `norm(&result, type, A)` with scalar allreduce.
- [x] ~~**Default Workspace**~~: removed 2026-04-24. All call sites use explicit `Workspace` / `Pipeline` / `Graph` (`ws.declare_tensor`, `p.declare_zero_tensor`, …). Free-function + singleton forms deleted; code is clearer without the hidden global.
- [x] **Chemistry fill API**: `declare_tensor_filled()` with `T.range(dim)` and `T.global(indices...)`. Parallelizable with OpenMP/TaskPool.
- [x] **Catch2 MPI seed sync**: Broadcast random seed from rank 0 so all ranks run tests in same order.
- [x] **MPI_Comm_free fix**: Communicator destructor checks `is_initialized()` before freeing.
- [x] **18 MPI integration tests**: Verified on np=1,2,4,6.

### Memory-Aware Scheduling
- [x] **Tensor::release()**: Free backing storage, return to deferred state. ProfileMemFree event.
- [x] **FreeInsertion pass**: Inserts Free nodes after each intermediate's last consumer. Size threshold (default 1MB). Re-execution safe.
- [x] **DataflowExecutor memory budget**: `set_memory_budget(bytes)` gates Materialize submissions. Free nodes wake blocked submissions via condition variable.

### Tensor I/O (.etn)
- [x] **Binary format**: FileHeader (64B), TensorEntry (160B), 64-byte aligned data. Entry table at end for append support.
- [x] **TensorFile**: Serial POSIX I/O: write, read, read_slice, multi-tensor, ReadWrite append mode.
- [x] **DistributedTensorFile**: MPI-coordinated single-file I/O using POSIX pwrite/pread + MPI Exscan/Gather.
- [x] **Checkpoint**: `checkpoint::save()`/`restore()` for Graph and Workspace. Distributed variants.
- [x] **GraphIO**: `read_etn()`, `write_etn()`, `checkpoint_etn()` as DiskRead/DiskWrite graph nodes.
- [x] **21 tests**: Format, round-trip (4 types), multi-tensor, slices, distributed, checkpoint, GraphIO, ReadWrite, query.
- [x] **Documentation**: `tensor_io.rst` with all APIs documented.
- [x] **Example**: `TensorIOExample.cpp` demonstrating all features.

### C++23 Backports / Upgrades
- [x] **CXX23 module**: `expected<T,E>`, `unreachable()`, `flat_map`, `flat_set`.
- [x] **expected retrofit**: Phases 1-5 across Comm, GPU, ComputeGraph modules.
- [x] **C++23 upgrade**: Bumped minimum standard from C++20 to C++23.
- [x] **Glaze integration**: Replaced nlohmann::json with Glaze v7.2.0 for JSON + compile-time reflection.

### Benchmark System
- [x] **publish_benchmark_result()**: All 12 benchmarks emit structured results to profiler server.
- [x] **ProfileAnnotate**: All benchmarks annotated with rank, pattern, dtype, elements, etc.
- [x] **LabeledSection0()**: All benchmark functions wrapped in profiler zones for annotation collection.
- [x] **Profiler thread ID fix**: `Profiler::current_thread_id()` replaces `std::hash<std::thread::id>` for annotation collection.
- [x] **Database schema shared**: SQL migrations in `devtools/benchmarks/migrations/*.sql`, used by both Python CLI and C++ viewer.
- [x] **Default DB path**: Python CLI and C++ viewer share `~/Library/Application Support/EinsumsProfileViewer/benchmarks.db`.
- [x] **Benchmark CLI**: `list-tests`, `delete-db`, `trend`, `scaling` subcommands.
- [x] **Renamed mlir→packed_gemm**: Labels and metrics in BenchmarkContraction/BenchmarkSortGemm.
- [x] **README**: `devtools/benchmarks/README.md` documenting all CLI commands and DB schema.

### Profile Viewer: Benchmark Panel
- [x] **BenchmarkDB**: C++ SQLite wrapper mirroring Python `models.py` API.
- [x] **BenchmarkPanel**: 5 tabs: Latest Run, Live Results, Trends, Scaling, Compare.
- [x] **Auto-record**: Benchmark results automatically stored to DB when viewer is running.
- [x] **Detail pane**: Click a row to see timing stats + parsed annotations in a side panel.
- [x] **Arrow key navigation**: Up/Down keys move selection in benchmark tables.
- [x] **Trend filtering**: Search box + label-specific metric radio buttons.
- [x] **Scaling plot**: Log-log ImPlot with error bars (min/max) and data table.
- [x] **DB panel layout**: Top-level "Benchmarks" tab alongside session tabs, "Open Benchmarks" button on welcome screen.

### Profile Viewer: Graph Hierarchy
- [x] **Graph parent metadata**: `pipeline_name`, `workspace_name`, `stage_name`, `stage_type`, `stage_index` on Graph.
- [x] **Pipeline::execute() propagation**: Metadata set on child graphs before execution.
- [x] **Workspace panel**: Single canvas with pipeline region boxes and graph zone boxes.
- [x] **Inter-graph wires**: Orange Bezier curves connecting tensors across graph zones (name-based matching).
- [x] **Loop-back wires**: Green curves for cyclic dependencies within loop graphs.
- [x] **Edge generation fix**: `to_json()` handles forward + loop-back edges correctly.
- [x] **Stage ordering**: Graphs sorted by `stage_index` within pipelines.
- [x] **Server shutdown drain**: 500ms drain loop processes pending viewer requests before exit.
- [x] **Graph registration at capture**: `register_graph()` called in `CaptureContext::end_capture()`.

### Profile Viewer: Editor Features
- [x] **Node dragging**: Click and drag nodes to reposition. Grid snapping on release (Ctrl to disable).
- [x] **Grid background**: Blueprint-style grid with minor/major lines.
- [x] **Selection glow**: Gold outline on selected nodes, brighter on dragged nodes.
- [x] **Node shadows + borders**: Drop shadow and dark border on all nodes.
- [x] **Right-click context menus**: Node info + "Auto Layout", canvas "Auto Layout All" + "Fit View".
- [x] **Properties panel**: Right-side panel showing node kind, label, timing, stage, editable parameters, input/output tensors.
- [x] **Editable properties**: InputDouble for prefactors, Checkbox for conjugate flags.
- [x] **Pin-to-pin wires**: Wires connect specific output/input pins, colored by dtype.
- [x] **Pin-to-pin wire dragging**: Click output pin to drag wire, drop on input pin to connect.
- [x] **Wire disconnect**: Click connected input pin to disconnect and re-drag from source output.
- [x] **Filled/hollow pins**: Connected pins filled, unconnected pins hollow.
- [x] **Drop target highlight**: Pins glow when a wire drag hovers over them.
- [x] **Rich node labels**: Real tensor names in einsum/permute/scale/axpy labels.
- [x] **Structured op data**: Prefactors, indices, conjugate flags in JSON and properties panel.

### Profile Viewer: Infrastructure
- [x] **ComputeGraphTypes module**: Shared header-only module (Enums, Ids, Descriptors, EinsumSpec, GraphData).
- [x] **Glaze metadata**: `glz::meta` declarations for all shared structs in `GraphDataMeta.hpp`.
- [x] **Git/build metadata**: `git_commit`, `git_branch`, `git_dirty`, `build_type` in profiler meta event.
- [x] **JSON response fix**: `response.data` as object handled correctly (no crash on `{"error":"unknown method"}`).
- [x] **Deadlock fix**: `drain_benchmark_results()` called before `lock_data()` to avoid double-lock.
- [x] **TaskGroup fix**: `wait_all()` uses timed wait; `submit_group` locks mutex before `notify_all()`.
- [x] **TensorLifetime fix**: Destruction canary in `GeneralTensor` for deterministic validation.
- [x] **println disambiguation**: Qualified `einsums::println` calls to avoid C++23 `std::println` conflict.

### Other
- [x] **Profiling**: SUMMA panels, MPI collectives, communication node annotations.
- [x] **DistributedTensor cleanup**: Deleted legacy classes, moved DistributionMap to Comm.
- [x] **make_einsum_executor**: Type-erased executor for arbitrary-rank einsums via StringDispatch.
- [x] **Performance tuning guide**: `performance_tuning.rst`.
- [x] **Documentation**: CLAUDE.md, distributed.rst, optimization_passes.rst, workspace.rst updated.
- [x] **Python TUI deleted**: Replaced by ImGui viewer's BenchmarkPanel.

## Needs GPU Hardware (NVIDIA/AMD)

- [ ] **GPU + InputSlicing mismatch**: InputSlicing modifies CPU tensor pointers via `begin_local_view`, but GPU shadows are allocated at original full size. GPU GEMM gets mismatched dimensions.
- [ ] **SUMMA + NCCL**: SUMMAExpansion uses CPU broadcasts and CPU einsum. For GPU tensors, broadcast panels need device memory and NCCL collectives.
- [ ] **GPU legacy cleanup**: ~12,900 lines of legacy GPU code (DeviceTensor, GPUMemory, GPUStreams, hipBLAS, hipBLASVendor) to delete once GPU abstraction module fully replaces them.
- [ ] **Ozaki FP64 emulation**: Designed (Ozaki Scheme II, CRT-based) for FP64 via FP32/FP16/FP8 tensor cores on NVIDIA GPUs. Not coded.
- [ ] **GPU PackedGemm**: Device-side packing for arbitrary-rank GPU contractions. Currently CPU-only.

## Future Improvements

- [ ] **Higher-rank chain restructuring**: `make_einsum_executor` infrastructure is in place, but dimension folding for rank-3+ needs stride validation, index mapping, and final output reshape. Currently analysis only.
- [ ] **Graph serialization**: Save/load a captured graph (nodes, edges, tensor metadata) to binary/JSON for replay, sharing, and visualization. `to_json()` exists for debugging.
- [ ] **Pipeline checkpoint integration**: Pipeline's Stage uses `std::variant<Graph, LoopNode>` which complicates iteration. Deferred from Checkpoint implementation.
- [ ] **Non-square SUMMA**: Outer-product handles non-square grids well. SUMMA with K-splitting on Pr≠Pc requires redistribution.
- [ ] **Block-cyclic distribution**: Replace balanced blocking with cyclic addressing for better load balance at 64+ ranks.
- [ ] **2.5D layers**: Add layers to ProcessGrid (Pr × Pc × c) for memory-bandwidth tradeoff at scale.
- [ ] **Fix ctest MPI exit code**: DistributedIntegration returns exit code 14 under ctest without mpirun.
- [ ] **MPI test infrastructure in CI**: Set up CI to run `mpirun -np 4` for distributed tests.

## Visual Graph Editor Path

### Completed
- [x] **Collapsing Materialize/Initialize nodes**: single "Tensor" node with init kind dropdown
- [x] **Comment nodes**: resizable, scrollable text, editable, drag from palette or context menu
- [x] **Storage wires**: green wires from Tensor nodes to compute nodes showing buffer ownership
- [x] **Draw layer channels**: channel 0 (bg/boxes), channel 1 (all wires), channel 2 (nodes/pins)
- [x] **Scaled fonts**: all text scales with canvas zoom
- [x] **Ctrl+scroll zoom**: mouse wheel no longer zooms by default
- [x] **Cross-graph pin filling**: pins filled when tensor has cross-graph producer/consumer
- [x] **Add Node from palette**: drag-and-drop from left panel (UE5-style) with categorized node types
- [x] **Delete node**: right-click node → Delete Node, with edge cleanup
- [x] **Save/Load/Save As compute graph**: `.cgraph` JSON files with ImGuiFileDialog, Ctrl+S quick save, separate Save As
- [x] **New Compute Graph**: File menu option creates blank editable session
- [x] **Editable mode**: live sessions are read-only; new/loaded graphs are fully editable
- [x] **Node palette**: left panel with Structure (Pipeline, Graph Stage), Compute, Linear Algebra, Data, Control, Annotation categories
- [x] **Variables panel**: workspace-level variable table (name, type, value) with reorder, validation, delete protection
- [x] **Resizable panels**: palette and properties panels have draggable splitters
- [x] **Tensor properties**: name, dtype (10 types), rank, symbolic dims with variable validation, init kind
- [x] **Pre-wired nodes**: new compute nodes come with placeholder input/output pins (`?` prefix)
- [x] **Wire connect updates pins**: connecting a wire replaces placeholder tensor IDs with real tensors
- [x] **Wire disconnect reverts**: disconnecting reverts to placeholder pins (all connection types)
- [x] **Output pin connection**: wires can connect to output pins for storage tensor assignment
- [x] **Contraction editor**: editable einsum spec string with format validation and rank checking
- [x] **Permute index editor**: editable c_indices/a_indices for Permute/Transpose nodes
- [x] **Per-node prefactors**: all node types show relevant parameters driven by registry
- [x] **Fixed pin enforcement**: Einsum/Gemm/Gemv/SVD/etc. have fixed input/output counts (wire-only)
- [x] **Workspace/Pipeline/Graph management**: add, delete, rename via context menu and palette drag-and-drop
- [x] **Cross-graph wire creation**: drag wires between nodes in different graph zones with scope enforcement
- [x] **Tensor ownership levels**: Graph/Pipeline/Workspace scope with visual rendering at correct level (shelves)
- [x] **Tensor ownership drag-and-drop**: drag tensors between workspace, pipeline, and graph zones to change scope
- [x] **Scope enforcement**: wire connections validate tensor scope (graph tensors can't be referenced cross-graph)
- [x] **Scope migration cleanup**: moving tensor to narrower scope disconnects out-of-scope references
- [x] **Context-sensitive property panels**: Workspace → Pipeline → Graph → Node, always visible
- [x] **Pipeline properties**: name, stages list, pipeline tensors, optimization pass configuration
- [x] **Graph zone properties**: name, type (graph/loop), nodes summary, tensors, edge counts
- [x] **Workspace properties**: name, pipelines list, workspace tensors, variables summary, validation warnings
- [x] **Optimization passes**: per-pipeline configuration (default or 25 individual pass toggles)
- [x] **Code generation panel**: live C++ code gen below canvas with Copy/Refresh, resizable horizontal splitter
- [x] **Template-driven code gen**: `NodeTypeDef::code_template` with `{in:N}`, `{out:N}`, `{c_indices}` etc. variable expansion
- [x] **String-based einsum code gen**: uses `EinsumFormatString` API (`"ij <- ik ; kj"`) instead of `Indices{}`
- [x] **Unified NodeTypeDef registry**: single serializable struct replacing old `NodeTypeInfo`, drives palette, rendering, properties, and code gen
- [x] **Node type JSON persistence**: loads from `~/Library/Application Support/EinsumsProfileViewer/node_types.json`, creates defaults if missing, saves on shutdown
- [x] **Node Type Editor**: View menu window for editing colors, pins, parameters, code templates (kind/category read-only)
- [x] **Export/Import node types**: File menu options for sharing custom node type definitions
- [x] **Reset node types**: View menu option to restore all defaults
- [x] **Validation on load**: checks dangling IDs, unconnected pins, scope violations, missing indices, incomplete tensors, undefined variables
- [x] **Refactored panels**: `render_palette_panel()`, `render_properties_panel()`, `render_pipeline_properties_panel()`, `render_graph_properties_panel()`, `render_workspace_properties_panel()`
- [x] **Graph polling**: viewer re-requests graphs every 2s while connected
- [x] **Workspace::materialize_all()**: materializes all deferred workspace tensors with proper init kinds
- [x] **Cross-level storage wires**: workspace/pipeline tensor → compute node output storage wires (unclipped)
- [x] **Distinct wire colors**: workspace→graph (orange), pipeline→graph (blue), graph→graph (yellow), storage (green)
- [x] **Ghost drag visual**: semi-transparent ghost node + drop target hint when dragging tensors between levels

### Next Steps: Code Generation
- [ ] **Code generation testing**: end-to-end: build a graph visually → generate C++ → compile → run → verify results match expectations. Test cases needed:
  - Single einsum with workspace tensors
  - Multi-stage pipeline with pipeline tensors and graph-level intermediates
  - Loop stages with convergence conditions
  - Scale, Axpy, Permute operations
  - Custom optimization pass selection (non-default)
  - Cross-stage tensor sharing (pipeline-owned)
  - Workspace tensors used across multiple pipelines
  - Edge cases: unconnected pins, missing indices, zero-rank tensors
- [ ] **Export C++ to file**: "Export Code..." button/menu that writes generated code to a `.cpp` file
- [ ] **CMakeLists.txt generation**: generate build configuration alongside exported code
- [ ] **Loop configuration**: editable max iterations and convergence condition (currently hardcoded in code gen)
- [ ] **Variable dtype in code gen**: Variable nodes currently always emit `double`; should use the dtype from the output tensor

### Next Steps: ComputeGraph Library
- [x] **String-based permute API**: `cg::permute("ji <- ij", &C, A)` with compile-time validation, prefactors, rank-3+, graph capture, runtime string dispatch
- [ ] **`from_json()` / `build_op_data()`**: reconstruct real `Graph`/`Node`/`EinsumDescriptor` objects from saved editor data for live testing

### Next Steps: Editor UX
- [ ] **Undo/redo**: command pattern for all graph modifications (add/remove node, wire connect, property edit, tensor move)
- [x] **Multi-select**: box select, Shift+click to add to selection, Delete to remove, bulk move
- [x] **Search/filter in palette**: type to filter node types as the palette grows
- [ ] **Node search**: Ctrl+F to find a node by name/kind across all graphs, highlight and pan to result
- [ ] **Keyboard shortcuts panel**: show all shortcuts (Ctrl+S, Ctrl+C/V, Ctrl+scroll, etc.) in help dialog
- [x] **Snap wires to nearest pin**: auto-snap with green highlight when wire endpoint is near a valid target
- [x] **Tooltips on pins**: hover a pin to see tensor name, dtype, dims without selecting the node
- [x] **Real-time validation indicators**: yellow warning triangle badge on nodes with issues, hover for details
- [x] **Minimap**: bottom-right corner overlay with pipeline/zone/node rendering, viewport indicator, click-to-navigate
- [x] **C++ syntax highlighting**: code panel uses ImGuiColorTextEdit with C++ language definition
- [x] **Copy/paste nodes**: Ctrl+C/V to duplicate nodes at mouse position, preserves ownership level
- [x] **Auto-layout improvements**: better graph layout algorithm (Sugiyama/layered) for cleaner node arrangement
- [x] **Drag-to-reorder stages**: drag graph zones within a pipeline to change stage_index ordering
- [x] **Zoom-to-fit selection**: double-click a pipeline/zone to zoom and fit it to the canvas
- [ ] **Recent files list**: File menu shows recently opened .cgraph files
- [x] **Auto-save**: periodically save to a temp file, recover on crash

### Medium-Term: Visual Polish
- [x] **Tensor dims on nodes**: show dimension info directly on the node visual (e.g., "Rank 2: [NAO, NAO]")
- [x] **Wire routing**: route wires around nodes using control point deflection
- [x] **Wire flow animation**: animated dots traveling along wires showing data flow direction
- [ ] **Color themes**: light/dark/custom themes for the canvas and nodes
- [x] **Node grouping**: user-defined groups with colored background regions and labels
- [ ] **Pin type colors**: different pin colors by tensor dtype visible at a glance (partially done, dtype colors exist)
- [ ] **Collapsed nodes**: double-click title bar to minimize node to just its title bar
- [ ] **Wire labels on hover**: show tensor name on wire when hovering (not just on nodes)
- ~~**Execution order numbers**~~: removed because execution order is determined by optimization passes and dataflow executor at runtime, not by visual graph topology
- [ ] **Grid snap visual**: faint grid overlay showing snap points when dragging nodes
- [ ] **Smooth pan/zoom**: animated transitions when using Fit View or click-to-navigate on minimap
- [ ] **Node badges**: small icons on nodes showing GPU target, loop status, or custom annotations
- [ ] **Wire bundling**: group parallel wires between the same pair of nodes into a single thicker wire

### Medium-Term: Advanced Features
- [x] **Tensor shape inference**: auto-compute output tensor dimensions from input shapes + index patterns
- [ ] **Cycle detection**: real-time validation of graph topology (prevent cycles in forward graphs)
- [x] **Template library**: save/load common graph patterns (e.g., "SCF iteration", "MP2 energy", "similarity transform")
- [ ] **Diff view**: compare two `.cgraph` files side by side with visual diff highlighting added/removed/changed nodes
- [ ] **Version history**: auto-save snapshots on significant changes, browse/restore previous versions
- [ ] **Live editing via server**: send "modify_node" requests from viewer to running process, see results in real-time
- [ ] **Batch operations**: apply a transform to all nodes of a type (e.g., change all Einsum prefactors, set all tensors to float32)
- [ ] **Custom node types**: allow users to define entirely new node kinds via the Node Type Editor with custom pin layouts, code templates, and validation rules
- [x] **Subgraph encapsulation**: select a group of nodes → "Create Macro" → collapses into a reusable single node with exposed pins
- [ ] **Graph comparison**: overlay a live session graph with an editor graph, highlight differences
- [ ] **Index pattern auto-complete**: suggest valid index patterns based on connected tensor ranks
- [x] **Tensor memory estimation**: show estimated memory usage per tensor and total workspace memory in the properties panel
- [ ] **Dependency highlighting**: click a node to highlight all nodes it depends on (ancestors) and all nodes that depend on it (descendants)
- [ ] **Critical path highlighting**: show the longest dependency chain through the graph (potential bottleneck)
- [x] **Node documentation**: attach markdown documentation to node types, shown in the properties panel and node catalog
- [ ] **Graph statistics dashboard**: node count, edge count, tensor count, estimated FLOPs, memory usage summary

### Completed (2026-04-06)
- [x] **Pattern template system**: reusable subgraph snippets (save selection, palette drag-and-drop, ID remapping on instantiation)
- [x] **Built-in templates**: Matrix Multiply, Symmetric Orthogonalization, Fock Build with proper intermediates and edges
- [x] **Template collapse/expand**: collapsed templates become real "Template" nodes using standard rendering/interaction infrastructure
- [x] **Template intermediate tensors**: instantiation creates Tensor nodes for intermediates, handles name conflicts with suffixes
- [x] **Dimension inference**: bidirectional propagation through Einsum contraction patterns (fills missing dims always, overwrites with infer_dims flag)
- [x] **Per-tensor infer_dims**: checkbox in Tensor properties, auto-enabled for template intermediates
- [x] **Shelf node free movement**: workspace/pipeline tensors drag in place (no ghost), positions persisted
- [x] **Output pin disconnect**: click connected output starts re-drag; right-click context menu "Disconnect Output" submenu
- [x] **Shelf selection fix**: pipeline shelf nodes no longer all highlight when one is selected
- [x] **Contraction pattern sync**: property panel auto-refreshes when inline node editor changes the pattern
- [x] **Cache rebuild after interaction**: graph caches rebuilt post-interaction so tooltips/labels update same frame
- [x] **Collapsed template wire hiding**: all wire types (edges, storage, cross-level, inter-graph, minimap) skip collapsed member nodes
- [x] **Collapsed template zone sizing**: bounding box excludes hidden member nodes
- [x] **Multi-select right-click**: preserves selection when right-clicking an already-selected node
- [x] **Scrollable palette**: fixed search bar with scrollable content area below
- [x] **String-based permute API**: `PermuteFormatString`, `parse_permute_spec()`, `dispatch::string_permute()`, 5 tests passing
- [x] **`--einsums:profile:wait-for-viewer`**: blocks at startup until viewer connects, extended 3s shutdown drain
- [x] **Glaze unknown-key fix**: `error_on_unknown_keys = false` for all server JSON parsing (was rejecting every message)
- [x] **Live graph re-layout**: detects node/edge/tensor count changes on each poll, triggers re-layout
- [x] **Layout height-based splitting**: layers taller than 300px auto-split into sub-columns (grid)
- [x] **Layout compaction**: per-layer top-alignment replaces broken median-center approach
- [x] **Code generation improvements**: variable dtype from tensor, Export C++ to file, loop max_iterations/convergence, string-based permute template
- [x] **Tensor slot validation**: `Graph::execute()` checks slot pointers before running, catches cross-pipeline misuse
- [x] **libclang code analyzer**: in-process AST analysis of generated C++ code via libclang C API, preamble caching, background thread
- [x] **Bidirectional code↔graph sync**: Edit mode: edit code → graph updates (patterns, prefactors, values, ownership migration, new nodes/pipelines/stages)
- [x] **Variable ownership dropdown**: Variables panel Owner column is now an editable dropdown, migrates nodes between levels
- [x] **Shelf/Variable node deletion**: right-click Delete works for workspace/pipeline shelf nodes
- [x] **Tensor node output pin fix**: clicking Tensor/Variable output pins no longer disconnects (only compute nodes disconnect)
- [x] **Default tensor properties**: new Tensor nodes get dtype=double, rank=2 instead of blank
- [x] **Keyboard shortcut guard**: shortcuts suppressed when text widgets have focus (code editor, rename fields)
- [x] **Duplicate library warning fix**: removed redundant imgui_lib from link libraries
- [x] **Standalone graph rendering unified**: all graphs render through render_workspace_panel, render_graph_panel deleted (~280 lines removed)
- [x] **Examples updated**: BasicCapture, MixedOperations use Workspace/Pipeline + string-based APIs; include paths fixed across 8 examples
- [x] **Dead code cleanup**: 12 unused constants, 1 unused function, render_graph_panel deleted
- [x] **CodeSyncManager extraction**: code→graph sync logic moved from GraphPanel.cpp (590 lines) to CodeSyncManager.cpp
- [x] **for_each_level helper**: const and non-const overloads; 7 three-level iteration patterns migrated across GraphPanel.cpp, PropertyPanels.cpp, CodeSyncManager.cpp
- [x] **next_unique_node_id, insert/extract_tensor_at_level**: moved out of anonymous namespace with header declarations for cross-TU use
- [x] **Layout compaction**: per-layer top-alignment replaces broken median-center approach

### Completed (2026-04-05 to 2026-04-06)
- [x] **Template-based code gen**: `{{workspace_name}}`, `{{workspace_variables}}`, `{{workspace_tensors}}`, `{{#pipelines}}`, `{{#stages}}` block syntax
- [x] **Code gen template editor**: View menu, full-text editor with placeholder reference, Apply/Reset buttons
- [x] **Template persistence**: loaded from/saved to `~/Library/Application Support/EinsumsProfileViewer/codegen.cpp.tmpl`
- [x] **clang-format integration**: generated code auto-formatted via `clang-format --style=file`
- [x] **Tensor dims on nodes**: visual "Rank N: [dim0, dim1, ...]" on Tensor node body when rank is set
- [x] **Dim name validation**: checks Variable nodes at all ownership levels (not just state.variables)
- [x] **Variable click-to-edit**: value displayed as text, click to reveal InputDouble (matches prefactor pattern)
- [x] **Shelf rendering unified**: shelf nodes now use shared render_node_frame/render_node_body
- [x] **Layout compaction**: per-layer Y compaction eliminates gaps from median positioning
- [x] **Dynamic shelf height**: uses actual node sizes instead of hardcoded constant
- [x] **Tighter layout spacing**: kPadding 60→20, kNodeSpacingY 30→16, zone/pipeline padding reduced
- [x] **Correct tensor scoping in code gen**: graph-level tensors declared on pipeline (not inside stages)

### Completed (2026-04-03 to 2026-04-05)
- [x] **Node grouping**: user-defined groups with colored background regions, auto-detect contained nodes, drag/resize
- [x] **Multi-select**: box select, Shift+click toggle, Delete to remove, bulk move with snap
- [x] **Auto-layout improvements**: Sugiyama-style with barycenter cross-minimization, group-aware
- [x] **Variable nodes**: scalar variables with connectable output pin, UE5 Blueprint-style inline value editing
- [x] **Parameter pins**: C_pf, AB_pf, alpha, beta, factor as connectable input pins on compute nodes
- [x] **Click-to-edit prefactors**: inline label + value display, click to reveal InputDouble
- [x] **Click-to-edit contraction**: click pattern text to reveal InputText
- [x] **Cross-graph auto-promote**: wiring across graphs auto-promotes tensor to Pipeline/Workspace scope
- [x] **imgui_md integration**: replaced custom markdown renderer with MD4C-based CommonMark parser, BeginTable for tables
- [x] **Node type merge on load**: new built-in node types auto-added, structural fields updated on startup
- [x] **GraphLayoutFile v2 migration**: version field checked on load, v1→v2 adds missing param pin placeholders
- [x] **Variable ownership**: Variables can be workspace/pipeline/graph scoped, shown in unified variable panel
- [x] **Tensor memory estimation**: per-tensor and total memory in properties panels

### Completed: Code Quality (2026-04-05)
- [x] **render_workspace_panel split**: 3,776 → 1,531 lines (60% reduction) via 8 extracted sub-functions
- [x] **render_graph_panel rewrite**: 625 → 385 lines, reuses shared render_node_frame/render_node_body
- [x] **ProfileData.cpp Glaze migration**: 402 → 180 lines, manual jval() extraction → glz::read_json
- [x] **Export.cpp Glaze migration**: 290 → 113 lines, manual JSON building → glz::write_json
- [x] **Unified validate_node()**: single function used by batch validation and inline badges
- [x] **Tensor migration consolidated**: insert_tensor_at_level + extract_tensor_from_level
- [x] **GraphPanelState helpers**: find_pipeline_store, get_or_create_pipeline_store, is_param_pin, etc.
- [x] **WorkspaceCtx**: shared context with coordinate transforms, graph caching, zone iteration, drop-target detection
- [x] **GraphCache**: per-frame tensor map + geometry caching, 10+ hot-path calls migrated
- [x] **Data-driven pass checkboxes**: kPassDefs array replaces 3 copies of 26-line checkbox lists
- [x] **dtype_to_cpp helper**: replaces 3 copies of dtype mapping blocks
- [x] **ProfileDataMeta.hpp**: shared glz::meta for all profile types (ProfileNode, ProfileMeta, etc.)
- [x] **PipelineTensorStore unified**: alias to GraphPanelState::PipelineTensors
- [x] **PinInfo/NodeGeom shared**: moved to GraphPanelInternal.hpp
- [x] **Static locals → PerGraphView**: did_drag, ctx_menu state moved to view
- [x] **g_node_defs reduced**: 12 of 15 calls migrated to state.find_node_type()
- [x] **const_cast eliminated**: src_tensor_ptr made non-const from the start
- [x] **Dead code removed**: y_overlaps, duplicate structs, duplicate copyright headers (33 files)
- [x] **~37 const qualifiers**: from clang-tidy misc-const-correctness
- [x] **14 duplicate constants**: moved to file scope
- [x] **Drop target helper**: WorkspaceCtx::hit_test_drop() replaces 4 duplicate detection blocks
- [x] **Error logging**: parse_message reports malformed JSON to stderr

### In Progress: Plugin/Module Architecture
- [ ] **Phase 0: Infrastructure**: IPanel interface, PanelContext, PanelRegistry, SelectionBus headers; services/ directory; CMakeLists.txt restructure
- [ ] **Phase 1: Simple panels**: Hotspots, MicroArch, Log, Documentation migrated to IPanel
- [ ] **Phase 2: Stateful panels**: Roofline, Gantt, Timeline, FlameGraph, TaskPool migrated
- [ ] **Phase 3: Coupled panels**: Tree, Detail, Source, Assembly migrated with SelectionBus
- [ ] **Phase 4: Complex panels**: GraphPanel wrapped, Benchmark, Compare migrated
- [ ] **Phase 5: Thin main.cpp**: generic loops, auto-generated menus, ~500 lines target
- [ ] **Phase 6: Dynamic loading**: dlopen support, plugin SDK shared library, external plugin discovery

### Long-Term Vision
- [ ] **Full visual workflow builder**: create entire Workspace/Pipeline/Graph hierarchy visually (UE5 Blueprint style)
- [ ] **Node catalog**: searchable palette with documentation, examples, and parameter descriptions per node type
- [ ] **Emscripten/web build**: WebSocket bridge to profiler server, viewer runs in browser
- [ ] **Python bindings for code gen**: generate Python/pybind11 code from visual graphs
- [ ] **Multi-user collaboration**: shared editing sessions with conflict resolution via CRDT or OT
- ~~**AI-assisted graph building**~: suggest node connections, auto-complete index patterns, recommend optimizations based on tensor shapes
- [ ] **Performance prediction**: estimate FLOP count and memory usage from graph structure before running, compare against hardware roofline
- [ ] **Hardware targeting**: generate code optimized for specific hardware (CPU/GPU/distributed) based on HardwareProfile, show target-specific warnings
- [ ] **Graph animation/replay**: step through execution order, show data flow animation, visualize tensor values at each stage
- [ ] **Integrated terminal**: compile and run generated code from within the viewer, show output/errors in a panel
- [ ] **Profiler integration**: run generated code with profiling, display timing data overlaid on the visual graph nodes
- [ ] **Plugin system**: allow third-party extensions to add new node types, code gen backends, and analysis passes
- [ ] **Graph optimization suggestions**: analyze graph structure and suggest optimizations (e.g., "these two einsums could be fused", "this tensor could be computed in-place")
- [ ] **Multi-backend code gen**: generate not just Einsums C++ but also NumPy, PyTorch, TensorFlow, or raw BLAS code from the same visual graph
- [ ] **Collaborative annotations**: shared comments and review markers on nodes/wires for team workflows
- [ ] **Regression testing integration**: save expected output tensors with the .cgraph file, verify generated code produces matching results

## Einsums Studio

### Completed (2026-04-15 to 2026-04-17)

**Testing and diagnostics**
- [x] **Catch2 Test Explorer**: `TestRunner` (ctest JSON discovery + queued QProcess execution), `TestExplorerWidget` (left-nav tree grouped by dotted name, status icons, rerun-failed button, context menu), `TestOutputWidget` (QPlainTextEdit + CatchOutputHighlighter for performance + ANSI-colored output + clickable file:line links)
- [x] **TestCaseDetector**: regex scanner for `TEST_CASE`, `TEMPLATE_TEST_CASE{,_SIG}`, `TEMPLATE_PRODUCT_TEST_CASE`, `SCENARIO`, `TEST_CASE_METHOD`, and `EINSUMS_*` variants
- [x] **Gutter test markers**: ▷/▶ icons in `LineNumberEditor` gutter, click to run single test, cursor-anchored via `QTextCursor` so edits shift them; right-click context menu with Run / Debug / Copy Name; status updates from `testFinished` (blue running → green passed / red failed)
- [x] **TEST_CASE CodeLens**: `▶ Run` / `🐞 Debug` chips rendered above each test case via the generic CodeLens infrastructure; Debug path launches the exe under DAP with Catch2 filter + `--einsums:debug:no-attach-debugger`
- [x] **Failure location markers**: Catch2 `FAILED:` output parsed into red wavy underlines on assertion lines, with full message tooltip; cleared on `newTestBatch` signal
- [x] **Rerun failed**: Test Explorer button; iterates leaves with `Failed` status, dispatches batch via `run_all`
- [x] **newTestBatch signal**: `TestRunner` emits on every `run_one`/`run_all`; `TestOutputWidget` resets its summary; failure markers are cleared
- [x] **Test impact Tier 1: "Show Tests Covering This"**: right-click any symbol; `textDocument/references` → filter to files with Catch2 macros → identify enclosing `TEST_CASE` for each reference → results in Search Results pane

**Generic CodeLens framework**
- [x] **`LineNumberEditor::set_code_lens(category, line, items)`**: category-keyed, cursor-anchored, painted as rounded chips above the anchored line by a custom `QPlainTextDocumentLayout` subclass that reserves height on the preceding block
- [x] **Click dispatch**: `code_lens_clicked(category, line, item_id)` signal; `PointingHandCursor` on hover via viewport mouse-tracking
- [x] **LSP diagnostic chips**: one chip per diagnostic per line with `✘`/`⚠`/`ℹ` severity glyph; click requests `textDocument/codeAction` and applies the returned fix (single → immediate, multiple → menu)
- [x] **Build diagnostic chips**: build-log parser feeds per-line chips for files open in the editor; persisted across file reopens

**Diagnostics pipeline**
- [x] **Background build-log parser**: `BuildManager::parse_error_line` resolves absolute paths, publishes per-file diagnostics; `FileBrowserWidget` decorates the Projects tree with red/yellow badges aggregated from LSP (authoritative when file is open) and build counts
- [x] **Projects pane diagnostic badges**: per-file and per-directory aggregation in `ProjectFileSystemModel`; LSP wins over build for any file that's been analyzed

**Editor feel**
- [x] **Occurrence highlighting**: 300ms debounced `textDocument/documentHighlight` on cursor rest; read (blue tint) vs write (orange tint) backgrounds; clears immediately on cursor move
- [x] **Sticky scope headers**: LSP `documentSymbol` flattened into `ScopeHeader{start, end, label}`; `LineNumberEditor` paints pinned signature strips at the top of the viewport for every enclosing scope that's scrolled off-screen (up to 4 levels of nesting)
- [x] **Focus mode** (Cmd+Option+F): dims the viewport outside the innermost scope containing the cursor; repaints on cursor move
- [x] **Snippet session polish**: tabstops anchored by `QTextCursor` (track edits live), linked same-numbered tabstops (`${1:foo} ... $1 ... $1`) mirror edits to all occurrences with sibling-highlight overlays
- [x] **Right-click test-marker context menu**: Run / Debug / Copy Test Name entries, menu auto-deletes on close

**Session and navigation**
- [x] **Session restore**: `EditorPanelWidget::capture_session` / `restore_session` round-trip open files + cursors + active file via `ProjectContext::session_state` (persisted to `project.json`); restored after `project_opened`, captured on `project_closing` and `aboutToQuit`
- [x] **Reveal in Projects**: 🎯 button on `ViewToolBar`; publishes `EventBus::revealInTreeRequested`; `FileBrowserWidget` expands ancestors (with `fetchMore` + retry) and scrolls the target into view; `CodeBundle` activates the Projects nav pane so the scroll lands on a visible tree

### Refactoring Suite (Tier A: biggest power-user win)
Build out `textDocument/codeAction` + LSP rename into a proper refactoring UX on par with CLion. Clangd already returns most of the individual operations as tweak-class code actions; the gap is the UI. We need dedicated entry points, a "Refactor This..." menu that filters to `refactor.*`, and preview dialogs for multi-file rewrites.
- [x] **Refactor This... menu** (Ctrl+Alt+Shift+T): request `textDocument/codeAction` with `context.only = ["refactor"]`, show the results as a menu at the cursor.
- [x] **Extract Variable** (Ctrl+Alt+V): maps to clangd's `ExtractVariable` tweak; inline name prompt.
- [x] **Extract Function / Method** (Ctrl+Alt+M): clangd's `ExtractFunction` tweak; inline name prompt.
- [x] **Inline** (Ctrl+Alt+N): inline variable, inline function (where clangd supports it).
- [x] **Refactor preview dialog**: for refactors that touch multiple files, show a tree of affected files + hunks with checkboxes before applying. Prevents "oh that renamed something I didn't mean to".
- [x] **Safe Delete**: before deleting a symbol, find references; if any are outside the declaration file, warn with a list and an "apply anyway" button.

### Editor Tier (high value)
- [ ] **Horizontal scroll reaches past inlay hints**: `LineNumberEditor::apply_annotation_spacing` currently constrains each annotated line's layout to `viewport()->width()`, so long lines with inlay hints fall off the right edge and the horizontal scrollbar doesn't extend to reach them. An earlier attempt to use a large `setLineWidth` fixed the scroll but regressed paint perf (every paint walks visible blocks and rewrites their `QTextLayout`). The right fix is probably to compute each annotated line's natural width (base text + sum of letter-spacing extras) and feed that specific value, not a global large number. Bonus: cache the result per-block so repeated paints don't recompute.
- [x] **Peek definition** (Alt+F12): floating inline window showing the definition of the symbol at cursor without navigating away.
- [x] **Recent locations** (Ctrl+Shift+E): ring-buffer dialog of recent navigations, filterable, Enter jumps.
- [x] **Bookmarks**: F3 toggles at current line (or click the gutter slot); Ctrl+F3 opens the jump list. Persisted to `.einsums-studio/bookmarks.json`. Gutter slot renders a small pennant in the accent color when the file has any bookmarks.
- [x] **TODO / FIXME panel**: workspace-wide scan for `TODO`, `FIXME`, `XXX`, `HACK`. Output pane with file-grouped tree, Refresh + live filter, severity-colored tags. Auto-scans on project open; `view.show_todos` action.
- [x] **Structure-based selection** (Alt+Up / Alt+Down): LSP `textDocument/selectionRange` driven expand/shrink. Per-editor chain cached on first expand, invalidated when the user's selection diverges from the indexed range.
- [ ] **Parameter hints pinned inline**: signature popup is modal and dismisses; also show the active parameter (name + type) inline next to the open paren until the call is closed.
- [x] **Markdown preview**: for `.md`/`.markdown` files, a split preview pane driven by `QTextBrowser::setMarkdown`. Live re-renders on edit (200ms debounce) with proportional scroll-sync both ways. `Ctrl+Shift+V` toggles the preview.
- [x] **Zoom** (Ctrl+= / Ctrl+- / Ctrl+0): global editor font size, clamped to [8, 40], persisted to `Editor/FontSize` in `QSettings`. Applies to every open editor in both split groups.
- [ ] **Regex tester**: a small dialog/pane with a regex input and a text-to-test-against input, showing matches live. Useful when building Find-in-Files patterns.
- [ ] **Multi-root workspaces**: open two unrelated project roots in one window. Projects-pane becomes multi-root; LSP per root.

### Local History / Timeline (Tier A)
- [x] **Periodic snapshot on save**: every save copies the file into `.einsums-studio/local-history/<relpath>/<timestamp>.snap` (capped, e.g., 100 per file).
- [x] **Timeline widget**: editor right-click → "Show Local History" → dialog with a scrollable list of snapshots, DiffWidget against current, "Restore" button. (TODO: also hook from Projects tree, currently only from the editor.)
- [x] **Prune policy**: keep last 100 per file, drop anything older than 30 days; hardcoded for now, wire to Settings later.

### VCS Tier (Tier A: partial commits + inline hunks)
- [ ] **Partial commit / stage hunks**: diff view in Git Status pane gets per-hunk checkboxes; committing only stages the ticked hunks. Mirror CLion's commit panel.
- [x] **Discard hunk / revert hunk from gutter**: right-click the gutter strip → "Revert Hunk". Runs `git apply --reverse` on a single-hunk patch, then reloads the buffer and refreshes the gutter.
- [x] **Inline diff on gutter hover**: hover the gutter strip → floating QFrame showing the hunk's raw diff lines with +/- coloring.
- [x] Inline blame annotations: `view.toggle_inline_blame` fires `GitManager::blame()` on a worker, renders `author · date` at end-of-line via a new `InlineAnnotation::Blame` category. Skips uncommitted-hash lines, clears on edit.
- [ ] Git log viewer: commit history with graph, diff per commit
- [ ] Branch management: create/switch/merge branches from UI

### Debug Tier (continued)
- [ ] **Smart step into**: at a line with nested calls `foo(bar(baz()))`, prompt which call to step into. DAP's `stepIn` takes an optional `targetId`; `stepInTargets` request populates the picker.
- [ ] **Drop frame**: restart from the top of the current function. Most DAP servers support it (`restartFrame`).
- [ ] **Watch expression pinning**: pin a watch expression to stay visible in the Variables panel across step operations.

### AI (Tier D: debatable, but worth listing)
- [ ] **AI chat panel**: right-side dock with an LLM chat conversant in the current file / selection / enclosing symbol. Most useful for "explain this template error", "generate a Catch2 test for this function", and "why is this `einsum` dispatching to the generic backend?". Pluggable backend (OpenAI/Anthropic/local); API key in Settings.
- [ ] Multiple cursors: Ctrl+D to select next occurrence, Alt+Click for additional cursors
- [x] Snippet expansion: LSP completion items have insertText with $1/$2 placeholders; Tab to jump between
- [x] Linked snippet tabstops: same-numbered `$1` occurrences mirror edits via `contentsChange` hook
- [x] Semantic highlighting: LSP semantic tokens for more precise coloring than regex-based
- [ ] Minimap: scroll preview sidebar (like VS Code)
- [ ] Tab bar: visual tabs for open files per editor group (currently dropdown only)
- [x] File watcher reload prompt: QFileSystemWatcher detects external changes, prompts reload (partially wired)
- [x] Sticky scope headers: pin enclosing function/class/namespace signatures to the top of the viewport on scroll
- [x] Focus mode: dim everything outside the innermost scope (Cmd+Option+F)
- [x] Occurrence highlighting: `textDocument/documentHighlight` on cursor rest

### Git Tier
- [x] Inline blame annotations: `view.toggle_inline_blame` fires `GitManager::blame()` on a worker, renders `author · date` at end-of-line via a new `InlineAnnotation::Blame` category. Skips uncommitted-hash lines, clears on edit.
- [ ] Git log viewer: commit history with graph, diff per commit
- [ ] Branch management: create/switch/merge branches from UI

### Debug Tier
- [ ] Watch expressions: user-added expressions in Variables pane
- [ ] Inline debug values: show variable values at end of lines when stopped (annotation category exists: DebugValue)
- [ ] Memory viewer: hex dump of memory at address
- [ ] Conditional breakpoints: right-click breakpoint to add condition
- [ ] Disassembly view: fill in the Assembly/IR stub with actual disassembly from DAP

### Testing Tier
- [ ] **Coverage gutter**: per-line red/green strip in `LineNumberEditor` driven by gcov/llvm-cov/lcov output. Parse the report after a test run, attach counts to each line via the same per-file metadata we already track for git line changes. Click a red line to jump to a test that covers a sibling line. Settings toggle to rebuild with `--coverage` flags, or consume an existing `*.gcov`/`*.info` file.

### Profiler Tier
- [ ] Roofline model: arithmetic intensity vs FLOP/s (stub exists)
- [ ] Micro-architecture analysis: hardware counter visualization (stub exists)
- [ ] Differential flame graph: compare two profiles, show regressions
- [ ] Memory profiling: allocation tracking timeline

### Quality of Life
- [x] Recent files: Ctrl+E MRU dropdown of recently opened files, deduped by path, persisted per project to `.einsums-studio/recent-files.json`.
- [x] Persistent session restore: reopen last workspace with tabs/cursors on launch (via `ProjectContext::session_state`)
- [x] Integrated terminal "Run" output: build/run output in terminal tabs (not just build pane)
- [ ] Drag-and-drop tabs: between editor groups
- [ ] Custom keybindings: rebindable shortcuts in settings
- [ ] Extension marketplace: discover and install plugins (longer term)
- [ ] F8 / Shift+F8 problem navigation: jump through diagnostics across open files
- [x] **Scratchpad buffer**: Ctrl+Shift+N (`file.new_scratchpad`) creates an in-memory editor tab with a synthetic `scratchpad://N` path, no disk read / LSP / git / test-marker work. Ctrl+S is a silent no-op and close-prompt skips the scratchpad. Future: "Save As" conversion.
- [ ] **Paste cleanup**: on paste, optionally strip common clutter: leading `>>> `/`$ ` prompts, `N: ` line-number prefixes, trailing whitespace. Settings toggle (off by default). When code is pasted from Stack Overflow / terminal transcripts it arrives clean.
- [ ] **Scroll-sync split**: two editor panes side-by-side that scroll in lockstep by line number. Poor man's diff review; also useful for header/implementation pairing. Toggle via a button on the split's title bar.
- [ ] **Copy with context**: Cmd+Shift+C copies the selection plus the enclosing function signature (and optionally the enclosing class/namespace) as a comment header. When you paste the snippet elsewhere it arrives self-documenting.

### Novel Features: Einsums-specific
These are features that genuinely leverage what makes Einsums Studio different from a generic IDE. Ordered roughly by expected payoff + buildability.

- [ ] **Einsum expression visualizer**: cursor on an `einsum(Indices{...}, C, Indices{...}, A, Indices{...}, B)` call → inline diagram (or side panel) showing the contraction pattern, free vs reduction indices, rank per tensor. Parsed from source text (no runtime needed). The most domain-specific feature we can add.
- [ ] **Algorithm dispatch indicator**: inline annotation on every `einsum` call showing which backend the compile-time dispatcher picks: `→ dgemm`, `→ packed_gemm`, `→ generic`. Data source is the same trait machinery that drives `Dispatch.hpp`; optionally confirmed against profiler trace when available. Makes the backend selection visible without digging through logs.
- [ ] **Profile overlay / hot-path heatmap**: after a profiler run, paint a per-line heatmap column showing ms-per-line (or sample count). Pulls from the existing profiler bundle's sample data. Click a hot line to open the profiler detail pane scoped to it.
- [ ] **Auto-benchmark diff on save**: when a saved function is covered by a benchmark, re-run that benchmark in the background and drop a `±N%` chip next to the function signature vs. the previous saved run. Uses the benchmark DB that already exists.
- [ ] **ComputeGraph visual editor in Studio**: graduate the viewer's standalone graph editor into a proper Studio bundle so you can edit `cg::Graph` code as a DAG and round-trip to source. Most of the editor already exists in the profile viewer; this is mainly a bundle port.
- [ ] **Build-time heatmap**: parse ninja's `-d stats` / `.ninja_log` per-TU durations, decorate a right-edge column in the editor with compile-time color. Pairs with the build-log parser we already have. Instant "why is this file slow to build".
- [ ] **Contraction rewriter**: select a chain of einsums → suggest algebraic reorderings that change cost (associativity, common subexpression sharing, fuse + hoist). Static-only analysis.
- [ ] **Vectorization / auto-vec report overlay**: compile with `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize`; parse the diagnostics, map them to source lines, render `✓`/`✗` chips in a right-edge column (or inline annotation) with the reason on hover ("vectorized VF=4, UF=2" or "not vectorized: memory dependence"). Makes auto-vec opportunities and failures visible without reading compiler output. Especially valuable for a numerical library where vectorization is load-bearing. Optionally extend to `-fopt-info-vec` for GCC.

### Novel Features: General IDE
- [ ] **Minimap with diagnostics**: thin column on the right with the whole file's shape + red/yellow dots at diagnostic locations; click to scroll. Doubles as diagnostic navigator.
- [ ] **Test impact map Tier 2**: live version of the manual "Show Tests Covering This" lookup: track edited symbols, on each save compute affected tests and badge them in the Test Explorer.
- [ ] **Test flakiness tracker**: record per-test pass/fail history, badge flaky tests in the explorer.
- [ ] **Incremental tests on save**: run impacted tests in the background whenever a file is saved, show inline pass/fail chips on affected TEST_CASE lines.
- [ ] **Regression test synthesizer**: after a bug fix, generate a Catch2 test from the failing input + expected output captured during the debug session.
- [ ] **Error history pane**: running list of every diagnostic seen this session, even ones already fixed. Useful when you want to go back and reread something that vanished after a successful build.
- [ ] **Focus/Zen polish**: expand the current focus mode with a toggle for "only the function under the cursor is editable" (hard focus).
- [ ] **Clipboard history ring**: Cmd+Shift+V cycles through recent clipboard entries.

### Remote Development (VSCode/CLion-style)
Run the IDE locally but operate on code that lives on a remote host: SSH to the box, run clangd / build / test / debug there, keep the UI fast and local. Several architectural seams already exist that make this tractable.

**Pragmatic Tier 3 (~4 weeks full-time, working daily driver):**
- [ ] **SSH transport**: persistent `libssh2`/`QSsh` channel, JSON-RPC framing (reuse LSP framer), reconnection on drop
- [ ] **`IProcessRunner` split**: `LocalProcessRunner` (current) + `SshProcessRunner` (prepends invocation over the SSH session). `GitManager`, `GitHubManager`, `LspClient` spawn, `BuildManager`, `TestRunner`, `DapClient` all come along for free
- [ ] **`IFileSystem` abstraction**: `LocalFileSystem` + `RemoteFileSystem` (SFTP-backed). Audit `QFile`/`QFileSystemModel`/`QDir`/`QDirIterator` usages, route through the interface
- [ ] **File watching over the wire**: remote inotify/FSEvents watcher, tunneled notifications; replaces `QFileSystemWatcher` when in remote mode
- [ ] **Path + URI translation**: all LSP/DAP messages carry absolute paths; remote paths won't match local mount points. Translate at the transport boundary on both sides
- [ ] **Remote terminal**: vterm connects to a PTY on the remote via `ssh -t`
- [ ] **Credentials + connection UI**: host picker, SSH key/passphrase/agent selection, known-hosts handling

**Full VSCode parity (3-6 months extra):**
- [ ] **`einsums-studio-server` binary**: headless backend auto-deployed to the remote on first connect. Version-matched client/server upgrade flow. Sessions survive disconnects (queued RPC, client-side reconciliation on reconnect)
- [ ] **Offline session resilience**: close the laptop mid-build, come back, build output resumes streaming from where it stopped
- [ ] **Latency-hiding everywhere**: audit every `QDir::entryList()` / stat / read for round-trip-per-panel-render; batch and cache
- [ ] **Multi-window, multi-host**: single workspace spanning remotes, session reuse across windows
- [ ] **Settings/theme sync**: per-host overrides, roaming preferences

**Why it's feasible at all:**
- `ProcessRunner` already centralizes process spawning (from the earlier audit)
- LSP's JSON-RPC framing is reusable for the transport
- The profiler bundle is already remote-capable (TCP)
- Editor/UI/theme stay local unchanged

**Why it takes real time even at Tier 3:**
- Every "open a file" path that's synchronous becomes a round-trip; the UI code needs an async audit
- Reconnection/offline semantics are the hard part; VSCode spent years on this
- Corporate SSH configs (multi-hop, VPN, agent forwarding, certificate auth) eat a lot of polish time

Recommended order: pragmatic Tier 3 first to validate the seams and the workflow, then decide if the polish delta is worth upgrading to a proper server binary.

# Editor
- [x] Problems panel: right click on row and apply fix.
- [x] Problems panel: select multiple rows and apply all fixes given.
- [x] Are the diagnostics from LSP being processed on the main thread?
- [x] If a Project is already open and the user selects to open a new Project, first as if it should be opened in a new Studio instance or replace the current on.
- [x] Ensure multiple instances of Studio can be run.
- [x] The test runner executable target discovery is messing up on pytests registered inside ctest. It is picking up `python3` as the executable. This results in `cmake --target python3` being called and `ninja: error: unknown target 'python3'`.
- [x] Terminal pane will automatically scroll to the right when an output line is wider than the panel. I don't want it to do that I want it to start scrolled to the left and I will scroll to the right manually if desired.
- [x] The Clear button on the Log panel should become a Trashcan icon like on the Build panel.
- [x] The Test nav panel should detect when ctests are skipped by default and grey out the test from the list and not run it automatically. If the user explicitly highlights it and clicks run then run it.
- [x] The code editing documents that offer previewing a rendered document should default to not showing the preview. Provide the user a button on the view toolbar to show/hide the preview pane.
- [x] In the Claude panel, if Claude is outputting anything the window is scrolled to the bottom to always show the update. I am unable to scroll up to view something from earlier until Claude is done generating output.
- [x] In the Claude panel (and probably in the Terminal panel, too), the Tab key is not being sent to the terminal app. For example, Claude frequently presents a list of Yes or No and the option to hit Tab to amend. When I hit Tab it seems like focus is lost or something.
- [x] Surround with's namespace template is inserting "ns" for a default namespace name
- [x] When a file is automatically reloaded it looses syntax highlighting.
- [x] The File, Session, and View menus are very disorganized. Let's fix that.

## ComputeGraph
  Small follow-ups (mirrors of work already done):
  1. Bind read_etn / write_etn (full-tensor graph variants) to Python. They live in the same
  header as the slice variants and have the same template shape, so it would just need
  INSTANTIATE_AS annotations across the four dtypes. Currently full-tensor graph reads from
  Python require building a slab covering the whole tensor, which works but is awkward.
  2. Bind the no-deps cg.custom(label, fn) overload. Codegen would need to emit a static_cast
  to disambiguate free-function overloads; currently we have to drop the EXPOSE on it and
  require a tensor argument always.

  Stubgen polish:
  3. slab: Slab instead of slab: Any in io.pyi. Stubgen's TypeTranslator doesn't yet recognize
  the bound Slab class as a Python type for its parameters.

  Optional shape variants:
  4. Bind Graph::add_loop(label, max_iter, cond) -> Graph& (the subgraph-returning overload)
  for a context-manager Python pattern: with g.add_loop("tiles", 4, cond) as body_g: with
  cg.capture(body_g): .... Useful when the body shape is dynamic in a way that doesn't fit a
  def body() cleanly.

  Out of scope for #790:
  5. Wiring IOPrefetch to recognize slab DiskRead/DiskWrite nodes for async overlap on the
  DataflowExecutor; that's a different milestone.
