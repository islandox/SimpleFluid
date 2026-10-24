# SimpleFluid

A finite-volume computational fluid dynamics code for incompressible Boussinesq
natural convection, built on [Trilinos](https://trilinos.github.io) for distributed-memory,
GPU-portable parallelism.

## Project Status

SimpleFluid is in active development. Its incompressible Boussinesq foundation
covers mesh infrastructure, finite-volume equations, turbulence, physical heat
sources and material properties. The reacting stack uses supplied catalogs,
conservative reactions and schema-indexed species/population inventories.
A separate native low-Mach backend owns phase internal energy and conservative
multispecies transport. Each path has a defined support boundary; numerical
implementation does not establish physical validation of a supplied closure.
Remaining acceptance and multiphysics work is tracked in `TODO.md`.

The pressure–velocity, Boussinesq, turbulence, and optional-physics stacks now
run directly on `MeshHandle`/`FieldStored` data for supported mapped meshes,
without reconstructing a legacy mesh. Existing callers that supply the legacy
`Mesh` backend retain their established object identity and behavior through a
separate compatibility path.

The reacting stack has two distinct thermal owners: a weak Boussinesq field
adapter, including an explicit finite-solvent/vapor reference approximation,
and a native conservative low-Mach phase-energy solver. Sources, heat and
reservoir transfers are declared independently. No enabled path infers H2
chemistry, solvent, gas yield or a heat-to-gas conversion by default.

**Turbulent bubbly-flow scope:** the single-continuum RANS models can be coupled
to schema-owned field populations with sequential void/material feedback in
supported configurations. This is not an Euler–Euler two-fluid model. Separate
gas momentum, interphase-force closures and quantitative turbulent bubbly-flow
validation are outside that claim.

| Capability | Status |
| ---------- | ------ |
| Mesh & geometry infrastructure | ✅ |
| Scalar, vector, and tensor fields (Tpetra) | ✅ |
| Native `MeshHandle`/`FieldStored` FVM and solver path | ✅ |
| Orthogonal diffusion operator | ✅ |
| Non-orthogonal correction (explicit/implicit/hybrid) | ✅ |
| Momentum equation (transient + convection + diffusion) | ✅ |
| Pressure–velocity coupling (SIMPLE/PISO/PIMPLE) | ✅ |
| Coupled Krylov solver (block Schur + MueLu AMG) | ✅ |
| Optional NOX nonlinear velocity-pressure solver (native fixed orthogonal mesh, frozen physical coefficients) | ✅ |
| Rhie–Chow collocated stabilization | ✅ |
| Verification suite (cavity smoke cases, Poiseuille, MMS) | ✅ |
| External OpenFOAM profile-comparison workflow | ✅ |
| Boussinesq natural convection examples | ✅ |
| Performance benchmarks & scaling tests | ✅ |
| Physical heat sources & material-property fields | ✅ |
| Solid subdomains & transient heat conduction | ✅ |
| Prescribed volumetric heat deposition | ✅ |
| Conservative catalog/stoichiometric source kernel | Implemented; supplied application laws |
| Six two-equation RANS closures and wall treatments | ✅ |
| Scalar void, boiling, and material feedback infrastructure | 🚧 |
| Schema-indexed species fields and named populations | Defined support boundary; closure validation is application-specific |
| Weakly coupled RANS plus gas-bubble populations | 🚧 |
| Fixed-grid planar volume budget and weak Boussinesq integration | 🚧 |
| Structured planar geometry-motion/GCL substrate | ✅ |
| Constrained solver-integrated planar ALE and generalized continuity | 🚧 |
| Reusable scalar/vector/tensor cell transfer, including conservative supported overlaps | ✅ |
| Native conservative multispecies low-Mach flow | Closed orthogonal scope; no general interface/cross-mesh occupancy claim |
| Full turbulent Euler–Euler bubbly flow | ⬜ |
| General reaction-source and species-provider contract | Implemented; explicit providers required |

## Governing Equations

The default Boussinesq flow path solves the incompressible Navier–Stokes
equations with the Boussinesq approximation for buoyancy-driven flows:

$$
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\mathbf{u})
= -\nabla p + \nu\nabla^2\mathbf{u} + \rho\beta g(T - T_{\mathrm{ref}})
$$

$$
\nabla\cdot\mathbf{u} = 0
$$

$$
\frac{\partial T}{\partial t} + \nabla\cdot(\mathbf{u}T) = \alpha\nabla^2 T
$$

The constrained Boussinesq planar-ALE path advances conservative old/new-volume
storage with the mesh-relative flux
$\phi_{\mathrm{rel}}=\phi_{\mathrm{abs}}-\phi_m$ and enforces the integrated low-Mach target
$\sum_f\phi_{\mathrm{abs},f}=Q_{V,c}$. Its supported model matrix and discrete
equations are documented in
[`docs/modeling/planar_free_surface_volume_budget.md`](docs/modeling/planar_free_surface_volume_budget.md).

For a solid region, the velocity term is absent and the conservative thermal
equation is

$$
\rho_s c_{p,s}\frac{\partial T_s}{\partial t}
= \nabla\cdot(k_s\nabla T_s) + \dot q_s.
$$

## Numerical Methods

### Discretization

- **Collocated** finite-volume method on supported hexahedral and
  triangular-prism meshes
- **First-order upwind** convection (implicit) and **Backward Euler** time
  integration by default
- Opt-in constant-step **BDF2** and bounded **linear-upwind** deferred
  correction for legacy and native mapped weighted-scalar transport, plus the
  native mapped physical-temperature path
- Gradient reconstruction via cached **least-squares** stencils, with a
  selectable **Gauss-linear** path for pressure and turbulence gradients

### Diffusion Operator

Three non-orthogonal treatments, selectable at runtime:

| Mode | Description |
| ---- | ----------- |
| `explicit` | Orthogonal two-point flux implicit + non-orthogonal cross-diffusion deferred |
| `implicit` | Full diffusion Jacobian including least-squares gradient neighbors |
| `hybrid` | Implicit operator with explicit correction iterations |

### Pressure–Velocity Coupling

| Mode | Description |
| ---- | ----------- |
| `SIMPLE` | Predictor → pressure correction → velocity correction → flux correction |
| `PISO` | One momentum predictor + multiple pressure corrections per step |
| `PIMPLE` | Outer nonlinear loop with inner PISO corrections |
| `coupledKrylov` | $\begin{bmatrix}A_u & G \\ D & C\end{bmatrix}$ system (including Rhie–Chow stabilization) with a block Schur preconditioner and MueLu AMG on the Schur complement |
| `coupledNonlinear` | Optional NOX/Thyra nonlinear solve with analytic convection derivatives, native Schur preconditioning, and transactional acceptance; see [supported scope and configuration](docs/architecture/coupled_nonlinear_solver.md) |

The coupled Krylov solver uses Belos **block GMRES** with an Ifpack2/MueLu
block-preconditioning strategy for robust convergence on challenging meshes.

The default coupled representation is assembled CRS. The opt-in
`TimeStepperOptions::coupled_operator_backend = CoupledOperatorBackend::BlockComposite`
uses the existing momentum, gradient, divergence and stabilization blocks
without constructing the monolithic matrix. Independently,
`coupled_workspace_policy = CoupledWorkspacePolicy::StreamedProducts` releases
sparse products component by component. Both choices preserve the assembled
momentum/Schur preconditioners. See the [architecture, supported scope and
measured tradeoffs](docs/architecture/coupled_operator_backends.md).
These options do not implement scalar matrix-free transport. The
[supported-solver qualification contract](docs/architecture/coupled_operator_backends.md#numerical-consistency-and-qualification)
covers all four coupled choices, both pressure-gradient schemes, mesh families,
physical-model regressions, and constrained ALE. Coupled MueLu reuse refreshes
numeric state, and timestepper continuity assembly uses the same interpolation
as final Rhie–Chow flux reconstruction.

### Segregated Linear Solves

Segregated momentum, temperature, turbulence-scalar, and pressure equations
use RHS-norm-scaled convergence. Momentum, temperature, and turbulence
scalars retain the preceding accepted field as the next initial guess.
`LinearSolverOptions` selects `gmres`, `bicgstab`, or `cg` and the `none`,
`jacobi`, `ilu0`, `ilut`, `dic`, `gaussSeidel`, `symmetricGaussSeidel`, or
`MueLu` preconditioner. CG requires a symmetric positive-definite matrix and
compatible preconditioner; ILU0, ILUT, and forward Gauss–Seidel are rejected
with CG. DIC requires a symmetric matrix and positive incomplete pivots.
The general transport operators and the row-gauge-fixed pressure operator can
be nonsymmetric.

Acceptance uses the recomputed true residual against the requested tolerance.
Near-roundoff GMRES residual gaps can trigger compensated CRS residual checks
and bounded defect correction within the original iteration budget. These
checks do not relax the linear tolerance or physical conservation gates.

`FluidSolver::set_linear_solver_options()` changes subsequent momentum and
transported-scalar solves. Pressure projection has an independent policy
through `set_pressure_linear_solver_options()`, allowing its cached matrix and
preconditioner to be reused without forcing the same choice on changing
transport matrices.

### Rhie–Chow Stabilization

Pressure-weighted face-flux interpolation prevents checkerboard pressure modes
on collocated grids. Compatible with all four pressure–velocity coupling modes.

## Features

### Mesh

- CRTP-based mesh hierarchy with type-erased `MeshHandle` (via `std::variant`)
- Runtime-switchable mesh types:
  - **Orthogonal Cartesian 3D** — structured hexahedral cells
  - **Orthogonal Cylindrical 3D** — polar-structured hexahedral cells
  - **Unstructured mesh** — STK-free hexahedral and triangular-prism
    connectivity; serial directly and distributed after
    `MeshPartitioner`/`PartitionedMesh` adaptation
  - **Semi-structured XY×Z** — polygonal XY × structured Z prisms with native
    serial/MPI ownership through XY graph partitions crossed with Z slabs,
    shared XY reordering, compact local topology, initially contiguous owned
    cell/face Tpetra maps, ghost layers, and axial ALE through `MeshHandle`;
    see [distribution and support](docs/semi_structured_parallel.md)
  - **Compact multi-region** — conforming/coarse-fine composition, including
    distributed extruded and swept R–Z providers with common affine Z motion
  - **STK adapter** — `HEX_8` and `WEDGE_6` meshes via Exodus II files;
    other volume topologies are rejected during assembly
- Cartesian/cylindrical [multidimensional decomposition](docs/architecture/orthogonal_decomposition.md)
  balances cell counts first, uses geometric widths for rank-grid/cut selection,
  and creates contiguous owned cell/face Tpetra maps with compact halo topology
  and arithmetic owned-cell/face geometry lookup
- Owned + ghost cell decomposition for distributed-memory assembly
- CRS-style neighbor connectivity for FVM stencil construction
- Kokkos-based geometry storage — portable across CPU and GPU backends
- Boundary condition support via sideset / side-part name mapping
- Compact solid-cell subdomains with independent distributed maps and named
  synthetic boundaries at solid/non-solid interfaces

### Mesh Generation

- Programmatic mesh generation for **box**, **cylinder**, and **sphere** domains
- External `HEX_8` and `WEDGE_6` mesh loading through Trilinos/STK
- Configuration-driven via the built-in typed key-value `Database`
- [`AnnularSemiStructuredMeshFactory`](docs/annular_meshes.md) — programmatic
  graded annular meshes with independent inner/outer wall rings, Delaunay
  triangular-prism bulk, hex wall layers, and optional swept bottom corners;
  returns a serial/MPI `MultiRegionMesh` with native polygonal volumes

### Fields

- **Cell-centered** scalar and vector fields backed by Tpetra distributed vectors
- **Face-centered** scalar and vector fields backed by Tpetra distributed vectors
- `FieldStored` scalar, vector, and row-major tensor cell storage, plus scalar
  and vector face/boundary-face storage
- Mesh-aware owned/overlap maps, component and bulk host views, owned-face
  indexing, and ghosted data exchange
- `BoundaryFaceField` compatibility storage for sideset-indexed boundary data
- [`MeshToMeshTransfer`](docs/mesh_to_mesh_transfer.md) — reusable scalar,
  vector, and tensor cell transfer with centroid interpolation or conservative
  supported overlaps, explicit intensive/extensive quantities, partial-coverage
  reports, MPI donor exchange, and geometry-epoch validation

### Solvers & Equations

- `FluidSolver` — reusable transient incompressible pressure-velocity driver
- `IncompressibleIsothermalSolver` — transient constant-density flow with
  optional RANS transport and no temperature solve
- `BoussinesqSolver` — thermal natural-convection specialization
- `Reacting::LowMachMixtureSolver` — native conservative species/phase-energy
  flow with supplied mixture EOS and explicit closed-mesh/motion contracts
- `Reacting::ReactingMixtureCoupler` — external transport-first reaction,
  face-source and configured interphase controller for that native solver
- `IncompressibleMomentumEquation` — generic velocity-transport assembly
- `BoussinesqMomentumEquation` — incompressible momentum with buoyancy
- `PressureProjectionEquation` — pressure-correction / Poisson system
- `TemperatureDiffusionEquation` — energy equation assembly
- `SolidHeatConductionEquation` — conservative, zero-advection solid energy
  equation on a `SolidSubdomain`
- `LumpedInnerCavityModel` — lumped headspace temperature and humidity with
  liquid evaporation, wall condensation, natural convection, and enclosure radiation
- `CoupledPressureVelocitySolver` — monolithic block-Krylov solver
- `BelosLinearSolver` — unified interface to Trilinos iterative solvers
- Direct `MeshHandle` execution through `MeshFieldTraits` for mapped pressure,
  velocity, temperature, turbulence, material, and optional-physics fields
- Selected-first `MeshReorderingFactory` layouts for range-backed solid
  subdomains without parent-sized selection or reverse-index tables
- Legacy `Mesh` execution through synchronized compatibility fields, without
  replacing the mesh supplied by the caller
- Runtime residual reporting for momentum, pressure, and continuity
- Named volumetric heat sources and updateable physical material fields

### RANS Turbulence and Wall Treatment

- Runtime standard, RNG, and realizable k-epsilon plus standard, BSL, and SST
  k-omega closures, with laminar mode retaining no turbulence state
- Automatic distributed Poisson wall distance for BSL/SST; an explicit
  positive uniform override remains available for controlled comparisons
- Resolved low-Re SST and standard/realizable k-epsilon treatments with smooth
  walls and molecular wall transport; resolved k-epsilon applies
  $\epsilon=2\nu k/y^2$ in wall-adjacent cells
- Standard high-Re k-epsilon treatment with smooth or equivalent sand-grain
  roughness and either constant turbulent-Prandtl or Jayatilleke heat transfer
- Optional signed OpenFOAM-style Boussinesq production,
  $G_b=C_b\beta(\nu_t/Pr_t)\mathbf{g}\cdot\nabla T$, with stable production
  and destruction splitting in both transported equations; the epsilon C3
  orientation follows OpenFOAM's incompressible `buoyancyTurbSource`, not its
  opposite compressible `buoyantKEpsilon` convention
- Globally reduced per-patch y+ statistics and a rebuild-only boundary-layer
  controller with damping, height bounds, and distributed mesh-quality gates

The principal database selectors are `turbulence_model`, `wall_treatment`,
`wall_boundaries`, `wall_distance_boundaries`,
`wall_distance_non_orthogonal_treatment`, `wall_roughness_model`,
`wall_roughness_heights`, `wall_roughness_constants`, `wall_thermal_law`, and
`turbulence_buoyancy_model`. Roughness arrays are aligned with
`wall_boundaries`. Automatic distance always includes every configured no-slip
velocity patch; explicit names augment that set and must also be no-slip.

The opt-in `SSTKOmegaSAS` selection adds a SAS omega source to the existing
**SST-1994** parent. See [the mathematical contract, configuration and supported
scope](docs/modeling/sst_sas.md) and [qualification status](docs/modeling/sst_sas.md#qualification-status).
The `sst_sas_activation` example is a small, isothermal transient source check
with optional diagnostics. The [supported model paths](docs/modeling/sst_sas.md)
cover cylindrical and serial semi-structured meshes, mixed slip and translational
periodicity, corrected Gauss-linear gradients, and accepted-state transactions
for compatible fixed-grid multiphysics models. Isothermal and physical Boussinesq
SAS also support optional NOX velocity-pressure coupling within its
[mesh, boundary and physics limits](docs/architecture/coupled_nonlinear_solver.md).
Source implementation and solver regression do not
establish physically validated scale-resolving turbulence.

### Turbulent Bubble-Flow Scope

`MultiphysicsSolverCoupler` can combine a RANS-enabled `BoussinesqSolver` with
`ReactingGasMode::Species`, supplied chemistry and named populations.
The flow/RANS stage supplies the carrier flux; normal typed chemistry and
accepted source heat precede temperature, followed by thermodynamic refresh
and accepted material/void feedback. Explicit gas-only reference applications
may select the separately guarded post-temperature schedule.

This remains a loose single-continuum approximation. Turbulence storage and
transport are not a full liquid-fraction-weighted two-fluid system. There is
no independent gas momentum/continuity solve, drag/lift/virtual-mass or wall-
lubrication model, turbulent dispersion or bubble-induced turbulence closure.
Representative bins do not provide a resolved size distribution. Native
low-Mach phase-energy support does not imply RANS qualification of that backend.

The regression fixtures contain causal turbulence, solution-advection, supplied
multiple-gas source, slip, inventory, feedback and rollback oracles. Their
existence is not a claim that every combination/rank count has passed, or that
an application chemistry or turbulent bubbly-flow model is physically validated.

### FVM Operators

- `DiffusionSystem` — scalar/vector orthogonal diffusion assembly
- `TransportSystem` — semi-implicit scalar/vector, weighted, temperature, and
  physical-momentum convection–diffusion assembly
- `NonOrthogonalCorrection` — cross-diffusion flux decomposition ($S_f = E_f + T_f$)
- `CellOperators` — cached scalar/vector gradient, divergence, and Laplacian
  reconstruction for legacy and stored fields
- `FaceFlux` — face interpolation, normal fluxes, cell balances, and
  FieldStored/legacy Rhie–Chow interpolation
- `MatrixOperators` — sparse diffusion, upwind, and pressure-Poisson assembly
  helpers for legacy and mapped meshes
- Mapped boundary and transport caches — reusable boundary locations,
  least-squares geometry, and compatible transport-matrix graphs

### Data & Utilities

- Typed key-value `Database` (int, real, string, bool, and vector variants)
- `RandomAccessView` — lightweight non-owning view with STL-compatible iterators
- Generic `vec3` 3D vector class with arithmetic operators
- Concept-based `TpetraTypePack` for clean Trilinos type management
- Mesh utility functions (centroid, face/cell type classification)

### I/O

- Cached-topology VTU output via `VTUWriter`, with appended binary arrays and
  rank-local pieces plus a `.pvtu` index for ParaView
- STK Exodus II `HEX_8` and `WEDGE_6` mesh input via `STKMesh`

### Parallelism

- MPI distributed-memory parallelism via Trilinos/Tpetra
- Kokkos device-resident mesh geometry and device-local coupled-preconditioner
  packing (CPU Serial/OpenMP and accelerator backends); general FVM matrix
  assembly remains host-side
- Zoltan2 graph/hierarchical mesh partitioning

## Verification

The automated suite combines analytical checks, convergence tests, and
short-running physical smoke cases:

| Case | Description |
| ---- | ----------- |
| Lid-driven cavity | Re = 100, 1000 transient smoke cases; optional centerline-profile export |
| Poiseuille flow | Parabolic profile recovery in a channel |
| Manufactured solution | Method of manufactured solutions for Navier–Stokes |
| Skewed diffusion | Non-orthogonal mesh convergence with all three treatments |
| Natural convection cavity | Differentially heated square cavity |
| OpenFOAM comparison | Manual external profile comparison; automated configuration and boundary-condition check |
| OpenFOAM pitzDaily | Five-block standard-k-epsilon duct case with an authenticated fail-closed comparator; the physical acceptance manifest remains pending qualification |
| OpenFOAM Gaussian tank | 1000 W axisymmetric SST tank with matched 50 x 150 R-Z distributions and error figures |
| Native mesh path | Cartesian, cylindrical, serial/MPI semi-structured flow and axial ALE, serial unstructured, partitioned-unstructured MPI, coupled-Krylov, and optional-physics regressions without legacy conversion |
| Fixed-grid planar volume budget | Analytic/component, global/cellwise inventory, Boussinesq integration, and two-rank conservation checks; no moving-surface claim |
| Planar geometry/GCL substrate | Structured/extruded motion, transactions, epoch/cache staleness, quality, and serial/two-rank GCL/shared-face checks |
| Constrained planar ALE | Old/new-volume transport, mesh-relative flux, generalized continuity, moving-boundary, rollback, and conservation component/integration checks; no quantitative pool-dynamics validation |

The source tree contains focused turbulence, species/population and combined
regression fixtures. Fresh serial/MPI output, actual rank counts, skips and ABI
checks establish which configurations ran. Numerical fixtures do not establish
quantitative turbulent bubbly-flow or application-chemistry validation.

The cavity smoke tests do not currently assert agreement with Ghia et al. or
with bundled OpenFOAM profile data. See
`verification/openfoam/cavityFlow/README.md` for the cavity workflow and
`verification/openfoam/pitzDaily/README.md` for the pitzDaily workflow. The
matched Gaussian tank workflow and its interpretation are documented in the
[legacy tank case directory](verification/openfoam/heatedSolutionTank/README.md).
Matched steady and transient checks for the supported dispersed-bubble
transport and constrained planar-ALE models are documented in
[`verification/openfoam/README.md`](verification/openfoam/README.md).
They include independent OpenFOAM finite-volume reference applications,
physical-time and sample-coverage checks, and conservation acceptance gates.
Both case families require `SIMPLEFLUID_ENABLE_IF97=ON` and share IF97
liquid-water reference properties with OpenFOAM; ALE retains the supported
linear Boussinesq density approximation.

## Examples

Pre-built example executables:

| Example | Description |
| ------- | ----------- |
| `natural_convection_box` | 3D heated box with Boussinesq convection |
| `natural_convection_cylinder` | Cylindrical domain natural convection |
| `natural_convection_sphere` | Spherical domain natural convection |
| `natural_convection_boundary_layer_box` | Box with thermal boundary layer resolution |
| `natural_convection_shiri` | MPI-capable annular natural convection with standard k-epsilon and adaptive steady-state search |
| `pitz_daily` | OpenFOAM pitzDaily geometry with transient isothermal standard k-epsilon transport |
| `reacting_solution_tank_demo` | Supplied multiple-gas chemistry with independent Gaussian heat deposition |
| `heated_solution_tank_sst` (legacy target name) | 1000 W Gaussian tank SST case for matched OpenFOAM R-Z verification |
| `constant_power_cylinder_vessel` | Finite gas and solvent/vapor inventories with coordinated bulk/wall Boussinesq heat ownership |
| `planar_free_surface_verification` | Five analytic fixed-grid volume-budget cases with CSV diagnostics |
| `planar_ale_verification` | Four solver-integrated planar-ALE conservation cases: heating, gas generation, complete two-gas escape, and rollback |
| `dispersed_bubble_verification` | IF97 reference-water steady source/escape balance and transient bubble transport with matched OpenFOAM references |
| `planar_ale_comparison` | IF97 reference-water linearized thermal expansion and source-off steady equilibrium with matched OpenFOAM moving-mesh references |
| `bottom_heated_bubbly_convection` | IF97 reference-water bottom heat/H2 source with solved buoyant plume and return flow; matched OpenFOAM transient and x–z figures |

Examples use explicit C++ providers, `Database` controls, documented environment
inputs, or a combination of these. Enabled generic chemistry needs supplied
catalog/mechanism/material state rather than a flat-file fallback. Smoke settings
reduce mesh size and run length; production defaults can be substantially more
expensive. The explicitly physical H2/IF97 reference applications retain those
names and their documented carrier approximation. Synthetic multiple-gas cases
and finite vapor examples are not application-level physical validation.

## Build

Maintainers and contributors should also read
[`MAINTENANCE.md`](MAINTENANCE.md) for architecture boundaries, focused test
commands, extension paths, numerical review checks, profiling, and Doxygen
generation.

### Prerequisites

- C++23 compiler (GCC ≥ 13 or Clang ≥ 17)
- CMake ≥ 3.21
- Trilinos with:
  - Kokkos, Teuchos, Tpetra
  - STK (IO, Mesh, Topology, Util)
  - Belos, Ifpack2, MueLu, Zoltan2
  - Optional: NOX together with ThyraCore and ThyraTpetraAdapters (NOX must be built with Thyra support)
- MPI (OpenMPI or MPICH)
- Boost headers with Multiprecision (`cpp_int`)
- Google Test (required whenever testing is enabled; testing defaults to `ON`)

CMake does not enforce version floors for Trilinos, MPI, Boost, or GTest.

### Configure & Build

```bash
# Using CMake presets (recommended)
cmake --preset GCC-ninja-multi
cmake --build --preset GCC-Release

# Or manually
cmake -B build/manual -G "Ninja Multi-Config" \
  -DTrilinos_DIR=/path/to/trilinos/lib/cmake/Trilinos
cmake --build build/manual --config Release
```

The checked-in GCC and LLVM presets place their build trees under `build/gcc`
and `build/llvm`, respectively. A manually configured tree uses whichever path
was passed to `-B`.

The optional water material library is enabled with
`-DSIMPLEFLUID_ENABLE_IF97=ON` (default `OFF`). It provides SI water/steam
properties, saturation data, and a liquid material-field adapter through
`SimpleFluid::IF97`. CMake fetches a pinned CoolProp IF97 release if no local
header is supplied. See [IF97 water properties](docs/modeling/if97_water.md)
for offline configuration, usage, and solver integration boundaries.

The optional NOX/Thyra velocity-pressure solve is enabled with
`-DSIMPLEFLUID_ENABLE_NOX=ON` (default `OFF`) and requires the corresponding
Trilinos packages and adapters. See the
[nonlinear solver guide](docs/architecture/coupled_nonlinear_solver.md) for
configuration and supported mesh/physics combinations.

### Run Tests

```bash
cmake --build --preset GCC-Debug
ctest --preset GCC-Debug
```

### Run Examples

```bash
./build/gcc/bin/Release/natural_convection_box
./build/gcc/bin/Release/natural_convection_cylinder
```

## Performance Benchmarks

Build the standalone benchmark executable and run the local Debug-safe suite:

```bash
cmake --build --preset GCC-Debug --target simplefluid_benchmark
ctest --preset GCC-Debug -L benchmark
```

The benchmark writes one CSV row per measured repetition. Rows include solver
configuration, mesh and MPI sizes, non-orthogonality angles, nonlinear and
Krylov iterations, residuals, setup/solve/total wall time, process memory,
compiler/build metadata, and the Git revision.

The canonical cases are `diffusion_nonorthogonal` and `lid_driven_cavity`.
The legacy `pressure_velocity` spelling remains accepted as an input alias for
`lid_driven_cavity`; emitted CSV rows always use the canonical name.
The cavity workload advances one startup step at nominal Re = 100 in a unit
box, with a unit-speed `ymax` lid and slip front/back boundaries, so it is an
extruded quasi-two-dimensional performance case rather than a steady-profile
validation.

The coupled Krylov implementation reuses compatible matrix graphs, static
reconstruction geometry, Schur-product storage, numeric preconditioner state,
Belos state, and preconditioner scratch vectors. Its default rebuild policy
invalidates this state when an operator graph changes; an explicit always-
rebuild policy remains available for comparison and diagnosis.

Run the larger profiling preset with frame pointers and debug symbols:

```bash
cmake --build --preset GCC-RelWithDebInfo --target simplefluid_benchmark
./build/gcc/bin/RelWithDebInfo/simplefluid_benchmark \
  --preset release-profile \
  --output build/gcc/benchmarks/release-profile.csv
```

Run strong-scaling measurements sequentially:

```bash
mpiexec -n 1 ./build/gcc/bin/Release/simplefluid_benchmark --preset mpi-strong --output build/gcc/benchmarks/mpi-strong.csv
mpiexec -n 2 ./build/gcc/bin/Release/simplefluid_benchmark --preset mpi-strong --output build/gcc/benchmarks/mpi-strong.csv
mpiexec -n 4 ./build/gcc/bin/Release/simplefluid_benchmark --preset mpi-strong --output build/gcc/benchmarks/mpi-strong.csv
```

Run weak-scaling measurements with approximately `32x32x8` cells per rank:

```bash
mpiexec -n 1 ./build/gcc/bin/Release/simplefluid_benchmark --preset mpi-weak --output build/gcc/benchmarks/mpi-weak.csv
mpiexec -n 2 ./build/gcc/bin/Release/simplefluid_benchmark --preset mpi-weak --output build/gcc/benchmarks/mpi-weak.csv
mpiexec -n 4 ./build/gcc/bin/Release/simplefluid_benchmark --preset mpi-weak --output build/gcc/benchmarks/mpi-weak.csv
```

Use `--case`, `--configuration`, `--nx`, `--ny`, `--nz`, `--shear`,
`--repetitions`, and `--warmups` for focused runs. Existing CSV files are
appended only when their header matches the current schema.

## Physical Sources And Materials

`BoussinesqModelOptions` enables conservative temperature transport with
density (`kg/m³`), specific heat (`J/(kg K)`), dynamic viscosity (`Pa s`),
thermal conductivity (`W/(m K)`), and named heat-source fields (`W/m³`).
Missing viscosity and conductivity values are derived from the legacy
kinematic viscosity and thermal diffusivity.

```cpp
SimpleFluid::BoussinesqModelOptions model;
model.reference_density = 1000.0;
model.density = 1000.0;
model.specific_heat_capacity = 4180.0;
model.dynamic_viscosity = 1.0e-3;
model.thermal_conductivity = 0.6;

SimpleFluid::BoussinesqSolver<> solver(
    mesh, boundary_conditions, time_options, linear_options, model);
solver.add_temperature_source("qdot_external", 2.0e5);
```

Material and source updater callbacks run once at the beginning of each time
step, with material updates preceding source updates. The same options can be
parsed from flat `Database` keys using
`boussinesq_model_options_from_database()`.

Auxiliary fields are opt-in for solution output:

```cpp
solver.write_solution_vtu(
    "solution.vtu",
    {.include_sources = true,
     .include_material_properties = true});
```

## Solid Vessel Heat Conduction

`MeshReorderingFactory::selected_cells_first()` evaluates a solid selector by
stable geometry ID and centroid, then reorders local cell ordinals into a
selected-owned prefix and a selected-ghost prefix. MPI ownership requires two
ranges rather than one range spanning owned and ghost cells. Geometry IDs,
Tpetra global IDs, face order, and ownership are preserved. The factory
consumes a uniquely owned mutable handle and reorders it in place, avoiding a
second parent-sized mesh/indexer; pass the handle with `std::move` before
constructing fields or caches. Shared ownership is rejected when a permutation
is needed. If every rank is already selected-first, the original ordering is
retained.

The reordered handle is intended to back the returned range layout and its
subdomains. Reordered legacy/STK handles are rejected by `FluidSolver` because
that compatibility path mirrors fields in the legacy mesh's original ordinal
order. Native structured handles do not use that compatibility path.

`SolidSubdomain` consumes that range certificate and builds compact
owned/overlap cell and face maps. Cell membership and parent/subdomain
translation are arithmetic; the subdomain retains no parent-sized selection,
cell-reverse, or face-reverse arrays. Its only reverse face search is over the
solid-sized face records. Native structured handles reuse their existing
local/global indexer and add no persistent parent-sized permutation. Because
explicit unstructured and STK storage cannot be physically reordered, those
parent handles retain one forward and one reverse local-cell permutation; the
subdomain does not duplicate them. A face between selected and unselected
cells becomes a boundary named `solid_interface` by default; the face owner
and normal are oriented outward from the solid. Selection runs only on owned
cells and is imported to ghosts, so the same API works when the solid crosses
MPI partitions or the parent is a legacy mesh wrapped in `MeshHandle`. The
parent must retain every face-neighbor of a selected owned cell (normally one
ghost layer); insufficient overlap is rejected.

`SolidHeatConductionEquation` advances the physical solid energy balance with
no advective-flux argument. It reuses the temperature operator's variable
conductivity interpolation, non-orthogonal treatment, matrix cache, and
publish-on-convergence behavior:

```cpp
using Pack = SimpleFluid::DefaultTpetraTypes;
using Solid = SimpleFluid::SolidSubdomain<Pack>;
using Reordering = SimpleFluid::MeshReorderingFactory<Pack>;

// parent_mesh may contain fluid and solid cells. Here the outer radial cells
// form the vessel wall. Reordering happens before any field or cache is bound
// to the resulting mesh layout.
auto wall_layout = Reordering::selected_cells_first(
    std::move(parent_mesh),
    [](Pack::global_ordinal_type, const Reordering::Vec3& center) {
        return std::hypot(center.x, center.y) >= inner_wall_radius;
    });
auto wall = std::make_shared<Solid>(std::move(wall_layout));

// Selecting every cell is also supported by constructing Solid(parent_mesh).

SimpleFluid::ScalarCellFieldStored<Pack, Solid> wall_temperature(
    wall, 300.0, "solid_temperature");

SimpleFluid::BoussinesqModelOptions material_options;
material_options.reference_density = 7850.0;
material_options.density = 7850.0;                 // kg/m^3
material_options.specific_heat_capacity = 500.0;  // J/(kg K)
material_options.thermal_conductivity = 16.0;      // W/(m K)
SimpleFluid::TimeStepperOptions time_options;
SimpleFluid::MaterialPropertyFields<Pack, Solid> material(
    wall, material_options, time_options);

SimpleFluid::BoundaryConditionSet boundaries;
boundaries.temperature["solid_interface"] = {
    SimpleFluid::BoundaryConditionType::Dirichlet, 350.0};
boundaries.temperature["rmax"] = {
    SimpleFluid::BoundaryConditionType::Dirichlet, 300.0};

SimpleFluid::SolidHeatConductionEquation<Pack> conduction(
    wall, boundaries);
conduction.advance(
    wall_temperature, 0.1, material, wall_temperature,
    [](Pack::local_ordinal_type) { return 0.0; }); // W/m^3
```

Missing temperature boundaries are homogeneous Neumann. A Neumann value is
the outward temperature gradient in K/m, so a prescribed inward wall heat
flux is converted consistently with its sign and the local conductivity. Use
`set_boundary_conditions()` to replace prescribed interface or exterior data
between time steps.
Exterior convection is configured through `boundaries.convection[patch]`,
with a constant coefficient, Churchill–Chu vertical-plate natural convection,
or Churchill–Bernstein cylinder crossflow. The physical temperature operator
assembles the Robin wall resistance implicitly; correlation coefficients are
updated from the old thermal state at each advance. See
[exterior convection](docs/modeling/exterior_convection.md) for configuration,
equations, applicability, and time-lagging details. Do not also assign a
`temperature` boundary to the same patch.

Automatic two-way temperature/flux continuity with a fluid solve is not yet
implemented; the current solid interface remains a standalone or
prescribed-boundary solve, not a full conjugate-heat-transfer driver.

`LumpedInnerCavityModel` supplies a separate, extensible headspace model above
a liquid free surface and inside vessel walls. It advances uniform moist-air
temperature and humidity and returns per-surface mass, heat, and energy rates
for external coupling. Field batches support facewise or area-averaged sampling
and refresh per-face heat-flux boundaries on coupled thermal subdomains.
See [lumped inner cavity](docs/modeling/lumped_inner_cavity.md)
for the conservation contract, natural-convection options, radiation view
factors, and a pool/wall example.

## Reacting Gas Sources in Solution

The [source contract](docs/modeling/reacting_gas_sources.md) uses immutable
species catalogs, mass/element-conserving mechanisms, finite donors, explicit
rate-volume bases or integrated extents, and independent external heat and
material reservoirs. Species names and catalog order do not select physics.
The old ideal-gas-source and two-population chemistry fallbacks and fixed-H2
public inventory APIs are retired without aliases.

`ReactingGasModel` owns actual schema-indexed species fields and explicitly
named population counts. Its external `MultiphysicsSolverCoupler` retains the
shared solver and owns optional advancement, source/escape histories and
accepted-state rollback. The solver still owns flow and scalar temperature.
Supply matching liquid/gas EOS or the distinct fixed-reference Boussinesq
binding; flat database input alone cannot construct an enabled catalog/provider
configuration. The C++ mode is `Species`, with `solution_transport` for the
solution carrier.

Normal field sources advance before temperature and apply their raw accepted
heat once. The finite bulk/wall Boussinesq adapter reserves actual solvent,
reaction donors, scalar superheat and void capacity under one declared
`rho_ref*cp*T`/latent owner. Independent scalar boiling cannot simultaneously
own those species. Gas-only H2/IF97 references instead declare their external
carrier, material/count reservoirs and zero-applied-heat post-temperature
schedule explicitly; these are documented reference approximations.

`Reacting::LowMachMixtureSolver` and `ReactingMixtureCoupler` provide the
separate native closed, orthogonal multispecies path. It conserves actual
species, phase internal energy and carrier momentum, with common thermodynamic
pressure, common or separate phase temperatures, prescribed phase drift and
explicit dispersed counts. Supplied same-carrier diffusion/Fourier face laws
and configured local interphase heat/material stages have distinct conservative
receipts. The phase-energy owner applies pressure work once; pressure roots
reevaluate constitutive properties without reintegrating source events.

The native model uses zero surface tension and does not provide evolving gas
momentum, a capillary energy equation or general interface dynamics. The weak
field adapter's carried escape enthalpy/momentum are traces, not conservative
phase debits. See the [model/API guide](docs/reacting-gas-models.md), the
[thermal-owner guide](docs/modeling/reacting_bubble_boiling.md), and
[native architecture](docs/architecture/low_mach_reacting_flow.md) for the
support and rejection matrices. Independent thermal deposition still uses
named `add_temperature_source()` or `PrescribedHeatSource` W/m3 inputs;
`ScalarSourceDecayTransport` remains a standalone neutral transport utility.

The Gaussian source demo uses explicitly supplied multiple-gas chemistry;
the constant-power cylinder uses the coordinated finite bulk/wall vapor
approximation. Reference and synthetic workloads preserve their own documented
physical oracles. Their output is not physical validation of a general
reacting solution or permission to combine unsupported energy owners.

## Planar Free-Surface Volume Budget and ALE

The optional Boussinesq surface controller separates physical solution volume,
submerged gas volume, clear/pool level and headspace. With schema-owned fields,
`LiquidMassInventory` follows the already transported Liquid+Dissolved mass
and pure-liquid EOS density; it performs no second mass transport. Accepted
chemical/external/interphase events remain distinct from actual transport
escape. Only physical gas escape reaches headspace/vent, while dissolved
outflow is a separate solution-mass sink.

Fixed-grid closed coupling requires an explicitly supplied mixture headspace
with matching catalog, real initial species and declared pressure/temperature
bounds. Its prescribed-temperature pressure closure is not a headspace energy
equation. Unbound scalar/global-constant-mass volume-budget components remain
separate approximations. See
[species-bound surface ownership](docs/architecture/reacting_species_surface.md)
and [planar volume budgets](docs/modeling/planar_free_surface_volume_budget.md).

The weak field source context uses actual EOS liquid volume for liquid-based
rates and geometric control volume for bulk-based inputs. It does not claim
local total phase occupancy. The surface controller closes its global pool/
headspace budget; it neither clips liquid volume nor substitutes void-reduced
mixture density for pure-liquid density. Histories and opt-in free-surface
fields publish accepted states only.

The constrained Boussinesq `planarALE` path retains laminar dimensional sensible
storage, Backward Euler, a constant-area vessel, vented headspace, a finite
species-bound liquid follower, complete solution advection and the moving-top
escape boundary. Absolute carrier flux belongs to pressure/continuity;
transport uses a distinct mesh-relative flux and accepted-old/new volumes.
Actual upwind material-volume receipts, physical sources and geometric GCL
corrections retain separate ownership. Geometry, flow, thermal/species/count
fields, source/history/headspace and clock accept or roll back together.

Cartesian-axis and supported cylindrical/semi-structured axial motion remain
subject to the motion and boundary contracts. Closed moving headspace,
independent boiling/finite Bouss vapor, RANS, arbitrary nonadiabatic moving
boundaries, non-Backward-Euler integration and unsupported mesh/motion
combinations reject. Read the owning architecture/matrix before selecting
additional physics; source snapshots alone do not authorize motion.

Native `LowMachMixtureSolver` prescribed planar ALE is a different explicit
transaction: the caller/controller supplies live motion authority and a target,
while conservative phase transport and pressure-conditioned phase work retain
accepted-old/trial-new geometry. Preview restores accepted geometry before
returning an immutable trial. Commit/restoration do not reevaluate source/EOS
laws. Its closed, orthogonal, inviscid/slip-wall and nonperiodic limits do not
establish a Boussinesq boiling/free-surface combination.

Neither planar path is a general moving-interface or cross-mesh phase-occupancy
model. VoF and general AMR are outside this reacting-flow support matrix.

## Dependencies

| Dependency | Role | Version | Requirement |
| ---------- | ---- | ------- | ----------- |
| **MPI** | Distributed-memory parallelism | MPI-3.0+ | Required |
| **Trilinos** | Distributed sparse linear algebra, mesh I/O, iterative solvers, and preconditioners | 17.0+ | Required |
| **Boost (headers)** | Multiprecision integers for exact geometry predicates | Not specified | Required |
| **LAPACK** | Dense linear algebra | 3.9+ | Required |
| **BLAS** | Low-level vector/matrix operations | 3.9+ | Required |
| **GTest** | C++ unit testing framework | 1.12+ | Optional |

### Required Trilinos components

| Component | Role | Requirement |
| --------- | ---- | ----------- |
| **Kokkos** | On-node parallelism (CPU/GPU portability) | Required |
| **Teuchos** | Smart pointers, parameter lists, timers | Required |
| **Tpetra** | Distributed sparse linear algebra | Required |
| **Thyra** | Abstract interfaces for operators, vectors, and nonlinear models | Optional |
| **STK** | Unstructured mesh I/O (Exodus format) | Required |
| **Belos** | Krylov subspace iterative solvers | Required |
| **Ifpack2** | Algebraic preconditioners (block smoothers) | Required |
| **MueLu** | Algebraic multigrid (AMG) for pressure/Poisson | Required |
| **NOX** | Coupled nonlinear solver | Optional (Depends on Thyra) |
| **Zoltan2** | Graph/hierarchical mesh partitioning | Required |

## Compact region meshes

`MultiRegionMesh` composes Cartesian, cylindrical, straight-extruded, swept R–Z,
and native unstructured regions through the existing `MeshHandle`/FVM path.
Compact implicit regions support MPI ownership, axial affine ALE,
conforming/coarse-fine seams, and translated periodic patches. Explicit unstructured constituents remain
serial-only in composites. Implicit traversal uses allocation-free ranges;
regular interfaces retain descriptor-sized correspondence. See [region mesh contracts,
example and storage accounting](docs/region_meshes.md).

## Documentation

| Topic | Guide |
| --- | --- |
| Development, builds, tests, and ABI boundaries | [Maintainer guide](MAINTENANCE.md) and [CMake guide](cmake/README.md) |
| External material providers and C ABI | [Material provider guide](docs/material-c-api.md) |
| Implementation status and remaining acceptance gates | [Roadmap](TODO.md) |
| Annular wall layers and swept corners | [Annular mesh guide](docs/annular_meshes.md) |
| Region composition, MPI, and ALE | [Region meshes](docs/region_meshes.md) |
| Conservative cell-field projection | [Mesh-to-mesh transfer](docs/mesh_to_mesh_transfer.md) |
| Coupled linear and nonlinear solvers | [Operator backends](docs/architecture/coupled_operator_backends.md) and [NOX solver](docs/architecture/coupled_nonlinear_solver.md) |
| Native conservative reacting flow | [Species, phase energy, pressure and transactions](docs/architecture/low_mach_reacting_flow.md) |
| Reacting gas sources in solution | [Species, source, energy, and conservation contract](docs/modeling/reacting_gas_sources.md) |
| SST-SAS configuration and supported paths | [Model contract and qualification](docs/modeling/sst_sas.md) |
| Gas sources, boiling, and moving pool volume | [Bubble/boiling model](docs/modeling/reacting_bubble_boiling.md) and [planar ALE](docs/modeling/planar_free_surface_volume_budget.md) |
| External reference comparisons | [OpenFOAM workflows](verification/openfoam/README.md) |

## License

This project is distributed under the MIT License.

## AI Assistance

Parts of this repository may be developed with assistance from AI coding tools,
including OpenAI Codex and DeepSeek API.

Codex-assisted commits may include:

`Co-authored-by: Codex <noreply@openai.com>`

DeepSeek-assisted commits may include custom trailers such as:

`AI-assisted-by: DeepSeek API`  
`AI-model: deepseek-v4-pro / deepseek-v4-flash`
