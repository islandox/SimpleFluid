# SimpleFluid

A finite-volume computational fluid dynamics code for incompressible Boussinesq
natural convection, built on [Trilinos](https://trilinos.github.io) for distributed-memory,
GPU-portable parallelism.

## Project Status

SimpleFluid is in active development with a tested incompressible Navier–Stokes
solver for natural convection. The foundation through Phase 12 is substantially
implemented, covering mesh infrastructure, verification, performance
benchmarking, physical heat sources, updateable material properties, prescribed
fission heating, and the baseline radiolytic-gas source. Numerical verification
gaps and later multiphysics acceptance work remain tracked in `TODO.md`.

The pressure–velocity, Boussinesq, turbulence, and optional-physics stacks now
run directly on `MeshHandle`/`FieldStored` data for supported mapped meshes,
without reconstructing a legacy mesh. Existing callers that supply the legacy
`Mesh` backend retain their established object identity and behavior through a
separate compatibility path.

**The advanced Phase 14.1 two-population radiolytic-bubble model has broad
implementation and focused conservation/transport coverage, but remains under
physical validation and acceptance.**

**Turbulent bubbly-flow scope:** the single-continuum RANS models and the
radiolytic bubble-population model can run in the same `BoussinesqSolver` with
sequential void-to-density/buoyancy feedback. This is not a full Euler–Euler
two-fluid formulation and is not yet a validated turbulent bubbly-flow model.
Permanent serial and exact-two-rank regressions exercise that combined path,
including causal changes from turbulence, dissolved-hydrogen advection, and
void-dependent density feedback; a checked-in combined user example and
quantitative bubbly-flow validation remain open.

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
| Rhie–Chow collocated stabilization | ✅ |
| Verification suite (cavity smoke cases, Poiseuille, MMS) | ✅ |
| External OpenFOAM profile-comparison workflow | ✅ |
| Boussinesq natural convection examples | ✅ |
| Performance benchmarks & scaling tests | ✅ |
| Physical heat sources & material-property fields | ✅ |
| Prescribed fission power-density profiles | ✅ |
| Baseline ideal-gas radiolytic source | ✅ |
| Six two-equation RANS closures and wall treatments | ✅ |
| Scalar void, boiling, feedback, and precursor infrastructure | 🚧 |
| Two-population radiolytic-bubble model | 🚧 |
| Weakly coupled RANS plus radiolytic bubble populations | 🚧 |
| Full turbulent Euler–Euler bubbly flow | ⬜ |
| In-memory TH/neutronics feedback and coupling scaffold | 🚧 |

## Governing Equations

The code solves the incompressible Navier–Stokes equations with the Boussinesq
approximation for buoyancy-driven flows:

$$
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\mathbf{u})
= -\nabla p + \nu\nabla^2\mathbf{u} + \rho\beta g(T - T_\text{ref})
$$

$$
\nabla\cdot\mathbf{u} = 0
$$

$$
\frac{\partial T}{\partial t} + \nabla\cdot(\mathbf{u}T) = \alpha\nabla^2 T
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
| `coupledKrylov` | Monolithic $\begin{bmatrix}A_u & G \\ D & 0\end{bmatrix}$ system with a block Schur preconditioner and MueLu AMG on the Schur complement |

The coupled Krylov solver uses Belos **block GMRES** with an Ifpack2/MueLu
block-preconditioning strategy for robust convergence on challenging meshes.

### Segregated Linear Solves

Segregated momentum, temperature, turbulence-scalar, and pressure equations
use RHS-norm-scaled convergence. Momentum, temperature, and turbulence
scalars retain the preceding accepted field as the next initial guess.
`LinearSolverOptions` selects `gmres`, `bicgstab`, or `cg` and the `none`,
`jacobi`, `ilu0`, `ilut`, or `MueLu` preconditioner. CG must only be selected
when both the matrix and preconditioner preserve a symmetric positive-definite
system; ILU0 and ILUT are not generally CG-compatible. The general transport
operators and the row-gauge-fixed pressure operator can be nonsymmetric.

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
  - **Unstructured mesh** — STK-free polyhedral connectivity; serial directly
    and distributed after `MeshPartitioner`/`PartitionedMesh` adaptation
  - **Semi-structured XY×Z** — 2D unstructured × 1D structured prisms,
    currently serial-only
  - **STK adapter** — `HEX_8` and `WEDGE_6` meshes via Exodus II files;
    other volume topologies are rejected during assembly
- Owned + ghost cell decomposition for distributed-memory assembly
- CRS-style neighbor connectivity for FVM stencil construction
- Kokkos-based geometry storage — portable across CPU and GPU backends
- Boundary condition support via sideset / side-part name mapping

### Mesh Generation

- Programmatic mesh generation for **box**, **cylinder**, and **sphere** domains
- External `HEX_8` and `WEDGE_6` mesh loading through Trilinos/STK
- Configuration-driven via the built-in typed key-value `Database`

### Fields

- **Cell-centered** scalar and vector fields backed by Tpetra distributed vectors
- **Face-centered** scalar and vector fields backed by Tpetra distributed vectors
- `FieldStored` scalar, vector, and row-major tensor cell storage, plus scalar
  and vector face/boundary-face storage
- Mesh-aware owned/overlap maps, component and bulk host views, owned-face
  indexing, and ghosted data exchange
- `BoundaryFaceField` compatibility storage for sideset-indexed boundary data

### Solvers & Equations

- `FluidSolver` — reusable transient incompressible pressure-velocity driver
- `BoussinesqSolver` — thermal natural-convection specialization
- `IncompressibleMomentumEquation` — generic velocity-transport assembly
- `BoussinesqMomentumEquation` — incompressible momentum with buoyancy
- `PressureProjectionEquation` — pressure-correction / Poisson system
- `TemperatureDiffusionEquation` — energy equation assembly
- `CoupledPressureVelocitySolver` — monolithic block-Krylov solver
- `BelosLinearSolver` — unified interface to Trilinos iterative solvers
- Direct `MeshHandle` execution through `MeshFieldTraits` for mapped pressure,
  velocity, temperature, turbulence, material, and optional-physics fields
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

### Turbulent Bubble-Flow Scope

`BoussinesqSolver` can configure a RANS closure and
`sheng2024TwoPopulation` radiolysis together. The pressure–velocity and RANS
systems advance first; the bubble inventories then use the projected liquid
face flux plus the configured rise/slip velocity, after which reconstructed
void can refresh mixture density and buoyancy feedback for subsequent solver
operations.

This is a loose, single-continuum coupling. The current turbulence equations
use full-cell storage and liquid face-flux advection rather than
liquid-volume-fraction-weighted phase equations. There is no separate gas
velocity or continuity equation, or interphase momentum-source closures for
drag, lift, virtual mass, and wall lubrication. Turbulent dispersion,
bubble-induced turbulence, and a full bubble-size distribution are likewise
absent. Those capabilities belong to the deferred Euler–Euler program in
`TODO.md`.

The shipped `pitz_daily` and `fissile_solution_tank_sst` examples exercise
RANS without radiolytic bubbles, while the fissile-solution smoke examples
exercise radiolysis without RANS. A permanent serial and exact-two-rank
regression advances both subsystems in a sheared flow and checks turbulence,
dissolved-hydrogen advection, fission production, bubble slip, hydrogen
balance, and void-dependent density feedback. There is not yet a checked-in
combined user-facing example or quantitative bubbly-flow validation case.

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
| Native mesh path | Cartesian, cylindrical, serial semi-structured, serial unstructured, partitioned-unstructured MPI, coupled-Krylov, and optional-physics regressions without legacy conversion |

The turbulence and radiolytic-bubble subsystems have separate focused serial
and MPI coverage plus a permanent combined serial and exact-two-rank
regression. A checked-in combined user example and a quantitative turbulent
bubbly-flow benchmark remain open.

The cavity smoke tests do not currently assert agreement with Ghia et al. or
with bundled OpenFOAM profile data. See
`verification/openfoam/cavityFlow/README.md` for the cavity workflow and
`verification/openfoam/pitzDaily/README.md` for the pitzDaily workflow. The
matched Gaussian tank workflow and its interpretation are documented in
`verification/openfoam/fissileSolutionTank/README.md`.

## Examples

Pre-built example executables:

| Example | Description |
| ------- | ----------- |
| `natural_convection_box` | 3D heated box with Boussinesq convection |
| `natural_convection_cylinder` | Cylindrical domain natural convection |
| `natural_convection_sphere` | Spherical domain natural convection |
| `natural_convection_boundary_layer_box` | Box with thermal boundary layer resolution |
| `natural_convection_shiri` | MPI-capable annular natural convection with standard k-epsilon and adaptive steady-state search |
| `pitz_daily` | OpenFOAM pitzDaily geometry with transient standard k-epsilon transport |
| `fissile_solution_tank_demo` | Cylindrical fissile-solution smoke case with Gaussian fission power |
| `fissile_solution_tank_sst` | 1000 W Gaussian tank SST case for matched OpenFOAM R-Z verification |
| `constant_power_cylinder_vessel` | Cylindrical vessel smoke case with uniform fission power, radiolytic gas, and boiling |

Examples use `Database` configuration, documented environment controls, or a
combination of both. Their CTest smoke settings intentionally reduce mesh size
and run length; production defaults can be substantially more expensive. None
is currently a combined turbulent radiolytic-bubble example.

## Build

Maintainers and contributors should also read
[`MAINTENANCE.md`](MAINTENANCE.md) for architecture boundaries, focused test
commands, extension paths, numerical review checks, profiling, and Doxygen
generation.

### Prerequisites

- C++23 compiler (GCC ≥ 13 or Clang ≥ 17)
- CMake ≥ 3.21
- Trilinos 17+ with:
  - Kokkos, Teuchos, Tpetra
  - STK (IO, Mesh, Topology, Util)
  - Belos, Ifpack2, MueLu, Zoltan2
- MPI (OpenMPI or MPICH)
- Google Test (for unit tests)

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

## Fission Power Source

`FissionPowerSource` registers the specialized temperature source
`qdot_fission` in W/m³. Constant power density, normalized three-axis Gaussian
profiles, and caller-provided cell fields are supported:

```cpp
SimpleFluid::FissionPowerSourceOptions fission;
fission.profile = SimpleFluid::FissionPowerProfile::Gaussian;
fission.total_power = 1.0e6;
fission.center = {0.0, 0.0, 1.0};
fission.standard_deviation = {0.25, 0.25, 0.5};
solver.configure_fission_power_source(fission);
```

Gaussian values are sampled at cell centroids and normalized with the
distributed discrete volume integral, so `integrated_power()` equals the
requested total power. Cylindrical axisymmetry is obtained by placing the
center on the cylinder axis and setting equal X and Y widths.

The equivalent flat `Database` keys are:

```text
fission_power_mode = disabled | constant | gaussian
fission_power_density
fission_total_power
fission_center = [x, y, z]
fission_standard_deviation = [sigma_x, sigma_y, sigma_z]
```

For externally prepared fields, use `initialize_from_power_density()` to copy
W/m³ values or `initialize_from_shape()` to normalize a nonnegative shape to a
total power. The model owns its copied profile. A programmatic time multiplier
may scale the base profile once per step at `t_n`; the retained
`qdot_fission` field is the value used for that step.

`qdot_fission` is included by
`SolutionOutputOptions::include_sources` and is additive with other named
temperature sources. The in-memory Phase 20 scaffold can import owned-cell
power, register `T_liquid`, `alpha_g`, `rhoFeedback`, and precursor fields,
export deterministic volume-averaged snapshots, and drive a callback-based
outer loop with thermal-hydraulic subcycles and returned-power exchange. It is
not a production external-neutronics protocol or a neutronics solver;
file-based exchange remains future work.

## Radiolytic Gas Models

The optional radiolysis subsystem provides a bounded ideal-gas void-source
model and an advanced two-population hydrogen-bubble model. The latter includes
dissolved hydrogen, bubble populations, pressure-sensitive properties,
transport, escape, and diagnostic inventory accounting, but is not yet fully
accepted as a validated physical model.

The delayed-neutron precursor model conservatively advances
`alpha_l * C_i` with the projected liquid face flux, optional diffusion, and
exact constant-source/decay integration. Its globally reduced diagnostics
separate source addition, decay, boundary outflow, transport positivity
adjustment, and balance error; distributed inputs are validated collectively.

See [`docs/radiolytic-gas-models.md`](docs/radiolytic-gas-models.md) for the
advanced radiolysis model and
[`docs/modeling/radiolytic_bubble_boiling.md`](docs/modeling/radiolytic_bubble_boiling.md)
for the scalar void, boiling, feedback, precursor, and output workflow.

The fissile-solution tank smoke cases build as `fissile_solution_tank_demo`
and `constant_power_cylinder_vessel`.
After `cmake --build --preset GCC-Debug`, run
`build/gcc/bin/Debug/fissile_solution_tank_demo` or
`build/gcc/bin/Debug/constant_power_cylinder_vessel` and inspect the generated
VTU files in ParaView. Use the corresponding `LLVM-Debug` preset and
`build/llvm/bin/Debug` path for the LLVM build. The Gaussian tank demo keeps
boiling disabled by default; the constant-power cylinder variant enables
radiolytic gas plus bulk and wall boiling so the generated VTU contains the
coupled gas-source and latent-heat fields.

## Dependencies

| Dependency | Role |
| ---------- | ---- |
| **MPI** | Distributed-memory parallelism |
| **Kokkos** | On-node parallelism (CPU/GPU portability) |
| **Teuchos** | Smart pointers, parameter lists, timers |
| **Tpetra** | Distributed sparse linear algebra |
| **STK** | Unstructured mesh I/O (Exodus format) |
| **Belos** | Krylov subspace iterative solvers |
| **Ifpack2** | Algebraic preconditioners (block smoothers) |
| **MueLu** | Algebraic multigrid (AMG) for pressure/Poisson |
| **Zoltan2** | Graph/hierarchical mesh partitioning |
| **LAPACK** | Dense linear algebra |
| **BLAS** | Low-level vector/matrix operations |
| **GTest** | C++ unit testing framework |

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
