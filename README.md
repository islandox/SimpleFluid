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

**The advanced Phase 14.1 two-population radiolytic-bubble model is implemented
in part and remains under verification.**

| Capability | Status |
| ---------- | ------ |
| Mesh & geometry infrastructure | ✅ |
| Scalar & vector fields (Tpetra) | ✅ |
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
| Scalar void, boiling, feedback, and precursor infrastructure | 🚧 |
| Two-population radiolytic-bubble model | 🚧 |
| TH/neutronics multiphysics coupling | ⬜ |

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
- **First-order upwind** convection (implicit)
- **Backward Euler** time integration
- Gradient reconstruction via **least-squares** on extended stencils

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

### Rhie–Chow Stabilization

Pressure-weighted face-flux interpolation prevents checkerboard pressure modes
on collocated grids. Compatible with all four pressure–velocity coupling modes.

## Features

### Mesh

- CRTP-based mesh hierarchy with type-erased `MeshHandle` (via `std::variant`)
- Runtime-switchable mesh types:
  - **Orthogonal Cartesian 3D** — structured hexahedral cells
  - **Orthogonal Cylindrical 3D** — polar-structured hexahedral cells
  - **Semi-structured XY×Z** — 2D unstructured × 1D structured prisms
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
- `FieldStored` with mesh-aware construction and ghosted data exchange
- `BoundaryFaceField` for sideset-indexed boundary data

### Solvers & Equations

- `FluidSolver` — reusable transient incompressible pressure-velocity driver
- `BoussinesqSolver` — thermal natural-convection specialization
- `IncompressibleMomentumEquation` — generic velocity-transport assembly
- `BoussinesqMomentumEquation` — incompressible momentum with buoyancy
- `PressureProjectionEquation` — pressure-correction / Poisson system
- `TemperatureDiffusionEquation` — energy equation assembly
- `CoupledPressureVelocitySolver` — monolithic block-Krylov solver
- `BelosLinearSolver` — unified interface to Trilinos iterative solvers
- Runtime residual reporting for momentum, pressure, and continuity
- Named volumetric heat sources and updateable physical material fields

### FVM Operators

- `DiffusionSystem` — scalar/vector orthogonal diffusion assembly
- `TransportSystem` — semi-implicit convection–diffusion assembly
- `NonOrthogonalCorrection` — cross-diffusion flux decomposition ($S_f = E_f + T_f$)
- `CellOperators` — gradient, divergence, Laplacian reconstruction
- `FaceFlux` — Rhie–Chow face-flux interpolation
- `MatrixOperators` — sparse matrix assembly helpers

### Data & Utilities

- Typed key-value `Database` (int, real, string, bool, and vector variants)
- `RandomAccessView` — lightweight non-owning view with STL-compatible iterators
- Generic `vec3` 3D vector class with arithmetic operators
- Concept-based `TpetraTypePack` for clean Trilinos type management
- Mesh utility functions (centroid, face/cell type classification)

### I/O

- VTU (VTK Unstructured Grid) output via `VTUWriter` — compatible with ParaView
- STK Exodus II `HEX_8` and `WEDGE_6` mesh input via `STKMesh`

### Parallelism

- MPI distributed-memory parallelism via Trilinos/Tpetra
- Kokkos on-node parallelism (CPU Serial, OpenMP, CUDA, HIP, SYCL)
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
| OpenFOAM pitzDaily | Five-block standard-k-epsilon duct case with velocity-profile comparison |

The cavity smoke tests do not currently assert agreement with Ghia et al. or
with bundled OpenFOAM profile data. See
`verification/openfoam/cavityFlow/README.md` for the cavity workflow and
`verification/openfoam/pitzDaily/README.md` for the pitzDaily workflow.

## Examples

Pre-built example executables:

| Example | Description |
| ------- | ----------- |
| `natural_convection_box` | 3D heated box with Boussinesq convection |
| `natural_convection_cylinder` | Cylindrical domain natural convection |
| `natural_convection_sphere` | Spherical domain natural convection |
| `natural_convection_boundary_layer_box` | Box with thermal boundary layer resolution |
| `pitz_daily` | OpenFOAM pitzDaily geometry with transient standard k-epsilon transport |
| `fissile_solution_tank_demo` | Cylindrical fissile-solution smoke case with Gaussian fission power |
| `constant_power_cylinder_vessel` | Cylindrical vessel smoke case with uniform fission power, radiolytic gas, and boiling |

Each example is configured via a `Database` object and runs a short transient
simulation with VTU output.

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
cmake --preset Local
cmake --build --preset Release

# Or manually
cmake -B build -G "Ninja Multi-Config" \
  -DTrilinos_DIR=/path/to/trilinos/lib/cmake/Trilinos
cmake --build build --config Release
```

### Run Tests

```bash
cmake --build --preset Debug
ctest --preset Debug
```

### Run Examples

```bash
./build/bin/Release/natural_convection_box
./build/bin/Release/natural_convection_cylinder
```

## Performance Benchmarks

Build the standalone benchmark executable and run the local Debug-safe suite:

```bash
cmake --build build --config Debug --target simplefluid_benchmark
ctest --test-dir build -C Debug -L benchmark --output-on-failure
```

The benchmark writes one CSV row per measured repetition. Rows include solver
configuration, mesh and MPI sizes, non-orthogonality angles, nonlinear and
Krylov iterations, residuals, setup/solve/total wall time, process memory,
compiler/build metadata, and the Git revision.

Run the larger profiling preset with frame pointers and debug symbols:

```bash
cmake --build build --config RelWithDebInfo --target simplefluid_benchmark
./build/bin/RelWithDebInfo/simplefluid_benchmark \
  --preset release-profile \
  --output build/benchmarks/release-profile.csv
```

Run strong-scaling measurements sequentially:

```bash
mpiexec -n 1 ./build/bin/Release/simplefluid_benchmark --preset mpi-strong --output build/benchmarks/mpi-strong.csv
mpiexec -n 2 ./build/bin/Release/simplefluid_benchmark --preset mpi-strong --output build/benchmarks/mpi-strong.csv
mpiexec -n 4 ./build/bin/Release/simplefluid_benchmark --preset mpi-strong --output build/benchmarks/mpi-strong.csv
```

Run weak-scaling measurements with approximately `32x32x8` cells per rank:

```bash
mpiexec -n 1 ./build/bin/Release/simplefluid_benchmark --preset mpi-weak --output build/benchmarks/mpi-weak.csv
mpiexec -n 2 ./build/bin/Release/simplefluid_benchmark --preset mpi-weak --output build/benchmarks/mpi-weak.csv
mpiexec -n 4 ./build/bin/Release/simplefluid_benchmark --preset mpi-weak --output build/benchmarks/mpi-weak.csv
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
temperature sources. Neutronics feedback and file-based power import remain
future work.

## Radiolytic Gas Models

The optional radiolysis subsystem provides a bounded ideal-gas void-source
model and an advanced two-population hydrogen-bubble model. The latter includes
dissolved hydrogen, bubble populations, pressure-sensitive properties,
transport, escape, and diagnostic inventory accounting, but is not yet fully
accepted as a validated physical model.

See [`docs/radiolytic-gas-models.md`](docs/radiolytic-gas-models.md) for the
advanced radiolysis model and
[`docs/modeling/radiolytic_bubble_boiling.md`](docs/modeling/radiolytic_bubble_boiling.md)
for the scalar void, boiling, feedback, precursor, and output workflow.

The fissile-solution tank smoke cases build as `fissile_solution_tank_demo`
and `constant_power_cylinder_vessel`.
After `cmake --build --preset Debug`, run
`build/bin/Debug/fissile_solution_tank_demo` or
`build/bin/Debug/constant_power_cylinder_vessel` and inspect the generated VTU
files in ParaView. The Gaussian tank demo keeps boiling disabled by default;
the constant-power cylinder variant enables radiolytic gas plus bulk and wall
boiling so the generated VTU contains the coupled gas-source and latent-heat
fields.

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
