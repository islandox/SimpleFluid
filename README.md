# SimpleFluid

A finite-volume computational fluid dynamics code for incompressible Boussinesq
natural convection, built on [Trilinos](https://trilinos.github.io) for distributed-memory,
GPU-portable parallelism.

## Project Status

SimpleFluid is in active development with a **verified incompressible Navier–Stokes**
solver for natural convection. All Phases 0–8 of the planned roadmap are complete (~37 kLOC
source, ~11 kLOC tests), covering mesh infrastructure through verification against
OpenFOAM.

**Phases 9–10 (performance profiling and multiphysics coupling) remain in the roadmap.**

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
| Verification suite (cavity, Poiseuille, MMS, OpenFOAM comparison) | ✅ |
| Boussinesq natural convection examples | ✅ |
| Performance benchmarks & scaling tests | ⬜ |
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

- **Collocated** finite-volume method on arbitrary unstructured meshes
- **First-order upwind** convection (implicit) with optional deferred correction
- **Crank–Nicolson** and **backward Euler** time integration
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
| `coupledKrylov` | Monolithic $\begin{bmatrix}A_u & G \\ D & 0\end{bmatrix}$ system with block Schur preconditioner and MueLu AMG on the Schur complement |

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
  - **STK adapter** — arbitrary polyhedral meshes via Exodus II files
- Owned + ghost cell decomposition for distributed-memory assembly
- CRS-style neighbor connectivity for FVM stencil construction
- Kokkos-based geometry storage — portable across CPU and GPU backends
- Boundary condition support via sideset / side-part name mapping

### Mesh Generation

- Programmatic mesh generation for **box**, **cylinder**, and **sphere** domains
- External mesh file loading through Trilinos/STK
- Configuration-driven via the built-in typed key-value `Database`

### Fields

- **Cell-centered** scalar and vector fields backed by Tpetra distributed vectors
- **Face-centered** scalar and vector fields backed by Tpetra distributed vectors
- `FieldStored` with mesh-aware construction and ghosted data exchange
- `BoundaryFaceField` for sideset-indexed boundary data

### Solvers & Equations

- `BoussinesqSolver` — transient natural convection driver
- `BoussinesqMomentumEquation` — momentum equation assembly
- `PressureProjectionEquation` — pressure-correction / Poisson system
- `TemperatureDiffusionEquation` — energy equation assembly
- `CoupledPressureVelocitySolver` — monolithic block-Krylov solver
- `BelosLinearSolver` — unified interface to Trilinos iterative solvers
- Runtime residual reporting for momentum, pressure, and continuity

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
- STK Exodus II mesh input via `STKMesh`

### Parallelism

- MPI distributed-memory parallelism via Trilinos/Tpetra
- Kokkos on-node parallelism (CPU Serial, OpenMP, CUDA, HIP, SYCL)
- Zoltan2 graph/hierarchical mesh partitioning

## Verification

The solver is verified against analytical and reference solutions:

| Case | Description |
| ---- | ----------- |
| Lid-driven cavity | Re = 100, 1000 — benchmarked against Ghia et al. |
| Poiseuille flow | Parabolic profile recovery in a channel |
| Manufactured solution | Method of manufactured solutions for Navier–Stokes |
| Skewed diffusion | Non-orthogonal mesh convergence with all three treatments |
| Natural convection cavity | Differentially heated square cavity |
| OpenFOAM comparison | Matching case cross-validation with OpenFOAM |

## Examples

Pre-built example executables:

| Example | Description |
| ------- | ----------- |
| `natural_convection_box` | 3D heated box with Boussinesq convection |
| `natural_convection_cylinder` | Cylindrical domain natural convection |
| `natural_convection_sphere` | Spherical domain natural convection |
| `natural_convection_boundary_layer_box` | Box with thermal boundary layer resolution |

Each example is configured via a `Database` object and runs a short transient
simulation with VTU output.

## Build

### Prerequisites

- C++23 compiler (GCC ≥ 13 or Clang ≥ 17)
- CMake ≥ 3.16
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
cd build
ctest --output-on-failure
```

### Run Examples

```bash
./build/bin/Release/natural_convection_box
./build/bin/Release/natural_convection_cylinder
```

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
