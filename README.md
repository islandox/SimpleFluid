# SimpleFluid

SimpleFluid is a C++23 finite-volume CFD library built on [Trilinos](https://trilinos.github.io) for distributed-memory, GPU-portable parallelism, focused on incompressible Boussinesq natural convection and reacting-flow applications.

## Project Overview

SimpleFluid provides:
- Mesh infrastructure with runtime-switchable types (Cartesian, Cylindrical, Unstructured, Semi-structured XY×Z).
- High-performance finite-volume operators for scalar, vector, and tensor fields.
- Robust pressure–velocity coupling (SIMPLE/PISO/PIMPLE) and block-Krylov linear solvers.
- Advanced physics closure support: RANS turbulence, reacting-flow species transport, and Boussinesq thermal modeling.

## Documentation

For deep-dive architectural and modeling details, refer to the documentation:

| Category | Description |
| --- | --- |
| [Architecture](docs/architecture/) | API contracts, solver couplings, and system design |
| [Modeling](docs/modeling/) | Physical assumptions, chemical mechanisms, and constitutive closures |
| [Guides](docs/guides/) | Implementation guides, parallel distribution, and performance contracts |
| [Verification](docs/verification/) | Numerical qualification gates and convergence contracts |

## Governing Equations

The default flow path solves the incompressible Navier–Stokes equations with the Boussinesq approximation:

$$
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\mathbf{u}) = -\nabla p + \nu\nabla^2\mathbf{u} + \rho\beta g(T - T_{\mathrm{ref}})
$$
$$
\nabla\cdot\mathbf{u} = 0
$$
$$
\frac{\partial T}{\partial t} + \nabla\cdot(\mathbf{u}T) = \alpha\nabla^2 T
$$

## Quick Start

### Build
SimpleFluid uses CMake presets for simplified configuration:

```bash
# Configure (using provided GCC preset)
cmake --preset GCC-ninja-multi

# Build (Release mode)
cmake --build --preset GCC-Release
```

Detailed build configuration and dependency requirements are documented in `cmake/README.md`.

### Run Tests
```bash
ctest --preset GCC-Debug
```

## License
This project is distributed under the MIT License.

## Maintenance
Maintainers and contributors must follow the [Maintainer Guide](MAINTENANCE.md) for review discipline, commit style, and ABI boundaries.
