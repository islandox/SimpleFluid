# SimpleFluid
Simple fluid code for convection in vessels

## Applications
- natural convection
- with or without boiling
- bubbles in dispersed phase

## Features

### Mesh
- Finite-volume mesh for hybrid **triangular-prism** and **hexahedral** cell types
- Owned + ghost cell decomposition for distributed-memory assembly
- CRS-style neighbor connectivity for FVM stencil construction
- Kokkos-based geometry storage — portable across CPU and GPU backends
- STK mesh I/O for reading external meshes (Exodus format)
- Boundary condition support via sideset / side-part name mapping

### Mesh Generation
- Programmatic mesh generation for **box**, **cylinder**, and **sphere** domains
- External mesh file loading through Trilinos/STK
- Configuration-driven via the built-in typed key-value `Database`

### Fields
- **Cell-centered** scalar fields backed by Tpetra distributed vectors
- **Face-centered** scalar fields backed by Tpetra distributed vectors
- Ghosted data exchange via Tpetra import/export

### Data & Utilities
- Typed key-value `Database` (int, real, string, bool, and vector variants)
- `RandomAccessView` — lightweight non-owning view with STL-compatible random-access iterators
- Generic `vec3` 3D vector class with arithmetic operators
- Concept-based `TpetraTypePack` for clean Trilinos type management
- Mesh utility functions (centroid, face/cell type classification)

### Parallelism
- MPI distributed-memory parallelism via Trilinos/Tpetra
- Kokkos on-node parallelism (CPU serial, OpenMP, CUDA, HIP, SYCL)

## Dependencies

| Dependency | Role |
|---|---|
| **MPI** | Distributed-memory parallelism — used by Trilinos/Tpetra for cross-process communication |
| **Kokkos** | On-node parallel programming model — enables portability across CPU (Serial, OpenMP) and GPU (CUDA, HIP, SYCL) backends |
| **Teuchos** | Trilinos utility library — provides smart pointers (`RCP`), parameter lists, timers, and exception handling |
| **Tpetra** | Distributed linear algebra — parallel sparse/dense vectors, CRS graphs, and maps used to build the linear systems |
| **STK** | Sierra Toolkit — mesh database (`STKMesh`, `STKIO`, `STKTopology`, `STKUtil`) for reading, storing, and querying unstructured mesh data (Exodus format) |
| **Belos** | Trilinos iterative linear solvers — provides Krylov subspace methods (CG, GMRES, etc.) for solving assembled systems |
| **NOX** | Trilinos nonlinear solvers — Newton-based methods for nonlinear PDE systems |
| **LAPACK** | Linear Algebra PACKage — dense linear algebra routines (eigenvalue decompositions, factorizations) |
| **BLAS** | Basic Linear Algebra Subprograms — low-level vector and matrix operations used by LAPACK and Kokkos |
| **GSL** | GNU Scientific Library — general-purpose numerical routines (interpolation, integration, special functions) |
| **GTest** | Google Test framework — C++ unit testing (build-time dependency only) |

## License
This project is distributed in MIT License.