# TODO: Coupled FVM Momentum–Pressure Solver with Implicit Non-Orthogonal Terms

## Phase 0 — Baseline infrastructure

- [x] Add `VectorField`, `ScalarField`, `FaceField` abstractions if missing.
- [x] Add cell/face geometry storage if missing:
  - cell center
  - face center
  - face area vector
  - owner/neighbour cells
  - cell volume
  - vector `dPN = xN - xP`
- [x] Add least-squares gradient reconstruction for scalar and vector fields if missing.
- [x] Add unit tests for gradient reconstruction on linear fields if missing.

## Phase 1 — Orthogonal diffusion operator

- [x] Implement implicit orthogonal Laplacian assembly:
  \[
  \Gamma_f (\nabla \phi)_f\cdot S_f
  \approx
  \Gamma_f \frac{\phi_N-\phi_P}{|d_{PN}|^2}(S_f\cdot d_{PN})
  \]
- [x] Support scalar and vector Laplacian.
- [x] Add Dirichlet/Neumann boundary contributions.
- [x] Verify with manufactured Poisson solution on orthogonal mesh.

## Phase 2 — Explicit non-orthogonal correction

- [x] Decompose face area:
  \[
  S_f = E_f + T_f
  \]
  where
  \[
  E_f = \frac{S_f\cdot d_{PN}}{d_{PN}\cdot d_{PN}} d_{PN}
  \]
- [x] Add explicit correction:
  \[
  \Gamma_f (\nabla \phi)_f\cdot T_f
  \]
- [x] Add `nNonOrthogonalCorrectors` loop.
- [x] Verify convergence on skewed manufactured meshes.

## Phase 3 — Fully implicit non-orthogonal operator

- [x] Implement full diffusion residual:
  \[
  R = \nabla\cdot(\Gamma\nabla\phi)
  \]
- [x] Assemble full Jacobian approximately:
  - orthogonal two-point stencil implicit
  - least-squares non-orthogonal stencil implicit
- [x] Store expanded sparse graph including gradient neighbours.
- [x] Compare:
  - explicit correction loop
  - fully implicit non-orthogonal matrix
  - hybrid implicit/explicit option
- [x] Add runtime switch:

    ```yaml
    nonOrthogonalTreatment: explicit | implicit | hybrid
    ```

## Phase 4 — Momentum equation

- [x] Assemble transient term.
- [x] Assemble semi-implicit convection:

  - first-order upwind implicit
  - optional high-order explicit correction later
- [x] Assemble viscous diffusion using Phase 1–3 operator.
- [x] Add body force term, especially buoyancy:
  [
  \rho g
  ]
- [x] Add momentum predictor solve.

## Phase 5 — Pressure–velocity coupling baseline

- [ ] Implement SIMPLE:

  - solve momentum predictor
  - build pressure correction equation
  - correct pressure
  - correct velocity
  - correct face flux
- [ ] Implement PISO:

  - one momentum predictor
  - multiple pressure corrections
- [ ] Implement PIMPLE driver:

  - outer nonlinear loop
  - inner PISO pressure corrections
- [ ] Add residual reporting for:

  - momentum
  - pressure
  - continuity imbalance

## Phase 6 — Fully coupled Krylov system

- [ ] Assemble block system:

  ```latex
  \begin{bmatrix}
  A_u & G \\
  D   & 0
  \end{bmatrix}
  \begin{bmatrix}
  U \\ p
  \end{bmatrix}
  =
  \begin{bmatrix}
  b_u \\ 0
  \end{bmatrix}
  ```

- [x] Use Tpetra sparse matrix/vector backend.
- [x] Use Belos FGMRES or GMRES as primary Krylov solver.
- [ ] Implement block Schur preconditioner:
  [
  S \approx D \operatorname{diag}(A_u)^{-1} G
  ]
- [ ] Use Ifpack2 for simple block smoothers.
- [ ] Use MueLu AMG for pressure/Poisson-like Schur block.
- [ ] Add solver switch:

  ```yaml
  pressureVelocityCoupling: SIMPLE | PISO | PIMPLE | coupledKrylov
  ```

Tpetra is the Trilinos package for distributed sparse matrices/vectors, Belos provides Krylov solvers, and MueLu is the Trilinos multigrid preconditioning package suitable for Poisson/convection-diffusion-like systems. ([trilinos.github.io][1])

## Phase 7 — Rhie–Chow / collocated-grid stabilization

- [ ] Implement pressure-weighted face flux interpolation.
- [ ] Add checkerboard-pressure test.
- [ ] Verify pressure field remains smooth on collocated meshes.
- [ ] Ensure stabilization is compatible with:

  - SIMPLE
  - PISO
  - PIMPLE
  - coupled Krylov mode

## Phase 8 — Verification cases

- [ ] Lid-driven cavity, Re = 100 and 1000.
- [ ] Poiseuille flow.
- [ ] Manufactured incompressible Navier–Stokes solution.
- [ ] Skewed-mesh diffusion test.
- [ ] Natural convection square cavity.
- [ ] Compare against OpenFOAM for one matching case.

## Phase 9 — Performance tests

- [ ] Measure iteration count vs mesh non-orthogonality.
- [ ] Compare:

  - explicit non-orthogonal correction + AMG
  - implicit non-orthogonal matrix + GMRES
  - coupled Krylov + Schur preconditioner
- [ ] Track:

  - nonlinear iterations
  - Krylov iterations
  - wall time
  - memory
  - continuity residual
- [ ] Add MPI scaling test.

## Phase 10 — Integration with TH/neutronics coupling

- [ ] Add density update:
  [
  \rho = \rho(T,\alpha_g)
  ]
- [ ] Add viscosity update:
  [
  \mu = \mu(T,\alpha_g)
  ]
- [ ] Couple fission heat source into energy equation.
- [ ] Couple velocity field to delayed-neutron precursor transport.
- [ ] Add outer multiphysics coupling loop.

## Acceptance criteria

- [ ] Orthogonal Poisson test shows expected convergence.
- [x] Skewed mesh test converges with both explicit and implicit non-orthogonal treatment.
- [ ] SIMPLE/PISO/PIMPLE produce mass-conservative velocity fields.
- [ ] Coupled Krylov solver reaches lower continuity residual than segregated SIMPLE for same tolerance.
- [ ] Natural-convection benchmark remains stable with temperature-dependent density.
- [ ] Solver options are selectable from input file without recompilation.

[1]: https://trilinos.github.io/tpetra.html?utm_source=chatgpt.com "Tpetra: Parallel sparse linear algebra"
