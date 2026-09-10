# SST-1994 with a SAS omega source

`SSTKOmegaSAS` is an opt-in source extension to SimpleFluid's original
Menter-1994 SST. It is not an SST-2003 replacement or a physically validated
scale-resolving turbulence model. Ordinary `SSTKOmega` keeps its vorticity
viscosity limiter, `20 beta_star k omega` k-production cap, computed Menter
gamma coefficients, blending, cross diffusion, buoyancy and wall treatment.

## Reference and equations

The algebraic reference is **OpenFOAM Foundation v13**:

- [kOmegaSSTSAS.C, Qsas](https://cpp.openfoam.org/v13/kOmegaSSTSAS_8C_source.html)
- [kOmegaSSTSAS.H, coefficients and literature reference](https://cpp.openfoam.org/v13/kOmegaSSTSAS_8H_source.html)
- [kOmegaSSTBase.C, parent formulation](https://cpp.openfoam.org/v13/kOmegaSSTBase_8C_source.html)

The reference header cites Egorov and Menter (2008), *Development and
Application of SST-SAS Model in the DESIDER Project*. The implementation here
is independently written mathematics; it has no OpenFOAM runtime dependency
and vendors no OpenFOAM source. The pointwise reference script below evaluates
the published expressions independently with 60-digit Decimal arithmetic.

For the existing positive, bounded transported k and omega, and the **active
parent** beta_star, kappa, and F1-blended beta and gamma:

\[
S_{ij}=\tfrac12(\partial_j U_i+\partial_i U_j),\quad
S^2=2\sum_{ij}S_{ij}^2,\quad H_U=|\nabla^2\mathbf U|_2,
\]
\[
L=\frac{\sqrt{k}}{\beta_*^{1/4}\omega},\quad
B=\beta/\beta_*-\gamma>0,\quad
\Delta=C_\Delta V^{1/3},
\]
\[
L_{vK,flow}=\frac{\kappa\sqrt{S^2}}{H_U},\quad
L_{vK,grid}=C_s\Delta\sqrt{\frac{\kappa\zeta_2}{B}},\quad
L_{vK}=\max(L_{vK,flow},L_{vK,grid}),
\]
\[
P=\zeta_2\kappa S^2(L/L_{vK})^2,\quad
D=\frac{2C}{\sigma_\Phi}k\max(|\nabla\omega|^2/\omega^2,|\nabla k|^2/k^2),
\]
\[
Q_{raw}=\max(P-D,0),\qquad Q_{applied}=\min(Q_{raw},\omega/(f_t\Delta t)).
\]

SAS uses **strain**, although the parent viscosity limiter uses vorticity.
P, D, Qraw and Qapplied have units s^-2. L and the three von Karman lengths
have units m. H_U has units (m s)^-1. There is no density, void-fraction or
cell-volume multiplier in the supplied omega source. The existing
density-divided full-cell scalar transport integrates it with volume once.
The clipped net P-D is explicit; the existing implicit destruction and signed
cross-diffusion/buoyancy splitting remain intact.

The H_U=0 limit is evaluated analytically: P=0. There is **no curvature epsilon**
with an implicit length or time scale. The equivalent computation
`sqrt(S2)/Lvk = min(H_U/kappa, sqrt(S2)/Lvk_grid)` avoids 0/0 and overflow from
forming a large L/Lvk ratio. Intermediate algebra uses long double and rejects
unrepresentable finite-state results instead of turning invalid state into a
plausible source. Unbounded or unrepresentably large diagnostic flow/selected lengths are encoded as the
largest finite `real_t`; this encoding never enters the source calculation.
Affine roundoff curvature gives quadratically small production, without a
regularization-driven activation. Nonfinite state, inadmissible coefficients,
B<=0, nonpositive volumes/filter widths/timesteps are errors.

OpenFOAM v13's complete SST parent has different gamma defaults, a strain-based
viscosity limiter and a different production configuration. Only identical
**supplied source inputs and coefficients** can be compared pointwise here.
No whole-flow OpenFOAM agreement or exact OpenFOAM-parent mode is claimed.

## Configuration

The flat `Database` selector is `turbulence_model = "SSTKOmegaSAS"`.
There are no new aliases. Existing selections and defaults remain valid.
Typed settings live in `TurbulenceModelOptions::sas` (`SSTSASOptions`).

| Flat key (prefix `turbulence_sas_`) | Default | Type, unit and validity |
| --- | --- | --- |
| `enabled` | true | bool; false bypasses all SAS derivatives, filter storage and outputs |
| `diagnostics` | false | bool; allocate/register the eleven fields below only when active and true |
| `zeta2` | 3.51 | real, dimensionless, finite and >0 |
| `sigma_phi` | 2/3 | real, dimensionless, finite and >0 |
| `c` | 2 | real, dimensionless, finite and >=0 |
| `cs` | 0.11 | real, dimensionless, finite and >0 |
| `delta_multiplier` | 1 | real, dimensionless, finite and >0 |
| `time_limiter` | true | bool; false applies Qraw without the numerical cap |
| `cap_time_fraction` | 0.1 | real, dimensionless, finite and >0, including when cap disabled |

For example, the complete key for `cs` is `turbulence_sas_cs`. Ill-typed values
are rejected by `DatabaseOptionReader`; coefficients are checked even when the
source is disabled. Disable with the boolean, not singular coefficients.

There is no independent SAS kappa. SAS reads the actual SST coefficient. With
`resolvedLowReSST`, the existing `wall_kappa` and `wall_beta_1` settings
coordinate the parent and wall law; SAS sees the resulting parent beta and
gamma. With no wall treatment, the existing parent defaults apply, including
kappa=0.41. This preserves the parent configuration rather than adopting the
reference's independently configured SAS and SST constants.

`TimeStepperOptions::physical_time` defaults to true. A false value rejects
active SAS before either supported solver starts its trial step. Direct
`TurbulenceModel::advance` callers likewise supply the physical-time boolean
(last argument, default true) and an actual physical timestep. The repository's
pitzDaily and Shiri pseudo-transient search drivers explicitly call
`validate_time_mode(false)`. Such searches may produce an SST initialization,
but cannot use active SAS. The cap is a numerical startup safeguard; disabling
it does not turn pseudo-time into physical time.

## Derivatives, geometry and transport

`FVM/VectorLaplacian.hh` forwards to the shared host implementation in
`FVM/details/VectorLaplacian.hh`. It sums unit-coefficient component diffusion
fluxes and divides by the cell volume. On each face, with center displacement d
and outward area S, it uses `(S dot d)/|d|^2` times the component difference,
plus the reconstructed gradient dotted with the remaining area vector.
Interior gradients are averaged with the neighbor, including synchronized
partition ghosts. This is the full explicit non-orthogonal correction, not a
second first-derivative reconstruction, Laplacian of speed, full Hessian norm,
or variable-viscosity momentum diffusion.

The model reuses its existing velocity/k/omega least-squares gradient machinery.
Active SAS initially requires the verified least-squares path; the parent
Gauss-linear option remains available with the source disabled. SAS imports the current
owned velocity into existing candidate velocity storage before its stencil,
and synchronizes the velocity-gradient ghosts. The Laplacian result and filter
width are read only on owned cells and need no halo. Geometry is kept in a
SAS-only `TransportGeometryCache`; filter widths, Laplacian storage and optional
diagnostic fields are reused. The existing transport caches and graphs are
unchanged. A changed geometry epoch fails explicitly; this fixed-grid feature
does not refresh or accept moving-grid histories.

The derivative operates on global Cartesian vector components, including native
cylindrical sectors and closed annuli. For a radial curved face the SAS flux
uses the surface integral of the normal: the mesh midpoint-normal/arc-area
product is multiplied by `sin(dtheta/2)/(dtheta/2)`. This closes the area-vector
sum and preserves affine and rigid-rotation zero curvature. It adds no
cylindrical-component metric terms and does not alter the parent transport
operators. Manufactured refinement and transient swirl evidence are recorded
in [the extension report](sst_sas_extensions.md).

Native `SemiStructuredXY_Z` uses the same planar-face operator directly, with
its existing serial-only ownership contract. Skewed triangular extrusions have
manufactured derivative/refinement and transient solver coverage. Slip and
periodic velocity conditions remain rejected pending their separate extensions. A cylindrical full
annulus has connected interior angular faces, not a boundary-value periodic
patch. Source-disabled SAS retains the established parent paths.

Supported active derivative paths are planar HEX_8/WEDGE_6 polyhedral legacy
meshes, native Cartesian, cylindrical and serial semi-structured handles, native planar unstructured handles (with the
existing partitioned adapter in MPI), and handles of existing STK meshes.
Native execution does not construct a legacy mesh. Prescribed velocity/no-slip
and homogeneous Neumann outlet conditions are supported. Prescribed face
values and non-orthogonal corrections are honored. Quadratic orthogonal
interior differences are exact; the existing two-point Dirichlet boundary flux
has first-order boundary error. The manufactured skewed-prism refinement
measures a volume-weighted norm including boundary cells, not a claim of exact
quadratic derivatives everywhere.

Cube-root cell volume is an **initial SimpleFluid filter policy**, not a
universal SAS requirement. It ignores directional anisotropy in stretched
cells. For thin extrusions it depends on the chosen thickness; for
axisymmetric representations an arbitrary azimuthal thickness cannot qualify
three-dimensional turbulent resolution. These are numerical-resolution
limitations, not coefficient-tuning instructions.

Equation-by-equation scope: momentum and the two turbulence equations retain
their existing Backward Euler and implicit first-order upwind path. SAS adds no
BDF2 or linear-upwind turbulence option. Existing opt-in higher-order weighted
scalar and native temperature capabilities do not imply corresponding
momentum/turbulence accuracy. No DES length replacement, third equation, SGS
viscosity, or manual reduction of nu_t is introduced.

## Accepted steps and diagnostics

`TurbulenceModel.tcc` computes SAS once from the pre-transport k/omega and the
current projected velocity. Both turbulence solves follow the complete
pressure-velocity correction loop, once per physical step. Their old-time
fields remain the preceding accepted k/omega. The post-solve closure refresh
updates viscosity and effective properties without re-evaluating SAS or
replacing the assembled record.

When requested, `output_fields()` publishes:

| Field | Meaning at the last accepted assembly |
| --- | --- |
| `sas_L` | Parent turbulence length |
| `sas_Lvk_flow`, `sas_Lvk_grid`, `sas_Lvk` | Flow length, grid lower bound, and selected length |
| `sas_delta` | Cube-root-volume filter width |
| `sas_production`, `sas_damping` | Unclipped P and D |
| `sas_Q_raw` | Nonnegative uncapped net source |
| `sas_Q_applied` | Source supplied to the accepted omega transport RHS |
| `sas_grid_limiter_active` | 1 exactly when grid length > flow length |
| `sas_time_limiter_active` | 1 exactly when enabled cap < Qraw |

`model.sas_statistics()` is available without allocating all diagnostic fields.
It reduces **owned cells only**: active cell fraction uses Qapplied>0, cap-active
fraction uses the strict cap condition above, extrema refer to Qapplied, and
volume measures use actual V. The integral is sum(V Qapplied), the mean divides
by sum(V), and active/capped volume fractions divide selected volumes by total
volume. A strict zero threshold can count negligible roundoff production; the
source extrema and integral quantify its significance. Statistics are invalid
until the first accepted step.

Candidate fields, source diagnostics and statistics publish only after both
solves and all derived-property checks succeed. `snapshot()`/`restore()` retain
the complete accepted turbulence state, wall data, gradients, effective
properties and source record and reject snapshots from another configuration
or geometry epoch. `restore_transported_state()` is a field-only restart: it
rebuilds the existing dependent state and clears SAS history (`valid=false`,
zero diagnostic fields), since no assembled source was supplied. Reconfiguration
rebuilds the registry; ordinary SST/laminar and disabled SAS expose no SAS fields.

The isothermal and Boussinesq solvers capture an accepted-state transaction
when SAS is active, including flow fields, material fields, turbulence and
public step/continuity diagnostics. Boussinesq also captures temperature and
source fields, so a later temperature rejection restores the preceding SAS
record. Initial solver scope is fixed-grid single-phase isothermal or
Boussinesq flow, including the parent signed buoyancy source and resolved SST
wall policy. The existing high-Re k-epsilon wall policy remains incompatible.
Active SAS currently rejects phase/inventory/free-surface/material-feedback
models in Boussinesq; these require their own complete transaction integration.
The existing laminar-only planar-ALE restriction remains. This does not add
conservative ALE turbulence, bubble-induced turbulence, interphase momentum,
boiling, transition, or a new turbulent heat-flux closure.

## Reproduction and evidence

From the repository root, with the configured dependencies:

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_MAX_TEST_PROCS=2
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASSource testSSTSASModel sst_sas_activation
ctest --test-dir build/gcc -C Debug -R 'SSTSAS|sst_sas_activation' --output-on-failure
python3 verification/sst_sas/pointwise_reference.py --check
python3 verification/sst_sas/compare_serial_mpi.py build/gcc/bin/Debug/sst_sas_activation
build/gcc/bin/Debug/sst_sas_activation /tmp/sas_activation.vtu
```

The last command writes the existing parallel VTU/PVTU output with all SAS
fields. The example is a two-step, 6x6x6 isothermal source-activation fixture
with smooth initial circulation, no-slip boundaries, and a controlled uniform
wall-distance override. It needs no IF97. Its `small_fixture` serial/MPI
registrations are independent of enlarged water-comparison defaults. It is
not a wall-resolved or sustained-turbulence benchmark. No external reference
manifest or tolerance is changed; pitzDaily physical qualification stays pending.

See [the implementation verification record](sst_sas_verification.md) for the
starting revision, commands actually executed, results and limitations.

Status must remain distinct: **implemented → component-tested →
solver-regression-tested → externally compared → physically validated**.
Pointwise algebra comparison and serial/MPI consistency do not establish
whole-flow external agreement or physical scale resolution. Longer 3D
assessment, spatial/time refinement, spectra, resolved-energy partition and
appropriate experimental or qualified reference comparisons remain open.
