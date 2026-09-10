# SST-SAS support extensions

This work starts from clean `feature/turbulence` at
`8cf294944dd9c65112cf02040cd812a770cc90c6`, with the initial source extension
already committed. The initial focused GCC baseline passed 20 registrations.
Each support addition is tested and committed separately. All evidence here
concerns component correctness and fixed-grid transient regression; it does
not establish physical scale-resolution validation or change the SST-1994
parent, source units, source algebra, or external-comparison gates.

## Cylindrical geometry

Native cylindrical sectors and closed annuli use stored global Cartesian
velocity components. The SAS vector Laplacian now integrates the radial face
normal using the chord factor `sin(dtheta/2)/(dtheta/2)`, instead of treating
its arc area and midpoint normal as an integrated vector. Existing parent
momentum, SST, scalar transport and mesh geometry are unchanged.

`SSTSASDerivativesTest.CylindricalSectorsAndClosedAnnuli` checks uniform flow,
affine shear, rigid rotation, scalar gradients, quadratic vector curvature,
nonzero SAS transport, disabled-source equality, diagnostics and restoration.
Its volume-weighted quadratic Laplacian L1 errors decrease from 0.898220 to
0.492935 on a sector and 0.858865 to 0.461980 on a full annulus as the radial,
azimuthal and axial resolution doubles. Both serial and two-rank paths pass.
`SASSupportedPathsTest.CylindricalCartesianComponentsAdvanceTransientSwirl`
checks two accepted isothermal steps, nonzero SAS, positivity and continuity.
The existing positive-inner-radius restriction remains; this does not add
axis cells or qualify a thin angular extrusion as resolved 3D turbulence.

Commands:

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASModel testSASSupportedPaths
ctest --test-dir build/gcc -C Debug -R '^SSTSAS' -E '_2procs' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^SSTSAS_2procs$' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^SASSupportedPathsTest.Cylindrical' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^SASSupportedPaths_2procs$' --output-on-failure
```

MPI registrations run with host network access. The SAS component selection
contains 18 passing serial registrations; the explicit MPI registration also
passes. The cylindrical solver registration passes in serial and on two ranks.

## Semi-structured geometry

`SemiStructuredXY_Z` now runs SAS directly through its native handle/fields.
No legacy mesh or extra derivative approximation is introduced. Skewed
triangular extrusions pass affine/vector/scalar derivative checks, quadratic
curvature refinement, nonzero transport, disabled-source equality, restart and
rollback tests. A native isothermal prism case accepts two steps with nonzero
SAS and controlled continuity. The two focused serial registrations passed:

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASModel testSASSupportedPaths
ctest --test-dir build/gcc -C Debug -R '^(SSTSASDerivativesTest.SemiStructured|SASSupportedPathsTest.SemiStructured)' --output-on-failure
```

The underlying handle remains serial-only for this mesh family, so these
geometry-specific bodies skip under MPI. This extension does not change mesh
ownership or qualify arbitrary extrusion thickness as a 3D resolution scale.

## Slip boundaries

The SAS derivative now enforces `U_n=0` and `dU_t/dn=0`. The normal boundary
flux is retained; tangential diffusion is zero. `CellGradientCache` has an
optional cached boundary-displacement provider, preserving its original
constructor and default arithmetic. Active SAS creates the additional cache
only for configured slip patches, using the normal foot rather than treating
the projected owner velocity as a full Dirichlet value at a skewed centroid.
The provider survives cache refresh. Ordinary SST/source-disabled SAS do not
select this new reconstruction; the analytic parent SST closure is unchanged.

The slip tests protect affine tangential shear and normal-velocity gradients
on native/legacy boxes and skewed prisms, so neither treating slip as no-slip
nor dropping its whole vector flux can pass. Quadratic fields activate SAS.
Cartesian slip and cylindrical no-slip/slip transient cases check bounded
fields and continuity. GCC builds and the 27-registration serial selection
pass; MPI runs use the existing two SAS registrations:

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASModel testSASSupportedPaths testCellGradientCache
ctest --test-dir build/gcc -C Debug -R '^(SSTSAS.*Test\.|SASSupportedPathsTest\.|CellGradientCacheTest\.)' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^(SSTSAS_2procs|SASSupportedPaths_2procs)$' --output-on-failure
```

## Periodic topology and geometry

Native Cartesian meshes now expose the indexer's existing periodic-axis and
wrapped MPI ownership machinery. Connected face normals, cell-center vectors,
face distances and VTU endpoint images follow that topology. Legacy
translational face pairing now stores the neighbor image displacement;
MeshHandle preserves it. Diffusion and Rhie-Chow both consume the wrapped
vector, avoiding inconsistent projection coefficients at the seam. These are
geometry corrections, not changes to SST coefficients or source algebra.

Native Fourier-gradient/Laplacian refinement, legacy/native paired-image
agreement, VTU cell-volume geometry and nonzero periodic transient SAS pass.
The existing non-adjacent-cell query rejection is preserved. Unpaired periodic
patches fail collectively; a peer rank receives the established propagated
runtime error rather than the originating invalid_argument. The legacy
fixture supplies its partners locally; native periodic meshes exercise MPI
seam halos. Rotational sector transformations and one-cell periodic native
axes are outside this API.

The focused GCC serial selection passes 23 registrations and all three MPI
registrations pass. A final ten-registration geometry/periodic subset also
passes after retaining the original overflow and adjacency guards. Before the fix,
the transient continuity check exposed unwrapped Rhie-Chow geometry; the
original continuity tolerance was retained. Commands:

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASModel testSASSupportedPaths testOrthogonalCartesian3D testFvmOperators testCellGradientCache testStoredPressureFaceFluxCache
ctest --test-dir build/gcc -C Debug -R 'Periodic|OrthogonalCartesian3DTest|StoredPressureFaceFluxCacheTest' -E '_[24]procs' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^(SSTSAS_2procs|SASSupportedPaths_2procs|StoredPressureFaceFluxCache_2procs)$' --output-on-failure
```

## Gauss-linear discretization

The new `turbulence_gradient_scheme` database key accepts `leastSquares`
(default) and `gaussLinear`, matching the typed option. Active SAS uses a
linearity-preserving Gauss-linear coordinate-moment correction. It normalizes
by volume and solves a dimensionless, pivoted 3x3 system locally; it rejects
singular moments, respects mixed slip/Neumann constraints and uses integrated
curved-face area vectors and wrapped periodic displacements. Constant offsets
cancel before reconstruction. This is a documented SAS numerical policy,
not a claim of equality with an uncorrected OpenFOAM Gauss gradient. Ordinary
SST and source-disabled SAS retain their previous reconstruction paths.

Manufactured affine gradients/curvature and quadratic refinement cover native
and legacy boxes, skewed prisms, cylindrical annuli, semi-structured extrusions
and periodic Fourier fields. Runtime checks cover disabled-source equivalence,
restart/diagnostics, slip flow and signed Boussinesq production with a nonzero
SAS source. The Gauss/option selection initially passed 20 serial registrations;
final selection passes 21 serial registrations, and both MPI registrations
pass, including the added periodic and buoyancy checks:

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASModel testSASSupportedPaths testTurbulenceModelOptions
ctest --test-dir build/gcc -C Debug -R '(SSTSAS|SASSupportedPaths).*Gauss|TurbulenceModelOptionsTest' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^(SSTSAS_2procs|SASSupportedPaths_2procs)$' --output-on-failure
```

## Material feedback

SAS now supports the existing temperature/constant material-feedback model.
Its accepted mirrors are captured with the model's snapshot API in addition
to the existing material, flow and turbulence snapshots. No feedback closure
or turbulence transport weighting changes. The regression demonstrates changed
density/viscosity and nonzero SAS, then induces overflow only in post-temperature
feedback. All published model/material/turbulence fields and time restore,
and the corrected input can retry successfully. The focused serial test and
its dedicated two-rank registration both pass.

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSASSupportedPaths
ctest --test-dir build/gcc -C Debug -R '^SASSupportedPathsTest.MaterialFeedback' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^SASMaterialFeedback_2procs$' --output-on-failure
```
