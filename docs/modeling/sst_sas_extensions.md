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
