# SST-SAS implementation verification — 2026-09-10

This record applies to the uncommitted SST-SAS implementation on
`feature/turbulence`, starting from
`812fc63518df3918ea46c0876440d1a7d49ed3c8`. The starting working tree was clean.
The checkout contained nine commits after the supplied GitHub reference
`ba220971273fd2794c8296ac49d7dc1899676378`; it was not reset, stashed, committed
or pushed. The existing SST-1994 implementation, shared presets, water fixture
scales/labels and external acceptance manifests were preserved.

## Baseline and environment

The task supplied the status of GitHub Actions run `34324281164` at the
reference revision: Ubuntu GCC Debug/Release passed, and Ubuntu/macOS LLVM
Debug/Release failed during compilation. Those historical failures have not
been diagnosed here and are not characterized as SAS failures.

Local dependencies matched the checked-in presets, including the GCC PIC and
LLVM libc++ Trilinos 17.2 installations under `~/local/mathlibs`. No shared
preset was edited. Local compilers were GCC 16.2.1 (20260810) and Clang 22.1.8;
Kokkos used its Serial host backend with MPI. Debug builds used two compiler
jobs on a 23 GiB host.

Before editing existing implementation files, the seven requested turbulence
targets (equations, model, options, model MPI, scalar transport, walls and
Boussinesq integration) built successfully with both GCC and LLVM. The GCC
baseline selection passed 85 tests and skipped one exact-two-rank body in its
86 serial registrations. The initial LLVM selection exposed an old
`testTurbulenceBuoyancy` executable with a stale template ABI; rebuilding that
target made all six buoyancy tests pass. It also exposed sandbox MPI launcher
failures before test execution. The two requested baseline MPI registrations
passed when rerun with host network access. These observations do not establish
an entirely green GitHub baseline or a complete baseline suite.

Baseline commands included:

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_MAX_TEST_PROCS=2
cmake --build --preset GCC-Debug --parallel 2 --target testTurbulenceEquations testTurbulenceModel testTurbulenceModelOptions testTurbulenceModelMultiRank testTurbulenceScalarTransportEquation testTurbulenceWallTreatment testTurbulentBoussinesqSolver
cmake --build --preset LLVM-Debug --parallel 2 --target testTurbulenceEquations testTurbulenceModel testTurbulenceModelOptions testTurbulenceModelMultiRank testTurbulenceScalarTransportEquation testTurbulenceWallTreatment testTurbulentBoussinesqSolver
ctest --test-dir build/gcc -C Debug -R 'Turbulence|TurbulentBoussinesq|SSTKOmega|BSLKOmega|KEpsilon|StandardKOmega' -E 'Buoyancy|_2procs' --output-on-failure
ctest --test-dir build/llvm -C Debug -R '^(Turbulence|TurbulentBoussinesq)' --output-on-failure
cmake --build --preset LLVM-Debug --parallel 2 --target testTurbulenceBuoyancy
ctest --test-dir build/llvm -C Debug -R '^TurbulenceBuoyancyTest\.' --output-on-failure
ctest --test-dir build/llvm -C Debug -R '^(TurbulenceModel_2procs|TurbulentBoussinesq_2procs)$' --output-on-failure
```

## Implementation locations

- `src/equations/turbulence/SSTSASSource.hh`: independent local algebra,
  coefficient/input checks, analytic zero-curvature limit, cap and statistics.
- `src/FVM/VectorLaplacian.hh` and `src/FVM/details/VectorLaplacian.hh`:
  unit-coefficient component diffusion with full non-orthogonal correction.
- `TurbulenceModel.hh/.cc/.tcc` and `TurbulenceEquations.hh`: canonical runtime
  selection, Menter/SST branches, flat options, lazy work/cache ownership,
  explicit omega assembly, accepted source records and snapshot/restart rules.
- `TurbulenceWallTreatment.hh`: internal visibility for the wall-evaluation
  copy constructor used by snapshots, matching the existing move constructor.
- `TimeStepperOptions.hh`, `BoussinesqSolver.tcc`,
  `IncompressibleIsothermalSolver.tcc`, and `FieldStateTransaction.hh`:
  physical-time qualification and accepted-state restoration after rejected
  active-SAS steps, including failures after turbulence has advanced.
- `src/equations/unitTests/testSSTSASSource.cc`, `testSSTSASModel.cc`,
  `sst_sas_reference.csv`, `testTurbulenceModelOptions.cc`, and
  `src/solvers/unitTests/testTurbulentBoussinesqSolver.cc`: layered evidence.
- `src/examples/sst_sas_activation.cc`, the pitzDaily/Shiri steady-search
  guards, and the two affected CMakeLists files: executable and registrations.
- `verification/sst_sas/pointwise_reference.py` and `compare_serial_mpi.py`:
  independent algebra and partition comparisons. README, TODO and
  [the modeling document](sst_sas.md) describe configuration and qualifications.

## Executed final checks

These are counts for each command, not disjoint totals. MPI counts refer to
CTest registrations containing multiple GTest bodies; serial-only bodies skip
inside the corresponding MPI executables by design.

| Check | Actual result |
| --- | --- |
| GCC Debug selected turbulence/SAS suite, example and ELF boundary | 122 registrations: 121 passed, one expected serial skip |
| LLVM Debug same selection | 119 registrations: 118 passed, one expected serial skip |
| Additional GCC cache, options, isothermal, steady-search, ALE and example regressions | 76 registrations: 73 passed, three expected serial skips |
| Explicit GCC two-rank registrations | 8/8 passed |
| Explicit LLVM two-rank registrations | 5/5 passed |
| Independent Decimal reference table | Six rows matched; C++ pointwise comparisons passed on both compilers |
| Serial/two-rank transient comparison | All seven reduced quantities agreed with rtol=1e-9, atol=1e-11 |
| VTU diagnostic inspection | All eleven names present in serial output, both rank-local VTUs and the PVTU index |
| `git diff --check` | Passed |

The broad regex also selected four additionally built isothermal/steady-monitor
registrations in GCC and one previously built boundary-condition registration
in LLVM. This explains the net count difference; all SAS registrations ran on
both compilers. The expected serial skips in the main selections are the existing
combined RANS/bubble exact-two-rank test. The broader serial skips require
multiple ranks for parallel output, rank-divergent selections or unsupported
multi-rank semi-structured configuration; the ALE MPI registration also ran.

The following build commands and test selections were executed after the
implementation (incremental repetitions were used after corrections):

```sh
cmake --build --preset GCC-Debug --parallel 2 --target testSSTSASSource testSSTSASModel testTurbulenceEquations testTurbulenceModel testTurbulenceModelOptions testTurbulenceModelMultiRank testTurbulenceScalarTransportEquation testTurbulenceWallTreatment testTurbulenceBuoyancy testTurbulentBoussinesqSolver testIncompressibleIsothermalSolver testStoredTransportReuse testGeometryEpochCaches testCellGradientCache testTimeStepperOptions testSteadyStateSearch testBoussinesqPlanarALE pitz_daily natural_convection_shiri sst_sas_activation
cmake --build --preset LLVM-Debug --parallel 2 --target testSSTSASSource testSSTSASModel testTurbulenceEquations testTurbulenceModel testTurbulenceModelOptions testTurbulenceModelMultiRank testTurbulenceScalarTransportEquation testTurbulenceWallTreatment testTurbulenceBuoyancy testTurbulentBoussinesqSolver sst_sas_activation
ctest --test-dir build/gcc -C Debug -R 'SSTSAS.*Test\.|Turbulence|TurbulentBoussinesq|SSTKOmega|BSLKOmega|KEpsilon|StandardKOmega|sst_sas_activation_small|simplefluid_elf_export_boundary' -E '_2procs' --output-on-failure
ctest --test-dir build/llvm -C Debug -R 'SSTSAS.*Test\.|Turbulence|TurbulentBoussinesq|SSTKOmega|BSLKOmega|KEpsilon|StandardKOmega|sst_sas_activation_small|simplefluid_elf_export_boundary' -E '_2procs' --output-on-failure
ctest --test-dir build/gcc -C Debug -R 'StoredTransportReuse|GeometryEpochCache|CellGradientCache|TimeStepperOptions|SteadyState|IncompressibleIsothermal|BoussinesqPlanarALE|pitz_daily.*smoke|natural_convection_shiri_smoke' -E '_2procs' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^(SSTSAS_2procs|SSTSASActivation_2procs|TurbulenceModel_2procs|TurbulenceBuoyancy_2procs|TurbulentBoussinesq_2procs|IncompressibleIsothermalSolver_2procs|StoredTransportReuse_2procs|GeometryEpochCaches_2procs|BoussinesqPlanarALE_2procs)$' --output-on-failure
ctest --test-dir build/llvm -C Debug -R '^(SSTSAS_2procs|SSTSASActivation_2procs|TurbulenceModel_2procs|TurbulenceBuoyancy_2procs|TurbulentBoussinesq_2procs)$' --output-on-failure
python3 verification/sst_sas/pointwise_reference.py --check
python3 verification/sst_sas/compare_serial_mpi.py build/gcc/bin/Debug/sst_sas_activation
git diff --check
```

`GeometryEpochCaches_2procs` is not a registered test; its regex alternative
selected nothing. Geometry-epoch rejection was exercised by the serial cache
test and the new SAS MPI model test. MPI commands were run with working host
network access after the baseline launcher failures. VTU output was generated
in fresh `/tmp/simplefluid-sas-output-*` and
`/tmp/simplefluid-sas-mpi-output-*` directories and its headers inspected.

Two implementation-time issues were corrected rather than weakening gates.
The rollback regression initially injected a failure during pre-step source
refresh; it now uses two individually finite heat sources whose sum overflows
only in the subsequent temperature assembly, after both turbulence solves.
LLVM's ELF audit found six emitted wall-evaluation copy constructors from the
snapshot path; they received internal visibility, and both final export audits
passed. An interim LLVM compilation also crashed with malformed-source
messages while a test source was being updated; rebuilding with sources held
fixed succeeded. This is not attributed to the historical GitHub LLVM failures.
No failing final check is being classified as a pass.

## Numerical evidence and limits

The 6x6x6 isothermal example, after two physical steps of 0.001 s, reported:

| Reduced quantity | Serial | Two ranks |
| --- | --- | --- |
| Active-cell fraction, Qapplied>0 | 1 | 1 |
| Cap-active fraction | 0 | 0 |
| Minimum applied source (s^-2) | 3.6607425658362276 | 3.6607425658362303 |
| Maximum applied source (s^-2) | 522.8973502293618 | 522.8973502293611 |
| Volume-mean applied source (s^-2) | 203.1480285889810 | 203.1480285889813 |
| Volume-integrated applied source (m3/s2) | 203.1480285889817 | 203.1480285889817 |
| Volume-integrated omega (m3/s) | 2.668089603137136 | 2.668089603137141 |

The tests additionally exercise runtime cap activity, density/volume-independent
one-step omega response, old-input source recording across closure refresh,
bitwise source-disabled parent equivalence for native and legacy fields,
nonzero source invariance under offsets/rotations (including transformed face
fluxes and boundary values), resolved SST wall coordination/automatic distance,
rank-local invalid data, exact snapshot restore, field-only restart clearing,
and reconfiguration. The Boussinesq regression has both nonzero SAS and direct
buoyancy production, finite fields and continuity control, and a downstream
rejection followed by successful retry. A three-outer-corrector PIMPLE fixture
checks the analytic one-step k decay, preventing accidental repeated physical
turbulence advancement.

Manufactured uniform, affine-shear and rigid-rotation vector fields preserve
zero curvature to roundoff. Orthogonal interior quadratic vector Laplacians
and affine scalar gradients match their analytic values. The skewed-prism
quadratic volume-weighted L1 error fell from **1.15266 to 0.633455** for 4^3 to
8^3 source-grid subdivisions; boundaries are included. Native and legacy
operator paths, native partitioned unstructured faces and stale geometry
rejection were exercised. This is component evidence for the documented
operator scope, not a universal accuracy bound on arbitrary mesh distortion.

No full repository suite, Release matrix, macOS build, accelerator test,
long-running three-dimensional turbulence assessment, whole-flow OpenFOAM
comparison, or experimental validation was run. Active cylindrical,
SemiStructuredXY_Z, slip, periodic, Gauss-linear and additional multiphysics
SAS paths remain explicitly unsupported. The source and fixed-grid transient
integration are implemented, component-tested and solver-regression-tested;
whole-flow external comparison and physical scale-resolution validation remain
open. Existing pitzDaily qualification and external-comparison tolerances were
not relaxed.
