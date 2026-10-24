# VoF verification contracts

Floating-point arithmetic is the primary VoF path. This index owns its
reproducible numerical assertions and focused test entry
points. The modeling guides and [general AMR design](../architecture/general_amr.md#qualification-status)
own support and qualification status. Raw commands, source/runtime manifests,
XML, logs and timing samples belong in local build artifacts. Historical passes
remain scoped to their recorded source and do not imply a new run after edits.

## Focused Debug workflow

Use the compiler and dependency locations from a configured repository preset.
The configured build requires C++23, Trilinos with Tpetra/Belos/Ifpack2, MPI and
Google Test. Configure with `cmake --preset GCC-ninja-multi` when needed.

```sh
cmake --build --preset GCC-Debug --target \
  testGeometricVoF testAdaptiveVoF testAdaptiveVoFThreeDimensional \
  testVoFPhysicalSources testConservativeVoFTransport testVoFSolver \
  testAdaptiveVoFSolver testVoFLowMach testAdaptiveVoFLowMach \
  testReactingVoFCoupler testBubbleRepresentation
ctest --test-dir build/gcc -C Debug -N -R 'VoF|BubbleRepresentation'
ctest --test-dir build/gcc -C Debug \
  -R '^(GeometricVoF|AdaptiveVoF|VoFPhysicalSources|ConservativeVoFTransport|VoFSolver|ReactingVoFCoupler|BubbleRepresentation)' \
  --output-on-failure
ctest --test-dir build/gcc -C Debug \
  -R '^(ConservativeVoFTransport_2procs|ConservativeVoFTransport_EmptyRank_33procs|AdaptiveVoFPhaseBudget_EmptyRank_9procs|AdaptiveVoF_2procs|AdaptiveVoFThreeDimensional_2procs|VoFPhysicalSources_2procs|AdaptiveVoFSolver_2procs|AdaptiveVoFLowMach_2procs|ReactingVoFCoupler_2procs|BubbleRepresentation_2procs|AdaptiveVoFLowMach_EmptyRank_9procs|BubbleRepresentation_EmptyRank_9procs)$' \
  --output-on-failure
```

Local runs exceeding available MPI slots require explicit launcher
oversubscription. The direct transport fixtures have at most 32 cells, so 33
ranks guarantee empty owners regardless of partitioning; their tests assert the
observed empty-owner count. For example:

```sh
mpiexec --map-by :OVERSUBSCRIBE -n 33 build/gcc/bin/Debug/testConservativeVoFTransport
mpiexec --map-by :OVERSUBSCRIBE -n 9 build/gcc/bin/Debug/testAdaptiveVoF \
  --gtest_filter='AdaptiveVoFTest.MicroscopicPhaseOutflowCannotOverdrawCanonicalSubfaceDonors'
```

Inspect actual CTest registrations before narrowing a run. Direct executable
runs with `--gtest_filter` and `--gtest_output=xml:<local-path>` preserve numerical
properties. An MPI runtime unable to create local sockets must be rerun in an
environment supporting normal host MPI; startup failure is not a numerical test.

## Numerical assertions

| Contract | Reproducible coverage |
|---|---|
| Passive momentum before projection | `ConservativeVoFTransportTest.SplitCounterexampleHasBoundedConservativeImplicitTracesForEveryFirstAxis`: original 4×4×2, spacing (1,1,100), dt=0.1, density ratio 1000; every first sweep, pure/vanishing phases, local face ledger and total momentum. |
| Accepted incompressible momentum | `VoFSolverTest.SplitPhaseMassPreservesPassiveMomentumBounds` and `AdaptiveVoFSolverTest.PassiveTransverseMomentumRemainsBoundedAcrossSplitPhaseTransport`: original initial data, rotating sweeps, native publication, coarse/fine transport, uniform velocity and MPI ownership crossings. |
| Low-Mach limit | `AdaptiveVoFLowMachTest.SourceFreeConstantDensityLimitPreservesPassiveTransverseMomentumBounds`: source-free, constant phase density, thermal recovery, preview/commit, mass and physical momentum gates. |
| Microscopic phase flux | `GeometricVoF.MicroscopicPhaseOutflowCannotOverdrawItsDonor` and `AdaptiveVoFTest.MicroscopicPhaseOutflowCannotOverdrawCanonicalSubfaceDonors`: all sweep orders, strictly nonnegative retained traces, conservative roundoff-only donor budgets, coarse/fine faces and empty-owner participation. |
| Microscopic carrier conditioning | `ConservativeVoFTransportTest.EquilibratedMicroscopicCarrierPreservesConstantConcentration` retains concentration through ordinary and subnormal storage without reciprocal overflow; the coarse/fine translation fixture checks every accepted passive inventory and the original unscaled residual. |
| Extreme carrier scales | `ConservativeVoFTransportTest.EquilibratedStationaryTracerHasAnOverflowSafePhysicalNormRatio` retains the cases whose former norm bound overflowed. `AdaptiveVoFSolverTest.StationaryMicroscopicColumnPreservesItsPassiveConcentration` and `StationarySubnormalColumnPreservesItsPassiveConcentration` retain exact stationary tracers with fractions 1e-200 and 1e-311. |
| Physical residual stopping | `AdaptiveVoFSolverTest.MicroscopicColumnAdvectionUsesTheOriginalPhysicalResidualTarget` checks the independent periodic implicit-upwind solution for the moving 4×4×2 tracer, original physical residual, conservation and checkpoint/rejection gates. `ConservativeVoFTransportTest.PhysicalDefectCorrectionHonorsTheSharedIterationBudgetAndRecovers` forces a correction, checks exhaustion of a one-iteration budget, and changes RHS on the retained operator. |
| ConstantSlip front | `ReactingVoFCouplerTest.ConstantSlipExtendsThroughEmptyLiquidAndReplay`: isolated relative front speed, independent implicit upwind profile, nonnegative inventories, number/mole conservation and exact replay. |
| Empty particle closure | `ReactingVoFCouplerTest.ParticleSlipUsesThePopulatedSideAtAnEmptyFront`: no invented empty-cell radius; front flux retains evaluated populated-side slip. |
| Depleted particle closure | `AdaptiveVoFLowMachTest.DepletedPopulationUsesPostSourceSupportForSlipAndTimestep` checks the isolated 0.02 m/s face slip after complete removal on one side, the source-drift timestep limit, conservation, MPI seams and retained-trial commit after rejection. |
| Population identity | `ReactingVoFCouplerTest.ReinitializationRebindsRequiredPopulationNames` and required-name removal coverage: source targets micro after reordering, missing names reject collectively, accepted state remains intact and old trials reject. |
| Geometry and remap | Geometric/adaptive FVM suites: PLIC identities, independent curved fractions, curvature, canonical coarse/fine signed fluxes, periodic/corner closure, remap conservation and metadata validation. |
| Translated reconstruction coverage | `AdaptiveVoFTest.TranslatedNarrowDomainReconstructsConstantFractions` constructs the narrow nonperiodic domain at origins 0 and ±1e6. `RepresentedCoverageRejectsMissingTranslatedQueryVolume` retains detection of a two-coordinate-ULP seam hole. |
| AMR capillary horizon | `AdaptiveVoFSolverTest.MovingPlanarCapillaryInterfaceCrossesACoarseFineSeam` advances for 0.75 s; `IndependentlyIntegratedStaticBubbleRetainsCapillaryBalanceOverFiniteDuration` advances an independently integrated sphere for 0.002 s in a uniformly refined interface neighborhood. `CurvedMovingCapillaryInterfaceCrossesACoarseFineSeam` uses active curvature at density ratio ten over 0.125 s with two fine-cell widths of translation and asserts actual phase passage. All assert pressure, phase-volume drift and parasitic velocity in their stated regimes. |
| Physical acceptance and transactions | Solver, low-Mach and representation suites: physical pressure/divergence, capillary balance, thermodynamic work/history, conservative representation changes, checkpoint/replay and forced failure without publication. |

The passive bound applies to the force-free, divergence-free transverse fixture.
It does not restrict velocity in a general pressure-driven or forced flow.
Tolerances are stated beside the independent numerical assertions in each test.

Curvature requires independently integrated mixed-cell fractions. Pure cells
return zero curvature; a binary center-point voxelization can therefore produce
zero curvature throughout a curved object. A zero-curvature binary initialization
cannot establish a static-bubble pressure or capillary-current qualification.

## Performance comparisons

Build the opt-in `adaptive_vof_geometry_benchmark` and
`adaptive_vof_transport_benchmark` and `adaptive_vof_refinement_benchmark` targets in the same configuration for both
source snapshots. Record arguments, compiler, dependency build, mesh/block sizes,
population/rank counts, repetitions, linear/corrector work, timings and memory.
First fix numerical behavior, then freeze that corrected baseline before
optimizing. Compare complete geometric outputs and physical acceptance results
before interpreting timing changes.

Geometry metadata payload estimates and measured process peak RSS are different
quantities. Debug measurements diagnose allocation/work changes; they do not
qualify optimized production speedups. Physical bubble rise, long-time capillary
behavior, empirical gas closures and application coupling need separate studies.

The disabled `CurvedCoarseFineSeamStaticBubbleQualification` case preserves the
stronger static-sphere test with the curved interface directly crossing the
refinement seam at density ratio 10,000. Its parasitic-velocity criterion is an
open qualification requirement. Run it explicitly with
`--gtest_also_run_disabled_tests --gtest_filter='*CurvedCoarseFineSeamStaticBubbleQualification'`;
passing the default capillary cases does not qualify this configuration.

## General native AMR and flow checks

Select only the affected family from this map; inspect current registrations
before executing. These commands are reproduction entry points, not a report
that a campaign ran during documentation maintenance.

| Scope | Targets and independent gates |
| --- | --- |
| Hierarchy and construction | `testCartesianAMR`, `testExtrudedAMR`, `testLocalProductAMR`, `testPolyhedralAMR`, `testMixedAMR`: ancestry, exact source observations, complete inverse families, closure and transactional rejection |
| Native authority boundary | Source/geometry checks are indexed in [ExactGeometry](../architecture/exact_geometry.md#focused-verification); they qualify source import separately from FP numerical geometry and remap |
| Distributed migration | `testDistributedAdaptiveKey`, `testDistributedAdaptivePackets`, `testDistributedAdaptiveAuthority`, `testDistributedAdaptiveClosure`, `testDistributedAdaptiveBalance`, `testDistributedAdaptiveState`, `testDistributedCartesianKernel`, `testDistributedProductKernel`, `testDistributedMixedKernel`, `testFrozenNative`, `testSparseNativeMesh`, `testSparseNativePeriodic`, `testDistributedNativeOperators`, `testSparseOwnerExchange`: owner-local state, ghost completeness, exact contact authorities, final packet cost and observed empty owners |
| Primary FP geometry/transport | `testBoundedConvexGeometry`, `testBoundedPlanarFace`, `testBoundedPhaseArithmetic`, `testPolyhedralReconstruction`, `testGeneralVoF`, `testGeneralVoFFinite`, `testFiniteVoFRemap`: bounded constructions, signed PLIC support, refinement requests, canonical transfers, independent phases and history |
| General flow | `testGeneralVoFFlowWorkspace`, `testConservativeVoFTransport`, `testGeneralVoFPhiClosure`, `testGeneralVoFCurvature`, `testGeneralVoFSolver`: physical residuals, independent phase mass, original physical horizons, rollback/regrid and cache invalidation |
| Optional exact backend | `testExactGeneralVoF`, `testExactPolyhedralReconstruction`, `testExactConservativeVoFTransport`, `testExactGeneralVoFFlowWorkspace`, `testExactGeneralVoFPhiClosure`, `testExactGeneralVoFCurvature`, `testExactGeneralVoFSolver`: rational operations, exact storage/history and explicit cross-policy boundaries; registered with the `exact` label |

For example, a reconstruction-only change can use:

```sh
cmake --build --preset GCC-Debug --target testPolyhedralReconstruction
ctest --test-dir build/gcc -C Debug -R '^PolyhedralReconstruction\.' --output-on-failure
ctest --test-dir build/gcc -C Debug -R '^PolyhedralReconstruction_(2|4)procs$' --output-on-failure
```

Full flow qualification must name the physical methods explicitly:
`PhysicalHydrostaticPressureBalancesSkewNativeCentersOverFourSteps`,
`IndependentlyIntegratedReflectedCylinderRetainsFiniteDurationCapillaryBalance`,
`ClosedSymmetryViscousModeRetainsAnalyticFiniteDurationRate`, and
`NonzeroSkewCoarseFinePhasePassagePreservesIndependentInventories`.
The coarse/fine fixture uses 50 native cells after closure and ten 0.002 s
steps. The cylinder retains its 0.02 s endpoint; the refined viscous fixture
uses 8×8×2 cells and ten 0.002 s steps with its fixed 5% rate gate. Setup,
shortened prefixes, operator-only diagnostics and timeouts cannot replace these
physical assertions. Use distinct per-rank XML paths and count logical methods
separately from rank executions. Explicit exclusions and known failures must
remain visible; consult the design's qualification table before reporting a pass.

<a id="proposed-fp-geometry-acceptance-gates"></a>

## Floating-point acceptance gates

**Status: [ordinary FP is the default and bounded certification is opt-in](../architecture/general_amr.md#floating-point-geometry-and-arithmetic).
The finite modes have focused geometry, reconstruction, arithmetic, projection and remap coverage;
full coupled-flow and performance qualification remain open.** Earlier
rational-route results are documented in [ExactVoF](../architecture/exact_vof.md)
and do not qualify the FP implementation. `testBoundedConvexGeometry`,
`testBoundedPlanarFace`, `testPolyhedralReconstruction`,
`testBoundedPhaseArithmetic`, `testGeneralVoFFinite`, `testFiniteVoFRemap` and
the finite Flow/Phi tests provide separate component gates. Consult actual
run manifests for source/configuration, ranks and method results; declarations
or a component pass are not evidence that the full horizon completed.

Mode regressions distinguish ordinary scalar admission from explicit bounded
certification, while checking the same physical invariants and rejecting mixed
mode state/packets. Ordinary checks do not establish an interval certificate.
Keep exact-utility and conservative-accounting oracles distinct from physical
resolution requirements. The mandatory arbitrary subcell-phase reconstruction
success cases have been retired; ordinary-volume fits retain complementary,
reversed-normal and pure-endpoint coverage. Historical tiny-fit results are not
future production gates. Add separate resolution/refinement cases for the
finite numerical path and compare admitted reconstructions against independent
geometric or analytic references. A changed policy must be named in its evidence.

| Gate | Required qualification evidence |
| --- | --- |
| Arithmetic routing and exact-work limits | Assert that ordinary and certified finite geometry/ledger evaluation performs no rational construction, volume/moment or ledger operations. Count native-authority imports and reference/oracle work separately. Verify the ordinary default, explicit certification, cap exhaustion and rejection of mixed modes; no failure may switch modes |
| Volume and moment bounds | Analytic boxes/tetrahedra, oblique rational cuts, translated/scaled cells and extreme ranges; the independent reference lies inside every claimed enclosure, and the propagated error fits the unchanged physical budget. No moment is required solely to replay a qualified structural coverage proof |
| Construction and topology | Shared edge/face identity, complementary cuts, tangencies, collinear prefixes, positive slivers and real holes/overlaps remain checked in both modes. Ordinary geometry uses scalar predicates; bounded certification additionally requires original-provenance enclosure decisions. Ordinary results must not claim a certified enclosure |
| PLIC fitting | `PolyhedralReconstruction.RationalSupportFitsComplementaryOrdinaryVolumes`, `FarMinoritySupportRetainsOffsetPrecisionForBothLabelsAndNormals` and the pure-endpoint case cover bounded FP fitting, nonbinary support, complementary labels and reversed normals; arbitrary thin-phase fit success is not required |
| Physical resolution and refinement | Finite reconstruction tests distinguish supported neighborhoods, insufficient phase/stencil support and small grazing cuts; test scaling, actionable refinement requests and unchanged inventory on rejection. This necessary support/model screen is not a full thickness/curvature or application-convergence test. Automatic refinement/reinitialization, exhausted-refinement handling and independent convergence remain open |
| Inventory and zero-phase accounting | Independent gas/carrier storage, dominant-phase cancellation, small carrier products, pure phases and zero-reference residuals remain accounting checks. They do not require physical reconstruction of an underresolved phase; refinement or rejection must preserve accepted inventory rather than erase mass |
| Small intermediate contributions | Exercise individually unrepresentable contributions accumulated with scaling/compensation and propagated bounds, including many pieces whose total is significant. No contribution vanishes from the accounting/error budget; an uncertain sign is not clamped. Accept only established nonnegative, conservative budgets; otherwise reject atomically without rational volume fallback |
| Canonical conservation | One shared phase transfer per face across MPI, opposite incidences, compatible coarse/fine sums, source-free constant/pure states, donor availability and recipient occupancy; check local and global extensive budgets independently |
| Ordinary face construction | `OrdinaryVoFSweep` checks signed represented-volume defects against independent cell-derived allowances, unchanged canonical controls and rejection under smaller or invalid budgets. Integrated coverage must verify owner-supplied canonical subface counts, identical shared-face diagnostics across ranks and combined cell closure within the original volume allowance |
| Boundedness and repairs | Nonnegative phase amounts and supported CFL/capacity bounds; any permitted roundoff correction is shared-face conservative and bounded by derived uncertainty; no independent cell clipping or residual renormalization |
| Structural coverage certificates | Independently demonstrate containment, non-overlap and completeness for each admitted construction class before skipping repeated moment/pair checks; balanced gap/overlap examples must still reject. Cover signed overlays separately from positive partitions |
| Coupled physical acceptance | Compatible phase/mass/momentum ledgers, original pressure/divergence and physical-residual targets, and the named hydrostatic, capillary, viscous and coarse/fine horizons above |
| Provenance and transactions | Topology/geometry/partition and policy invalidation, rejected cache preparation, collective failure with asserted empty owners, stale/foreign owner evidence, and unchanged accepted state/time/history after rejection |
| Accumulated error and policy isolation | Track error across repeated steps, conservative remap and repartition without resetting it; reject cross-policy adoption or silent resource-triggered switching; optional-policy contracts are owned by [ExactVoF](../architecture/exact_vof.md) |
| Bounded retries and compiler coverage | In explicit certification mode, exercise decisive, ambiguous, range-limited and unsupported-environment cases, and exhaust work/storage budgets. In ordinary mode, verify scalar finite/domain checks and the unchanged physical acceptance criteria. Keep reference/oracle runs separate and compiler/environment coverage explicit; never enable exact geometry as a fallback |

The general-native targets listed above provide the starting regression map.
Inspect their current CMake registrations, run affected Debug serial cases first,
then relevant MPI modes with observed empty owners. A new FP acceptance path
needs its own independent resolution, error and convergence tests. Test separately
that spatial underresolution requests refinement, while a resolved construction
with unresolved arithmetic rejects with an arithmetic diagnostic. A reference
calculation cannot bypass the resolution gate or rescue a failed FP acceptance
budget. Optional exact-API verification is owned by
[ExactVoF](../architecture/exact_vof.md#verification-and-historical-qualification),
[ExactTransport](../architecture/exact_transport.md#focused-verification) and
[ExactGeometry](../architecture/exact_geometry.md#focused-verification).
The finite backend and refinement-request API are implemented;
an automatic adaptation controller and full physical resolution qualification
remain open. Tangent remap cases whose newly tiny phase cannot meet its own
arithmetic accuracy budget reject atomically; this is an arithmetic limitation,
not a phase cutoff or proof of spatial underresolution.
Full flow qualification includes both
long capillary/viscous methods; any exclusion remains an open named boundary.

After correctness gates pass, compare matched Release source/runtime snapshots
over complete physical horizons in serial and at two/four ranks. Preserve the
50-cell coarse/fine case's ten `0.002`-second steps through `0.02`, and add
longer runs to expose accumulated drift. Record initialization and step costs,
repetitions, per-rank wall/CPU time, fragment/pair counts, bounded FP retries and
rejections, certificate/measure-cache reuse and process peak RSS. Report native
import and selected reference costs independently. Verify zero rational
construction/measure/ledger work in finite numerical steps; count native-source
authority import separately and do not infer zero work from a low sampling
percentage. Report instrumentation settings and overhead; timing runs must
not overlap other builds or
numerical jobs. No operation-count, memory or universal scaling claim may be
inferred from an inclusive CPU profile alone.

## Benchmark setup and reference data

A benchmark may use verified reference data or independent analytic phase
initialization. Regenerating rational reference volumes in every run is not a
requirement for the primary FP path. Keep reference-oracle checks in explicitly
selected correctness work. Preserve the same physical initial data and error
criteria when changing a fixture.

Measure fixture construction/adaptation, native authority import, solver
construction, explicit initialization and physical steps separately, alongside
complete-case wall time. A timer spanning all setup stages cannot be reported
as `initialize()` alone. The solver constructor currently initializes a default
state, so a following explicit initialization repeats part of the preparation.
Exact-API instrumentation is optional; bounded geometry/ledger counters are
built in. Describe timing instrumentation and concurrent workloads accurately.
