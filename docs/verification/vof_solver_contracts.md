# VoF verification contracts

This index owns reproducible numerical assertions and focused test entry
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
  testRadiolyticVoFCoupler testBubbleRepresentation
ctest --test-dir build/gcc -C Debug -N -R 'VoF|BubbleRepresentation'
ctest --test-dir build/gcc -C Debug \
  -R '^(GeometricVoF|AdaptiveVoF|VoFPhysicalSources|ConservativeVoFTransport|VoFSolver|RadiolyticVoFCoupler|BubbleRepresentation)' \
  --output-on-failure
ctest --test-dir build/gcc -C Debug \
  -R '^(ConservativeVoFTransport_2procs|ConservativeVoFTransport_EmptyRank_33procs|AdaptiveVoFPhaseBudget_EmptyRank_9procs|AdaptiveVoF_2procs|AdaptiveVoFThreeDimensional_2procs|VoFPhysicalSources_2procs|AdaptiveVoFSolver_2procs|AdaptiveVoFLowMach_2procs|RadiolyticVoFCoupler_2procs|BubbleRepresentation_2procs|AdaptiveVoFLowMach_EmptyRank_9procs|BubbleRepresentation_EmptyRank_9procs)$' \
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
| ConstantSlip front | `RadiolyticVoFCouplerTest.ConstantSlipExtendsThroughEmptyLiquidAndReplay`: isolated relative front speed, independent implicit upwind profile, nonnegative inventories, number/mole conservation and exact replay. |
| Empty particle closure | `RadiolyticVoFCouplerTest.ParticleSlipUsesThePopulatedSideAtAnEmptyFront`: no invented empty-cell radius; front flux retains evaluated populated-side slip. |
| Depleted particle closure | `AdaptiveVoFLowMachTest.DepletedPopulationUsesPostSourceSupportForSlipAndTimestep` checks the isolated 0.02 m/s face slip after complete removal on one side, the source-drift timestep limit, conservation, MPI seams and retained-trial commit after rejection. |
| Population identity | `RadiolyticVoFCouplerTest.ReinitializationRebindsRequiredPopulationNames` and required-name removal coverage: source targets micro after reordering, missing names reject collectively, accepted state remains intact and old trials reject. |
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
| Hierarchy and construction | `testCartesianAMR`, `testExactExtrudedConstruction`, `testExtrudedAMR`, `testLocalProductAMR`, `testPolyhedralAMR`, `testMixedAMR`: ancestry, exact source observations, complete inverse families, closure and transactional rejection |
| Geometry and remap | `testConvexCellGeometry`, `testExactPolyhedralComplex`, `testExactExplicitProvider`, `testConservativeAMRRemap`, `testProductAMRRemap`, `testPolyhedralAMRRemap`, `testMixedAMRRemap`: actual native certificates, positive tiny pieces, coverage, moments, independent phases and paired quantities |
| Distributed migration | `testDistributedAdaptiveKey`, `testDistributedAdaptivePackets`, `testDistributedAdaptiveAuthority`, `testDistributedAdaptiveClosure`, `testDistributedAdaptiveBalance`, `testDistributedAdaptiveState`, `testDistributedCartesianKernel`, `testDistributedProductKernel`, `testDistributedMixedKernel`, `testFrozenNative`, `testSparseNativeMesh`, `testSparseNativePeriodic`, `testDistributedNativeOperators`, `testSparseOwnerExchange`: owner-local state, ghost completeness, exact contact authorities, final packet cost and observed empty owners |
| General geometry/transport | `testExactSweptGeometry`, `testDepartureFan`, `testPolyhedralReconstruction`, `testExactPhaseTransfer`, `testGeneralVoF`: exact contacts versus positive overlaps, signed PLIC support, donor fans, finite provenance and optional exact-policy separation |
| General flow | `testGeneralVoFFlowWorkspace`, `testConservativeVoFTransport`, `testGeneralVoFPhiClosure`, `testGeneralVoFExactPressure`, `testGeneralVoFCurvature`, `testGeneralVoFSolver`: physical residuals, conditioned sampled directions, independent phase mass, original physical horizons, rollback/regrid and cache invalidation |

For example, a reconstruction-only change can use:

```sh
cmake --build --preset GCC-Debug --target testPolyhedralReconstruction
build/gcc/bin/Debug/testPolyhedralReconstruction
mpiexec -n 2 build/gcc/bin/Debug/testPolyhedralReconstruction
mpiexec -n 4 build/gcc/bin/Debug/testPolyhedralReconstruction
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

## Proposed FP geometry acceptance gates

**Status: planned verification for the
[FP geometry proposal](../architecture/general_amr.md#proposed-floating-point-geometry-for-finite-vof),
not implemented behavior or a newly executed campaign.** The integrated
integer/FP predicate and lazy-distance optimizations still construct exact
geometry and run exact coverage/measure checks. Their recorded results do not
qualify approximate constructions, FP moments or a replacement coverage contract.

Keep exact-utility and conservative-accounting oracles distinct from physical
resolution requirements. The mandatory arbitrary subcell-phase reconstruction
success cases have been retired; ordinary-volume fits retain complementary,
reversed-normal and pure-endpoint coverage. Historical tiny-fit results are not
future production gates. Add separate resolution/refinement cases for the
proposed numerical path and compare admitted reconstructions against independent
geometric or analytic references. A changed policy must be named in its evidence.

| Gate | Required evidence before integration |
| --- | --- |
| Arithmetic routing and exact-work limits | Assert that the proposed finite backend performs no rational construction, volume/moment or ledger operations, including cache preparation, MPI serialization and retries. Exercise bounded FP retry, the sign-only exact exception and cap exhaustion; no hidden whole-geometry retry or policy switch. Reference/oracle work is explicit and measured separately |
| Volume and moment bounds | Analytic boxes/tetrahedra, oblique rational cuts, translated/scaled cells and extreme ranges; the independent reference lies inside every claimed enclosure, and the propagated error fits the unchanged physical budget. No moment is required solely to replay a qualified structural coverage proof |
| Construction and topology | Shared edge/face identity, complementary cuts, coplanarity, tangencies, collinear prefixes, positive slivers and real holes/overlaps; bounded FP retries or sign-only exact predicates must resolve decisions from original provenance. Unresolved geometry rejects; an exact classification of rounded coordinates cannot certify the source construction |
| PLIC fitting | `IndependentPhaseVolumesAreUnchangedDuringFitting` and `RationalCellFitsComplementaryPlanesAndPureEndpoints` cover ordinary phase amounts, nonbinary cell support, complementary labels, reversed normals and pure endpoints; no arbitrarily thin phase-fit success is required |
| Physical resolution and refinement | Planned cases must distinguish adequately resolved features, genuinely underresolved thickness/curvature/stencils, and small cut fractions from resolved grazing interfaces. Verify unit/scale invariance, refinement request and recheck, exhausted/unavailable refinement, actionable diagnostics and atomic rejection without phase-mass deletion |
| Inventory and zero-phase accounting | Independent gas/carrier storage, dominant-phase cancellation, small carrier products, pure phases and zero-reference residuals remain accounting checks. They do not require physical reconstruction of an underresolved phase; refinement or rejection must preserve accepted inventory rather than erase mass |
| Small intermediate contributions | Exercise individually unrepresentable contributions accumulated with scaling/compensation and propagated bounds, including many pieces whose total is significant. No contribution vanishes from the accounting/error budget; an uncertain sign is not clamped. Accept only established nonnegative, conservative budgets; otherwise reject atomically without rational volume fallback |
| Canonical conservation | One shared phase transfer per face across MPI, opposite incidences, compatible coarse/fine sums, source-free constant/pure states, donor availability and recipient occupancy; check local and global extensive budgets independently |
| Boundedness and repairs | Nonnegative phase amounts and supported CFL/capacity bounds; any permitted roundoff correction is shared-face conservative and bounded by derived uncertainty; no independent cell clipping or residual renormalization |
| Structural coverage certificates | Independently demonstrate containment, non-overlap and completeness for each admitted construction class before skipping repeated moment/pair checks; balanced gap/overlap examples must still reject. Cover signed overlays separately from positive partitions |
| Coupled physical acceptance | Compatible phase/mass/momentum ledgers, original pressure/divergence and physical-residual targets, and the named hydrostatic, capillary, viscous and coarse/fine horizons above |
| Provenance and transactions | Topology/geometry/partition and policy invalidation, rejected cache preparation, collective failure with asserted empty owners, stale/foreign owner evidence, and unchanged accepted state/time/history after rejection |
| Accumulated error and policy isolation | Track error across repeated steps, conservative remap and repartition without resetting it; preserve `ExperimentalExact` authority and reject cross-policy adoption or silent resource-triggered switching |
| Bounded retries and compiler coverage | Directly exercise decisive, ambiguous, range-limited and unsupported FP cases. Compare retries/predicates with original inputs and independent references; exhaust their work/storage budget and verify collective rejection. Unsupported environments must select a qualified bounded FP route or reject, never silently enable full rational geometry. Keep actual compiler/environment coverage explicit |

The general-native targets listed above provide the starting regression map.
Inspect their current CMake registrations, run affected Debug serial cases first,
then relevant MPI modes with observed empty owners. A new FP acceptance path
needs its own independent resolution, error and convergence tests. Test separately
that spatial underresolution requests refinement, while a resolved construction
with unresolved arithmetic rejects with an arithmetic diagnostic. Neither an
exact predicate nor a successful reference reconstruction can bypass the
resolution gate or rescue a failed finite acceptance budget. Existing exact-API
and `ExperimentalExact` tests keep their own authority; do not weaken them to
match FP values. No resolution controller, refinement request or FP backend is
implemented by these design changes or the earlier test retirement.
Full flow qualification includes both
long capillary/viscous methods; any exclusion remains an open named boundary.

After correctness gates pass, compare matched Release source/runtime snapshots
over complete physical horizons in serial and at two/four ranks. Preserve the
50-cell coarse/fine case's ten `0.002`-second steps through `0.02`, and add
longer runs to expose accumulated drift. Record initialization and step costs,
repetitions, per-rank wall/CPU time, fragment/pair counts, bounded FP retries and
rejections, and exact work split into predicate/construction/measure/ledger
counts. Report operand-size distributions for any exact predicates, their cap
usage, certificate/measure-cache reuse and process peak RSS. Verify zero rational
construction/measure/ledger work in the proposed finite backend; do not infer it
from a low sampling percentage. Report reference-oracle and instrumentation
cost separately; uninstrumented timing runs must not overlap other builds or
numerical jobs. No operation-count, memory or universal scaling claim may be
inferred from an inclusive CPU profile alone.
