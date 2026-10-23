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
