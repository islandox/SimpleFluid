# VoF verification contracts

This tracked index identifies reproducible numerical checks. The
[review follow-up summary](vof_review_followup.md) records the tested source,
commands, measured outcomes and remaining qualification for the follow-up to
`8fbfb988`. Raw build logs, XML and CSV outputs remain local build artifacts.
Historical Stage 1–4 reports
referenced by earlier documentation were local ignored files and are unavailable
in a fresh checkout. Use the source tests and commands below to reproduce their
covered contracts; do not infer historical measured results from a link.

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
