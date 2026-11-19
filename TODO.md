# TODO: SimpleFluid Multiphysics Roadmap

**Current work (`feature/vof`):** qualify experimental source-free VoF on the
generic reacting-flow foundation.

## Roadmap dashboard

### Current priorities

Work through the [Stage 7 checklist](docs/verification/vof_solver_contracts.md#stage-7-scope-and-priorities)
in this dependency order:

1. Preserve the implemented Ordinary compatible projection and its independent
   pressure-work/energy checks while completing coupled-flow qualification.
2. Use the faithful native replays to distinguish optimizer defects from
   remaining model rejection; retain strict target-volume and work gates.
3. Resolve Q2 storage/accounting blockers and complete the required refinement
   sequence, separating capacity results from numerical convergence.
4. Complete required GeometricFace Q3 coupled motion; recheck DonorCell where
   shared operators change without erasing its displacement failure.
5. Complete the existing GeometricFace capillary horizon after shared-flow
   prerequisites pass; defer DonorCell curvature-support expansion.

### Status map

See [README.md](README.md) for capability overview and project status.
Detailed workstream status is tracked in [docs/verification/](docs/verification/).

- [x] **AMR storage Stage 6 — whole-pipeline scaling.** The serial and registered two- and four-rank gates cover
  Cartesian, Product, Polyhedral, Static with a refinable companion, and Mixed providers. They exercise fixed edits,
  migration, retained generations, and publication under the 512 MiB scaling fixture budget.
  [AMR heap attribution](docs/architecture/amr_heap_attribution.md)
- [x] **AMR storage Stage 7 — caloric exception exports.** The Solvers library owns the vtables and type information
  for `CaloricDomainUnresolved` and `ThermalCaloricRangeError`. The shared-library exception and export checks
  passed in GCC Debug and LTO-enabled RelWithDebInfo.
- [x] **Persistent geometry-demand storage and prepared-page policy.** The hierarchy maintains the exact owner union
  across snapshot roots. The serial trusted Cartesian stream prepares all 256 and 512 cells under the 64 MiB
  hierarchy budget. [Demand, admission and compatibility paths](docs/architecture/general_amr.md#persistent-geometry-demand-storage-and-prepared-page-policy)
- [x] **Persistent owner-local topology with certified contact/subface reuse.** Stage 8 covers Cartesian, Product,
  Polyhedral, Static and Mixed snapshots in serial and at two and four ranks. The four-rank cubic probes passed
  at 64 MiB for explicit-row N32 and canonical N16, and at 96 MiB for full canonical N32 in RelWithDebInfo.
  [Cubic halo measurements](docs/architecture/spatial_adaptive_closure.md#cubic-halo-storage)

## Active roadmap

See the specific project documentation for detailed implementation contracts and validation gates for active workstreams.
