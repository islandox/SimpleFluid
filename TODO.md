# TODO: SimpleFluid Multiphysics Roadmap

**Current work (`feature/vof`):** qualify experimental source-free VoF on the
generic reacting-flow foundation.

## Roadmap dashboard

### Current priorities

Work through the [Stage 7 checklist](#stage-7--general-vof-and-flow-qualification)
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

## Active roadmap

See the specific project documentation for detailed implementation contracts and validation gates for active workstreams.
