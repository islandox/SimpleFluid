# Geometry Epochs and Motion Transactions Contract

`PlanarALEMeshMotion` is the sole geometry mutator for the constrained solver-integrated `planarALE` path.

## Transactional Contract
- **Epoch Authority:** The geometry epoch is the authoritative numeric-cache revision. Any coordinate modification (via `PlanarALEMeshMotion`) must trigger a cache invalidation for dependent operators.
- **Lease Enforcement:** The exclusive controller lease and geometry epoch belong to the shared concrete mesh; every alias `MeshHandle` observes the same revision.
- **Snapshot/Restore:** Treat one ALE step as an accepted-state transaction:
    1. Snapshot all mutable fields, model ledgers, and history.
    2. Start the trial, perform assemblies.
    3. Accept only after all conservation and convergence gates pass.
    4. On exception, roll back geometry and restore all snapshots.

## Cache Invalidation
After a trial changes coordinates, the following must be refreshed:
- Cell-gradient and transport-geometry caches.
- Rhie–Chow face-flux geometry and boundary workspaces.
- Momentum and temperature transport coefficients and solver state.
- Pressure-projection matrices and predictor state.
- Coupled pressure--velocity numeric blocks.
- Model-owned transport geometry (liquid inventory, reacting gas).
- Geometry-derived output (VTU point coordinates).

For detailed ALE implementation, see `PlanarALEMeshMotion.hh`.
