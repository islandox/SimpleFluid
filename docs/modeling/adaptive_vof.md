# Adaptive Cartesian Volume of Fluid

`AdaptiveVoFSolver` extends the source-free, isothermal two-phase model described in the [uniform VoF guide](vof.md) to distributed dyadic Cartesian blocks. It relies on the [general AMR architecture](../architecture/general_amr.md) for mesh hierarchy and partitioning, and the [VoF transport contract](../architecture/vof_transport.md) for conservation guarantees.

## Hierarchy and Face Ownership

`Meshes::AMRGrid` defines a Cartesian box, root blocks, cell counts per block, periodicity, and a maximum refinement level.
- **Refinement Rules:** A maximum level difference of one is enforced between touching blocks (including edges and corners). Refinement replaces one block with eight children; coarsening requires a complete sibling group.
- **Flux Consistency:** Native canonical fine subfaces tile coarse faces exactly. Coarse cells consume the sum of fine subface fluxes directly without secondary area-fraction scaling. All levels advance with a single shared timestep (no subcycling).

## Interface Geometry and Transport

`VoF::AdaptiveVoF` reconstructs conservative PLIC planes on local block patches:
- **Stencils:** Neighbor samples restrict finer cell volumes and prolongate coarser cells into virtual stencils for height-function and PLIC surface-fit curvature evaluations.
- **Advection:** Each canonical face computes a signed swept gas volume by intersecting the donor PLIC plane with its subface prism. Liquid takes the complementary swept volume.
- **Compression:** Directional Weymouth–Yue compression uses a frozen indicator whose sum vanishes under the discrete continuity constraint.

## Flow and Transported Amounts

- **Momentum & Pressure:** Momentum uses the same phase face masses as the geometric update. At coarse/fine interfaces, least-squares gradients align adjacent cell-center values to the subface before computing normal differences.
- **Mobility:** Face mobility is defined as $d / (\rho_o d_o + \rho_n d_n)$, measured normal to the subface, ensuring consistent density-weighted impulses across refinement jumps.
- **Passive Scalars:** Registered passive amounts travel with their selected carrier's face flux using an implicit donor-concentration equation to maintain positivity even in cells with vanishing phase volume.

## Conservative Remapping

Regridding intersects old and new leaf cells. Refined gas occupancy is computed from the old PLIC geometry, while restriction sums child gas volumes.

`ConservedQuantity::amount` defines extensive amounts per owned cell, distributed according to their `InventoryCarrier`:

| Carrier | Remap Weight | Example |
|---|---|---|
| `BulkVolume` | Intersected control volume | Bulk extensive tracer |
| `GasVolume` | Intersected PLIC gas volume | Resolved gas moles |
| `LiquidVolume` | Complementary liquid volume | Dissolved species or dispersed bubbles in liquid |
| `MixtureMass` | $\rho_l V_l + \rho_g V_g$ | Momentum or mixture energy |

All remapped quantities must satisfy global extensive conservation checks independent of the flow solver's tolerances.

## Refinement and Coarsening Policy

- **Resolution Targets:** `refinement_plan()` targets 32 cells across the minimum resolved diameter by default.
- **Protection Zones:** The policy protects all mixed cells and pure-phase jumps with at least a four-cell stencil and a velocity-based motion buffer. High curvature, narrow films, and externally flagged wake regions can also trigger refinement.
- **Coarsening:** Only complete, uniform pure-phase sibling groups outside protected neighborhoods are eligible for coarsening, preventing the accidental merging of gas components or erasure of interfaces.

## Verification and Scope

This model supports Cartesian blocks, periodic boundaries, and slip walls. Moving meshes (ALE), prism/polyhedral adaptation, and reacting-gas coupling are covered by separate roadmap stages; see [vof_solver_contracts.md](../verification/vof_solver_contracts.md) for verification targets.
