# Capillary Geometry Model

The capillary geometry model provides a curvature-based closure for surface tension, integrated with the VoF solvers to determine pressure-jump conditions at phase interfaces.

## Modeling Approach

- **Surface Catalog:** The model operates on a detached, immutable surface catalog derived from the accepted phase authority. This catalog provides the geometric basis for curvature fitting independently of the transport backend.
- **Curvature Fitting:** Curvature is determined via local quadratic Monge-surface fits or filtered binary reconstruction. The solver requires a connected neighborhood of phase-samples, which must resolve the interface topology to be admissible.
- **Filtering:** To reduce spurious currents and provide physically consistent interface representation, binary gas components can be filtered. This uses a volume-constrained geometric smoothing technique, providing a continuous piecewise-linear surface description.

## Architectural Interface

- **Curvature Consumption:** The solver-owned pressure/capillary operator consumes the calculated cell curvature and interface normals. This operator is responsible for enforcing force-balance conditions, while the geometry model handles the reconstruction of the interface itself.
- **Support Admissibility:** The model enforces strict support connectivity requirements. If a cell cannot establish a connected neighborhood to support a robust curvature fit (e.g., due to topological ambiguity or insufficient resolution), it reports a `ResolutionRequired` failure.

## Integration Principles

- **Geometric Consistency:** The capillary fit is performed by the cell owner based on canonical neighbor-face incident data.
- **Decoupling:** Capillary surface construction does not alter physical inventories, phase history, or transport declarations. It is a diagnostic and force-calculation utility that runs in parallel with the solver’s transport cycle.
- **Arithmetic Integrity:** Like transport kernels, curvature reconstruction honors the collective `ArithmeticMode` (Ordinary/BoundedCertification/Exact) determined at the solver entry boundary.

## Verification
The validity of the capillary model is contingent upon satisfying physical pressure-jump conditions (Young-Laplace balance) and maintaining stable spurious velocity levels in quiescent conditions.

For the qualification of this model in coupled flows, see:
- [VoF Solver Contracts](../verification/vof_solver_contracts.md)
