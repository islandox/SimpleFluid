# Conservative Low-Mach Reacting Flow

The `Reacting::LowMachMixtureSolver` framework provides a unified, conservative approach to solving low-Mach, multi-phase reacting flows. It enforces local volume continuity through a mechanical pressure correction while maintaining absolute thermodynamic pressure consistency across phases.

## Core Modeling Principles

- **Phase Ownership:** Each cell maintains independent species moles and internal energy for every declared physical phase. Liquid and dissolved compartments share a single solution phase, while gas phases are handled via EOS providers.
- **Density Definition:** Density is defined fundamentally as the sum of phase masses divided by the control volume ($\sum m_i / V_{cell}$), rather than Boussinesq approximations or void-fraction-based corrections.
- **Pressure-Volume Coupling:** The thermodynamic pressure is determined by total volume closure under specified energy and composition. The mechanical pressure correction enforces volume continuity using a variable-density face coefficient.

## Pressure Correction and Continuity

To ensure stability at small time steps in dense mixtures, the mechanical pressure correction accounts for the caloric volume response ($\chi = -dV/dp$ at fixed inventory). This removes explicit pressure-volume feedback instability.

- **Pressure Matrix:** The pressure system incorporates $\chi/dt$ to provide numerical stability.
- **Accuracy Gates:** The system uses normalized RHS true-residual checks. Pressure Krylov solvers are subject to absolute physical residuals rather than arbitrary RHS-relative thresholds, ensuring that roundoff in tiny fluxes does not drive unnecessary iterations.
- **Continuity Requirements:** Final flow acceptance is subject to nonlinear EOS occupancy and physical continuity residuals.

## Conservative Transport Consistency

Transport is performed componentwise to ensure mass and inventory conservation.

- **Upwind Transport:** Uses backward Euler and first-order upwind schemes, ensuring monotonicity and preventing negative species/count values in the presence of tiny physical arrivals.
- **Consistency Checks:** Before accepting transported values, the system performs a local extensive balance check. If a cell fails local inventory balance due to numerical roundoff or flux misalignment, a synchronized upwind sweep is performed to restore consistency within the allocated iteration budget.
- **Drift Handling:** For multi-phase flows with prescribed slip/drift, physical momentum ($P = M \cdot u + \sum m_k w_k$) is accounted for explicitly, removing drift impulses from the velocity equation and restoring them into the physical ledger to maintain conservation.

## Reacting-Flow Coupling

The solver manages chemistry and source terms through a structured `CoupledReactionContext`:

1.  **Sequence:** Transport $\rightarrow$ Chemistry/Source evaluation $\rightarrow$ Thermodynamic/Pressure Correction $\rightarrow$ Final Acceptance.
2.  **Source Stage:** The controller invokes reaction kernels with pre-transported composition and energy.
3.  **Source Consistency:** Source evaluation is fenced and decoupled from the linear solver to ensure determinism. Gross events (birth/death/transfer) are managed as atomic extensive packets to prevent rounding errors from eroding inventory budgets.

For detailed information on these components, see:
- **Chemical Species Ownership:** [reacting_species_contract.md](reacting_species_contract.md)
- **Thermodynamic Coupling:** [reacting_interphase_thermal.md](reacting_interphase_thermal.md)
- **External Host API:** [external_host_reacting_flow.md](external_host_reacting_flow.md)
