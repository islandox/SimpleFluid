# Liquid-Side Bubble Mass Transfer Contract

The bubble mass transfer framework provides constitutive closures for species exchange between dispersed gas populations and the carrier liquid phase.

## Modeling Framework

- **Constitutive Selectors:** The system supports multiple Sherwood-number ($Sh$) correlations (e.g., `Hughmark`, `SphericalDiffusion`, `FengMichaelides`). Selectors define the mass-transfer coefficient $k_L$ based on local bubble diameter, relative slip, and material properties.
- **Thermodynamic Basis:** Equilibrium concentration is determined by the supplied fugacity and Henry’s law coefficients ($C_{eq} = H \cdot f$). Providers are responsible for maintaining thermodynamic consistency; the framework treats this as an authoritative input.
- **Evaluation Domains:** Selectors are validated against specific physical domains (e.g., $Re$, $Sc$, $Pe$ ranges). Out-of-bounds evaluation triggers explicit domain errors rather than silent extrapolation.

## Architectural Interface

- **Transport Coupling:** Film-stage mass transfer operates in conjunction with population transport. Slip velocities may be prescribed or dynamically calculated (e.g., Celata correlation) based on local carrier conditions.
- **Inventory Management:** The framework manages mass-event transfers between dissolved bins and dispersed populations. Events are subject to finite-donor limits to ensure species conservation without clipping or uncontrolled renormalization.
- **Atomicity:** Mass transfer events are processed as part of the coupled stage. The evaluation maintains a consistent internal state, ensuring that species receipts are correctly registered in the system's ledger.

## Integration Principles

- **Units:** All inputs and outputs must adhere to SI units.
- **Decoupling:** The mass-transfer evaluation is independent of bulk phase-energy solvers or interphase thermal heat-transfer closures. It focuses solely on the kinetic species exchange across the interfacial film.
- **Validation:** Constitutive laws must demonstrate consistency with established steady-state radial diffusion limits ($Sh=2$ for $Pe=0$) and respect their defined operational gate boundaries.
