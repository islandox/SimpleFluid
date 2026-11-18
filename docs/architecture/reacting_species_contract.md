# Reacting Species and Thermodynamics Contract

The reacting framework relies on `SpeciesCatalog` and `ReactionMechanism` as immutable shared definitions. These definitions govern the configuration fingerprints that protect field layouts, source packets, and checkpoint integrity.

## Extensive Inventories and Source Packets

- **State Ownership:** `CellSpeciesState` stores extensive moles per cell across liquid, dissolved, resolved gas, and dispersed populations.
- **Inventory Basis:** Liquid and dissolved compartments share a common physical solution volume and caloric state.
- **Source Ledger:** `SpeciesSourceLedger` tracks accepted integrated events. All chemistry evaluations and physical transfers start from a synchronized accepted state; retrying an interval must not accumulate sources or corrupt history.
- **Balance Checks:** The framework rejects joint donor overdraw, inconsistent mass/elemental balances, and nonfinite budgets during reaction advancements.

## Thermodynamic Ownership

- **Provider Interface:** `MixtureThermodynamics` provides composition-dependent density, energy, enthalpy, and transport coefficients. The framework assumes EOS providers expose consistent fugacity or chemical-potential closures.
- **Energy Conventions:** All phases and source mechanisms must adhere to a single `EnergyConvention`. Formation energy defined in the caloric EOS must not be double-counted as reaction heat.
- **Temperature Ownership:** By default, a common temperature is assumed. Separate phase temperatures (e.g., liquid vs. gas) require explicit phase-energy ownership. Empty phases must possess zero volume, species, and energy, with no fictitious material properties.

## Solver Adapters and Adaptations

- **Field Adapters:** Adapters convert schema-indexed molar concentrations into extensive mass inventories. They must ensure that extensive phase transport is performed once and is consistent with the global continuity gate.
- **Source Integration:** The controller manages chemistry and source callbacks. All source stages must start from the same accepted state, ensuring that transported inventories are correctly mapped to thermodynamic states before reaction rates are evaluated.
- **Distribution:** Distributed field allocation, schema fingerprints, and MPI configuration metadata must be synchronized across all ranks before any source-stage callback invocation.

## Numerical Acceptance

- **Independence:** Physical conservation gates (mass, species, energy) are enforced independently of linear solver tolerances.
- **Verification:** Numerical verification must be performed against manufactured closures. The framework does not infer experimental physical validation from a successful numerical integration of a prescribed reaction mechanism.
