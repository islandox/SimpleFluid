# Hydrogen and Specific Species Configuration

While the core reacting flow engine is species-agnostic, specific species like Hydrogen require dedicated configuration within the `SpeciesCatalog` and `ReactionMechanism` providers. This document outlines the best practices for defining such species.

## Catalog Specification
Species properties are defined as immutable shared definitions. For Hydrogen (and similar reacting species), the following must be provided:
- **Molar Mass:** Precise atomic/molecular weight in SI units (kg/mol).
- **Elemental Composition:** Correct count of atoms per molecule for balance checking during reaction advancements.
- **Phases:** Definition of allowed phases (e.g., gas, dissolved, etc.).

## Equation of State (EOS) and Transport
Specific species properties influence the mixture EOS and transport coefficients:
- **Gas Providers:** For Hydrogen, define the mixture EOS parameters correctly (e.g., specific heat ratio, compressibility).
- **Transport Coefficients:** Define binary diffusion coefficients and viscosity contributions within the `SpeciesCatalog`.

## Implementation Example
When adding a new species configuration, ensure that:
1. **Consistency:** All elemental balances (C, H, O, etc.) are verified against the reaction mechanism.
2. **Phase Definitions:** If the species is intended to exist in multiple phases (e.g., gaseous vs. dissolved), ensure the phase mapping is explicitly defined in the catalog.
3. **No Reserved Names:** The engine does not reserve names. Any species identifier is strictly enforced through the schema fingerprint of the `SpeciesCatalog`.
