# Reacting species and thermodynamics contract

The reacting implementation uses `Reacting::SpeciesCatalog` and
`ReactionMechanism` as immutable shared definitions. A species has a stable ID,
molar mass, elemental composition, and allowed phases. Ordered schema fingerprints
protect field layouts, source packets, checkpoints, and MPI configuration;
matching vector lengths or revision numbers alone are insufficient.

## Inventories and accepted source packets

`CellSpeciesState` stores extensive moles per cell in liquid, dissolved, resolved
gas, and named dispersed populations. Liquid and dissolved compartments share one
physical solution volume and caloric state. A population owns a separate bubble
count; its composition is the full species vector. No name or catalog index is
reserved for hydrogen, an inert gas, a donor, or a solvent.

Supplied chemistry evaluates rates with an explicit volume basis or supplies
integrated reaction extents. `advance_reactions` uses the same finite inventories
for every reaction and transfer. It rejects joint donor overdraw, invalid schemas,
nonfinite budgets, and inconsistent elemental or mass balances. It does not
provide a stiff chemistry mechanism. External material packets declare their
reservoir, species amounts, energy, and momentum. Independent signed external
heat remains separate from reaction and transported material energy.

`SpeciesSourceLedger` records the accepted integrated events. Preview/corrector
work starts from the same accepted state; retrying an interval cannot accumulate
sources or publish history. A caller commits fields and model ledgers together.

## Thermodynamic ownership

`MixtureThermodynamics` supplies composition-dependent density, internal energy,
enthalpy, thermodynamic derivatives, and transport coefficients in SI units.
Gas providers expose species fugacities or a chemical-potential closure. Built-in
ideal-mixture and supplied linear-liquid helpers share the same provider interface;
no nonideal correlation is inferred from a species name.

Every participating phase and source mechanism uses one `EnergyConvention`.
Formation energy in the caloric EOS must not also enter as reaction heat.
Sensible-energy storage uses the explicitly supplied reaction heat. Interphase
material carries its declared enthalpy once; additional conductive exchange is
paired between phases. Pressure-volume work belongs to the flow transaction.

Common temperature is the default. Separate liquid, resolved-gas, and population
temperatures require explicit phase-energy ownership, including cross-phase
chemistry allocation. A disappearing phase must have zero species, number,
volume, and internal energy. An empty phase has no fictitious density, heat
capacity, or temperature. Only explicitly bounded operand cancellation may clear
an empty-phase energy residual; physical heat cannot repair a failed budget.

## Field and solver adapters

`ReactingGasModel` stores schema-indexed molar concentrations per current bulk
volume. Its intensive-to-extensive conversion applies the control volume once.
Its model-owned liquid mass and pure-liquid density derive from all solution
species and their EOS. `LiquidMassInventory::initializeFromSpecies` binds a
cellwise reporting follower to these fields. `previewSpeciesMass` consumes the
already transported mass and independent source/outflow budgets; it cannot
transport that inventory a second time. Its trial density stays private until
commit, and ALE previews use accepted-old and trial-new volumes.

The Boussinesq controller consumes the accepted species thermal source once and
refreshes derived thermodynamic properties after temperature advancement without
re-evaluating chemistry. This backend retains the sensible-temperature and
Boussinesq approximations. Formation-energy storage requires the conservative
low-Mach backend. Typed reacting free-surface integration consumes the exact accepted event and
liquid transport receipts; see [the surface contract](reacting_species_surface.md).
The field adapter's general bulk-mixture diffusion remains gated. Native ordered
face transport and local interphase thermal stages have their own conservative
owners and support limits; component kernels do not enable every combination.

Experimental VoF solvers can be built explicitly for source-free development.
The reacting VoF coupler remains excluded until it migrates to the shared
species/source configuration boundary. Source-free finite geometry/flow tests
do not qualify multispecies reacting-VoF execution. Shared geometric and
finite-volume utilities remain available independently.

## Distributed and numerical acceptance

Configuration presence, schema, population layout, and collective control-flow
metadata agree before distributed field allocation. Ranks without owned cells
participate in the same validation and collective sequence. Rank-local provider
failure rejects the complete transaction. Accepted physical inventories,
geometry, time, and diagnostic histories survive failed attempts unchanged.

Mass, species, energy, and occupancy gates use physical quantities independently
of linear solver tolerances. Tiny occupied phases do not inherit unit-sized
absolute tolerance floors. Numerical verification exercises supplied manufactured
closures; it does not establish experimental validation of a reacting fluid.
