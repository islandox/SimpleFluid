# Reacting gas, finite vapor and thermal ownership

Chemical gas production and solvent/vapor phase change have distinct material
and energy events. Neither heat deposition nor a scalar void increment defines
a reaction. The generic species API requires an explicit catalog, conservative
mechanism and supplied material laws; see
[the reacting model guide](../reacting-gas-models.md) and
[the source contract](reacting_gas_sources.md).

## Select one thermal owner

| Configuration | Thermal and material owner |
| --- | --- |
| Standalone scalar void/boiling | Low-order bounded scalar void and admitted latent sink; no general chemical inventory |
| `ReactingGasOptions.bouss_reference` plus `bouss_sources` | Finite schema species/vapor with fixed reference-volume and scalar `rho_ref*cp*T` storage |
| Native `LowMachMixtureSolver` plus configured interphase stages | Actual phase internal energy, supplied mixture caloric EOS and solver-owned pressure work |

Do not attach an independent `BoilingSourceModel` to a schema-owned reacting
inventory. The coordinated finite Boussinesq adapter supplies the positive
gas-plus-boiling path without that second material/latent owner. Its
approximation is separate from native phase-energy recovery; neither one
qualifies the other.

## Finite Boussinesq reference binding

`BoussReferenceVolume` explicitly declares identity, revision, catalog
fingerprint, liquid reference density, thermal reference density, constant heat
capacity and catalog-ordered gas molar volumes. These additive gas volumes are
fixed reference inputs, not an EOS or an inferred solvent. The field model
owns actual Liquid/Dissolved and named gas inventories even though its thermal
equation remains a scalar approximation.

`BoussCoordinatedSources` contains separately selected gas requests, solvent/
vapor binding, volume-only population processes and explicit population-number
events. `configure_bouss_sources(...)` replaces these sublaws without resetting
evolved inventories or void. Initialization and later configuration must keep
the same material ownership and immutable schema.

The material sequence is:

1. Transport actual species and paired population counts.
2. Execute the actual reaction/external-source input. An infeasible actual
   source rejects; it is not rescaled to fit void capacity.
3. Reserve the explicitly requested forward gas-producing reaction extents
   against remaining finite original donors and admit them in declared order.
4. Admit bulk/wall solvent-vapor proposals using the remaining donor, thermal
   and scalar gas-volume capacity. Incoming credit cannot finance competing
   requests in the same reservation.
5. Apply explicit number events consistent with the accepted material endpoint.
   No automatic birth, death or disappearance is inferred.

`BoussVaporBinding` names one real species present as liquid solvent and vapor,
its vapor compartment/population, saturation temperature, latent heat and time
scale. It may also declare a signed scalar-storage sensible correction.
Evaporation transfers the same material amount from Liquid to vapor and
condensation reverses it. Available superheat/subcooling and finite stocks
bound the admitted event. A scalar cap cannot delete material or supply a
missing donor.

The net evaporation cooling is
`dm*(latentHeat-sensibleCorrection)` and must be positive. The accepted latent
and sensible channels enter `rho_ref*cp*T` once; condensation reverses their
sign. The complete material/energy receipt, not the requested amount, drives
publication. Do not add mixture latent enthalpy, carried sensible traces or
pressure work again.

## Wall heat admission

Each `BoussWallHeatBinding` names a physical patch, nonnegative inward heat
flux and evaporation fraction. The coupling owner captures actual canonical
owned-face heat, area, interval and geometry epoch before admission. The
declared flux must match the temperature equation's immutable boundary input.

The full boundary heat is already supplied through the temperature boundary
condition. It is never added again as a volumetric source. Only the admitted
latent/sensible conversion changes the scalar thermal budget. Bulk and wall
proposals share finite solvent and the declared proportional-request admission
policy; rejected wall demand leaves unspent heat in the same scalar owner.
Raw receipts retain patch and geometry-face provenance.

The initial wall path requires closed zero-carrier-flux walls and known
prescribed heat or compatible Neumann input with valid conductivity. Unknown
Dirichlet, Robin or convective heat cannot be guessed before the solve.
Boundary data/conductivity, source configuration or geometry changes invalidate
the retained heat snapshot. Local failures are fenced before halo exchange or
global reductions, including on empty owners.

## Boussinesq support and rejection limits

The coordinated adapter requires fixed-grid physical sensible-temperature
storage with its declared constant `cp` and thermal density. It uses zero
surface tension, zero/prescribed slip and closed material boundaries. Its
reference-volume population callback deliberately contains no EOS/fugacity
point.

It rejects supplied phase EOS alongside this binding, EOS film laws, the
general EOS population callback, variable-`cp`/external material-updater storage,
independent boiling, `coupledNonlinear`, free surface/ALE and a native phase-U
thermal participant. Use the distinct supported path rather than disabling
one owner silently. Scalar mirror bounds must match; an unrelated scalar
collapse cannot remove schema-owned gas.

The separate `PrescribedCarrierAfterTemperature` reference schedule rejects
both finite Boussinesq bindings. That schedule is gas-only with individually
zero applied chemical/external/transfer heat; it is not an alternative route
for latent cooling.

Material feedback can use accepted temperature and gas void under its selected
Boussinesq or mixture approximation. It supplies neither an independent gas
momentum equation nor drag, lift, virtual mass or turbulent dispersion.
Gas escape enthalpy/momentum in the weak field adapter remain carried traces,
with no implied conservative phase-energy/momentum debit.

## Native phase-energy interphase transfer

The native low-Mach controller consumes actual transported inventory, chemistry
and ordered face stages before freezing configured interphase event amounts.
Each material stage reserves gross outgoing species from its actual predecessor;
later stages may use earlier admitted credit, but simultaneous incoming credit
does not fund a donor. Number events run after final material changes.

Supplied true partial-molar enthalpies and their temperature/pressure
derivatives determine paired material energy at each pressure-conditioned
anchor. Signed local heat slots provide separate paired heat exchange.
Common-temperature mode records internal caloric redistribution; separate
temperatures retain each occupied phase's own energy. An absent phase has no
invented temperature or heat capacity.

The solver alone applies `-p*(Vfinal-Vadv)`, using the actual transported
physical phase-volume packet. Kinetics, chemistry and count events are not
reintegrated during pressure roots. Frozen plans, active constitutive values
and final thermal evaluations have separate receipts, and commit/restore
invoke none of those laws. See
[the native thermal contract](../architecture/reacting_interphase_thermal.md).

This backend uses closed orthogonal meshes, common phase pressure and zero
surface tension. Prescribed native planar ALE retains accepted-old/new storage,
relative phase flux and explicit live motion authority. It is not the
Boussinesq free-surface/vapor adapter. Upstream complete depletion without owned
scalar source-decay work, pressure-conditioned transfer across faces, evolving slip
momentum, capillary energy and general interface dynamics remain outside its
support boundary.

## Standalone scalar controls

`ScalarVoidFractionModel` and `BoilingSourceModel` remain available for their
documented low-order approximation without a general reacting inventory.
They share admitted void capacity and latent sink; requested boiling and
discarded scalar increments are not accepted material production.
Bulk boiling uses available sensible superheat. Wall boiling uses prescribed
heat flux, not RPI heat-flux partitioning. The scalar model has no advective
population balance and rejects nonzero reserved scalar slip.

The established boiling controls include `enable_bulk_boiling`,
`enable_wall_boiling`, `saturation_temperature`, `boiling_activation_delta_t`,
`boiling_time_scale`, `latent_heat`, `gas_density`, `wall_evaporation_fraction`,
`wall_heat_flux` and `wall_boiling_patches`. Scalar controls include `alpha_min`,
`alpha_max`, `initial_alpha_g`, `alpha_collapse_time` and `alpha_diffusivity`.
They do not create a catalog or configure the finite adapter automatically.

Generic reacting setup is C++ provider-based. A flat database cannot construct
a mechanism, mixture EOS or finite Boussinesq binding; enabled generic parsing
therefore rejects without explicit setup. The retired fixed-H2/yield/model
keys and modes have no compatibility aliases.

## Output, replay and evidence

Schema species fields and keyed chemical/external/interphase/physical-escape
ledgers carry material provenance. `BoussAcceptedReceipt` retains requested and
admitted gas extents, vapor/number events and original thermal channels.
Additional `bouss_evaporation_moles_rate`, `bouss_condensation_moles_rate`,
`bouss_latent_power`, `bouss_sensible_power` and `bouss_vapor_volume_source`
fields report the reference adapter. They do not replace accepted receipts or
define native phase energy.

Coupled snapshots and rejected steps retain flow, temperature, fields,
inventories, source/heat receipts, time and histories atomically. A failed
admission does not permit a partial latent debit or silent reinitialization.
Use fresh focused serial/MPI output, actual owner counts and skips to establish
execution evidence. Numerical source/energy/replay fixtures are not physical
validation of boiling, nucleation or interfacial constitutive laws.

Explicit H2/IF97/OpenFOAM references may retain those physical names and supplied
correlations. Their documented external-carrier approximation does not imply
finite-water inlet conservation or general latent ownership. VoF remains
excluded, with no experimental-enable path provided in this source snapshot;
bounded-FP and exact-rational geometry paths remain deferred.
