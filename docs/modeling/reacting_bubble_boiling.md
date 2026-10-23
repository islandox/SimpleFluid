# Reacting gas, boiling, and material feedback

The [general reacting-gas design](reacting_gas_sources.md) treats gas
production in a solution as application-supplied reactions with explicit
species, stoichiometry, finite reactants, and heat accounting. Chemical gas
production and liquid-to-vapor phase change have separate inventory and
energy ledgers, even when both contribute to the same occupied volume.

This historical documentation path remains valid for existing links. The
`Reacting*`, `PrescribedHeat*`, and H2-specific names below are current
implementation identifiers. General reaction providers and arbitrary species
remain planned; the existing energy-yield specialization cannot be made
generic through material callbacks alone. The current source-driven models
are coupled by `MultiphysicsSolverCoupler` to a shared Boussinesq solver.
They are not a full Euler-Euler two-fluid method.

## General coupling design

A reaction provider supplies integrated species changes and net reaction heat
under a declared volume and energy convention. It must distinguish gas
released from finite dissolved/reactant stocks from external species
injection. A boiling provider separately transfers solvent mass to vapor and
accounts for latent heat. Neither provider can reserve the same reactant,
sensible energy, or void capacity twice. Gas composition and thermodynamic
properties determine occupied volume; void fraction is a consequence of those
inventories on a conservative path.

All providers participate in the coupler's accepted-state transaction.
Preview combined reaction, phase-change, thermal, and volume effects from the
same accepted state. Accept only with consistent species, mass, energy, and
volume budgets; reject or recompute a smaller physical step on exhausted
reactants. A scalar void bound alone does not establish chemical conservation.
Supporting mixed reaction gas and vapor in a population therefore requires
composition-aware inventories and transfer laws before enabling the currently
rejected combinations.

Material feedback uses the accepted temperature, composition, gas occupancy,
and pressure with supplied validity-bounded properties. The design requires
no particular source of reaction energy or external application model. General
source/feedback adapters remain planned.

## Legacy implementation: model choices

- `idealGasSource`: computes `S_alpha_reaction` from the legacy power-density
  input, temperature, pressure, gas yield, and release efficiency.
- `sheng2024TwoPopulation`: reconstructs void fraction from dissolved
  hydrogen plus microbubble and large-bubble inventories.
- Low-order scalar `alpha_g`: aggregates `S_alpha_reaction`, `S_alpha_boil`, and
  optional collapse into `S_alpha_total`, then applies an explicit bounded
  update.
- Boiling source: adds explicit bulk and wall boiling alpha sources and a
  positive latent heat sink `latentHeatSink` in W/m3. Bulk boiling is capped
  by the sensible superheat available over one timestep. Bulk and wall
  sources are then limited together by remaining scalar void capacity, with
  the same factor applied to `latentHeatSink`.

The boiling-model update accepts the canonical scalar-void model and any
already-reserved legacy gas source in one call. This keeps `alpha_g`, both
bounds, and collapse behavior in one authoritative object. It publishes only
the admitted boiling source and its matching latent sink; the former unbounded
two-argument update is no longer a supported coupling path.

Collapse is limited to the gas removable without crossing `alpha_min` during
the current timestep:

```text
S_alpha_collapse = min(alpha_g / tau_c,
                       (alpha_g - alpha_min) / dt).
```

Boiling can reuse that realizable capacity, but cannot claim collapse that the
bounded scalar update would otherwise discard.

## Legacy implementation: configuration keys

`reacting_transport_solver_tolerance` optionally sets the relative linear
solve tolerance for the existing gas finite-volume inventory transport (default
`1e-10`). The typed option is `ReactingGasOptions::transport_solver_tolerance`.
Graded meshes with nearly zero early escape may require tighter algebraic
accuracy to satisfy the unchanged roundoff-level inventory checks; the
bottom-heated boundary-layer fixture uses `1e-14`. This is separate from
`local_ode_tolerance`, which controls local kinetics/radius calculations.

Boiling keys:

```text
enable_bulk_boiling
enable_wall_boiling
saturation_temperature
boiling_activation_delta_t
boiling_time_scale
latent_heat
gas_density
wall_evaporation_fraction
wall_heat_flux
wall_boiling_patches
```

Scalar void keys:

```text
alpha_min
alpha_max
initial_alpha_g
alpha_collapse_time
alpha_diffusivity
```

The low-order scalar void model does not yet have an advective transport
path. Direct `ScalarVoidFractionOptions` therefore reject a nonzero reserved
`constant_slip_velocity`, while the scalar model's flat database parser does
not consume that key. In a shared flat database, `constant_slip_velocity`
belongs to the bubble-population model's operational rise-velocity
configuration.

Material feedback keys:

```text
density_feedback_model = constant | boussinesqTemperatureOnly | boussinesqVoid | mixture
viscosity_feedback_model = constant
min_density
min_viscosity
gas_density
```

## Legacy implementation: numerical ordering

For each Boussinesq step:

1. Refresh material properties and temperature sources from the previous
   accepted state.
2. Advance velocity and pressure.
3. For the ideal gas-source specialization, compute `S_alpha_reaction` from the
   authoritative scalar `alpha_g` and its bound.
4. Compute boiling, limit it against timestep energy and remaining void
   capacity, then update the scalar void field.
5. Advance temperature with all heat sources minus the accepted
   `latentHeatSink`.
6. When selected, advance the two-population model and mirror its reconstructed
   void into the scalar publication field. The mirror derives `S_alpha_total`
   from its actual old/new published state.
7. Refresh feedback material fields for output and the next step.

Other independently enabled legacy physics stages are outside this gas and
boiling contract; this ordering is not a generic reaction-network API.

## Legacy implementation: output fields

Use `SolutionOutputOptions`:

- `include_sources`: writes temperature sources plus `S_alpha_reaction`,
  `S_alpha_boil`, `S_alpha_total`, and `latentHeatSink`.
- `include_reacting_gas_fields`: writes `alpha_g`, `alpha_l`, and
  diagnostics from the existing H2 specialization.
- `include_material_properties`: writes material fields plus `rhoFeedback`
  and `muFeedback` when feedback is configured.

## Legacy implementation: feedback mapping

`FeedbackMap::volume_weighted_average` maps owned CFD cells into named
feedback cells. Multiplying a mapped average by its mapped volume preserves
the original scalar volume integral for coarsened cells.
`FeedbackFieldRegistry` exports deterministic snapshots with stable field and
coarse-cell ordering. Registry contents, mesh identity, options, and exchanged
sizes are validated collectively for distributed runs.

The existing `FeedbackMap::import_power_density` and
`PlaceholderOuterCouplingDriver` remain a specialized in-memory power/feedback
scaffold; the imported field retains the name `qdot_external`. They are not a
general species-source adapter and do not establish reaction/heat convergence
or an external communication protocol.

## Legacy implementation: OpenFOAM transport verification

The paired [`dispersedBubbleFlow` verification cases](../../verification/openfoam/dispersedBubbleFlow/README.md)
compare production microbubble transport with an independent OpenFOAM
finite-volume reference. The steady case balances uniform bubble production
with top escape; the transient case tracks clearance of an initial bubble
inventory. Both use a prescribed constant rise speed and suppress population
conversion and dissolution to isolate the shared transport equations.
The checks cover matched cell profiles, inventories, and source/escape
conservation. They do not validate gas momentum, interphase forces, turbulence,
or the full legacy kinetics model.
Both cases require the optional IF97 library and use liquid-water reference
properties and surface tension at 300 K and 101325 Pa absolute. OpenFOAM
uses the same material snapshot. The fixture gas remains hydrogen; water
vapor density is not substituted for its ideal-gas/capillary bubble model.

## Legacy implementation: limitations

- No separate gas momentum equation is solved.
- The scalar `alpha_g` path has no population balance.
- The Sheng 2024 model uses two representative bubble populations, not a full
  size distribution.
- Boiling with the Sheng 2024 model is rejected until vapor mass can be added
  conservatively to its bubble inventories.
- Finite scalar collapse with the Sheng 2024 model is likewise rejected until
  removal is coupled to bubble inventories and escape accounting.
- Boiling is explicit and does not subcycle; its accepted source is limited by
  timestep sensible energy, scalar alpha capacity, gas-source reservation, and
  configured collapse.
- Wall boiling uses a prescribed heat flux and does not implement RPI heat
  flux partitioning.
- Viscosity feedback currently supports a constant model with a safety floor.
- General reactions, multiple generated gas species, composition feedback,
  and conservative reaction-gas/vapor coupling require implementation and
  separate qualification.
- The existing outer loop is an in-memory callback scaffold and does not
  establish an application-specific physical validation claim.
