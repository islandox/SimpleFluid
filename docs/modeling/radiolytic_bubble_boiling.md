# Radiolytic Bubble, Boiling, and Feedback Models

This note documents the simplified source-driven gas models currently wired
into the Boussinesq solver. These models are engineering building blocks for
fissile-solution-tank studies; they are not a full Euler-Euler two-fluid
method.

## Model Choices

- `idealGasSource`: computes `S_alpha_rad` from fission power, temperature,
  pressure, gas yield, and release efficiency.
- `sheng2024TwoPopulation`: reconstructs void fraction from dissolved
  hydrogen plus microbubble and large-bubble inventories.
- Low-order scalar `alpha_g`: aggregates `S_alpha_rad`, `S_alpha_boil`, and
  optional collapse into `S_alpha_total`, then applies an explicit bounded
  update.
- Boiling source: adds explicit bulk and wall boiling alpha sources and a
  positive latent heat sink `latentHeatSink` in W/m3. Bulk boiling is capped
  by the sensible superheat available over one timestep. Bulk and wall
  sources are then limited together by remaining scalar void capacity, with
  the same factor applied to `latentHeatSink`.

The boiling-model update accepts the canonical scalar-void model and any
already-reserved radiolysis source in one call. This keeps `alpha_g`, both
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

## Configuration Keys

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

Precursor keys:

```text
precursor_group_count
precursor_decay_constants
precursor_initial_concentrations
precursor_source_terms
precursor_power_yields
precursor_effective_diffusivity
```

## Numerical Ordering

For each Boussinesq step:

1. Refresh material properties and temperature sources from the previous
   accepted state.
2. Advance velocity and pressure.
3. For ideal radiolysis, compute `S_alpha_rad` from the authoritative scalar
   `alpha_g` and its bound.
4. Compute boiling, limit it against timestep energy and remaining void
   capacity, then update the scalar void field.
5. Advance temperature with all heat sources minus the accepted
   `latentHeatSink`.
6. When selected, advance the two-population model and mirror its reconstructed
   void into the scalar publication field. The mirror derives `S_alpha_total`
   from its actual old/new published state.
7. Advance the conserved delayed-neutron precursor inventories
   `alpha_l * C_i` with the projected liquid face flux, apply zero-flux
   diffusion with coefficient `alpha_l * precursor_effective_diffusivity`,
   integrate constant source and decay exactly, and reconstruct `C_i` from the
   updated liquid fraction. Each group records globally reduced source, decay,
   boundary-outflow, transport-positivity, and balance diagnostics.
8. Refresh feedback material fields for output and the next step.

## Output Fields

Use `SolutionOutputOptions`:

- `include_sources`: writes temperature sources plus `S_alpha_rad`,
  `S_alpha_boil`, `S_alpha_total`, and `latentHeatSink`.
- `include_radiolytic_gas_fields`: writes `alpha_g`, `alpha_l`, and
  radiolytic diagnostics.
- `include_material_properties`: writes material fields plus `rhoFeedback`
  and `muFeedback` when feedback is configured.
- `include_precursor_fields`: writes `C_i` and `S_C_i` precursor fields.

## Feedback Mapping

The Phase 20 interface is an in-memory coupling scaffold, not a production
external-neutronics driver. `FeedbackMap::volume_weighted_average` maps owned
CFD cells into named feedback cells by volume-weighted averaging. Multiplying
the mapped average by the mapped cell volume preserves the original scalar
volume integral for coarsened cells. `FeedbackMap::import_power_density`
copies an externally supplied owned-cell `qdot_fission` vector into a cell
field after checking size, finiteness, and non-negativity.

`FeedbackFieldRegistry` references legacy or native scalar fields on one mesh
and provides named registration for `T_liquid`, `alpha_g`, `rhoFeedback`, and
one or more `C_i` fields. It exports deterministic `MappedFeedbackSnapshot`
objects with stable field and coarse-cell ordering. The callback-driven
`PlaceholderOuterCouplingDriver` imports an initial power vector, invokes a
thermal-hydraulics callback for configured subcycles, exports feedback, calls
a placeholder neutronics update, imports the returned power, and repeats.
Registry contents, mesh identity, options, and exchanged field sizes are
validated collectively for distributed runs.

The scaffold deliberately owns neither solver and defines no file/network
protocol. A production external-neutronics adapter, convergence policy, and
coupled physical validation remain future work.

## OpenFOAM Transport Verification

The paired [`dispersedBubbleFlow` verification cases](../../verification/openfoam/dispersedBubbleFlow/README.md)
compare production microbubble transport with an independent OpenFOAM
finite-volume reference. The steady case balances uniform bubble production
with top escape; the transient case tracks clearance of an initial bubble
inventory. Both use a prescribed constant rise speed and suppress population
conversion and dissolution to isolate the shared transport equations.
The checks cover matched cell profiles, inventories, and source/escape
conservation. They do not validate gas momentum, interphase forces, turbulence,
or the full radiolytic kinetics model.
Both cases require the optional IF97 library and use liquid-water reference
properties and surface tension at 300 K and 101325 Pa absolute. OpenFOAM
uses the same material snapshot. The gas remains radiolytic hydrogen; water
vapor density is not substituted for its ideal-gas/capillary bubble model.

## Limitations

- No separate gas momentum equation is solved.
- The scalar `alpha_g` path has no population balance.
- The Sheng 2024 model uses two representative bubble populations, not a full
  size distribution.
- Boiling with the Sheng 2024 model is rejected until vapor mass can be added
  conservatively to its bubble inventories.
- Finite scalar collapse with the Sheng 2024 model is likewise rejected until
  removal is coupled to bubble inventories and escape accounting.
- Boiling is explicit and does not subcycle; its accepted source is limited by
  timestep sensible energy, scalar alpha capacity, radiolysis reservation, and
  configured collapse.
- Wall boiling uses a prescribed heat flux and does not implement RPI heat
  flux partitioning.
- Viscosity feedback currently supports a constant model with a safety floor.
- Precursor transport uses the projected liquid face flux supplied by the
  Boussinesq step. It does not independently reconstruct a distinct phase
  velocity or model precursor exchange with a gas phase.
- The Phase 20 outer loop is an in-memory callback scaffold; it is not a
  production external-neutronics transport, protocol, or convergence driver.
- Full SILENE validation is deferred until point-kinetics or neutronics
  coupling is available.
