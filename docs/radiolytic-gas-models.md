# Radiolytic Gas Models

SimpleFluid provides one runtime-selectable radiolysis subsystem with three
modes:

- `disabled`
- `idealGasSource`
- `sheng2024TwoPopulation`

The subsystem is owned by `BoussinesqSolver` only after
`configure_radiolytic_gas(...)` is called. An unconfigured solver retains the
pre-Phase-12 execution path.

## Public units

All public options and fields use SI:

| Quantity | Unit |
| --- | --- |
| Temperature | K |
| Absolute pressure | Pa |
| Dissolved concentration | mol/m3 liquid |
| Bulk molar inventory | mol/m3 bulk |
| Radiolytic yield | mol/J |
| Radius | m |
| Time | s |
| Dynamic viscosity | Pa s |
| Density | kg/m3 |
| Henry coefficient, `C = H p` | mol/(m3 Pa) |

Published empirical conversions are isolated in
`RadiolyticGasProperties.hh`. Winter et al. (2020) Eq. (32) explicitly
defines `C_U` in mol/m3 and temperature in K, so
`mean_fission_fragment_let` uses the public concentration directly. The
surface-tension fit also uses `C_U` directly in mol/m3 and converts the
public temperature from K to degrees Celsius at the call site.

There is a source discrepancy in that fit. Sheng et al. (2024) Table 2
prints a positive `T_C^2` coefficient, while Winter et al. (2020) Eq. (24)
and Winter et al. (2022) Eq. (13) print it as negative. The runtime
`sheng2024` selector follows the governing Sheng table verbatim. The choice
is explicit rather than inferred from coefficient magnitudes.

## Property provenance

| Implementation | Primary location | Convention or guard |
| --- | --- | --- |
| Mean fission-fragment LET | Winter 2020, Eq. (32), PDF p. 12 | `C_U` in mol/m3, `T` in K |
| Nucleation yield correction | Winter 2020, Eq. (30), PDF p. 10 | `0.5 < G_H2 < 4.5` molecules/100 eV |
| Sheng surface tension | Sheng 2024, Table 2, PDF p. 8 | `C_U` in mol/m3, `T_C` in C |
| Hydrogen diffusivity | Winter 2022, Eq. (45), PDF p. 8 | input K, output m2/s |
| Bubble rise velocity | Winter 2022, Eqs. (15a-d), PDF p. 5 | SI implementation of the Celata relation |

## Phase 12

The ideal-gas source is

```text
S_alpha_rad =
    (1 - alpha_g) eta G qdot_fission R T / p_abs.
```

The source is zero at zero fission power and when `alpha_g >= alpha_max`. It
is limited by both `max_source_alpha_rate` and
`(alpha_max - alpha_g) / dt`. In the Boussinesq solver, the Phase 14 scalar
model owns the authoritative `alpha_g` field and bound; ideal radiolysis reads
that state and mirrors it only for its model-local diagnostics.

When no scalar-void model was explicitly configured, ideal-radiolysis bounds
seed the solver's implicit scalar model before the first step. Explicit
`ScalarVoidFractionOptions` take precedence. Reconfiguring radiolysis after
time advancement never resets the evolved scalar state or its bounds.
Standalone ideal-mode calls likewise must use the `advance` overload that
supplies authoritative `alpha_g` and `alpha_max`; the shorter overload is
reserved for disabled and two-population modes.

## Phase 14.1 state

The advanced model stores dissolved hydrogen conservatively as

```text
I_H2 = alpha_l C_H2.
```

It transports and reacts the following bulk quantities:

- `I_H2`
- `N_micro`, `M_micro`
- `N_large`, `M_large`

`C_H2` is reconstructed from `I_H2 / alpha_l`. Empty bubble populations have
zero radius and zero void diagnostics; population floors are never used as
divisors.

In this two-population mode, the radiolysis inventories own the reconstructed
void state and the scalar model is only a common publication mirror. Their
configured alpha bounds must match so mirroring cannot clip conserved bubble
state.

The model implements the following Sheng et al. relationships:

- Henry/Laplace equilibrium and critical concentration, Eqs. (9)-(10)
- pressure-corrected nucleation radius, Eqs. (11)-(15)
- two-population production, conversion, and dissolution, Eqs. (23)-(30)
- bubble equation of state and void reconstruction, Eqs. (33)-(35)
- experimental local inertial-pressure update based on Eq. (32)

Bubble radii are positive roots of

```text
4 pi r^3 / 3 (p_l + 2 sigma / r) = zeta R T.
```

The solver uses bracketed bisection over the configured radius interval and
reports failed brackets or iterations.

## Numerical ordering

For each configured two-population solver step:

1. Refresh material properties and fission power.
2. Solve velocity and pressure correction.
3. Advance temperature.
4. Reconstruct or prescribe absolute thermodynamic pressure.
5. Transport dissolved and bubble inventories.
6. Integrate local production, conversion, growth, and dissolution.
7. Reconstruct radii, concentrations, and void fraction.
8. Evaluate experimental inertial pressure for the following timestep.

Transport/local-kinetics splitting is first order. Local linear decays are
analytic and the remaining rates use bounded subcycles. `maximum_subcycles`,
clipping, pressure-floor events, and radius failures are exposed through
`RadiolyticGasStepStatistics` and reduced globally across MPI ranks.

In ideal mode, `S_alpha_rad` is the source applied to the low-order scalar
model. In two-population mode it is the model-owned net bounded-void rate,
`(alpha_g_new - alpha_g_previous) / dt`; it includes reconstruction, transport,
and escape and can therefore be negative. The common scalar mirror derives
`S_alpha_total` independently from its own published old/new state rather than
copying this internal history.

The no-advection dissolved mode is the default paper-compatibility mode.
Optional CFD advection uses the projected liquid face flux. Bubble number and
moles use the same category face flux. `general` bubble transport retains the
full CFD liquid flux. `axial` reconstructs only the liquid z-velocity at each
face for paper-style one-dimensional compatibility. Rise velocity is added
in the positive z direction in both modes. Configured free-surface patches
are outflow-only for hydrogen and bubble-count escape.

## Bubble rise velocity

`celata2007` implements Winter et al. (2022) Eqs. (15a-d):

```text
v_b = sqrt(8 r_b g / (3 C_D))
C_D = max(
    24 (1 + 0.15 Re^0.687) / Re,
    8 Eo / (3 (Eo + 4)))
Eo = 4 g r_b^2 (rho_l - rho_g) / sigma
Re = 2 rho_l v_b r_b / mu_l
```

The coupled positive root is found by bracketed bisection. Gas density enters
`Eo`, exactly as printed in Winter Eq. (15c); it is not inserted into
Eq. (15a). Celata et al. (2007) Table 2 reports an experimental envelope of
diameter 0.5-4 mm, `Re` 200-1500, and `Eo` 0.1-3.5. The pure property result
reports whether a point lies inside that envelope, but the runtime model
allows explicit extrapolation for the much smaller radiolytic populations.

## Pressure

Pressure modes are:

- `constant`: uniform `reference_pressure`
- `prescribedHistory`: piecewise-linear absolute-pressure history
- `reconstructed`:
  `p_ref + p_gauge - volume_mean(p_gauge)`, where the solver gauge pressure is
  stored in Pa
- `inertial`: retained absolute-pressure field updated weakly after kinetics

The incompressible pressure-correction field is never interpreted directly as
absolute pressure. The inertial option does not change the projection
equation. Pressure-floor events are reported in step statistics.

## Conservation

The per-step hydrogen balance is

```text
inventory_error =
    H2_after + H2_escaped - H2_before - H2_produced.
```

The inventory is the domain integral of
`I_H2 + M_micro + M_large`. Microbubble dissolution and category conversion
transfer moles between these fields without deleting them.

Free-surface escape is evaluated from the accepted unknown of the implicit
upwind transport solve. `H2_escape_molar_rate` and
`bubble_escape_number_rate` are cell-volumetric rates localized to the
free-surface owner cells, and their global integrals times `dt` reproduce the
per-step escaped inventories.

When the raw reconstructed void exceeds `alpha_max`, population inventories
remain unchanged. `alpha_g_raw`, `alpha_g_excess`, and bounded `alpha_g`
publish the distinction. A concentration publication ceiling similarly does
not modify `I_H2`; excluded dissolved inventory is reported separately.

## Configuration

Options are parsed from flat snake-case `Database` keys. The principal keys
are:

```text
enable_radiolysis
radiolytic_bubble_model
radiolytic_pressure_mode
dissolved_hydrogen_transport_mode
bubble_transport_mode
radiolytic_heaviside_mode
bubble_rise_velocity_model
surface_tension_model
hydrogen_diffusivity_model
hydrogen_yield_mol_per_j
gas_release_efficiency
reference_pressure
gas_constant
alpha_min
alpha_max
max_source_alpha_rate
henry_coefficient
surface_tension
hydrogen_diffusivity
atmospheric_pressure
uranium_concentration_mol_per_m3
hydrogen_yield_molecules_per_100_ev
microbubble_lifetime
large_bubble_dissolution_time
micro_to_large_conversion_coefficient
smooth_heaviside_width
constant_slip_velocity
bubble_gas_density
bubble_gravity
rise_velocity_tolerance
max_rise_velocity_iterations
initial_dissolved_hydrogen
initial_micro_number_density
initial_micro_moles
initial_large_number_density
initial_large_moles
min_radius
max_radius
min_population
max_population
max_concentration
local_ode_tolerance
max_radiolytic_subcycles
max_radius_iterations
pressure_history_times
pressure_history_values
radiolytic_free_surface_patches
```

Enabled modes require an explicit positive molar yield and an explicit finite
source-rate limit. Advanced mode also requires Henry, property, nucleation,
radius, population, and concentration inputs.

Advanced mode reconstructs void from conserved bubble inventories. It is
therefore rejected with active boiling or finite scalar collapse until those
mechanisms can add or remove the corresponding inventories conservatively.

## Output

`SolutionOutputOptions::include_sources` writes `S_alpha_rad`.

`SolutionOutputOptions::include_radiolytic_gas_fields` writes `alpha_g`,
`alpha_l`, absolute pressure, all advanced state fields, reconstructed
properties, and diagnostic rates.

## Limitations

- Chemistry is H2-only and intended for short pulse studies.
- The two populations are representative radii, not a size distribution.
- `F = 1e-4 Pa^-1 s^-1` is a SILENE-calibrated default, not a universal
  coefficient.
- The inertial pressure update is experimental and weakly coupled.
- Full SILENE validation requires point kinetics or neutronics coupling.
- Celata validity diagnostics use the experimental envelope in the primary
  paper; radiolytic microbubble use is an explicit extrapolation.
- The positive quadratic surface-tension coefficient follows Sheng Table 2
  despite the negative coefficient printed in both cited Winter papers.

## References

- H. Sheng et al., Annals of Nuclear Energy 206 (2024) 110668,
  <https://doi.org/10.1016/j.anucene.2024.110668>.
- F. Winter et al., Annals of Nuclear Energy (2020) 107379,
  <https://doi.org/10.1016/j.anucene.2020.107379>.
- F. Winter et al., Annals of Nuclear Energy (2022) 108614,
  <https://doi.org/10.1016/j.anucene.2021.108614>.
- G. P. Celata et al., Experimental Thermal and Fluid Science (2007),
  <https://doi.org/10.1016/j.expthermflusci.2006.06.006>.
