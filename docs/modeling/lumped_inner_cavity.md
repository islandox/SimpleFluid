# Lumped inner cavity above a liquid pool

For a vented cavity whose volume changes with a moving pool, use the additive
`VentedLumpedCavityModel` in `equations/VentedLumpedCavityModel.hh`. It advances
perfectly mixed dry air at supplied ambient pressure. Its pure
`preview(dt, new_volume, surfaces)` returns sensible surface transfers, signed
vent mass/enthalpy and $p(V_\mathrm{new} - V_\mathrm{old})$ work. Inflow carries ambient
enthalpy; outflow carries candidate cavity enthalpy. It uses $h = c_p T$,
$H = m c_p T$ and $U = H - pV$, with an implicit enthalpy balance. Water exchange
is rejected and radiation is not part of this sensible-only model.

`solvers/ThermalStepParticipant.hh` supplies the typed transaction interface for
coupling external solid/cavity inventories to `BoussinesqSolver` planar ALE.
Register the participant before configuring `HeadspaceTemperatureMode::External`
when it owns cavity temperature. Every preview starts from the same substep
baseline; normalized thermal residuals must be at most one before collective
preflight and a nonthrowing commit. Step and interval rollback include immutable
boundary-flux and participant snapshots. The existing closed moist-air model
below retains its fixed-volume contract.

`LumpedInnerCavityModel` (`equations/LumpedInnerCavityModel.hh`, linked through
`SimpleFluid::Equations`) represents a closed, fixed gas volume above a liquid
free surface and inside vessel walls. Air has one temperature and one humidity
ratio. An extensible list supplies each surface's name, area, temperature,
convection closure, moisture behavior, and emissivity. Surface temperatures are
prescribed over each step; the model returns transfers for an external liquid
and wall solve. The field-batch adapter samples coupled subdomains and refreshes
physical heat-flux boundaries on `TemperatureDiffusionEquation` and
`SolidHeatConductionEquation`. The caller controls coupling iterations, liquid
inventories, and acceptance; there is no automatic `BoussinesqSolver` driver.

## State, properties, and conservation

All inputs and outputs use SI units. Humidity ratio is $w = m_v/m_d$, kg of
vapor per kg of dry air, and differs from relative humidity. The dry-air mass
$m_d$ and gas volume $V$ remain fixed. With ideal gases,

$$
p_v=m_vR_vT/V,\quad p=(m_dR_d+m_vR_v)T/V,\quad RH=p_v/p_{sat}(T).
$$

Pressure evolves with temperature and evaporation; this is not an isobaric,
vented headspace. The caller supplies a pure saturation-pressure function, so
water-property ownership remains outside the cavity model. For example,
`SimpleFluid::IF97Water::saturation_pressure` can be used when linking the
optional IF97 component. It is not a mandatory dependency. The default heat
capacities and gas constants are approximate constant air/water properties;
they are configurable and are not inferred from the liquid material provider.

The constant caloric model uses a common reference temperature $T_r$:

$$
h_l(T)=c_{pl}(T-T_r),\qquad
h_v(T)=L_r+c_{pv}(T-T_r),\qquad
L(T)=h_v(T)-h_l(T),
$$
$$
U=m_d(c_{pd}-R_d)(T-T_r)+m_v[h_v(T)-R_vT].
$$

The gas balances are

$$
\frac{dm_v}{dt}=\sum_i\dot m_i,\qquad
\frac{dU}{dt}=\sum_i\left[h_iA_i(T_i-T)+\dot m_i h_v(T_i)\right].
$$

Transported vapor carries enthalpy across the gas control surface. Its latent
energy is stored in the vapor inventory, rather than being added again as a
sensible-air heat source. With a fixed volume, evaporation can also change
temperature through the difference between transported enthalpy and stored
internal energy. The associated liquid-volume change and wall motion are
neglected in this model.

## Surface exchange

Positive rates leave a surface. `LiquidSurface` supports evaporation and
condensation; `CondensationOnly` supports cold-wall condensation without
allowing an untracked wall-film inventory to evaporate. `Impermeable` disables
moisture transfer. For supplied gas-film coefficient $k_m$ in m/s,

$$
\dot m_i=\frac{A_i k_{m,i}}{R_vT_{film,i}}
              [p_{sat}(T_i)-p_v],\qquad T_{film,i}=(T_i+T_{old})/2.
$$

For condensation-only walls this expression is restricted to nonpositive
values. A zero coefficient disables the exchange. This is a dilute gas-film
pressure-difference closure: no boiling, Stefan-flow correction, solution water
activity, bulk fog, or automatic mass/heat-transfer analogy is included. Supply
coefficients appropriate to the geometry and conditions. Relative humidity
above one is reported without clipping vapor mass; boundary condensation may
remove vapor, but bulk supersaturation is not resolved into droplets.

Convection reuses `ConvectionBoundary`: either constant nonnegative `h` or the
existing Churchill–Chu vertical-plate natural-convection correlation. The
ambient temperature is always replaced by the cavity temperature. See
[exterior convection](exterior_convection.md) for that correlation's formula,
property inputs, and supported Rayleigh range. This is a mean plate approximation
to a vessel wall, not a general enclosed-gap correlation. Do not apply it to a
horizontal liquid surface or lid. Such surfaces can supply an effective $h$, or
provide `convection_coefficient(T_surface, T_air)` for a validated horizontal or
enclosure-specific closure. This callback overrides `convection`, must return
a finite nonnegative coefficient, and is evaluated once per surface per call.

Radiation is diffuse-gray surface-to-surface exchange with transparent gas.
The pure numerical kernel is also available as `GrayDiffuseEnclosure` in
`equations/GrayDiffuseEnclosure.hh`. It takes areas, symmetric exchange areas
$C_{ij} = A_i F_{ij}$, and patch emissivities, and retains its LAPACK factorization
for repeated temperature evaluations. Its shifted-emission interface accepts
$\sigma * (\bar{T^4} - T_\mathrm{ref}^4)$ and returns shifted irradiation as well as
pairwise-conservative outward heat rates. It owns no gas or liquid state.
Pass a row-major $N \times N$ view-factor matrix in the same order as the surface list:

$$
J_i=\epsilon_i\sigma T_i^4+(1-\epsilon_i)\sum_jF_{ij}J_j,
\qquad Q_{rad,i}=\sum_j A_iF_{ij}(J_i-J_j).
$$

View factors must be finite, in $[0,1]$, sum to one in each row, and obey
$A_i F_{ij} = A_j F_{ji}$ (relative geometric tolerance $1\times 10^{-10}$). Concave walls may
have nonzero self-view factors. Emissivities must be in $(0,1]$; a numerically
singular radiosity system is rejected. The dense system uses Trilinos'
`Teuchos::LAPACK` LU factorization and solve (`GETRF`/`GETRS`), with column-major
internal storage; the public view-factor input remains row-major. An empty
matrix explicitly disables radiation. Pairwise exchange conserves radiative energy; radiation is not
deposited directly in the air. These view-factor and enclosure assumptions
follow the [COMSOL description of radiative exchange](https://www.comsol.com/blogs/introduction-to-computing-radiative-heat-exchange).
The transparent-gas approximation excludes significant water-vapor absorption.

Each surface reports the following rates:

| Member | Meaning and coupling use |
|---|---|
| `vapor_mass_rate` | Signed water transfer, kg/s; subtract from liquid or accumulate condensate |
| `sensible_heat_rate` | $ h A (T_\mathrm{surface} - T_\mathrm{air}) $, W |
| `latent_heat_rate` | $ \dot m L(T_\mathrm{surface}) $, W |
| `radiation_heat_rate` | Net radiation to the other surfaces, W |
| `heat_rate` | Sensible + latent + radiation, W; use when liquid enthalpy transport is already accounted for |
| `energy_rate` | `heat_rate` + $\dot m h_\mathrm{l}(T_\mathrm{surface})$, W; total surface energy loss using this enthalpy reference |

The sum of `energy_rate` over all surfaces equals `air_energy_rate` to floating
point accuracy. For wall condensate, the difference between `energy_rate` and
`heat_rate` accounts for the collected liquid's enthalpy. Do not apply both
rates to the same energy balance or remove liquid enthalpy twice.

## Advancement and external coupling

`evaluate()` returns instantaneous rates without changing state. `preview(dt)`
solves backward-Euler water-mass and gas-internal-energy balances without
mutation; `advance(dt)` commits that result. Coefficients $h$ and the film
mass-transfer conductance are frozen at the old air temperature. Vapor pressure,
air temperature and mass are coupled implicitly. Rates in a returned step are
the discrete rates used in its balances, not a fresh evaluation with updated
coefficients. Multiply by `dt` to obtain the transferred mass and energy.

The solver brackets nonnegative vapor mass and then solves the monotone energy
residual in the configured air-temperature bracket. The default bracket is
273.15–373.15 K; change it only when the supplied properties remain applicable.
Out-of-bracket solutions, invalid geometry/properties, callback failures, and
failure to close the balances throw before committing state. `set_state()` also
validates before assignment. The returned mass and energy residuals are in kg
and J, respectively. Time integration is first order; implicit stability is not
a temporal accuracy guarantee.

A coupling driver should preview a step, apply equal and opposite transfers
to its liquid and wall models, check available liquid and coupled convergence,
then accept all states together (the accepted cavity state can be applied with
`set_state(step.state)`). Finite liquid availability is caller-owned: there is
no silent evaporation cap. Condensate return to the liquid pool is also a
caller decision. If coupling fails, discard the preview. The original scalar
surface API has no mesh or communicator: MPI callers using that API must reduce
surface data first and synchronize acceptance. The field-batch API below handles
collective sampling and cavity acceptance. Callbacks must be pure and provide
consistent properties on every participating rank.

## Face/cell fields, batched entries, and boundary refresh

Include `equations/LumpedInnerCavityCoupling.hh` and bind each patch using
`InnerCavityFieldBatch::from_field(temperature, batch_id, properties, sampling)`.
The batch owns the mesh, communicator, boundary ID, closure properties, and field
reader once. Its `FaceEntry` contains only face ID, batch slot, area, and temperature.
The reader captures only the field reference and receives mesh/batch context
from the batch when called. `InnerCavityFieldInput` collects these batches for
multi-subdomain evaluation; shared binding metadata is never stored per face.
The integer-ID factory is local. The optional name overload resolves the name
collectively exactly once at binding, including on ranks without local faces.
`inner_cavity_batch_id(mesh, name)` exposes that setup operation separately.
Sampling and refresh use direct batch-ID/slot access with no repeated name scan.
The temperature field must outlive the batch. Both scalar `FaceField`/`CellField`
and their `FieldStored` counterparts are supported, including `MeshHandle` and
`SolidSubdomain` meshes. Face fields supply interface temperatures directly.
Cell fields supply the boundary owner's current owned temperature: this is an
explicit owner-cell approximation, without wall-resistance reconstruction or
gradient extrapolation. Ghost temperatures are never included or required.
The batch's `properties.area` and `properties.temperature` are replaced by mesh
areas and field samples; other closure properties remain caller-supplied.

An `InnerCavityFieldInput` holds an ordered list of batches from one or several
subdomains with identical or congruent communicators (`MPI_IDENT`/`MPI_CONGRUENT`).
Reordered rank groups are rejected. The input's communicator coordinates all
collectives; endpoint wrappers need not be the same object. It may mix field
and mesh types and the following sampling modes:

| Mode | Cavity surfaces and distribution |
|---|---|
| `Facewise` (default) | One surface per globally owned boundary face, retaining nonlinear evaporation and radiation |
| `AreaAverage` | One surface per batch with $A = \sum{A_f}$ and $T = \frac{\sum{A_f T_f}}{A}$; resulting extensive rates are returned to faces in proportion to area |

Patch averaging is a modeling approximation: evaluating saturation pressure,
natural convection, and fourth-power radiation at the mean temperature differs
from integrating those terms over nonuniform faces. Both modes preserve the
integrated rate actually used in their cavity balance.

Call `cavity.evaluate(input)`, `cavity.preview(dt, input)`, or
`cavity.advance(dt, input)`. Every call reads current field values and geometry;
constructing a batch does not freeze its field. All batch operations are
collective, including on ranks with no local patch faces. Owned faces are
counted exactly once. Averaged batches reduce only $\sum A$ and $\sum A T$;
their face IDs, areas and temperatures are never gathered globally. Facewise
batches cache global ID ordering and local-to-global mappings, rebuilding them
collectively when any rank's sampled face IDs/order changes. Only current areas
and temperatures are communicated on subsequent facewise samples. Numeric
geometry and field data are always read anew, including after motion/replay.
Immutable batch configuration is validated once on the first operation. Duplicate
batches, unknown/globally empty patches,
inconsistent configuration/state/time steps, invalid samples, callback failures,
and rank-inconsistent computed rates fail before `advance` commits state.

The replicated `exchange.surfaces` array is ordered by batch, then ascending
global face ID for facewise batches. Averaged batches occupy one slot. A supplied
view-factor matrix uses that order; a change in sampled topology requires a new
matrix. Geometry-dependent view factors remain caller-owned. The complete dense
radiation problem is replicated, making this API suitable for small boundary
sets; use averaged batches for a compact enclosure model where appropriate.

Closures stay in batch storage; numerical samples borrow them during a call.
`exchange.surfaces` contains compact numeric rates, without repeated names or
callbacks. For scalar calls its order matches the supplied surface list.
`exchange.boundaries` holds each batch's mesh identity, integer ID, and **local
owned** face transfers with batch slots and global IDs for refresh validation.
It includes batches even on empty ranks. Do not sum the replicated
`exchange.surfaces` across ranks; sum local boundary-face rates when auditing
distributed coupling.

`refresh_inner_cavity_boundary_conditions(cache, exchange, transfer)` converts
face rates into outward heat fluxes, $q_\mathrm{out} = \mathrm{rate}/A$, in a mesh-bound
`FVM::MeshBoundaryCache`. `Heat` selects `heat_rate`; `TotalEnergy` selects
`energy_rate`. It validates face coverage, current areas and mesh identity and
updates only the matched patches, preserving unrelated cached patches.
Refresh is collective and rejects stale mappings without changing the cache.

`TemperatureDiffusionEquation::set_boundary_heat_flux(cache)` and
`SolidHeatConductionEquation::set_boundary_heat_flux(cache)` retain an independent
copy for callers supplying mutable caches. Their
`set_boundary_heat_flux_snapshot(shared_ptr<const Cache>)` methods instead adopt
an immutable snapshot without copying arrays. Keep no mutable aliases to a
published snapshot. `boundary_heat_flux()` returns a shared boundary checkpoint.
The preferred overload
`refresh_inner_cavity_boundary_conditions(equation, cache, exchange, transfer)`
uses a caller-owned `shared_ptr<const Cache>`: it stages one cache copy and shares
the accepted snapshot between equation and caller. Previous snapshots remain
valid for rollback. The mutable-cache convenience overload requires a separate
equation copy, but avoids the previous nested staging copy.
Updates preserve matrix/graph and Krylov storage when only flux values change;
changing the overridden batch set invalidates operator storage collectively.
Physical reassembly still invalidates numeric preconditioners on the coupled
path, since material properties and time steps may change independently of flux.
Overrides replace the complete
temperature or convection condition on the selected batches; other patches
retain their existing conditions. The physical operator uses the same effective
conductivity, including cell or boundary overrides, to convert
$dT/dn = -q_\mathrm{out}/k$. Nonzero heat flux requires positive conductivity and zero
carrier flux through the coupled face. Diffusivity-only APIs reject these
physical flux boundaries. Passing an empty mesh-bound cache clears all flux
overrides; replacing a solid equation's `BoundaryConditionSet` also clears them.

For example, with caller-owned `liquid_face_temperature`, `wall_cell_temperature`,
`liquid_temperature_equation`, and `solid_conduction`:

```cpp
using Batch = SimpleFluid::InnerCavityFieldBatch;
SimpleFluid::InnerCavityFieldInput input(comm, {
    Batch::from_field(liquid_face_temperature, free_surface_batch_id, pool,
                     SimpleFluid::InnerCavitySampling::Facewise),
    Batch::from_field(wall_cell_temperature, inner_wall_batch_id, wall,
                     SimpleFluid::InnerCavitySampling::AreaAverage)
});

// Initialize each SP<const Equation::boundary_cache_type> with an empty,
// mesh-bound cache. Refresh shares one immutable snapshot with each equation.
const auto trial = cavity.preview(dt, input); // Optional view factors as third argument.
SimpleFluid::refresh_inner_cavity_boundary_conditions(
    liquid_temperature_equation, liquid_flux_cache, trial.exchange,
    SimpleFluid::InnerCavityEnergyTransfer::Heat);
SimpleFluid::refresh_inner_cavity_boundary_conditions(
    solid_conduction, solid_flux_cache, trial.exchange);
// Advance both thermal subdomains, account for water/condensate inventories,
// and check the coupled candidate. Accept all states only when those checks pass.
cavity.set_state(trial.state);
```

The helper does not advance subdomains, alter their mass inventories, or provide
a transaction spanning multiple equations. On a rejected coupled trial, the
driver restores subdomain states and their previous flux caches together.

## Example: liquid pool and lumped vessel wall

The following uses the optional IF97 material callback. Transfer coefficients
and geometry below are illustrative; qualify them for the actual vessel.

```cpp
#include "equations/LumpedInnerCavityModel.hh"
#include "materials/IF97Water.hh"
#include <array>

SimpleFluid::LumpedInnerCavityOptions options;
options.volume = 0.1;
options.dry_air_mass = 0.115;
options.saturation_pressure = SimpleFluid::IF97Water::saturation_pressure;
const double initial_temperature = 298.15;
const double initial_rh = 0.5;
const double initial_w = initial_rh * options.saturation_pressure(initial_temperature)
    * options.volume / (options.vapor_gas_constant * initial_temperature * options.dry_air_mass);
SimpleFluid::LumpedInnerCavityModel cavity(options, {initial_temperature, initial_w});

SimpleFluid::InnerCavitySurface pool;
pool.name = "liquid-free-surface";
pool.area = 0.2;
pool.temperature = 310.15;
pool.moisture_exchange = SimpleFluid::CavityMoistureExchange::LiquidSurface;
pool.mass_transfer_coefficient = 0.002;
pool.convection.heat_transfer_coefficient = 3.0; // supplied horizontal-surface h
pool.emissivity = 0.96;

SimpleFluid::InnerCavitySurface wall;
wall.name = "vessel-wall";
wall.area = 0.8;
wall.temperature = 293.15;
wall.moisture_exchange = SimpleFluid::CavityMoistureExchange::CondensationOnly;
wall.mass_transfer_coefficient = 0.002;
wall.convection.heat_transfer_coefficient = 2.0; // effective mean wall coefficient
wall.emissivity = 0.8;

std::array surfaces{pool, wall};
// Pool sees only the enclosing wall. Wall also sees itself.
std::array view_factors{0.0, 1.0, 0.25, 0.75};
const double dt = 0.1;
const auto trial = cavity.preview(dt, surfaces, view_factors);
// Transfer dt * trial.exchange.surfaces[i].vapor_mass_rate and the appropriate
// energy/heat rate to each coupled inventory, then accept the complete step.
cavity.set_state(trial.state);
```

The focused `testLumpedInnerCavityModel` tests analytic sensible relaxation,
mass/energy closure, evaporation and condensation, gray enclosure resistance,
multiple-surface radiation, equilibrium, natural-convection reuse, extension
callbacks, and transactional failures. They qualify the implementation's
balance contract, not a particular vessel correlation or experimental case.
`testLumpedInnerCavityCoupling` additionally checks both sampling modes, legacy
and mapped fields, mixed subdomains, live refresh, conservation through a solid
solve, empty ownership, and collective failures in serial and with two MPI ranks.
