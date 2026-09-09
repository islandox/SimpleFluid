# Optional IF97 water material library

Enable `SIMPLEFLUID_ENABLE_IF97` to build `SimpleFluid::IF97`. The option is
off by default. The property library itself has no Trilinos or MPI dependency;
the field adapter uses the existing SimpleFluid equation interfaces.

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_IF97=ON
cmake --build --preset GCC-Debug --target testIF97Water testIF97Material
ctest --test-dir build/gcc -C Debug -R '^IF97' --output-on-failure
```

CMake uses a local `IF97.h` when it finds one, or downloads the SHA256-pinned
[CoolProp IF97 v2.2.1 release](https://github.com/CoolProp/IF97/releases/tag/v2.2.1).
For an offline build, provide
`-DSIMPLEFLUID_IF97_INCLUDE_DIR=/path/to/directory/containing/IF97.h`.
Local headers must be version 2.2.1 or newer; older releases have shared
scratch storage in saturation evaluation. The dependency stays private to
one translation unit, and no upstream build scripts or wrappers are run.
The upstream [MIT license](https://github.com/CoolProp/IF97/blob/v2.2.1/LICENSE)
is retained in the fetched source; redistributors must retain its notice.
The disabled configuration performs no IF97 discovery or download.

Link `SimpleFluid::IF97` for property queries. The `SimpleFluid` umbrella
also includes the target when the option is on.

```cpp
#include "materials/IF97Water.hh"

const auto water = SimpleFluid::IF97Water::liquid(298.15, 101325.0);
const auto nu = water.kinematic_viscosity();
const auto alpha = water.thermal_diffusivity();
const auto beta = SimpleFluid::IF97Water::liquid_thermal_expansion(298.15, 101325.0);
const auto boiling = SimpleFluid::IF97Water::saturation_at_pressure(101325.0);
// boiling.liquid.temperature [K], boiling.vapor.density [kg/m^3],
// boiling.latent_heat() [J/kg], boiling.surface_tension [N/m].
```

All temperatures are Kelvin, pressures are **absolute Pascal**, and all
properties use SI units: density kg/m³, isobaric heat capacity J/(kg K),
enthalpy J/kg, entropy J/(kg K), viscosity Pa s, and conductivity W/(m K).
Never pass gauge or kinematic solver pressure directly. Defining upstream's
`IAPWS_UNITS` flag is a compile error because it changes the pressure and
energy units. Thermodynamic reference values follow IF97.

`evaluate(T,p)` returns the stable homogeneous water/steam state in IF97
regions 1, 2, and 3. This combined thermodynamic/transport API accepts
273.15–1073.15 K and 611.213 Pa–100 MPa. It does not expose high-temperature
region 5, ice, metastable extrapolation, or mixtures. Region 3 uses the
upstream direct backward density approximation. These thermodynamic regions
come from [IAPWS R7-97(2012)](https://iapws.org/technical-guidance/release/IF97-Rev);
transport properties use the IAPWS correlations supplied by CoolProp IF97.

`liquid(T,p)` rejects vapor and temperatures at or above the critical point.
It selects saturated liquid when T is within 1e-7 K of saturation;
`evaluate` rejects that ambiguous band. `saturation_at_pressure(p)` returns
both phases separately for p below 22.064 MPa. Saturation-line and surface
tension queries also accept the critical endpoint, where surface tension
is zero. The 273.15 K saturation pressure uses the upstream rounded lower
bound, 611.213 Pa; its temperature round trip is accurate to 1e-5 K.
Non-finite or non-positive inputs throw `std::invalid_argument`; unsupported
states throw `std::out_of_range`. There is no fallback to synthetic water.

`liquid_thermal_expansion` computes `-(1/rho) d(rho)/dT` at constant pressure
with a second-order, 0.001 K finite difference. It switches to a one-sided
stencil at liquid domain and region 1/3 boundaries, and rejects an interval too narrow
to differentiate. It preserves water's negative expansion near freezing.
Use it for a reference-state Boussinesq approximation, rather than treating
a single coefficient as the nonlinear IF97 equation of state.

## Material fields and solver integration

`materials/IF97Material.hh` provides initial `BoussinesqModelOptions` and an
isobaric, pure-liquid callback. Link both `SimpleFluid::IF97` and
`SimpleFluid::Equations` when using this adapter.

```cpp
auto options = SimpleFluid::if97_liquid_model_options(298.15, 101325.0);
// Set this before constructing a solver when buoyancy should use the
// updated physical density, avoiding a second linear thermal correction.
options.density_feedback_enabled = true;
// Construct the solver with options, and initialize temperature in Kelvin.
solver.set_material_updater(
    SimpleFluid::make_if97_liquid_material_updater<Pack, MeshType>(101325.0));
```

The callback updates density, heat capacity, dynamic viscosity, and thermal
conductivity from cell temperatures at the prescribed pressure. It ignores
the gauge-pressure field. It validates all owned states before publishing
any values on that rank and performs no communication; the existing
`MaterialPropertyFields::update()` handles collective errors and ghost
synchronization. This local staging does not add a distributed transaction
to the generic callback interface. Solver-owned rollback remains necessary
after a failed global update.

Install callbacks before free-surface liquid-mass initialization. Do not
also install `MaterialFeedbackModel` modes that overwrite their density or
viscosity. These are pure-liquid values; void fractions, noncondensable gas,
and dissolved solutes need their own models. Saturated steam density is not
the density of radiolytic hydrogen/oxygen bubbles.

The current `planarALE` solver requires built-in
`BoussinesqTemperatureOnly` feedback and rejects custom material callbacks.
For its existing cases, IF97 can supply the reference density, cp, viscosity,
conductivity, and expansion coefficient for that supported linear closure.
Those cases must identify that approximation. Nonlinear IF97 feedback in
ALE requires a separate solver integration with consistent energy,
liquid-volume closure, and rollback; enabling this library does not enable it.
