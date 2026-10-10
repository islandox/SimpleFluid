# Exterior convection boundaries

The cylinder smoke example accepts `--exterior-convection=constant`, `natural`,
or `forced`. With no option it retains its original insulated radial boundary.
These options apply cooling directly to the fluid-domain boundary, without a
solid vessel wall; the natural option uses a vertical-plate approximation.

`BoundaryConditionSet::convection` applies exterior heat exchange to the
physical temperature path used by `SolidHeatConductionEquation` and dimensional
`BoussinesqSolver` configurations. Each entry names a mesh boundary patch:

```cpp
SimpleFluid::BoundaryConditionSet boundaries;
auto& exterior = boundaries.convection["rmax"];
exterior.correlation = SimpleFluid::ConvectionCorrelation::ChurchillChuVerticalPlate;
exterior.ambient_temperature = 293.15;       // K
exterior.characteristic_length = 1.0;       // full vertical height, m
exterior.fluid_thermal_conductivity = 0.026; // exterior fluid, W/(m K)
exterior.kinematic_viscosity = 1.6e-5;       // exterior fluid, m^2/s
exterior.thermal_diffusivity = 2.3e-5;       // exterior fluid, m^2/s
exterior.thermal_expansion = 1.0 / 300.0;    // exterior fluid, 1/K
exterior.gravity = 9.80665;                 // magnitude, m/s^2
```

The numbers above are illustrative supplied properties, not a built-in air
material model. Supply appropriate exterior-fluid properties at a representative
film temperature, `(T_surface + T_ambient)/2`; properties stay fixed until the
caller changes the configuration. They are separate from the solid/liquid
conductivity and material provider inside the mesh. No material-specific
property law is embedded in these closures.

For a specified coefficient, use `ConvectionCorrelation::Constant` and set
`heat_transfer_coefficient` in W/(m² K); zero means insulation. For cylinder
crossflow, use `ChurchillBernsteinCylinder`, set `characteristic_length` to the
outer diameter, and provide `free_stream_speed` in m/s plus the exterior
conductivity, viscosity, and diffusivity. Natural and forced models are selected
independently; there is no automatic mixed-convection combination.

| Selector | Geometry and formula | Supported input range |
|---|---|---|
| `Constant` | User-specified nonnegative `h` | Finite `h >= 0` |
| `ChurchillChuVerticalPlate` | Mean isothermal vertical-plate correlation, Eq. (9) of [Churchill and Chu (1975)](https://doi.org/10.1016/0017-9310(75)90243-4) | Positive properties; implementation conservatively restricts `0 <= Ra <= 1e12` |
| `ChurchillBernsteinCylinder` | Mean circular-cylinder crossflow correlation, Eq. (9) of [Churchill and Bernstein (1977)](https://doi.org/10.1115/1.3450685) | Positive properties; `Re Pr > 0.2` |

With exterior properties, `Pr = nu/alpha`, `Ra = g beta |Ts-Tinf| L^3/(nu alpha)`,
`Re = U L/nu`, and `h = Nu k_ext/L`, the implemented equations are

\[
Nu_{natural}=\left[0.825+
\frac{0.387 Ra^{1/6}}{[1+(0.492/Pr)^{9/16}]^{8/27}}\right]^2,
\]
\[
Nu_{forced}=0.3+
\frac{0.62 Re^{1/2}Pr^{1/3}}{[1+(0.4/Pr)^{2/3}]^{1/4}}
\left[1+(Re/282000)^{5/8}\right]^{4/5}.
\]

The primary papers were checked in their scanned text editions:
[vertical plate](https://www.scribd.com/document/273008782/Correlating-Equations-for-Laminar-and-Turbulent-Free-Conve)
and [cylinder crossflow](https://www.scribd.com/document/502579894/churchill1977).
These are mean correlations, applied facewise using each reconstructed surface
temperature and the configured full-body length. This is a local engineering
closure for nonuniform walls, not a local boundary-layer correlation. The plate
model does not infer cylinder-curvature corrections, orientation, or a length
from the mesh. It must not be applied to a horizontal lid/bottom as if it were a
vertical wall. Cylinder crossflow does not model axial flow, cylinder bundles,
blockage, or a resolved exterior flow. No radiation term is included.

## Discrete thermal contract

The outward heat flux is `q_out = h (Ts - Tinf)`, so the normalized Robin
condition is `dT/dn = (h/k_wall) (Tinf - Ts)`. Positive outward flux removes
energy. Both cooling and heating are supported. Unlike Neumann temperature
values, `heat_transfer_coefficient` is not a gradient in K/m.

For an orthogonal face, eliminating its surface temperature gives

\[
H = \frac{1}{d/k_{wall}+1/h},\qquad
q_{out}=H(T_P-T_\infty).
\]

The matrix diagonal receives `A H` and the right-hand side receives
`A H Tinf`. The conductivity is the same local or boundary-override value as
the physical diffusion operator. `h=0` and `k_wall=0` give zero heat exchange.
On skew faces the existing orthogonal projection and non-orthogonal correction
are both multiplied by the Robin resistance factor; gradient reconstruction
uses `(n + (h/k_wall)d).grad(T) = (h/k_wall)(Tinf - T_P)`.

Natural-convection `h` is recomputed on each physical advance. A bounded scalar
solve reconstructs the old surface temperature from the old cell temperature
and the cell-to-face conductance, then freezes `h` for the implicit temperature
solve. This is a lagged nonlinear coefficient, not a converged nonlinear Robin
solve at the new time level. On skew faces that old surface estimate uses the
two-point conductance; non-orthogonal terms enter the subsequent assembly.
The Rayleigh range check conservatively covers the full old-cell-to-ambient
temperature interval. `Ra=0` uses the continuous correlation limit and produces
zero flux when wall and ambient temperatures agree.

`evaluate_convection(boundary, surface_temperature)` exposes `h`, `Nu`, `Pr`,
`Ra`, and `Re` for inspection. `evaluate_convection_at_cell` exposes the same
evaluation after the old surface reconstruction.

Unknown patches, conflicting `temperature` and `convection` entries, invalid
properties, unsupported ranges, and nonzero carrier flux through a convection
wall are errors. Configuration and face-evaluation failures are propagated
collectively before the temperature solve; failed advances preserve the output
temperature. ALE walls must have zero **mesh-relative** carrier flux. Diffusivity-
only explicit/semi-implicit temperature APIs reject convection because they lack
physical conductivity and heat capacity. Existing temperature boundaries are
unchanged when no convection entry is configured.

The low-level physical temperature assembler also accepts normalized scalar
`Robin` conditions: `value = Tinf`, `robin_coefficient = h/k_wall` in 1/m.
Other scalar/vector equation families do not acquire general Robin support.
For changing exterior data between solid advances, use
`SolidHeatConductionEquation::set_boundary_conditions`. This boundary does not
implement automatic two-way fluid–solid interface coupling.
