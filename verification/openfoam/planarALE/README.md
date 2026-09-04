# Planar ALE: OpenFOAM and SimpleFluid

These serial fixtures compare **the same uniform thermal-expansion energy and
liquid-volume problem** at every accepted time, including an explicitly checked
source-off steady state. SimpleFluid runs its dimensional `BoussinesqSolver`
with the enabled `planarALE` path, mutable geometry, cellwise liquid mass,
pressure projection, conservative temperature transport, and vented headspace.
The independent `planarALEBudgetFoam` application uses OpenFOAM finite-volume
matrices and actually moves its mesh. It solves the corresponding reduced
energy/volume equations with zero relative flux. Neither implementation reads
the other's results or uses the analytic solution to advance its state.

This comparison covers thermal expansion, old/new-volume energy storage,
geometry, GCL, and liquid mass conservation. The OpenFOAM reference does not
solve momentum, and this is not a comparison against VOF/interface dynamics.
Pressure, velocity profiles, nonuniform convection, gas displacement, bubble
escape, boiling, turbulence, and spatial/time convergence studies are outside
this fixture's scope. See [the model support matrix](../../../docs/modeling/planar_free_surface_volume_budget.md).

## Matched physical and numerical inputs

| Input | Both implementations |
| --- | --- |
| Initial liquid domain | Cartesian `[0,1] × [0,1] × [0,1]` m |
| Mesh | `1 × 1 × 8` uniform hexahedral cells; bottom fixed; affine axial expansion |
| Vessel | Area `A = 1 m²`, bottom `0 m`, total height `2 m` |
| Initial state | `T₀ = 300 K`, pure-liquid density `ρ₀ = 10 kg/m³`, liquid mass `M = 10 kg` |
| Liquid density law | `ρₗ(T) = ρ₀[1 − β(T − T₀)]`, `β = 10⁻³ K⁻¹` |
| Heat capacity | `cₚ = 2 J/(kg K)` |
| Thermal boundaries | Adiabatic on all patches; no physical liquid boundary flux |
| Gravity and conduction | Zero |
| Heating | `q = 1000 W/m³` over the **accepted new liquid-domain volume** |
| Time discretization | Conservative Backward Euler; `Δt = 0.01 s` |
| Nonlinear geometry coupling | Up to 30 iterations; level change at most `10⁻¹³ m` |
| Transient case | 20 heated steps; compare `t = 0, 0.01, …, 0.20 s` |
| Steady case | Same heating, then five source-off steps; compare through `0.25 s` |

SimpleFluid sets wall velocity to no-slip, moving-top velocity to slip, top
pressure to zero gauge, and the remaining pressure boundaries to zero gradient.
The ALE boundary owner imposes the top's absolute mesh-normal flux. Its solved
relative face fluxes and absolute volume-continuity residual must each be below
`2 × 10⁻¹⁰ m³/s`. This verifies the zero-relative-flux reduction used by the
OpenFOAM reference. The latter has adiabatic temperature patches and advances
conserved cell masses on the affine moving mesh; it has no momentum boundary
conditions because it does not solve momentum. SimpleFluid's finite viscosity
does not enter the matched uniform energy and volume equations.

For each cell with conserved mass `M_c`, the reduced equations are

```text
(M_c cp T_c^(n+1) − M_c cp T_c^n) / dt = q^(n+1) V_c^(n+1)
L^(n+1) = sum_c [M_c / rhoLiquid(T_c^(n+1))] / A
z_point^(n+1) = z_reference L^(n+1) / L0
phi_relative = phi_absolute − phi_mesh = 0.
```

OpenFOAM assembles `fvm::ddt(rhoCp,T) + fvm::div(relativeHeatCapacityFlux,T)`
with `rhoCp = M_c cp / V_c` after `mesh.movePoints()`. The `Euler` scheme uses
the accepted old cell volumes and current new volumes, as shown in the
[OpenFOAM Euler time-scheme source](https://api.openfoam.com/2512/EulerDdtScheme_8C_source.html).
It solves temperature, recomputes the level from the pure-liquid density, and
iterates geometry until the level criterion is met. Its GCL residual comes
from the actual swept mesh flux and cell-volume change.

An independent Backward-Euler oracle checks each solution. With
`a = 1 − β(T^n − T₀)` and `b = q Δt/(ρ₀ cₚ)`, its stable small root is

```text
dT = 2 b / [a + sqrt(a² − 4 β b)]
T^(n+1) = T^n + dT
L^(n+1) = 1 / [1 − β(T^(n+1) − T₀)].
```

The continuous-time solution is not the acceptance oracle: both solvers use
the same first-order time discretization. At `t = 0.20 s`, the discrete
solution is approximately `310.053058614908 K` and `1.01015514892227 m`.
After heating stops, both drivers require **all five** subsequent steps to
satisfy `|ΔT| ≤ 2 × 10⁻⁸ K` and `|ΔL| ≤ 2 × 10⁻¹¹ m`, otherwise they exit
nonzero. The final warmed, expanded equilibrium is the steady verification
state. It is distinct from selecting an unconverged heated transient as steady.

## Run

From the repository root, with the usual SimpleFluid dependencies available:

```sh
SIMPLEFLUID_BUILD_CONFIG=Debug \
  verification/openfoam/planarALE/run_comparison.sh transient /tmp/planar-ale-results
SIMPLEFLUID_BUILD_CONFIG=Debug \
  verification/openfoam/planarALE/run_comparison.sh steady /tmp/planar-ale-results
```

The common [build environment](../../environments.sh) selects/builds
`planar_ale_comparison`; its default configuration is `RelWithDebInfo` when
`SIMPLEFLUID_BUILD_CONFIG` is omitted. The OpenFOAM launcher uses the active
OpenFOAM environment, or sources `OPENFOAM_BASHRC` (default
`/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc`). OpenFOAM v2606 is the exercised
version. It requires `wmake`, `blockMesh`, and `checkMesh`. Python 3 is the only
comparison-script dependency.

Every comparison creates a unique directory below the requested output root.
All OpenFOAM compilation and executable files remain within that directory;
the launcher does not install anything into the user's OpenFOAM application
directory. `blockMesh` and a successful `checkMesh` precede the solver. Logs,
both `history.csv` files, the final OpenFOAM mesh/temperature, and
`comparison.json` are retained there. The wrapper exits nonzero if a driver,
mesh check, coverage check, analytic/conservation gate, or comparison fails.

The individual drivers can also be run with explicit output directories:

```sh
verification/openfoam/planarALE/run_simplefluid.sh transient /tmp/ale-sf
verification/openfoam/planarALE/run_openfoam.sh transient /tmp/ale-of
```

## Outputs and acceptance

Both histories have one `sample=global` record at every declared time. They
write the solver owner's accepted physical time and check it against the
configured step schedule before accepting each record. They
contain mass-weighted temperature, pool level, actual mesh volume, total
liquid mass, sensible energy, and cumulative accepted-volume heat. Additional
columns give liquid-mass error, cumulative energy-budget error, maximum
cell GCL residual, and errors against the independent discrete temperature
and level oracle. Temperature uniformity is checked cell by cell in both
executables before the global record is accepted.

The [transient](transient.json) and [steady](steady.json) manifests declare
physical units, exact time/sample coverage, and fixed absolute tolerances.
The [shared comparator](../compare_verification.py) rejects duplicate,
missing, unexpected, nonfinite, and stale samples. It checks both solvers'
conservation/oracle residuals separately against zero, then compares the
physical quantities. Passing means agreement for this constrained numerical
fixture; it does not establish unrestricted ALE flow accuracy or physical
validation.
