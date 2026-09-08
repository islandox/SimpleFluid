# Bottom-heated bubbly convection in reference water

This case develops **liquid circulation from rest**, driven by a localized
bottom heat and hydrogen source. Both SimpleFluid and the independent
OpenFOAM application solve momentum, pressure, temperature, and gas transport.
It complements the earlier prescribed-flow and uniform-ALE conservation
fixtures with a resolved buoyant plume and downward return flow.

## Physical setup

| Input | Shared value |
| --- | --- |
| Domain | 20 mm wide × 20 mm tall, extruded 2 mm in y |
| Mesh | 28 × 1 × 36 graded Cartesian cells; 6 layers at each x/z wall, growth 1.25 |
| Initial state | Rest, 300 K, zero gas inventory |
| Water | IF97 reference at 300 K and 101325 Pa absolute; constant cp, mu, k and reference beta |
| Gravity | 9.81 m/s² downward in z |
| Source region | Middle third of the width, lowest eighth of the height |
| Heat deposition | 4 MW/m³ in the source region; 0.133333 W integrated |
| H2 production | 2e-7 mol/J × deposited power; 2.666667e-8 mol/s integrated |
| Bubble slip | Prescribed 5 mm/s upward relative to the solved liquid flux |
| Thermal boundaries | Side and top temperatures fixed at 300 K; bottom and y faces adiabatic |
| Liquid boundaries | No-slip side/bottom walls; impermeable slip top and y faces |
| Gas boundary | Escape only through the top; no incoming bubbles |
| Time | Backward Euler, dt=0.02 s, 0–20 s; output every 2 s |

The y direction represents a two-dimensional slice; front/back wall friction
is absent. The source is a volumetric heater/radiolysis surrogate inside the
lowest cells, not a resolved gas nozzle. Cold walls provide heat removal.
The 20-second run is a **developing transient**, not a claim of thermal steady
state. The earlier steady verification cases remain available separately.
The source bounds are cell faces on both the default 1008-cell grid and the
2080-cell finer grid. See [boundary-layer meshes](../BOUNDARY_LAYER_MESHES.md)
for spacings, regeneration, and the shared-mesh override. Water and source
properties remain identical when the grid changes.

## Matched model

Both sides use the shared IF97 material snapshot, checked by live IF97 queries
in SimpleFluid. The pure-liquid reference law is
`rho_l = rho0 [1 - beta (T - T0)]`. Gas affects the existing single-continuum
buoyancy through `rho_mix = (1-alpha_g) rho_l + alpha_g rho_H2`, with a fixed
ideal-hydrogen reference density `rho_H2 = p_abs M_H2/(R T0)`. Neither steam
density nor an independent gas momentum equation is used.

The liquid equations use constant reference-density inertia, `nu=mu/rho0`,
incompressible continuity, and acceleration `g (rho_mix-rho0)/rho0`. Momentum
is solved before temperature; then bubble transport and production advance,
and material feedback is refreshed for the next step. Temperature uses the
production fixed-grid contract: current frozen `C=rho_mix cp` in storage and
upwind capacity flux, physical conductivity, and the localized heat source.
OpenFOAM assembles `C*fvm::ddt(T) + fvm::div(phi_C,T) - fvm::laplacian(k,T)`;
`phi_C` uses upwind C, matching SimpleFluid. This is not nonlinear IF97
enthalpy transport or a phase-resolved energy model.

The source uses the production Sheng microbubble moment path. Microbubble
moles and number are transported with liquid flux plus upward slip, followed
by local production. Number production uses the temperature-dependent
nucleation correlation and capillary ideal-gas equation. Each implementation
independently reconstructs radius/void from the two moments. Conversion,
dissolution, and large bubbles are suppressed to isolate the shared model.
The nucleation model's required positive solute parameter is 1 mol/m³; the
material closure remains a dilute reference-water approximation, not a
validated uranyl-solution property or radiation-chemistry model. Absolute
bubble pressure is fixed at 101325 Pa; this small cavity omits its approximately
196 Pa hydrostatic variation in the gas EOS.

The OpenFOAM reference uses its finite-volume PISO momentum/pressure solve;
it does not prescribe velocity or import SimpleFluid flow fields. Both
solvers start with exactly zero liquid velocity. Neither solves turbulence,
interphase drag/lift, bubble-induced turbulence, or interface motion.

## Run and outputs

From the repository root:

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_IF97=ON
SIMPLEFLUID_BUILD_CONFIG=Debug \
  verification/openfoam/bottomHeatedBubblyConvection/run_comparison.sh \
  build/verification/bottom-convection
```

The launcher creates a fresh run directory, builds a local OpenFOAM reference,
checks its mesh, runs both solvers, compares complete matched histories, and
renders a `figures/index.html` gallery. `fields.csv` retains all x–z cells,
physical bounds, temperature, liquid density, gas fraction, and solved velocity
components. Figures show two-dimensional distributions and error fields;
time–height figures show the central x column. Matched CSV data contain every
cell. SVG is always generated, with PNG/PDF when `rsvg-convert` is available.

`history.csv` records temperature extrema/mean, gas holdup, maximum speed,
upward/downward velocity, inventories, integrated source, continuity, and a
discrete per-step thermal budget. `transient.json` declares comparison limits
for this coarse fixture. Differences in the PISO implementations and
collocated velocity reconstruction can remain; the field error figures expose
them. These are numerical comparisons, not experimental or grid-converged
validation. Figures are retained if the numeric comparison fails, and the
launcher still exits nonzero.

Each executable independently requires a heated bubbly plume **and downward
return flow**, finite/bounded fields (`290<T<320 K`, `alpha_g<0.02`), hydrogen
closure below 1e-13 mol, continuity below 1e-6 s⁻¹, and a thermal step residual
below 1e-6 J. The thermal residual uses the actual frozen transport capacity,
temperature increment, source energy, and outward wall conduction; it does
not claim conservation of an unsupported nonlinear mixture enthalpy.

The focused `bottom_convection_zero_source` CTest disables both sources and
requires the water to remain cold, gas-free, and stationary. Standalone
SimpleFluid options `--steps N` and `--source-scale S` support such controls;
paired comparisons always use the shared defaults.

Both executables support distributed MPI solves. SimpleFluid partitions the
Cartesian mesh along its largest cell-count direction; a matching OpenFOAM
decomposition uses z slabs for this grid. Each rank writes only its owned
cells, retaining the global coordinate-based sample IDs, under SimpleFluid's
`rankN` output directory or OpenFOAM's `processorN` case directory. Each rank's
history contains the same globally reduced diagnostics and conservation
budgets. Processor interfaces retain liquid and bubble transport, and the
pressure reference is applied only by the rank owning the first global cell.
The per-rank `timing.json` records elapsed and process CPU time for the time
loop, including its diagnostics and CSV output, after mesh and solver setup.

The opt-in [performance runner](../PERFORMANCE.md) selects two ranks and
pressure PCG/DIC for this 1,008-cell fixture while keeping transport policies
and all physical/linear tolerances unchanged. Explicit `--ranks` and
`--policy` arguments override those experiment settings. The ordinary paired
comparison launcher above retains its serial execution and solver defaults.

### Historical uniform-grid comparison (2026-09-08, GCC Debug / OpenFOAM v2606)

Before boundary-layer refinement, the full 1000-step comparison passed its declared limits and all independent
source, hydrogen, continuity, and thermal-budget gates. The matched field
output contains 2112 cell/time records per solver (192 cells, 11 output times).

| Quantity at 20 s | SimpleFluid | OpenFOAM |
| --- | ---: | ---: |
| Maximum temperature, K | 302.075148 | 302.002300 |
| Maximum upward velocity, m/s | 0.00758516 | 0.00756696 |
| Most negative vertical velocity, m/s | -0.00420224 | -0.00418708 |
| Maximum gas volume fraction | 0.000410162 | 0.000404717 |

Across all output times, the maximum cell temperature difference was
0.1450 K and the maximum velocity-vector difference was 0.0007764 m/s
(about 10.3% of the reference peak speed). Local relative errors are larger
near stagnation points and tiny gas-fraction tails; the figures and matched
CSV retain them. These spatial differences should not be confused with the
smaller differences between each solver's global extrema. The zero-source
control, 21 comparator checks, and 9 plotting checks also passed.

## Planar ALE boundary

This case uses a fixed mesh and fixed liquid surface. Current planar ALE
explicitly rejects spatial fission/gas profiles and nonadiabatic thermal
boundaries. Those guards remain intact. Enabling this plume in ALE requires
separate localized-source geometry, energy, gas-volume, and rollback work;
changing a diagnostic surface level would not provide that capability.
