# Dispersed microbubble transport: OpenFOAM versus SimpleFluid

These serial cases compare the production `RadiolyticGasModel` microbubble
number and hydrogen-mole transport with an independent OpenFOAM finite-volume
reference. Both use a prescribed carrier velocity and constant upward bubble
slip in the current weak single-continuum model. They do not solve separate
gas/liquid momentum equations or validate the general Euler–Euler regime,
drag correlations, polydispersity, coalescence, or boiling. Consequently,
stock `twoPhaseEulerFoam` is not a matched reference for these cases.

Run either complete comparison from any working directory:

```sh
verification/openfoam/dispersedBubbleFlow/run_comparison.sh steady
verification/openfoam/dispersedBubbleFlow/run_comparison.sh transient
# Optional second argument chooses the output root; each comparison gets a fresh run directory.
verification/openfoam/dispersedBubbleFlow/run_comparison.sh transient /tmp/bubble-comparisons
```

The launcher builds `dispersed_bubble_verification` through
`verification/environments.sh`. It also builds the bundled
`dispersedBubbleReferenceFoam` against OpenFOAM, generates a fresh `blockMesh`
case, runs both solvers, and writes `comparison.json`. Requires Python 3,
the configured SimpleFluid toolchain, and OpenFOAM development tools. The
reference is tested with OpenCFD OpenFOAM v2606; set `OPENFOAM_BASHRC` if its
`etc/bashrc` is elsewhere. OpenFOAM binaries and intermediate build files
remain inside the unique run directory. No download or stored reference
result is required.

`run_simplefluid.sh [mode] [output-directory]` and
`run_openfoam.sh [mode] [output-directory]` run one side. The latter requires
an empty output directory. Omitting the output argument creates a temporary
directory. Neither standalone run establishes cross-solver agreement.

## Shared problem and numerical method

[`reference.properties`](reference.properties) supplies both solvers' SI
inputs. A 1 m tall column of area 1 m² has 40 uniform cells, a 0.1 m/s
upward carrier velocity, and 0.4 m/s upward slip. Temperature is 300 K,
pressure 100000 Pa, and surface tension 0.07 N/m. Carrier motion is
prescribed; hydrodynamic pressure and velocity are not being compared.
SimpleFluid currently prevents bubble inflow at every boundary; only
`zmax` allows outgoing bubbles. The OpenFOAM boundary flux is identically
zero elsewhere. Thus the bottom supplies bubble-free carrier fluid.

Both solvers advance the conserved microbubble molar concentration `c`
and number density `N` using implicit Euler and first-order upwind at
`dt = 0.01 s`, then add the local production increment:

```text
transport: (c* - c_old)/dt + div(U_b c*) = 0
production: c_new = c* + q dt
            N_new = N* + (q / n_b) dt
U_b = U_liquid + U_slip = 0.5 m/s
```

The SimpleFluid path invokes `RadiolyticGasModel::advance`, including its
production finite-volume transport, local kinetics, radius reconstruction,
and escape accounting. Microbubble lifetime and large-bubble dissolution
time are `1e100 s`, so the computed microbubble decay is zero at this time
step; conversion is zero, both large-bubble moments and dissolved hydrogen
start at zero, and there is exactly one local kinetics substep. These
choices suppress interpopulation and dissolved-gas mass transfer while
retaining the actual production/transport path. No fields are overwritten
with an expected solution after an advance.

`q = release_efficiency × yield_mol_per_j × power_density`. All generated
hydrogen enters the microbubble moment, and the source number increment is
computed from the moles per newly formed bubble:

```text
n_b = (4 pi r_nuc³ / 3) (p + 2 sigma/r_nuc) / (R T)
alpha_g = N (4 pi r_nuc³ / 3)
```

The reference radius `r_nuc = 6.646002774709118e-8 m` was independently
evaluated from the Sheng/Winter nucleation correlation at the shared
temperature, concentration, yield, and pressure. SimpleFluid recomputes
and checks this value to `1e-12` relative tolerance before running. Both
initial moments use the same `n_b`; their ratio remains constant under
transport and production. OpenFOAM derives its own molar/number source
and void from the shared physical parameters, without reading any
SimpleFluid result. Its dimensioned fields represent mol/m³ and 1/m³.

## Steady production and escape

The steady case starts at `c = 1e-6 mol/m³` and turns on a uniform
`q = 1e-6 mol/(m³ s)` source. Sustained production is needed because the
current model permits no bubble inflow. It runs to the same physical
time `t = 10 s` on both sides; output times are 0, 2, 4, 6, 8, and 10 s.
The comparison includes the approach to steady state and the converged
profile.

The continuum solution is `c(z) = q z/U_b` and the steady outlet rate is
`q × column_volume = 1e-6 mol/s`. Each executable enforces the final
outlet balance to `2e-11 mol/s` and a per-step maximum concentration change
below `1e-12 mol/m³` for at least five consecutive steps. The expected upwind and source-splitting offset from
the continuum cell-center profile is
`q (dz/(2 U_b) + dt) = 3.5e-8 mol/m³`; each executable checks that bound
with 1% numerical allowance. In particular, an empty-column or zero-flux
"steady" result cannot pass these gates. The outlet rate uses the
transported state `c*`, before the current production increment.

## Transient bubble escape

The transient case starts uniformly at `c = 1e-5 mol/m³` and has zero
source. It runs to `t = 1 s`, with output at 0, 0.25, 0.5, 0.75, and 1 s.
The continuum solution is a bubble-free front translating upward at
`U_b`: `c = 0` below `z = U_b t` and the initial value above it. The
executable compares against exact cell averages of this front, with
normalized integrated absolute error below 0.12, allowing the documented
first-order numerical diffusion. By 1 s, escaped inventory must be
between 40% and 60% of the initial inventory, bracketing the exact 50%.

## Outputs and acceptance

Both solvers write `profiles.csv` for all 40 cells at every declared
physical output time and a global `history.csv` with inventory,
production, escaped moles, outlet molar rate, and convergence information.
Profiles include `c`, `N`, `alpha_g`, and independently computed hydrogen
and number conservation residuals. Hydrogen conservation means
`inventory + cumulative_escape - initial_inventory - cumulative_production`.

[`steady.json`](steady.json) and [`transient.json`](transient.json)
declare every required time, sample, quantity, tolerance, and conservation
gate. The shared comparator rejects missing, duplicated, stale,
non-finite, or mismatched rows; it does not interpolate between times.
It compares molar concentration, number density, and void point by point
with relative tolerance `2e-7`. The concentration absolute tolerance is
`2e-13 mol/m³`; the corresponding number and void tolerances are obtained
by dividing by `n_b` and multiplying by bubble volume, respectively
(`183867.7493 m⁻³` and `2.260870692e-16`). This gives all three equivalent
moments the same physical tolerance near the empty side of the front.
It independently requires each solver's global hydrogen residual to be
below `2e-13 mol` and number residual below `2e-8` relative. Solver exit
status also enforces analytic, conservation, boundedness, and steady or
transient behavior gates. This supplies an independent numerical
cross-check of the supported dispersed transport limit, not experimental
validation of a bubbly-flow closure.
