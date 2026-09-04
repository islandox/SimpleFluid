# OpenFOAM comparisons

These workflows run matched SimpleFluid and OpenFOAM cases and compare their
outputs. Each case documents its equations, physical parameters, boundaries,
time discretization, acceptance tolerances, and model limitations.

## Dispersed bubbles and planar ALE

| Model | Steady verification | Transient verification |
| --- | --- | --- |
| [Dispersed bubbles](dispersedBubbleFlow/README.md) | Uniform microbubble production balanced by top escape | Clearance of an initial microbubble inventory |
| [Planar ALE](planarALE/README.md) | Source-off equilibrium after thermal expansion | Uniform heating with conservative moving-mesh energy transport |

The bubble cases exercise SimpleFluid's production population transport with
prescribed rise speed. The ALE cases exercise its constrained, laminar,
constant-area free-surface solver. The OpenFOAM applications implement the
shared reduced equations using OpenFOAM finite-volume operators. Their source
is included with the cases: stock Euler-Euler or VOF solvers solve different
models and would require a different comparison contract. Agreement here is
numerical verification of the stated limits, not experimental validation of
turbulent bubbly flow or general free-surface dynamics.

Both case families use liquid-water reference properties from the optional
[`SimpleFluid::IF97` library](../../docs/modeling/if97_water.md), evaluated at
300 K and 101325 Pa **absolute**. The shared
[`reference_water.properties`](reference_water.properties) records the same
SI material inputs for OpenFOAM. SimpleFluid queries the enabled library and
checks that snapshot before advancing. Bubble surface tension also comes from
IF97; radiolytic hydrogen remains a noncondensable gas. The ALE case uses
IF97-derived constant reference coefficients and built-in linear Boussinesq
density feedback, not a nonlinear IF97 material callback.

The shared state has density `996.5580761 kg/m³`, heat capacity
`4181.097327 J/(kg K)`, dynamic viscosity `8.537422562e-4 Pa s`, conductivity
`0.6095012841 W/(m K)`, and expansion coefficient `2.743751849e-4 K⁻¹`.
Both implementations derive `nu = mu/rho` and `alpha = k/(rho cp)` from
the applied coefficients. Full-precision values and backend provenance are
kept in the snapshot. Its repeatable generator is
[`export_reference_water.cc`](export_reference_water.cc), linked only to
the IF97 material library; it exports no flow or temperature solution.

Load an OpenCFD OpenFOAM environment with `wmake`, `blockMesh`, and `checkMesh`
available, then run from the repository root:

```sh
# Example installation; use the path for your OpenFOAM installation.
. /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_IF97=ON

verification/openfoam/dispersedBubbleFlow/run_comparison.sh steady
verification/openfoam/dispersedBubbleFlow/run_comparison.sh transient
verification/openfoam/planarALE/run_comparison.sh steady
verification/openfoam/planarALE/run_comparison.sh transient
```

The launchers use `verification/environments.sh` to build SimpleFluid. The
default is GCC RelWithDebInfo; select Debug with
`SIMPLEFLUID_BUILD_CONFIG=Debug`, or use `SIMPLEFLUID_BUILD_DIR` for an already
configured custom build tree. These small comparison cases run in serial.
An optional second argument selects the output directory; consult each case
README for the generated files and reference-application build details.
`SIMPLEFLUID_ENABLE_IF97` remains OFF by default. In that configuration the
two water-case executables exit with an explicit enable-IF97 diagnostic,
and their solver CTests are not registered. No synthetic fluid is substituted.

The shared [`compare_verification.py`](compare_verification.py) requires
explicit expected times, sample identifiers, quantity tolerances, and
conservation tolerances from the case manifest. It rejects missing, duplicate,
unexpected, or nonfinite data before comparing quantities. Steady convergence
is checked by the case drivers; transient comparisons cover the declared
history instead of selecting each solver's latest available output.

The focused SimpleFluid and comparator tests do not require OpenFOAM:

```sh
cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_IF97=ON
cmake --build --preset GCC-Debug --target dispersed_bubble_verification planar_ale_comparison
ctest --test-dir build/gcc -C Debug --output-on-failure \
  -R '^(dispersed_bubble_verification_(steady|transient)|planar_ale_comparison_(steady|transient)|openfoam_verification_comparator)$'
```

Running these CTests alone does not establish OpenFOAM agreement. Use the four
comparison launchers above for that check; each returns failure if either
solver or the comparison fails.

## Other cases

- [Cavity flow](cavityFlow/README.md)
- [pitzDaily](pitzDaily/README.md)
- [Natural convection](naturalConvection/README.md)
- [Fissile solution tank](fissileSolutionTank/README.md)

These older workflows have their own convergence and interpretation limits.
