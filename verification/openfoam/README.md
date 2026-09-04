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

Load an OpenCFD OpenFOAM environment with `wmake`, `blockMesh`, and `checkMesh`
available, then run from the repository root:

```sh
# Example installation; use the path for your OpenFOAM installation.
. /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc

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

The shared [`compare_verification.py`](compare_verification.py) requires
explicit expected times, sample identifiers, quantity tolerances, and
conservation tolerances from the case manifest. It rejects missing, duplicate,
unexpected, or nonfinite data before comparing quantities. Steady convergence
is checked by the case drivers; transient comparisons cover the declared
history instead of selecting each solver's latest available output.

The focused SimpleFluid and comparator tests do not require OpenFOAM:

```sh
cmake --preset GCC-ninja-multi
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
