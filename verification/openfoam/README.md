# OpenFOAM comparisons

These workflows run matched SimpleFluid and OpenFOAM cases and compare their
outputs. Each case documents its equations, physical parameters, boundaries,
time discretization, acceptance tolerances, and model limitations.

The default water-case domains are ten times larger in x, y and z, with
original cell widths retained. See [mesh sizes and scaling](BOUNDARY_LAYER_MESHES.md)
and [focused validation](SCALED_CASES.md)
for the new 40,400-cell bubble, 8,400-cell ALE and 790,560-cell convection
fixtures. Historical performance and convergence tables describe scale-1
inputs and do not establish performance or mesh independence at the new size.

## Linear solver experiments

The three water verification executables accept optional solver overrides.
For example, after building the Release targets with IF97 enabled:

```sh
build/gcc/bin/Release/bottom_heated_bubbly_convection \
  --output /tmp/convection-pcg-dic-sgs \
  --pressure-solver pcg --pressure-preconditioner dic \
  --transport-solver bicgstab --transport-preconditioner sgs
```

The same options apply to `planar_ale_comparison`. The prescribed-flow
`dispersed_bubble_verification` accepts only the two transport options.
Omitting overrides retains the existing solver choices and tolerances.
`gs` selects one forward Gauss-Seidel preconditioning sweep; `sgs` selects
one symmetric sweep. Both use Ifpack2 relaxation with unit damping and zero
initial preconditioner output. These are preconditioners for BiCGStab, not
standalone smoothing solvers.

`pcg` is an alias for the existing CG backend. The pressure equation clears
off-diagonal gauge-row and gauge-column entries for CG, retaining the gauge
diagonal, neighbor diagonals and prescribed zero gauge value. DIC uses a
diagonal incomplete-Cholesky recurrence with unchanged off-diagonal factors.
It supports serial and distributed symmetric matrices with positive finite
pivots. Under MPI, GS/SGS use local sweeps with Jacobi coupling between
ranks. Forward GS, ILU0 and ILUT are rejected as CG preconditioners because
they do not generally preserve the required symmetry.

Each SimpleFluid executable writes `linear_solver_statistics.csv`, containing
per-step flow aggregate and gas solve/iteration counts and maximum achieved
true relative residuals. Flow counts combine momentum, pressure and thermal
solves. BiCGStab sums iterations from independently solved RHS columns;
other multi-RHS counts follow the selected Belos manager's semantics.
They are diagnostics rather than equivalent units of work across
different algorithms. The physical history and field comparison gates are
unchanged, and timing a failed solve does not establish a speedup.

The BiCGStab wrapper uses contiguous one-column buffers for multiple RHSs.
This avoids an installed Trilinos 17.2 nested-subview indexing defect exposed
when deflation removes the zero middle velocity component of a 2D case.
Each RHS retains the requested tolerance and iteration limit, and the
unchanged preconditioner is reused across columns within one solve call.

### Distributed DIC

`LinearPreconditioner::DIC` uses the matrix's Tpetra maps and Teuchos
communicator. Each rank retains its owned factor rows and column-map halo.
All cross-rank coefficients contribute to the diagonal recurrence and both
triangular sweeps; Tpetra imports exchange dependency values. A distributed
sparse transpose validates symmetry and supplies conjugate backward-factor
coefficients. Setup and application reject invalid rank-local inputs
collectively, including inconsistent maps, vector counts and zero-alpha
branches. Empty ranks and subcommunicators are supported.

Distributed factorization orders rows by ascending global ID, independent of
local row permutations. Serial factorization retains its existing local row
order. Thus the factors agree across decompositions when the serial map is
ordered by global ID. Each communication stage completes all locally ready
rows before exchanging remote dependencies. Factor setup records this
schedule for reuse by Belos PCG and subsequent preconditioner applications.

Communication cost depends on the partition and ordering: a contiguous
partition of a chain needs at most one stage per rank, while interleaved
ownership can require a stage per row. Each apply performs one forward and
one reverse halo exchange between stages. This is a dependency-ordered
preconditioner; MPI support does not imply efficient scaling for every mesh.
The comparison executables below support MPI and coordinate-based owned output. Distributed DIC itself is tested through Tpetra/Belos solves and
the production pressure-projection path on multiple ranks.

## Dispersed bubbles and planar ALE

For a bottom-driven flow with **solved liquid convection**, use
[`bottomHeatedBubblyConvection`](bottomHeatedBubblyConvection/README.md).
It adds localized heat and H2 production, gravity, cold walls, gas escape,
and a 20-second plume/return-flow comparison with two-dimensional field figures.
The fixed mesh accommodates physics currently rejected by the narrow ALE path.

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

Each comparison also writes a `figures/index.html` gallery with temperature,
liquid-density, gas-volume-fraction, and velocity distributions and relative
errors. SVG output needs only Python; PNG/PDF exports use `rsvg-convert` when
available. See [field figures and error definitions](WATER_FIELD_FIGURES.md)
for spatial/time histories, zero-reference handling, and velocity scope.
The cases use matched [boundary-layer meshes](BOUNDARY_LAYER_MESHES.md), with
native-factory coordinates shared by both solvers and a finer convection mesh
for sensitivity checks. Run-specific manifests retain the selected grid.

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
