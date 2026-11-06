# Native steady-residual and face-flux diagnostics

These diagnostics measure the native discrete equations. They do not qualify
physical accuracy or set acceptance tolerances. They are independent of the
linear solver's convergence report and the steady-search field-update criterion.

## Library interface

`FluidSolver`, `IncompressibleIsothermalSolver`, and `BoussinesqSolver` expose
`evaluate_diagnostics(SolverDiagnosticsOptions) const`. Evaluation is collective,
uses independent fields/equation workspaces, and does not advance the solution.
Set `include_cell_data` / `include_face_data` to retain owned-entity arrays.
An optional `pressure_coefficient_time_step` reconstructs pressure-weighted flux
with that value divided by reference density. Zero is permitted for a frozen
diagnostic control; it never changes the production timestep or flux policy.
The result records whether transport uses retained or reconstructed flux.

`enable_diagnostic_history(true)` enables accepted pressure-update/correction
history before stepping. It is off by default. A loaded snapshot cannot recover
historical flux or pressure corrections; those quantities are explicitly absent.
No zero-filled field is presented as an authenticated historical flux.

Steady assembly explicitly omits the storage term. It uses the same spatial
operators, boundary conditions, sources, stress treatment and model coefficients
as production. Turbulence diagnostics rebuild coefficients and wall data in a
scratch model for every closure. They do not reuse the last linear solve's stale
source fields. Retained versus rebuilt eddy viscosity is reported separately.

The residual is `A(q) q - b(q)`. Reports contain the signed integrated residual,
volume-weighted RMS and maximum residual density. The normalization denominator
is the volume-weighted RMS of `(sum_j |A_ij q_j| + |b_i|)/V_i` and is explicitly
labelled **algebraic_absolute_row_terms**. It is not a universal physical-error
scale or a permissible comparison tolerance. Zero/zero gives zero; a nonzero
residual with zero denominator is unavailable. Wall-cell constraints are excluded
from PDE norms and reported as `q - q_imposed`, in transported-field units.

Momentum residual density has units m/s²; k has m²/s³, epsilon m²/s⁴ and omega
1/s². Physical thermal assembly reports W/m³ and simple thermal assembly K/s.
Continuity uses integrated volume rate per cell, divided by cell volume (1/s),
against the configured volume-continuity source. Pressure history L2 norms are
unweighted discrete field norms, not residual-based convergence criteria.

For moving meshes the result identifies frozen geometry, retained mesh-relative
flux and available geometric-conservation/storage evidence. Missing history is
reported as unavailable. An instantaneous spatial imbalance in an unsteady or
SAS flow is not evidence that its transient solve failed. SAS retains its physical
timestep and source-cap semantics during evaluation.

## pitzDaily replay and short continuations

The pitzDaily example supports these opt-in environment variables:

| Variable | Meaning |
|---|---|
| `SIMPLEFLUID_PITZ_REPLAY_PREFIX` | Prefix of original `*_rankN.csv` cell files |
| `SIMPLEFLUID_PITZ_REPLAY_RANKS` | Original file count; default 3 |
| `SIMPLEFLUID_PITZ_DIAGNOSTICS_PREFIX` | Output prefix; absent disables runtime sampling/history |
| `SIMPLEFLUID_PITZ_DIAGNOSTIC_ONLY` | Replay and evaluate without a solver step |
| `SIMPLEFLUID_PITZ_DIAGNOSTIC_COEFFICIENT_DT` | Nonnegative frozen coefficient override |
| `SIMPLEFLUID_PITZ_DIAGNOSTICS_INTERVAL` | Accepted-step sampling interval; default 10 |
| `SIMPLEFLUID_PITZ_DIAGNOSTIC_FIELDS` | Export owned cell/face CSV arrays; default false |

The loader requires finite complete U,p,k,epsilon,nu_t and a unique bijection to
the configured mesh's cell centres. The 1e-12 m coordinate-index resolution is a
geometry identity check, not a physical tolerance. No interpolation or field
clipping is performed. Existing turbulence floor validation rejects invalid input.
File ranks may differ from replay ranks. Replay elapsed time starts at zero;
the campaign receipt separately records the original endpoint time and source.

JSON summaries are rank-zero files; CSV arrays are rank local. Per-step clipping
statistics are separated by k and epsilon because their units differ. Histories
and data products belong outside the source checkout.

The runner authenticates all four retained endpoints, refuses duplicate output
directories, records exact commands/settings and actual executable/library process
mappings, and never relaxes convergence controls after failure. The continuation
uses the endpoint's final timestep, not its original startup timestep. Its fixed
200-step budget is diagnostic evidence, not a renewed stationarity qualification.

Example (run in WSL with the usual SimpleFluid build environment):

```sh
cd /home/hbyin/.codex/worktrees/pitzdaily-evidence/SimpleFluid
. verification/environments.sh
export_build_env "$PWD"
cmake --build build/gcc --config RelWithDebInfo --target pitz_daily -j 2
python verification/openfoam/pitzDaily/run_native_diagnostics.py \
  --campaign ../qualification-target48900-20261022 \
  --output ../native-diagnostics-20261024 \
  --executable build/gcc/bin/RelWithDebInfo/pitz_daily --mode replay
python verification/openfoam/pitzDaily/run_native_diagnostics.py \
  --campaign ../qualification-target48900-20261022 \
  --output ../native-diagnostics-20261024 \
  --executable build/gcc/bin/RelWithDebInfo/pitz_daily --mode continue --steps 200
```

For profile comparisons, export the trusted existing sampling infrastructure from
`codex/pitzdaily-evidence` outside the checkout using `git archive`, load
OpenFOAM.com v2606, then use `--mode report --sampling-tools <exported-pitzDaily-dir>
--reference <reviewed-openfoam-case>`. This calls the existing common cellPoint
sampler and profile comparator; it does not introduce another sampling method.
The runner's `--mode verify` performs authentication without executing a solver.

`physical_acceptance.json` remains fail-closed and pending. Diagnostic residuals,
coefficient sensitivity and profile errors do not by themselves justify tolerances.


Compact input archives may omit duplicate VTU/sampling outputs. They retain the
original checksum index as `original-checksums.json`, its SHA-256 in
`input_scope.json` (`native-replay-input-subset-v1`), and an index of included
original files as `checksums.json`. The collector requires every included digest
to match the original index and still requires the complete native restart and
configuration/provenance files. Without this explicit scope, every original
indexed file remains mandatory. This packaging scope does not grant qualification.

## Pressure flux on nonorthogonal faces

For an interior face let `S` be the oriented area vector, `d` the centre-to-centre
displacement, `alpha` the pressure matrix's two-point diffusion weight and
`g` the interpolated cell pressure gradient. The integrated pressure derivative
is `alpha * (p_N - p_P) + g · (S - alpha*d)`. Subtracting `g · S` gives the
stabilization derivative `alpha * ((p_N - p_P) - g · d)`. An affine pressure
with its exact gradient therefore contributes zero stabilization, including
on a skewed face. A pressure jump unresolved by the cell gradients still
contributes. An adjustable Dirichlet face uses its boundary pressure, the
cell-to-face displacement, the owner gradient and the boundary diffusion weight.

The segregated projection keeps the two-point pressure matrix and iterates
the remaining face term until the full pressure-equation residual meets the
configured tolerance. Failure to converge is reported. Coupled continuity
assembly uses the projected displacement for the pressure-gradient contribution;
velocity flux continues to use the full face area. The historical 48,900-cell
results predate this operator change. They cannot qualify the corrected scheme.
