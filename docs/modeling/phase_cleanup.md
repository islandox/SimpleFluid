# Budgeted phase cleanup

`GeneralVoFOptions::phase_cleanup` defaults to enabled for Ordinary
`SweptPartition` and `GeometricFace` transport. Set `relative_cutoff = 0`
explicitly to disable cleanup. `DonorCell` retains a disabled automatic default
and rejects an explicit positive cutoff.
It deliberately changes represented phase inventories; recording those changes
does not make the operation exactly conservative. Strict PLIC offset acceptance
and the transport, momentum, occupancy and physical drift tolerances remain
unchanged.

For a cell with inferred mixed geometry, eligibility is
`0 < min(G,L) <= relative_cutoff * actual_geometry_volume` for the independently
stored gas and carrier volumes. Already validated `PureGas`/`PureCarrier`
declarations retain their independent minority inventories: their geometry needs
no relabeling, so they create no cleanup event or budget debit, including when a
pinned phase cap is zero. This protection also applies with an explicit positive
cleanup cutoff and cap. It does not admit a contradictory pure declaration or
change the budget rejection for an eligible inferred mixed cell. The proposed operation
sets the smaller phase to zero and adds its old amount to the dominant phase
using binary64 arithmetic. It never replaces the dominant amount with the cell
volume. Every correction is the difference between the actual published before
and after values. If the dominant addition rounds away, its correction is zero.

The enabled automatic cutoff is `1e-14`. Each automatic extensive `maximum_absolute_volume`
cap is `1e-13` times that phase's initial global inventory, before any cleanup.
These are cumulative run budgets in cubic metres, derived conservatively from
owned inputs only. The policy is a numerical control, not a universal resolution
criterion; applications can override the cutoff and either absolute cap.

Unconstructed options use `GeneralVoFPhaseCleanupOptions::automatic` (`-1`)
independently for the cutoff and each cap. Construction agrees these requests
collectively and resolves them once. `GeneralVoF::options()` and cleanup
checkpoints contain concrete values. An explicit cap of zero permits no spending
for that phase; it does not request automatic sizing. No positive budget floor is
introduced for a globally absent or dilute phase. In particular, deleting an
entire nonzero dilute phase can exceed its relative global allowance and reject.

The budget bounds direct inventory corrections. It does not bound the change in
future fluxes or interface position caused by pure/mixed reclassification;
accuracy and convergence still require their own physical checks.

## Accounting and admission

`phase_cleanup_state()` exposes replicated global totals:

| Field | Meaning |
| --- | --- |
| `signed_volume[p]` | Accumulated signed physical volume correction |
| `absolute_volume[p]` | Conservative upper debit for the sum of absolute actual corrections |
| `largest_absolute_volume[p]` | Largest absolute correction to one cell |
| `cleaned_cells` | Cumulative number of cell cleanup events, including repeated repairs |
| `policy`, `policy_bound` | Original cutoff and absolute caps bound to this run |

The absolute accumulator rounds upward when a positive addition would otherwise
underestimate spending; tiny additions cannot disappear from budget admission.
Opposite signed corrections do not refund this debit. Totals are finite numerical
records, not exact-arithmetic certificates. `latest_phase_cleanup()` retains the
local owned-cell keys, native GIDs, before/after endpoints and signed actual
deltas for the latest publication. Ghost cells never create another charge.

All owners propose a batch collectively against one global remaining allowance,
including empty owners. If either phase exceeds its remaining allowance, the
entire trial rejects. There is no rank-local allowance and no partial selection
based on rank or cell traversal order. A candidate carries tentative spending;
only accepting that candidate advances the accepted history. Abandoned and
rejected trials leave accepted phases, time and spending unchanged.

Raw transport positivity, signed balance, closure and independent global phase
balance are checked before cleanup. Cleanup then rechecks the original occupancy
admission, classifies cleaned cells as pure and publishes owner state to ghosts.
It does not inflate `PhaseRoundoff`, phase history or occupancy error allowances.
Raw inventory/mass diagnostics include the actual cleanup effect. Applications
must continue to enforce their original cumulative physical drift gates; the
cleanup ledger explains an intentional contribution to that drift, not a
replacement conservation test. When cleaning initial input, retain the original
input inventory as the physical drift reference, rather than silently choosing a
post-cleanup reference.

## Mass and momentum

The prepared ledger keeps raw and cleaned endpoint capacities separately, along
with per-cell phase corrections and
`cleanup_mass = rho_g * delta_G + rho_l * delta_L`. It derives raw capacity from
the retained pre-cleanup endpoints, avoiding subtraction of a tiny correction
from a rounded dominant mass. Phase validation checks the raw receipts against
those raw endpoints independently of the declared cleanup.

Momentum transport first solves the original equation using raw endpoint mass.
Cleanup then preserves that transported cell velocity and records the local
momentum change `delta_p = cleanup_mass * transported_velocity`. Subsequent
density, force and pressure operations use cleaned mass. The prepared
`cleanup_momentum` and solver's signed/cumulative momentum diagnostics expose
this physical change separately from transport residual and arithmetic error.
The raw transport residual plus the cleaned/raw momentum identity residual must
fit the original momentum tolerance. No cleanup magnitude is added to its
acceptance allowance.

## Initialization, continuation and remap

Construction applies the same owner-only admission before the first phase halo
publication. The seven-argument `GeneralVoF` constructor accepts a prior
`GeneralVoFPhaseCleanupState` before any new initialization cleanup. Save
`phase_cleanup_checkpoint()` with the accepted phase checkpoint, decode it with
`decode_phase_cleanup_checkpoint()`, and pass the recovered state to that
constructor together with compatible options. Automatic fields inherit
the checkpoint's bound values, including a previously disabled policy; caps are
never recalculated from restarted or remapped inventories. Explicit conflicting
values reject. The versioned finite codec
retains cumulative spending and the bound policy; malformed data, policy changes
and rank disagreement reject. Resuming with a fresh empty ledger would start a
new allowance and is not a valid continuation of an existing run.

This checkpoint serializes cleanup accounting only. It is not a complete solver
restart format: the application still owns restart geometry, accepted phases,
declarations, histories, clock and other model state.

Phase-only finite remap inherits the original ledger and can admit further
cleanup on the new candidate after raw remap checks. Ownership migration retains
spending without charging ghosts or copied inventories. Cleanup-enabled remaps
with coupled channels reject explicitly: correction of transported auxiliary
fields has no implemented contract. Solver regrid therefore requires explicit
cleanup opt-out on the geometry backends. Ordinary solver ownership-only
repartition instead uses phase-only migration and a separate keyed copy of
accepted flow fields. It keeps the cleanup checkpoint, current and cumulative
cleanup momentum, and clock unchanged; it neither performs a new cleanup
transaction nor recovers velocity by dividing transferred momentum by mass.
Failed migration leaves the complete accepted publication unchanged. See
[measured live balancing](../architecture/general_vof_balancing.md).

BoundedCertification, Exact and DonorCell retain disabled defaults and reject an
explicit positive cutoff. Disabled automatic policies resolve their caps to zero.
`GeometricFace` retains its configured identity and `AlgebraicFaceVolume`
prepared family when cleanup is enabled. Its final canonical phase streams remain
the raw transport authority; cleanup is a separately recorded correction after
raw acceptance.

The uncleaned microscopic fixed-normal PLIC problem may still reject. For the
enabled policy, eligible cleanup must remain within budget and publish its
result before any subsequent mixed-cell fit; budget exhaustion still rejects.
Successful microscopic zero-cutoff completion and wider plane representations
are optional robustness work. Preserve the captured inputs/local replay and
collective rejection without changing accepted state or calling the failed
physical horizon passed.

## Qualification and sensitivity

Default-policy qualification uses Ordinary SweptPartition with automatic cleanup;
GeometricFace's enabled policy requires its own evidence. Record automatic versus
explicit configuration and the resolved cutoff/caps. The current named swept
full50 uses automatic defaults, while controlled algebraic fixtures usually set
zero explicitly. Their recorded outcomes are not interchangeable; see
[regimes and evidence](../verification/vof_solver_contracts.md#stage-7-scope-and-priorities).

Cleanup sensitivity is a core acceptance task: compare matched complete mesh/time
studies under declared cutoff and cumulative-cap variations, including zero where
admissible. Keep the original physical initial inventory, oracle, endpoints,
tolerances and resource limits. Report raw physical drift, signed/absolute phase
corrections, mass/momentum changes, event counts, exhausted budgets and interface
accuracy/convergence. Report a strict-policy rejection as such; sensitivity does
not require successful reconstruction of every microscopic zero-policy state.
Small direct cleanup volume does not prove small future trajectory or flux error.

Cleanup-enabled coupled AMR/remap is an explicit capability extension. It needs
an auxiliary-channel correction/transfer contract, preserved original caps and
spending, independent material/momentum checks and atomic serial/MPI rollback.
Existing phase-only remap and ownership migration do not close this gate. Until
it is implemented and qualified, preserve coupled-channel rejection and label
raw-policy Q5 results separately. Complete solver restart remains separate work.
