# Budgeted phase cleanup

`GeneralVoFOptions::phase_cleanup` is an opt-in numerical policy for Ordinary
`SweptPartition` transport. Its default `relative_cutoff = 0` disables cleanup.
It deliberately changes represented phase inventories; recording those changes
does not make the operation exactly conservative. Strict PLIC offset acceptance
and the transport, momentum, occupancy and physical drift tolerances remain
unchanged.

For the independently stored gas and carrier volumes, eligibility is
`0 < min(G,L) <= relative_cutoff * actual_geometry_volume`. The proposed operation
sets the smaller phase to zero and adds its old amount to the dominant phase
using binary64 arithmetic. It never replaces the dominant amount with the cell
volume. Every correction is the difference between the actual published before
and after values. If the dominant addition rounds away, its correction is zero.

The caller supplies two extensive `maximum_absolute_volume` budgets, in cubic
metres, for the entire run. A cutoff around `1e-14` is an experimental starting
point, not a library default or a universal resolution criterion. Budget choice
must be small relative to the application's permitted raw phase-inventory drift.
The full50 test explicitly selects `1e-14` and caps each cumulative phase repair
at `1e-13` times its initial global inventory. These controls belong to that test;
they do not alter other callers.

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
constructor together with the original options. The versioned finite codec
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
fields has no implemented contract. Solver regrid therefore remains unsupported
with cleanup enabled. Ordinary solver ownership-only repartition instead uses
phase-only migration and a separate keyed copy of accepted flow fields. It keeps
the cleanup checkpoint, current and cumulative cleanup momentum, and clock
unchanged; it neither performs a new cleanup transaction nor recovers velocity
by dividing transferred momentum by mass. Failed migration leaves the complete
accepted publication unchanged. See [measured live balancing](../architecture/general_vof_balancing.md).
BoundedCertification, Exact, DonorCell and GeometricFace do not support this
policy and reject an enabled cutoff.

The uncleaned microscopic fixed-normal PLIC problem may still reject. The
integration obligation is to clean an eligible amount within budget, publish a
pure cell, and avoid requesting that mixed-cell fit. Wider plane representations
remain separate robustness work.
