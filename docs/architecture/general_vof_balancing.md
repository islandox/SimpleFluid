# Measured General VoF balancing

`GeneralVoFOptions::record_balance_work` enables optional bounded work capture
for Ordinary `SweptPartition` interface-local transport. It defaults to false.
`GeneralVoFSolver::rebalance()` uses the exact accepted state's last captured
step to propose and, when admitted, apply an ownership-only partition change.
The physical clock does not advance during migration.

## Measurements and forecast authority

A prepared `GeneralVoFWorkSample` pins its phase publication, control, complete
hierarchy observation, communicator, arithmetic/backend, timestep and stage.
Owned key/revision rows include zero-work cells. Measurements distinguish:

- Actual owner reconstruction calls and branch counts after local/peer reuse.
- Canonical face setup and aggregation, assigned entirely to `face.first`.
- Receiver mixed-plane remeasurement and request/reply payload envelopes.
- Per-face donor dependencies, with exact rank-deduplicated remote fan-out.

Compute weights use elapsed seconds consistently. Geometry operation and fit
counters are explanatory observations, not interchangeable time units or
hardware FLOPs. Capture scopes exclude MPI waits and capture allocation work.
Storage is admitted against existing source, preparation, exchange and prepared
ledger allowances. Cache-admission failures make the observation unsuitable for
a unique-work model rather than inventing a lower cost.

`plan_partition_proposal()` separates historical observation from its target:

- `ObservedSource` requires the exact sampled phase publication.
- `AcceptedSuccessor` authenticates the factory's candidate against the exact
  current publication and retains the predecessor stamps unchanged. Geometry,
  partition, communicator and key/revision layout must still match. Costs and
  dependencies are a one-step-lag forecast, not measurements of current phases.

A currently inferred two-phase cell requires an observed mixed model. A prior
pure observation cannot supply zero mixed-cell cost. This conservative check can
reject an inferred cell whose retained geometry is pure. Unknown receiver work
on a newly remote mixed donor also rejects the proposal. Existing receivers keep
their measured costs; new receivers use the selected mean or maximum estimate.
The maximum is conservative among observations, not a future execution bound.

The solver adds its own checks: exact accepted State and last-ledger identity,
clock/stage/timestep, material density and source geometry. Phase identity alone
is insufficient because reinitialization can change flow fields without changing
phase identity. Reinitialization keeps its existing clock-reset semantics and
invalidates captured work.

## Proposal and cost scope

The existing sparse adaptive heuristic proposes destinations using reconstruction
plus face compute. A VoF-specific qualifier recomputes donor requester sets and
receiver load under those destinations. The full predicted maximum rank load
must improve by the requested threshold. Rejection preserves the current state;
there is no optimality promise and no case-specific destination assignment.

This version accepts compute seconds only; communication and migration objective
coefficients must be zero. It reports canonical cuts, distinct remote consumers
and request/reply payload bytes without inferring bandwidth from MPI polling.
Payload diagnostics exclude exchange peer framing and MPI internals.

Native migration is qualified using actual grouped native records, anchored
families and application framing. The nested `AdaptiveBalancePlan` belongs to
the hierarchy/native domain. Its exact byte count is not total VoF or solver
state migration traffic. Actual migration duration is reported separately.
Acceptance establishes forecast compute improvement. Callers must also assess
the remaining physical horizon and amortize migration against measured step-time
savings; a short run can become slower despite a better compute distribution.
The API performs one explicit operation and provides no automatic rebalance cadence.

## Live ownership migration

Ordinary `repartition()` and an accepted `rebalance()` prepare detached native
resources and phase-only migration, then transfer accepted solver state by
persistent cell keys and canonical face identity. Key/revision coverage and
phase amounts, histories, declarations, occupancy provenance and the cleanup
checkpoint must remain unchanged. Migration creates no new cleanup batch or
cleanup momentum.

Accepted cell velocity, pressure, curvature, fraction, density, finite mass and
error evidence are copied without momentum/mass or pressure-volume recovery.
Derived storage is independently checked. Canonical face velocities, wide rates,
radii and optional wall-pressure evidence retain their orientation and meaning.
The original physical momentum and divergence gates remain enforced.

Time, pressure-step diagnostics and current/cumulative cleanup momentum are
preserved. Capillary catalogs are rebuilt with target native IDs while accepted
cell curvature is retained. Geometry-dependent resources and operators are
rebuilt. Old closure scratch accounting is not treated as new allocation or
physical inventory.

Publication uses the existing nonthrowing State/Resources swaps only after all
collective checks pass, including on empty owners. Failure preserves exact old
state, fields, clock, cleanup and work-observation eligibility. Identity
repartition is a validated no-op. Successful migration clears obsolete prepared
transport/work; another accepted captured step is required before rebalance can
reuse measurements. Externally retained immutable ledgers remain historical
objects and cannot be rebound to the new partition. Borrowed solver references
and local-indexed caller data must be refreshed after successful mutation.

Cleanup-enabled coupled-channel remap and solver regrid remain rejected. The
new ownership-only path does not broaden Exact/Bounded or reacting/ALE support.

## Evaluation

The canonical full50 partition remains the kernel regression. Separate
`GeneralVoFPartitionExperiment` cases compare a capture-enabled canonical
control with a live rebalance after the first accepted step. Both continue the
same ten-step physical horizon, use the original cleanup policy/gates, and
report proposal/migration overhead inside the complete workflow time. Per-rank
elapsed durations are gathered after the horizon; no synchronized MPI clocks
are assumed. Post-balance work is reported separately from the first measured
step. Dated runs and profiles belong in local build artifacts.
