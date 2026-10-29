# Conservative face transport for general-native VoF

The [shared transport contract](general_vof_transport.md) classifies both backends
below as algebraic face-volume transport and distinguishes them from geometric
swept-tetrahedra transport. A geometric PLIC candidate does not change that family.
The Ordinary swept backend also reuses the donor-cell stream/publication provider
for its authenticated interface-free specialization; its configured backend stays
`SweptPartition` and its prepared family records the algebraic execution. The
interface-local swept family also uses this donor-cell core as an admissible
low-order baseline and numerical support; its final signed ledger has the separate
publication and momentum contract described in the shared guide.

The opt-in `GeneralVoFTransportBackend::DonorCell` and `GeometricFace` backends
advance phase inventories on a fixed native mesh without swept solids,
donor/recipient overlays, or a complete geometric partition of each donor.
`SweptPartition` remains the default. `ArithmeticMode` remains an independent
selector; the new backends currently admit only `Ordinary` arithmetic.

The initial implementation supports source-free incompressible flow, constant
positive phase densities, admitted convex native cells, impermeable boundaries,
and synchronized AMR stepping. A remap is a separate fixed-clock operation and
retains its existing overlap geometry. Periodic images, open/inflow boundaries,
ALE within a substep, sources, level subcycling, and counterflow on one canonical
face are explicitly outside this backend. Nonzero exterior flux rejects.

## One prepared physical transport

`GeneralVoF::plan_face_transport` returns a detached, source-pinned `Trial`.
Its immutable `GeneralVoFPreparedTransport` records the source identity/control,
mesh partition generation, transaction, timestep, caller time-stage identity,
densities, canonical directed streams, owned old/new mixture-mass capacities,
and local diagnostics. The flow solver uses the accepted time's binary
representation as its stage identity. `GeneralVoFSolver::prepared_transport()`
exposes the last accepted preparation, including its immutable momentum operator
identity and component preparation/reuse counts.

Each nonzero stream has an incident donor and recipient and nonnegative
integrated volume `q`. It stores independently evaluated gas and liquid amounts,
with the larger flux obtained from `q` minus the smaller flux. Its mixture mass
is `rho_g*gas + rho_l*liquid`. These are integrated amounts, not rates. A vector
of streams allows a future multiple-stream contract; current input admits exactly
one direction per canonical fine face/substep. It represents a constant,
spatially uniform normal velocity on that face. A net flux with hidden reversal
or counterflow cannot be passed under this model. The node-trajectory transport
overload rejects a face backend rather than discarding those trajectory controls.

The first-cell owner authenticates shared controls and distributes one stream
record. Endpoint owners consume opposite incidence. Candidate planes are prepared
on donor owners, and endpoint owners supply limiter capacities to the canonical
face owner. Communication uses bounded sparse owner packets and retained native
adjacency; there are no all-retained-donor geometric searches. Coarse/fine faces
are the existing canonical fine tiles, without a second parent-face transfer.

Gas and liquid accepted inventories are always stored independently. Recipient
publication sums independent retained phase residuals and incoming final streams.
It never computes a minority inventory from cell volume minus the dominant phase.
The positive retained/incoming decomposition is also the sole momentum transfer
ledger; legacy geometric routes are not combined with limited physical streams.

Ordinary `GeometricFace` also supports
[budgeted phase cleanup](../modeling/phase_cleanup.md). It runs after raw transport
acceptance and records actual endpoint changes separately. The original physical
drift gates include those changes. Cleanup defaults to enabled for `GeometricFace`.
`DonorCell` retains raw algebraic inventories, defaults to disabled cleanup and
rejects an explicit positive cutoff. Enabling cleanup does not change the
configured backend or prepared transport family.
Momentum first uses raw endpoint mass, then records the cleanup mass correction
at the transported velocity. Ownership-only solver migration preserves cleanup
state and accepted flow fields. Cleanup-enabled coupled AMR/remap remains
unsupported.

## Admissible donor-cell baseline

For donor inventories `G,L`, the smaller low-order flux is measured directly as
`q*min(G,L)/(G+L)`; the other flux completes the prescribed bulk amount. This
normalizes the *stream composition*, not accepted cell inventories. The old
occupancy discrepancy stays in that donor's independent retained residuals.
Its supporting occupancy allowance therefore stays local as well.

Admission checks outgoing bulk volume against both geometric volume and `G+L`,
independent old phase exports against `G,L`, and canonical bulk continuity
against the existing volume tolerance. Continuity and occupancy have separate
diagnostics and gates. Invalid low-order data reject before geometric candidates
are attempted. No inventory clipping, phase source, or fraction repair is used.

Publication also enforces a fixed cumulative occupancy gate. With
`delta = G + L - geometric_volume`, the candidate must satisfy
`delta_new = delta_old - signed_canonical_bulk_transfer + publication_error`.
Admission uses the existing signed discrepancy and an independently calculated
publication uncertainty; it does not grant a fresh physical allowance each step.
Accumulated arithmetic provenance remains recorded separately and cannot enlarge
the fixed physical limit of `1e-13` times geometric cell volume. This physical
limit is separate from the PLIC target-volume tolerance and the per-increment
canonical continuity tolerance. Infeasible candidates reject before publication,
including when each individual transfer increment would pass by itself.
The nominal wide geometry, native bulk volume and actual binary64 published
sum each retain their own unchanged physical check. A conversion discrepancy
or rounding of the public sum cannot borrow the other reference's allowance.

Pure geometric hints never establish physical absence. A hinted pure cell with
a positive minority trace transports that trace using its physical composition.
Pure cells require no PLIC cut, swept solid, or overlay. Opposite pure neighbors
exchange phase through the same stream rule; a receiving pure cell can become
mixed. Pure hints survive only compatible incoming geometry, with exact absence
providing a fresh pure hint.

## Measure-only PLIC and geometric candidates

`PLIC::OrdinaryPlaneMeasure` caches the original cell boundary in translated,
scaled local coordinates. Volume queries clip individual original face loops
and integrate signed tetrahedral contributions using a reference on the cutting
plane. The cap has zero contribution, so no closed clipped polyhedron is built
or validated. Optional section area supplies the offset derivative. The status
and `numerical_error_estimate` are numerical quality indicators, not certified
bounds.

`PolyhedralReconstructionOptions::measure_only` is explicit and defaults to
false. `GeometricFace` enables it. The existing normal estimator and angular
objective remain; offset positioning uses a bracketed Newton/bisection solve,
evaluates the smaller cell phase directly, and rejects when the requested
accuracy is below its measurement or binary64 plane-representation floor.
BoundedCertification and Exact retain their separate algorithms. Existing
`ConvexCellGeometry::phases()` and explicit surface extraction remain available
for remapping and surface-sensitive consumers.

A geometric candidate integrates the covered area of the canonical face while
translating the donor plane with a locally reconstructed displacement. For each
donor, an area-weighted least-squares fit uses all authenticated incident normal
controls, including zero wall controls, to estimate the complete displacement.
The face predictor keeps its tangential components and replaces its normal
component with the prescribed signed `q/area`. Thus constant vector motion is
reproduced, one-dimensional crossings retain their authoritative displacement,
and tangential motion contributes to an oblique interface's face coverage.
The fit rejects deficient or nonfinite geometry and conditioning beyond
`reconstruction.maximum_condition`; these support failures do not select the
arithmetic fallback. Its temporary cache is charged to the existing byte budget.
For spatially varying flow this is a local predictor, not an exact trajectory or
a claim of second-order accuracy.

A strict swept-face extremum proof handles wholly covered faces first. It uses
compensated signs at every original native vertex and outward endpoint bounds
covering both wide plane translation and represented sample offsets. Only
unanimous strict signs establish an empty phase and assign the full phase its
canonical bulk amount. This avoids comparing two rounded area representations
for a geometrically full face. Contacts, crossings and unresolved signs continue
through measurement; no positive measured receipt is clipped. The proof does
not establish arbitrary sub-ULP crossing accuracy for the area integrator.

Otherwise both phase-covered areas are measured directly: the cell minority
need not be the outflow-face minority. Time intervals
are split at original face-vertex crossings. Two interior Gauss points integrate
the quadratic area on each interval and avoid assigning finite weight to a
face-parallel contact instant. Bulk `q` remains authoritative.

Only `GeometryErrorCode::UnresolvedArithmetic` permits fallback to the donor-cell
candidate. Affected canonical streams and bulk volume are counted. Invalid
geometry, stale ownership, insufficient physical reconstruction support, and
resource exhaustion retain their failures. A fallback is a loss of local order,
not evidence of geometric or physical qualification.

## Shared limiting and compatible histories

The final represented streams must still satisfy both old-phase export budgets.
If a donor fails only by a witnessed arithmetic discrepancy, a bounded local
replay may reduce coefficients on its adverse faces. The replay uses exactly
the production convex blend, minority preservation and bulk completion. It
retains only candidates passing both strict donor budgets and bounds the sum of
absolute per-face changes independently for gas and carrier by pointwise
operation-error evidence. It never publishes a newly zero coefficient or erases
a previously positive phase transfer, changes inventories, or damps unrelated
donors. The search has one
precision-derived attempt bound; it need not find a feasible correction or the
globally largest coefficient. Failure remains collective rejection.

Donor proposals are authenticated and completed at canonical face owners, which
re-evaluate the same blend and mass before redistribution. All original final
admission checks still run. This scalar arithmetic witness does not select
BoundedCertification or certify geometric accuracy; the actual final streams
remain the sole phase, mass and momentum authority.

The limiter blends low and high phase candidates with one `theta` per canonical
stream. It budgets all corrections before choosing coefficients. Endpoint gas
and liquid capacities constrain decreases from the admissible low-order update.
Two additional donor budgets constrain total explicit old gas/liquid export.
These prevent unsafe export of old concentrations or histories even when final
cell positivity alone would permit incoming material to finance an export.

Both phase endpoints are blended independently using a convex expression;
`theta=0` and `theta=1` preserve those endpoints exactly. This avoids cancelling
a tiny high-order minority in `low + (high-low)`. The smaller final flux is
retained directly and the larger completes `q`. Final donor exports are checked
again before publication. Independent phase error histories follow each final
phase transfer in proportion to the donor's old phase inventory, with outward
rounding for history accounting. Explicit tracer channels must use these same
phase transfers; this source-free solver does not introduce new species physics.

## Momentum preparation and atomic acceptance

The solver derives mass transfers only from final limited phases and verifies
prepared mass capacities against the phase-storage evaluation. The implicit
upwind equation is

\[
 (M^{n+1}+\sum F_{\mathrm{out}})u^{n+1}
 -\sum F_{\mathrm{in}}u^{n+1}_{\mathrm{donor}}
 =M^n u^n + S.
\]

Constant velocity follows from `A*1=M_old` when the shared mass ledger closes.
The finite transport workspace caches immutable coefficients, routed physical
rows, donor maps/import, matrix, and applicable preconditioner state. All ranks
agree on reuse. Matching three component solves build at most once; each RHS
still has fresh input validation, physical residuals, iteration/refinement gates,
and convergence evidence. A change to any numerical flux/capacity/control key
invalidates reuse. A changed advecting interval must be replanned from accepted
old phases; this solver does not mutate an accepted trial in place.

The solver coordinator publishes phases, velocity, pressure, time, diagnostics,
and the attached preparation only after the entire step succeeds. Failed phase,
momentum, or projection work leaves the accepted state untouched. Fixed-clock
remap/repartition creates a new mesh-bound workspace and does not remap a cached
transport operator as physical state.

Ordinary algebraic projection also checks transport readiness. A physical
pressure residual alone can leave a rate defect too large for the next phase
volume budget. With tolerance `tau`, Courant control `C`, signed incident rate
sum `net` and absolute incident rate sum `S`, the existing pressure correction
loop requires either `abs(net)/S <= tau/(2*C)` or
`abs(net)*dt/V <= tau/2`, alongside its original physical acceptance. The finite
interval alternative allows zero-circulation problems without requiring an
exact-zero roundoff tail. No interval-certified closure is invoked in Ordinary.
For every accepted algebraic state, `stable_timestep()` additionally bounds the
next interval by `tau*V/(2*abs(net))` on cells with nonzero net rate. This derived
bound also covers transferred face fields after fixed-clock remap. It retains
the original absolute-face-rate, viscous and capillary limits and cannot turn an
incomplete physical horizon into a pass. Projection and linear iteration caps
are unchanged.

Solver-owned readiness additionally checks the prospective cumulative occupancy
against the fixed physical occupancy limit, including publication uncertainty. Timestep
selection retains its existing physical and canonical restrictions; the final
transport admission checks the actual signed cumulative update. A reversing
canonical transfer can restore capacity, so a cumulative allowance must not be
treated as an unsigned, monotonically shrinking timestep cap. Exact zero-flow
identity publication incurs no new narrowing. Absolute history magnitudes cannot
replace the signed physical discrepancy or qualify an incomplete endpoint.

After a provisional physical projection, cumulative occupancy uses a compatible
pressure-rate target. Let `r=delta/V`, `S=sum(abs(Q))`, physical occupancy limit
`epsilon=1e-13`, and the unchanged physical divergence threshold be `P`. Each
cell reserves half of its original admissible rate capacity:

```
L = 0.5 * min(tau*S/(2*C), P*V, 2*epsilon*S/C)
mean = sum(L*r)/sum(L)
b = L*(r-mean)/(2*epsilon)
```

Zero total capacity gives a zero target. The weighted mean makes the target
closed-domain compatible; its reduction discrepancy is checked against derived
arithmetic error. The pressure solver then corrects `net-b` through the same
canonical rates and pressure forces, within the existing shared correction and
iteration budgets. A separate target-defect check prevents the original loose
physical threshold from skipping this correction. All original tests on raw
divergence, pressure residual, relative-or-finite-step readiness and final phase
publication remain mandatory. Gas, carrier, mass and momentum use the resulting
single canonical transport authority. No inventory is overwritten and no error
history is reset. A uniform global occupancy discrepancy cannot be removed by
closed conservative transfers. Actual timestep admission remains authoritative;
the target construction is not a universal guarantee of timestep growth.

Interior nonorthogonal pressure reconstruction retains its cell-center stencil.
Wall pressure and capillary traction instead use a separate conditioned linear
fit of shared internal-face values minus cell values at the actual face-centroid
displacements. This gives a one-sided pressure gradient across a density jump
and uses every canonical fine subface. The same linear response acts on the
density-weighted face fraction, preserving constant-curvature pressure balance.
Deficient nonconstant support rejects collectively; a structurally constant
response is zero. Unsampled slab pressure reactions remain explicit. Finite
certification propagates face-value uncertainty, and cache and temporary storage
remain inside the existing budgets.

Canonical continuity and cell pressure momentum still use distinct discrete
operators. The compact face correction is not the interpolation of the cell
pressure impulse, and the force map is not the adjoint of that interpolation.
Small canonical divergence therefore does not establish an energy-bounded cell
velocity. Long high-density-ratio flow remains unsupported where this mismatch
amplifies pressure work and reduces the stable timestep. A common projection
formulation must match the cell/face constraint, mass metric, pressure force,
wall reactions and stabilization while retaining the existing wall-pressure and
capillary contracts. Resetting face velocities or omitting wall traction does
not establish that formulation. The transport momentum residual and occupancy
admission remain necessary checks, distinct from this projection limitation.

## Explicit face-aligned surfaces

`face_interfaces()` returns canonical opposite-pure geometric faces with
centroid, area, and gas-to-liquid normal. It preserves trace inventories and
never manufactures mixed cells. Its bounded halo catalog includes interface
faces incident only to neighboring halo cells; deduplicate replicas by face ID
for global counts. The Ordinary API is independent of the transport backend.

The shared capillary catalog combines mixed sections and face-aligned patches
using canonical patch identities and connected support. Represented planar
surfaces require coherent normals, coplanarity and two independent tangent
directions. Curved binary surfaces use the explicitly selected, component-volume
preserving `FilteredBinary` model. Deficient support and unresolved topology still
reject. See the [capillary geometry model](../modeling/general_vof_capillarity.md)
for filtering, volume publication, diagnostics and support limits.

## Migration and qualification

Implementation is staged behind explicit backend selection. Registered focused
coverage includes moving pure phases at subnormal fluxes, traces, opposite-phase
receipts, CFL/continuity/boundary rejection, normal face crossings, limiter
capacity competition, density-jump constant momentum, preparation reuse, explicit
face surfaces, and planar capillary support under tangential MPI stripes.
Coupled-flow tests exercise pure motion, pure-liquid hydrostatics, and a
high-density-ratio skew coarse/fine passage. Test registration is not a pass;
current commands/results belong in local build logs.

The opt-in qualification manifests in `verification/vof/` prescribe complete
motion, coupled-flow and capillary endpoints before candidate runs. The motion
harness uses independent polygon/prism integration, spatial and temporal
refinement, midpoint excursion checks for return cycles, and canonical directed
stream theta statistics. It reports affected bulk volume separately from discarded
high-order correction. `DonorCell` is the conservative low-order comparison;
the sharper-interface accuracy gates apply to `GeometricFace`.

The configured default remains `SweptPartition`; promoting a different backend
would require an explicit decision and its relevant coupled accuracy/physical
gates. The [scoped matrix](../verification/vof_solver_contracts.md#stage-7-scope-and-priorities)
treats this opt-in backend as a capability extension, not a prerequisite for
qualifying the swept default. Raw controlled-study results do not establish
GeometricFace's cleanup-enabled default-policy accuracy or sensitivity.

A short passage does not qualify translation/rotation/deformation convergence or
curved-capillary equilibrium. Cleanup-enabled coupled AMR/remap, open boundaries
and counterstreams remain separate extensions; existing rejection remains required
until each is implemented and qualified. Matched whole-solver timing/RSS/scaling
is required only for performance claims, after numerical qualification. Geometry
and momentum preparation counts alone do not establish those claims.
