# Conservative face transport for general-native VoF

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
translating the donor plane at the prescribed constant face-normal velocity.
The velocity magnitude is `q/(area*dt)`. Both phase-covered areas are measured
directly: the cell minority need not be the outflow-face minority. Time intervals
are split at original face-vertex crossings. Two interior Gauss points integrate
the quadratic area on each interval and avoid assigning finite weight to a
face-parallel contact instant. Bulk `q` remains authoritative.

Only `GeometryErrorCode::UnresolvedArithmetic` permits fallback to the donor-cell
candidate. Affected canonical streams and bulk volume are counted. Invalid
geometry, stale ownership, insufficient physical reconstruction support, and
resource exhaustion retain their failures. A fallback is a loss of local order,
not evidence of geometric or physical qualification.

## Shared limiting and compatible histories

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

## Explicit face-aligned surfaces

`face_interfaces()` returns canonical opposite-pure geometric faces with
centroid, area, and gas-to-liquid normal. It preserves trace inventories and
never manufactures mixed cells. Its bounded halo catalog includes interface
faces incident only to neighboring halo cells; deduplicate replicas by face ID
for global counts. The API requires a face backend.

The initial capillary consumer admits an all-pure planar face surface only with
coherent normals, coplanarity, and two independent tangential support directions.
This admits a supported planar zero-curvature equilibrium. Curved binary
staircases, deficient support, and unresolved combinations of mixed sections and
face-aligned patches reject. A general unified curved surface fit remains open.

## Migration and qualification

Implementation is staged behind explicit backend selection. Registered focused
coverage includes moving pure phases at subnormal fluxes, traces, opposite-phase
receipts, CFL/continuity/boundary rejection, normal face crossings, limiter
capacity competition, density-jump constant momentum, preparation reuse, explicit
face surfaces, and planar capillary support under tangential MPI stripes.
Coupled-flow tests exercise pure motion, pure-liquid hydrostatics, and a
high-density-ratio skew coarse/fine passage. Test registration is not a pass;
current commands/results belong in local build logs.

The default must remain `SweptPartition` until relevant coupled physical gates
are met. A short conservative passage or robust geometry query does not establish
drop-translation accuracy, rotation/deformation convergence, curved capillary
equilibrium, or production scaling. Open boundaries, counterflow, unified curved
surface fitting, and broader physical accuracy qualification remain migration
gates. Geometry operation counts and momentum preparation counts are not
whole-solver timing claims.
