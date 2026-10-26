# General-native VoF transport families and preparation

General-native VoF has two transport families. The family describes how material
crosses a face; it is independent of `ArithmeticMode` and does not imply physical
qualification.

| Family | Backend | Transport construction |
| --- | --- | --- |
| Algebraic face-volume | `DonorCell` | Canonical integrated face volume with donor composition and endpoint/export budgets |
| Algebraic face-volume | `GeometricFace` | The same face-volume contract with measure-only PLIC candidates and conservative limiting |
| Algebraic face-volume | `SweptPartition`, Ordinary interface-free step | Canonical face-volume authority after collective homogeneous admission |
| Geometric swept-tetrahedra | `SweptPartition`, remaining cases | Authenticated node trajectories, corrected signed tetrahedral face sweeps, donor intersections and positive recipient partitions |

`SweptPartition` remains the default. `GeometricFace` uses geometry to estimate
phase coverage, but does not construct swept tetrahedra. Neither selecting a
family nor preparing its ledger silently changes arithmetic policy. The configured
backend stays `SweptPartition` when its homogeneous specialization reports the
actual `AlgebraicFaceVolume` family. Face-volume execution currently admits Ordinary
arithmetic only; the swept path retains its
explicit finite and Exact providers. See the [face-volume contract](conservative_face_vof.md)
and [floating-point ownership contract](vof_transport.md).

## Homogeneous specialization

`homogeneous_transport_eligible()` collectively checks the current source.
Algebraic transport is eligible only when
all retained cells explicitly declare pure gas or pure carrier, no retained mixed
surface exists, and no internal face joins opposite declarations. Periodic images
remain outside this specialization. Disconnected components may declare different
pure phases. Empty owners participate in the eligibility reduction.

The node-control `plan_transport` entry authenticates every supplied node velocity,
departure and face control before selecting the specialization. The explicit
`plan_homogeneous_transport(face_controls,dt)` entry rechecks eligibility and
authenticates face controls, timestep and canonical ownership. It exposes no
trajectory certificate. Exactly zero canonical flux retains the original accepted
source and its histories after those checks. The solver selects this entry before
attempting nodal interpolation, including in `stable_timestep()`.

A declaration is geometric evidence, not a phase-volume cutoff. Independently
positive traces remain transported material. An `Infer` cell or an opposite-pure
face excludes the specialization even if its current face flux is zero. Stage 2
specializes an entire interface-free step; joining local bulk regions to geometric
transport within a mixed domain is Stage 3 work.

For old independent inventories `G,L`, let `S=G+L`. A directed canonical bulk
transfer `q` carries the smaller phase as `q*min(G,L)/S`; the other phase completes
`q`. Retained gas and liquid are each their old inventory minus their own exports,
then incoming transfers are added independently. Only stream composition is
normalized. The accepted inventories are never reconstructed by complement or
renormalized, and the old occupancy discrepancy stays in the donor. Admission
checks `q` summed over all exports against both native volume and `S`, independent
phase exports, canonical continuity, boundaries and resource limits.

Both phase-history channels remain independent. For positive old phase inventory,
the inherited error follows the corresponding phase-transfer fraction. For an
exactly absent phase, its nonzero uncertainty follows `q/S`; the self-retained
share is `1-outgoing_bulk/S`, rather than `(native_volume-outgoing_bulk)/S`.
Shares, products and history sums round outward. Existing pure-fit evidence is
transferred with the minority-phase share (normalized bulk share when absent),
and publication uncertainty is added separately. This records transported evidence
without enlarging a physical tolerance or deriving a bound from a desired result.

The resulting momentum equation uses adjacent-cell implicit upwind coefficients
from those same phase mass transfers. This is a deliberate homogeneous momentum
discretization, including for nonuniform velocity. Canonical face volume is the
transport authority; there is no nodal subface-counterflow model on this path.
The solver retains its face-based advective, viscous and capillary timestep bounds,
canonical absolute-face-rate bound and outgoing-volume CFL. Fitted-node speed and
displacement gates remain specific to geometric swept execution. Supplied nodes
still undergo authentication through the original node-control entry, whose
globally zero-motion identity shortcut remains unchanged. No stationary trajectories
are fabricated. There are no swept tetrahedra, donor intersections, recipient
overlays or PLIC cuts on the homogeneous path.

## One detached finite ledger

`GeneralVoF::Trial::prepare_transport` binds a finite trial to positive constant
phase densities and a caller time-stage identity. Its immutable
`GeneralVoFPreparedTransport` retains the source pin, transaction, mesh identity,
selected backend/family and arithmetic mode, timestep, independently stored phase
records and histories, and owned old/new mixture-mass capacities. Empty owner
collections are valid and do not select an arithmetic policy or transport family.
The homogeneous swept trial retains phase streams privately until the caller binds
densities and time stage with `prepare_transport`; it exposes no dummy-density
preparation through `prepared_transport()`. The actual family is factory-authenticated
and is not inferred from an empty stream collection.
An Exact trial continues through its separate typed path.
The source and trial validation methods authenticate publication/context pins;
they are not an admission API for caller-mutated ledger copies. Numerical execution
consumes the immutable preparation produced by the trial factory.

Canonical signed physical face contributions and positive donor-to-recipient
route support have different meanings. Swept contributions can name nonincident
geometric donors and can oppose each other on a face; they must not be reinterpreted
as adjacent upwind streams or collapsed to a net volume. Face-volume preparations
also retain their directed streams and limiter diagnostics. Independent gas and
liquid amounts remain authoritative, including traces on declared pure geometry.

The solver obtains this preparation before the component solves, checks phase
publication and mass capacities, and derives momentum inputs from its retained
ledger. The finite momentum workspace validates routing and raw face agreement
when preparing the numerical operator, then reuses its immutable routing, imports,
coefficients and applicable preconditioner for matching component right-hand sides.
Every right-hand side still has its own input, iteration and original physical
residual checks. Both families preserve the existing coefficient-agreement requirement;
positive route support is not permission to change the accepted physical equation.

Only successful whole-step publication exposes the preparation, including its
momentum identity and build/reuse counts, through `GeneralVoFSolver::prepared_transport()`.
Failure leaves accepted phases, velocity, pressure, time, histories and the previous
preparation untouched. A new mesh or changed numerical coefficients invalidate
reuse; retained accepted preparations are immutable evidence, not mutable workspaces.

## Collective failure evidence

Finite VoF local-work boundaries retain bounded structured failure records. Each
record identifies the originating rank, stage and error category, original cause,
and the available cell/face/donor/recipient/construction context. Numerical scale,
budget and retry fields are reported only when known. Sweep construction can retain
its bounded tetrahedral coordinates; unavailable information is explicitly absent.

All participants first agree that a local operation failed, select a deterministic
origin, and distribute the bounded record before throwing. The originating local
exception remains accessible. Allocation/resource exhaustion, malformed metadata,
and typed unresolved geometry/arithmetic remain distinguishable. Legacy untyped
exceptions retain their original message; boundary categories are diagnostics,
not a new recovery or fallback policy. Diagnostic records do not introduce new
collectives inside rank-dependent local geometry traversal.
The helper does not change unrelated distributed AMR error handling. One compiled
FVM context provider shares the thread-local diagnostic stack across solver, FVM
and caller scopes; the export boundary permits only that exact bridge.

These records improve diagnosis; they do not make a failed geometric construction
acceptable. A zero-volume failure still rejects atomically under the same numerical
rules.

## Remaining repair stages

The [Stage 7 roadmap](../../TODO.md#stage-7--general-vof-and-flow-qualification)
separates the implemented shared foundation and homogeneous specialization from
the remaining Ordinary swept repairs: an interface-local signed face ledger,
aggregate fragment measurement, and unified capillary surfaces. The homogeneous
path does not replace partitions in mixed domains, coalesce fragments, or change
the configured default backend.

Binary-staircase filtering is a later explicit capillary model; it must preserve
component geometric volume without changing accepted cell inventories. Complete
motion/capillary horizons, accuracy/convergence targets and production scaling
remain separate qualification obligations. Focused preparation and failure tests
cannot establish those results.
