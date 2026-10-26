# General-native VoF transport families and preparation

General-native VoF has two transport families. The family describes how material
crosses a face; it is independent of `ArithmeticMode` and does not imply physical
qualification.

| Family | Backend | Transport construction |
| --- | --- | --- |
| Algebraic face-volume | `DonorCell` | Canonical integrated face volume with donor composition and endpoint/export budgets |
| Algebraic face-volume | `GeometricFace` | The same face-volume contract with measure-only PLIC candidates and conservative limiting |
| Geometric swept-tetrahedra | `SweptPartition` | Authenticated node trajectories, corrected signed tetrahedral face sweeps, donor intersections and positive recipient partitions |

`SweptPartition` remains the default. `GeometricFace` uses geometry to estimate
phase coverage, but does not construct swept tetrahedra. Neither selecting a
family nor preparing its ledger silently changes arithmetic policy. Face-volume
backends currently admit Ordinary arithmetic only; the swept path retains its
explicit finite and Exact providers. See the [face-volume contract](conservative_face_vof.md)
and [floating-point ownership contract](vof_transport.md).

## One detached finite ledger

`GeneralVoF::Trial::prepare_transport` binds a finite trial to positive constant
phase densities and a caller time-stage identity. Its immutable
`GeneralVoFPreparedTransport` retains the source pin, transaction, mesh identity,
selected backend/family and arithmetic mode, timestep, independently stored phase
records and histories, and owned old/new mixture-mass capacities. Empty owner
collections are valid and do not select an arithmetic policy or transport family.
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
residual checks. Stage 1 preserves the existing coefficient-agreement requirement;
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
separates the implemented shared foundation from the remaining Ordinary swept
repairs: homogeneous algebraic specialization, an interface-local signed face
ledger, aggregate fragment measurement, and unified capillary surfaces. In
particular, Stage 1 does not replace recipient partitions, add a homogeneous
shortcut, coalesce fragments, or change the default backend.

Binary-staircase filtering is a later explicit capillary model; it must preserve
component geometric volume without changing accepted cell inventories. Complete
motion/capillary horizons, accuracy/convergence targets and production scaling
remain separate qualification obligations. Focused preparation and failure tests
cannot establish those results.
