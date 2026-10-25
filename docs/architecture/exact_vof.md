# ExactVoF: optional rational reconstruction and inventories

Floating-point arithmetic is the primary VoF choice. Use the
[default VoF contract](vof_transport.md) and
[FP geometry policy](general_amr.md#floating-point-geometry-and-arithmetic)
for ordinary development and qualification. ExactVoF names this documentation
family; it does not introduce a new C++ class or change the default policy.

This document owns explicit rational reconstruction, exact-policy state and
compatibility contracts. [ExactTransport](exact_transport.md) owns rational
sweeps, remap, momentum and projection; [ExactGeometry](exact_geometry.md) owns
rational geometry, certificates and predicate acceleration. These paths require
explicit selection and cannot rescue an unresolved FP trial.

## Reconstruction

`PLIC::PolyhedralReconstruction<ArithmeticMode::Exact>` fits the normal and
offset against actual clipped physical-cell volumes. Its
`PLIC::PolyhedralPhaseSample<ArithmeticMode::Exact>` inputs retain exact geometry.
Native callers explicitly import their Ordinary source geometry before entering
this specialization; the fit uses the exact geometry's published centroid and
volume for initialization and normalization. A least-squares gradient supplies an initial normal;
every angular evaluation solves the target offset again. Conditioning,
optimizer convergence, target-volume residual and curved-stencil model defect
have separate checks. An underresolved or ill-conditioned stencil rejects.
Curved-model admission does not establish planar-exact reconstruction or
curvature convergence.

PLIC support extrema use exact vertices, retain the signed displacement from
the displayed anchor, and round search endpoints outward. Acceptance uses the
original exact minority-volume residual. Rounded pure fractions never authorize
discarding a positive phase.

Finite reconstruction selects the Ordinary or BoundedCertification instance of
`PLIC::PolyhedralReconstruction<Mode>`, with matching sample geometry. Exact
reconstruction is a separately selected reference or client API. Success on an arbitrary tiny phase is not a production
mesh-resolution requirement; accepted inventories still require conservative
accounting in either policy.

## Inventory policy and publication

Finite publication and rational geometry are distinct choices. The compatibility
route can form transient exact certificates from represented phase values while
publishing independent finite gas/carrier volumes once. It carries no rational
phase remainder into a subsequent finite step. The optional exact-inventory
policy has a different authority:

For general VoF, `GeneralVoFOptions::arithmetic_mode = ArithmeticMode::Exact`
selects the existing experimental exact-inventory path; `Ordinary` is the
default and `BoundedCertification` is a separate finite opt-in. The compatibility
`inventory_policy()` getter reports `ExperimentalExact` only for `Exact` and
`Finite` for both finite modes. The cold exact phase/error
payload is allocated only in the opt-in policy. Its ordinary finite views and
conversion bounds remain diagnostics; the cold payload is that mode's authority.
Source, result, capsules, imported records and accepted-control checks pin the
policy, including empty owners. Cross-policy adoption rejects; resource or
numerical failure never silently switches policy. The finite branch stores no
exact phase/error remainder, and transaction-local weighted momentum products
remain a separate concept in both modes.

Experimental exact handoff preserves its authoritative phases and provenance
through same-policy ownership migration and certified overlap transfer. Positive
owned exact inventories must still have representable positive scalar views;
tiny internal fragments are aggregated before that unchanged admission boundary.
The retained exact implementation is experimental: focused policy/API/remap
contracts do not qualify general physical use, performance or total storage.
Normalized history denominators, rational arithmetic and packets
can grow with accepted steps even after allocation slack and duplicate retention
are removed. It has no long-horizon, centimetric-bubble or production-scaling
qualification. The finite and experimental policies were delivered separately.
Prototype results do not qualify the integrated opt-in path; its focused policy tests
must remain distinct from its pending physical and performance gates.

## Source organization

Optional rational operations live in the owning component's `Exact` namespace:
`SimpleFluid::VoF::Exact`, `SimpleFluid::detail::Exact` or
`SimpleFluid::Exact`. Non-template definitions live in the owning compiled
`.cc` translation units; the solver's exact template implementation lives in
`details/GeneralVoFExactSolver.tcc`, included only by `GeneralVoFSolver.cc`.
Shared geometry, packet framing and collective admission
helpers remain common. This organization preserves the arithmetic compiler
flags and policy dispatch; exact operations are not a finite-path fallback.
Shared scalar arithmetic lives in `utils/BoundedFloatingPoint.hh`,
`utils/DirectedFloatingPoint.hh`, `utils/BoundedInterval.hh`, and
`utils/ExactRational.hh`. Geometry, phase accounting, and solver state remain in
their owning components. Private headers declare shared state and helper
interfaces; separate compiled translation units implement geometry, finite and
exact transport, projection, and adaptive solver subsystems. Each numerical
implementation retains its owning target's arithmetic compiler settings.

Explicit rational reconstruction uses `PLIC::PolyhedralReconstruction<ArithmeticMode::Exact>`;
the same template exposes Ordinary and BoundedCertification instances with
matching geometry inputs. Only the finite modes expose `assess_resolution`;
this refactor does not add their phase-support screen to the Exact algorithm.

## Verification and historical qualification

Exact-API checks are separated into `testExactGeneralVoF`,
`testExactPolyhedralReconstruction`, `testExactGeneralVoFSolver` and the
`testExactConservativeVoFTransport`, `testExactGeneralVoFFlowWorkspace`,
`testExactGeneralVoFPhiClosure` and `testExactGeneralVoFCurvature` kernel targets.
Their CTest cases carry the `exact` label. Exact solver policy checks use
`ExactGeneralVoFSolver_2procs` and `ExactGeneralVoFSolver_4procs`; inspect the
registered filters for each rank count. Primary executables retain floating-point
coverage. Choosing an exact expected-value oracle for a floating-point test does
not switch the production arithmetic policy.

The earlier implementation, with finite-default inventories and rational
geometry, completed the ten-step skew coarse/fine passage fixture to its
original 0.02 s endpoint with unchanged inventory, momentum, positivity and
divergence gates in serial and at two/four ranks. This supersedes the early
resource failures for that version of the fixture; it does not qualify the
replacement bounded FP backend. Selected hydrostatic and solver transaction
checks also have passing recorded coverage. The latest geometry-optimization
campaigns excluded the long reflected-cylinder and viscous methods; earlier
bounded passes and timeouts do not establish their qualification on every later
source. The optional exact policy has focused policy/API/history/remap coverage,
but no full physical, long-horizon, storage/RSS or scaling qualification.

## Recorded implementation and remaining work

The following entries were moved from the general VoF roadmap. Their checkmarks
record the earlier rational-route implementation and its scoped evidence;
they do not qualify the primary FP implementation or announce new test runs.

- [x] Complete the original ten-step skew coarse/fine phase passage to 0.02 s
      in serial/MPI2/MPI4 on the earlier rational-geometry implementation with
      its original physical gates. This historical qualification does not
      establish a pass or speedup for the replacement bounded FP backend.
- [x] Retain `ExperimentalExact` as a separately delivered, explicit opt-in
      policy with source/result/capsule pins and same-mode remap. Reject
      cross-policy adoption and silent fallback; finite history remains default.
- [x] Reuse authenticated preparation, immutable topology, volume-only queries
      and certification-local exact support planes without weakening proofs.
- [x] Integrate bounded integer signs and certified binary64/lazy-distance
      evaluation with recorded focused matrix and complete coarse/fine coverage.
      Retain exact fallback and compiler/mode guards; this remains rational
      construction/measurement, not a qualified FP geometry backend. This status
      reconciliation records existing implementation/evidence, not a new test run.
- [ ] Qualify optional exact-history physical accuracy, rational-history growth,
      storage/RSS, long-horizon performance and scaling. Focused policy tests
      do not close these gates.
