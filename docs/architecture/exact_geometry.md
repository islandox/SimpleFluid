# ExactGeometry: rational geometry and source authority

Floating-point numerical geometry is the primary VoF choice; its contract lives
in [general AMR](general_amr.md#floating-point-geometry-and-arithmetic).
This document owns the rational geometry APIs, certificates and accelerations
used by explicit [ExactVoF](exact_vof.md) and
[ExactTransport](exact_transport.md) clients.

## Native authority and the FP import boundary

The existing AMR/native-mesh substrate retains exact source authority.
`ConvexCellGeometry`, exact fragments and lossless source certificates remain
available independently of the chosen VoF numerical arithmetic. The hierarchy,
ancestry, canonical incidence and migration contracts remain in the
[AMR design](general_amr.md#implemented-certified-geometry-and-sparse-remap).

The FP importer obtains bounded coordinates and opaque original support keys
from that authority. Membership queries and coordinate/support exports are
bootstrap work and must be measured separately from numerical steps. Opaque
support keys identify provenance; they are not rational numerical coefficients
in FP geometry or ledger payloads. Native authority import does not enable an
exact numerical fallback for FP clipping, fitting or physical acceptance.

## Separation and wall trajectories

Private certified convex intersections first admit exact zero-volume separation
certificates: disjoint or touching axis intervals, then vertices entirely
on/outside a supporting plane of either certified shell. This check precedes
clipping scratch and shell copies for primary departure-tetrahedron/donor
intersections, fan disjointness, signed fragments and owner-column proofs.
Positive penetration, however small, cannot satisfy those certificates; absent
a certificate, the original exact intersection test runs. Every caller's pair
work charge, containment, independent volume/first moments and complete donor
coverage checks remain required. Fixed exact locals add no retained geometry
cache; existing uncached rational-temporary accounting remains explicit.
Source/native identity and revisions pin reconstruction support;
complete canonical node fans and fixed support rings supply physical neighbors,
subject to explicit stencil, halo, packet and opaque-storage capacity guards.
Exact wall-nullspace projection uses the complete authenticated catalog of
incident exterior walls. Shared raw node input and the full exact departure
must agree. Point, edge and corner wall constraints are exact; nearby planes
are never merged by tolerance. This constrains the departure while preserving
the physical inventory and canonical projected face volume.

## Measures, coverage and predicate acceleration

Exact volume-only queries omit unused first moments while using the same
oriented determinants. Admission and coverage proofs still calculate all three
moments. Exactly zero determinants and inactive normal axes can be skipped;
no positive piece is removed by a tolerance. Certification-local support planes
use the first nonzero exact outward fan normal, including collinear prefixes.
Optional caches account planes, ranges and rational limb capacities under live
scratch leases. Only typed optional-cache exhaustion selects the uncached path;
geometry, incidence and allocation failures propagate. Independent containment,
pairwise disjointness, volume and first-moment proofs remain mandatory.

### Integer signs, FP filters and lazy distances

The private integer helper evaluates rational support signs with checked
fraction-free integer sums and positive denominators. Its bounded 32,768-bit
workspace is leased before allocation. Unavailable capacity or checked overflow
releases that lease and uses the original rational expression. Reserve-callback
exceptions propagate. The result supplies only a sign, never approximate geometry.

The floating-point helper encloses rational plane distances with outward
binary64 intervals. Only an interval strictly on one side of zero certifies a
sign; contact, cancellation, subnormal/overflow or unsupported arithmetic falls
back to integer/rational evaluation. It is enabled only for guarded strict GCC
on x86-64 SSE2 with round-to-nearest, masked exceptions and FTZ/DAZ disabled.
Clang uses exact fallback. Every query rechecks the floating-point environment;
cached intervals cannot bypass that check. Control bits are never changed.

Lazy clipping materializes the original exact distance when a sign is uncertain
or an edge crosses. Exact intersection coordinates, equality, vertex identity,
incidence and moments therefore remain rational constructions. Scratch denial
releases optional storage before uncached retry. Neither helper changes physical
tolerances, mandatory resource limits, finite/exact policy or atomic rejection.

Recorded matched runs of the earlier rational-geometry backend show bounded
workload improvements, not universal speedups. That coarse/fine case costs
hundreds of seconds for 50 cells. Exact clipping,
moments and rational normalization remain substantial costs. Inclusive profile
rows overlap; sample shares are not operation counts, fallback rates or elapsed
time. Payload admission excludes backend/allocator/Krylov internals and is not
an RSS cap. Large migration and process-memory costs still need qualification.

## Reference-oracle use

Small exact/analytic comparisons can validate FP enclosures and structural
proofs. They are explicitly selected correctness work. Generating rational
reference volumes afresh for every benchmark invocation is a fixture choice,
not a conservative VoF requirement. Benchmarks can use verified reference data
or independent analytic initialization, with fixture preparation and solver
cost measured separately.

Potential future sign-only integration with the FP kernel would need to classify
the original construction from authenticated provenance, return only a discrete
decision, and charge bounded work/storage to the existing caps. An exact sign
of rounded coordinates cannot certify the original construction. The current
FP numerical kernel has no such fallback; an unresolved predicate rejects.

## Focused verification

`testConvexCellGeometry`, `testExactPolyhedralComplex`,
`testExactExplicitProvider` and `testExactExtrudedConstruction` cover the exact
source/geometry contracts. The source remap families
`testConservativeAMRRemap`, `testProductAMRRemap`, `testPolyhedralAMRRemap` and
`testMixedAMRRemap` retain their native-geometry authority. These tests do not
substitute for `testBoundedConvexGeometry` or the finite reconstruction,
transport and remap gates. Report policy, source revision and actual executions.
