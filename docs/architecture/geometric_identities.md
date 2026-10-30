# Geometric identity and serialization

PLIC distinguishes represented geometry, geometric identity, construction
provenance, and allocation ownership. Their equality relations are different.

`PointIdentity<Mode>`, `LineIdentity<Mode>`, `PlaneSupportIdentity<Mode>`, and
`DisplacementIdentity<Mode>` are immutable opaque values for Ordinary and Exact
planar geometry. Point identity describes represented coordinates, not a source
construction certificate. Ordinary supports retain their original construction
expression; coincident planes need not have equal supports. Exact support and
line identities retain their existing rational canonicalization.

Only displacement has a default value, representing zero. Missing supports use
`std::optional`. There are no implicit byte or arithmetic-mode conversions.
`promote_identity` explicitly promotes Ordinary points and displacements where
the Exact operation supports represented finite inputs. BoundedCertification
keeps its provenance-based API and its explicit Ordinary source-point bridge.
An Ordinary support may retain an Exact source-support key as ancestry through
the numerical-shell bridge; this does not certify rounded finite coplanarity.

Computational operations consume identities directly. `GeometryIdentityCodec.hh`
owns packet import/export. `canonical_key` and `canonical_face_key` serve a
separate stable key contract: key bytes, nesting, ordering, routing hashes, and
`PlaneAuthority` packing preserve existing AMR addresses for a fixed ABI,
partition, and rank count. They must never be replaced with a transport packet,
pointer address, or rounded coordinate. Key ordering retains the established
`char` comparison; routing hashes consume unsigned byte values.

Canonical face keys remove redundant collinear vertices and normalize loop
start. Incidence certificates retain inserted vertices and loop order. Consumers
must choose the representation required by their contract.

The typed API intentionally replaces raw-byte computational overloads. Geometry,
FVM, solver, and downstream consumers must be rebuilt together. The initial
typed migration retains the existing packet encoding; subsequent storage and
codec changes must preserve the independent canonical key contract.

`PlaneAuthority`, `DynamicPointProvenance`, and `GeometryAllocationLedger` remain
separate mechanisms. Identity equality does not establish shared allocation
ownership, and shared ownership does not establish geometric incidence.
