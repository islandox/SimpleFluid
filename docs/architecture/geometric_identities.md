# Geometric identity and serialization

PLIC distinguishes represented geometry, geometric identity, construction
provenance, and allocation ownership. Their equality relations are different.

`PointIdentity<Mode>`, `LineIdentity<Mode>`, `PlaneSupportIdentity<Mode>`, and
`DisplacementIdentity<Mode>` are immutable opaque values for Ordinary and Exact
planar geometry. Point identity describes represented coordinates, not a source
construction certificate. Ordinary supports retain their original construction
expression; coincident planes need not have equal supports. Exact support and
line identities retain their existing rational canonicalization.

Ordinary points and displacements hold finite coordinates inline. Lines hold
the originally computed axis, direction, and anchor without renormalization.
Their local arithmetic, native comparison, key size, and key hashing do not
allocate. Canonical export may allocate a byte vector. Scalar zero signs are
normalized as in the original codec. Hashing consumes unsigned canonical bytes.

Exact identities retain native rational tuples in compiled private payloads.
Ordinary support payloads retain typed axis, source polygon, input plane,
normalized plane, translation, retained Exact ancestry, and extrusion-DAG
variants. Variable-sized identities keep eager canonical key caches. Local
geometry operations consume native payloads; parsing is confined to checked
imports. Exact periodic normalization retains its upper-seam-only rule, while
Ordinary retains its existing interval-wrapping rule.

Default ordering compares native values and structured provenance:

| Identity | Lexicographic fields |
| --- | --- |
| Point / displacement | Native coordinate components |
| Ordinary line | Axis, direction, anchor |
| Exact line | Rational direction, moment |
| Exact support | Rational normal, offset |
| Ordinary support | Variant, then its native fields and ancestry |

Ordinary support variants order as axis, source polygon, input plane,
normalized plane, translation, retained Exact support, and extrusion DAG.
Translations compare parents before shifts without combining operations. DAGs
compare records before endpoints. Equal coordinates do not merge different
support constructions. Shared comparisons can propagate allocation failures
from arbitrary-precision arithmetic.

`LegacyCanonicalKeyLess` is explicitly deprecated. It provides the former
plain-`char` byte-order policy over the current canonical encoding; computational
callers use native ordering. It does not restore historical numbering or read
legacy packets. Exact swept side diagonals now use native current-point
lexicographic order, consistently across shared incidences.

Only displacement has a default value, representing zero. Missing supports use
`std::optional`. There are no implicit byte or arithmetic-mode conversions.
`promote_identity` explicitly promotes Ordinary points and displacements where
the Exact operation supports represented finite inputs. BoundedCertification
keeps its provenance-based API and its explicit Ordinary source-point bridge.
An Ordinary support may retain an Exact source-support key as ancestry through
the numerical-shell bridge; this does not certify rounded finite coplanarity.

Computational operations consume identities directly. `GeometryIdentityCodec.hh`
owns packet import/export. `canonical_key`, `canonical_face_key`, and
`canonical_cell_key` provide deterministic lossless keys under the current
schema. Key material and transport envelopes are separate. Neither pointer
addresses nor rounded coordinates define geometric identity. Native ordering
and compact tag encodings can change routing hashes and derived entity IDs;
consumers must not rely on previous numeric numbering.

Canonical face keys remove redundant collinear vertices and normalize loop
start. Incidence certificates retain inserted vertices and loop order. Consumers
must choose the representation required by their contract.

The typed API intentionally replaces raw-byte computational overloads. Geometry,
FVM, solver, and downstream consumers must be rebuilt together.

`GeometryCodecTags.hh` owns explicit `std::uint8_t` category and variant enums,
which are serialized as single bytes. Numeric payloads, indices, lengths, and
publication pins retain their required widths; hardware floating-point masks
are independent of codec tags.

Identity and geometry packets use a checked 27-byte envelope: an eight-byte
family signature, one byte each for version, kind, and arithmetic mode, and
eight bytes each for finite numeric profile and body length. Multibyte fields
are little-endian. Exact packets use profile zero; Ordinary packets bind
the radix, significand precision, and exponent range. Decoders reject foreign
kinds/modes/profiles, unsupported versions, noncanonical scalar or rational
representations, truncated/trailing data, and exceeded byte budgets. Zero
displacement has a complete envelope with an empty body; empty wire data is not
a displacement packet.
Canonical support/line checks retain the providers' normalization and ordering
rules, including Exact direction normalization, nonzero translated supports,
and ordered source/DAG records. Ordinary line validation preserves finite
normalization rounding; it does not recompute a chosen axis or demand an exact
unit component.

Canonical-face and incidence packets have different kinds. Cell certificates
and convex fragments also have different kinds. Nested canonical support tokens
inherit their enclosing packet version. Their stable key representation does
not provide a legacy geometry-packet import path.

Affected AMR and FVM records have their own version stamps, including records
whose optional geometry is absent. Ancestry keys retain their original schema.
Mixed join capsules carry a new geometry certificate and regenerate the canonical
key from it. Authority tile packets retain the leading numeric GID
word for collective ordinal patching, followed by their checked record stamp.
AMR/FVM geometry record stamps use an eight-byte family signature plus one-byte
version and kind. Extrusion and measure codecs use compact tags too, and their
native-fragment, frozen-part/batch, and partition containers are versioned
accordingly. Independent product ancestry capsules and pure topology codecs
retain their own schemas.

This is a clean format break: regenerate saved artifacts using affected packet
formats and rebuild every communicating process. There are no legacy packet
import paths or automatic migrations. Configuration signatures, keys, hashes,
and numeric entity IDs may change. Geometric equivalence, construction history,
deterministic ownership, and numerical acceptance gates remain the contracts.

`PlaneAuthority`, `DynamicPointProvenance`, and `GeometryAllocationLedger` remain
separate mechanisms. Identity equality does not establish shared allocation
ownership, and shared ownership does not establish geometric incidence.

`PlaneAuthority` carries a sealed typed original-support attachment alongside
its public key. Equality, ordering, and serialization still use only that key.
The leading key word packs the one-byte `AuthorityKeyTag` namespace in its high
eight bits and the payload length in its low 56 bits. This reserves the original
support namespace without confusing arbitrary opaque tokens with compact tags.
Semantic access checks every key word and padding byte against the attachment;
recognized original-support keys cannot manufacture missing attachments.
Authenticated imports rebind to the supplied source authorities. Generic
opaque authority tokens retain their existing behavior. Allocation traversal
counts the actual carrier owner even when a mutable key has been rejected; it
does not authenticate that key.

Ordinary supports and Exact identities share immutable backing allocations.
Copies retain the same owner; separately decoded equal identities remain
distinct owners. Translation retains its parent support without simplifying the
construction expression. There is no global interner or lazy face cache.

`GeometryAllocationLedger` admits geometry and shared identity owners in the
same scope, pins them against address reuse, and charges each owner once. Failed
admission poisons the scope. Standalone footprint queries include reachable
identity ancestry; `exclusive_storage_bytes()` on a planar face excludes those
shared identity allocations so a caller can admit them through its ledger.
Counts cover backing objects, buffer capacities, and ledger bookkeeping, not
allocator overhead, shared-pointer control blocks, or RSS. Cached full keys and
retained parent owners can increase the footprint of nested translations.
Native Exact payloads also include dynamically allocated integer limbs without
counting inline limbs twice. Import budgets bound both encoded records and
retained native payloads. Nested translation caches are preflighted cumulatively;
cell imports share a bounded construction operation and owner ledger across
decoded fields and reconstruction, with no separate default-budget fallback.

Packet length, exclusive retained capacity, and shared-owner admission are
separate quantities. In AMR, released packet storage may decrease the live
budget; an identity ledger keeps its own monotonic counter and transfers only
new admission charges into that live budget.
Closure request preparation pins admitted geometry until that request phase
ends. Per-leaf scratch release excludes every surviving ledger charge. Storage
reports count unique payloads and exclude their temporary reporting ledgers.
