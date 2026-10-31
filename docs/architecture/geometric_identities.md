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
variants. Native payloads and retained ancestry are authoritative; backing owners
cache canonical length and FNV hash, without retaining serialized key buffers.
Local geometry operations consume native payloads; parsing is confined to checked
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
support constructions. Native ordering can propagate allocation failures from
arbitrary-precision arithmetic. Identity equality, size and hash remain
allocation-free and nonthrowing: matching hashes require complete native field
comparison. Canonical emission streams native scalars and Exact integer limbs.
Sequential cursors preserve packed word ordering with bounded ancestry scratch;
deep traversal and final-owner release do not recurse through the parent chain.

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

Identity and geometry packets use the natural C++ layout of
`GeometryPacketHeader`. Family, version, kind, arithmetic mode, and numeric
profile each occupy one byte; body length is eight bytes. `sizeof` and
`offsetof` determine the serialized size and member positions, including
compiler-inserted inter-member and tail padding. On the current GCC/x86-64
layout the header is 16 bytes: fields at offsets 0--4, zero padding at 5--7,
and length at offset 8.

Serialization zeroes the header byte storage and writes members individually,
so no indeterminate C++ padding enters a packet. Decoders reject nonzero padding.
The length remains little-endian and uses `memcpy` with compile-time endian
conversion; an enclosing variable-length record need not provide an aligned
address. Padding aligns header-relative fields, not arbitrary enclosing buffers.

Numeric profile enums identify Exact rational data, binary64, x87 binary80,
or binary128. Ordinary selection matches radix, significand precision, and both
exponent limits exactly; unsupported formats reject instead of selecting an
approximation. Profiles describe arithmetic formats, not `sizeof(long double)`.
This header is ABI-dependent: communicating endpoints and saved artifacts need
compatible C++ member layouts. A matching numeric profile alone does not prove
layout compatibility, and the header does not carry a separate layout ID.
Decoders reject foreign kinds/modes/profiles, unsupported versions, noncanonical scalar or rational
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
Compact closure-span contacts and their request batches use distinct record
kinds from the older full-certificate closure payloads; a matching outer header
version does not make those payloads interchangeable.
AMR and FVM geometry record stamps use natural headers of one-byte family,
version, and kind fields. With their byte alignment these headers occupy three
bytes; the codecs still initialize and check any compiler-inserted padding.
Adaptive batch headers include their eight-byte dictionary count after the
one-byte family and version fields, using natural padding and the existing
little-endian count encoding. Dictionary admission checks still precede
allocation. Extrusion and measure codecs use compact tags too, and their
native-fragment, frozen-part/batch, and partition containers are versioned
accordingly. Independent product ancestry capsules and pure topology codecs
retain their own schemas.

Native multi-region partition packets likewise separate their one-byte family
enum from their one-byte schema version. Their natural C++ header includes the
eight-byte region count: on the current layout it occupies 16 bytes, with six
zero padding bytes before the count. The decoder checks family, version, padding,
and the count's remaining-byte bound before constructing the partition. This
count keeps the partition format's native byte order. Header fields are copied
individually with `memcpy`; decoding accepts unaligned packet addresses. These
header guarantees do not change the native representation of partition bodies.

These compact-tag formats exclude legacy pre-tag packet imports and automatic
migrations; legacy artifacts must be regenerated. Native identity storage,
immutable authorities and owning point exports preserve the current canonical
bytes, hashes, ordering and packet versions. Their source API changes require
rebuilding consumers, without regenerating compatible current-format packets.

`PlaneAuthority`, `DynamicPointProvenance`, and `GeometryAllocationLedger` remain
separate mechanisms. Identity equality does not establish shared allocation
ownership, and shared ownership does not establish geometric incidence.

`PlaneAuthority` is an immutable handle to a typed original support or shared
opaque words. `opaque()` cannot mint the reserved original-support namespace.
Copies share ownership; independently created equal authorities retain distinct
owners. Equality, ordering, and serialization preserve the original numeric
`uint64_t` word sequence, which differs from native identity and byte ordering.
The leading key word packs the one-byte `AuthorityKeyTag` namespace in its high
eight bits and the payload length in its low 56 bits. This reserves the original
support namespace without confusing arbitrary opaque tokens with compact tags.
Boundary import checks every word, declared length and padding byte against a
supplied authority registry, then reuses its immutable owner. Serialized words
alone never grant original-support authority. `words()` explicitly materializes
boundary data; packet writers stream words without retaining another key buffer.
Allocation traversal counts opaque owners or typed ancestry once per owner.

Ordinary supports and Exact identities share immutable backing allocations.
Copies retain the same owner; separately decoded equal identities remain
distinct owners. Translation retains its parent support without simplifying the
construction expression. There is no global interner or persistent key cache.

`GeometryAllocationLedger` admits geometry and shared identity owners in the
same scope, pins them against address reuse, and charges each owner once.
Immutable provenance-support vectors have their own allocation owners; copied
vertices and exported point histories share those buffers while the ledger
traverses every retained typed authority attachment. Mutation detaches before
changing a shared support sequence. Failed admission poisons the scope.
Standalone footprint queries include reachable identity ancestry; `exclusive_storage_bytes()` on a planar face excludes those
shared identity allocations so a caller can admit them through its ledger.
Counts cover backing objects, buffer capacities, and ledger bookkeeping, not
allocator overhead, shared-pointer control blocks, or RSS. A translation adds
one native owner rather than a copy of every ancestor's canonical key. Unique
retained storage grows linearly with depth; computing each new canonical hash
can still perform quadratic cumulative work across a complete chain.
Native Exact payloads also include dynamically allocated integer limbs without
counting inline limbs twice. Import budgets bound both encoded records and
retained native payloads. Import layers and canonical cursor scratch are
preflighted separately from borrowed wire data and charged to active operations;
cell imports share a bounded construction operation and owner ledger across
decoded fields and reconstruction, with no separate default-budget fallback.

Packet length, exclusive retained capacity, and shared-owner admission are
separate quantities. In AMR, released packet storage may decrease the live
budget; an identity ledger keeps its own monotonic counter and transfers only
new admission charges into that live budget.
Closure request preparation pins admitted geometry until that request phase
ends. Per-leaf scratch release excludes every surviving ledger charge. Storage
reports count unique payloads and exclude their temporary reporting ledgers.
