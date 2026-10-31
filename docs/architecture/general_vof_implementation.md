# General VoF implementation boundaries

The public `GeneralVoF` and `GeneralVoFSolver` interfaces retain their phase,
geometry, conservation and rollback contracts. Implementation modules separate
operations from arithmetic selection. Ordinary is the default; Bounded and
Exact remain explicitly selected paths with distinct numerical evidence.

## Phase transport and remap

`src/FVM/details/GeneralVoF/Transport` owns moving transport. `Remap` owns
finite hierarchy adaptation and ownership migration. Each has an independent
compiled provider with Ordinary and Bounded explicit instantiations. Exact
transport and Exact transfer retain their separate implementations and support
contracts; finite remap does not imply an Exact remap specialization.

`PhaseAllocation` owns the shared physical phase labels, signed fitting
redistribution and inherited uncertainty calculations. `Wire` owns shared
record encoding, without selecting communication order. Operation-specific
records and codecs stay with their operation.

The operation coordinators retain collective failure boundaries and owner
exchanges in their original order. Rank-local work must complete its collective
error propagation before the next exchange, including on empty owners. Local
preparation helpers do not silently initiate collectives.

Transport authenticates mode-checked source-pinned controls, selects algebraic
bulk or swept interface work, authenticates donor allocations, exchanges amounts
and validates the candidate. Prepared views retain the selected family's ledger
and diagnostics without inferring policy from nonempty records. Remap agrees
the schema and target, authenticates overlap requests, transfers contributions,
certifies recipient coverage and installs phase inventories and their evidence
together. A source-pinned trial remains responsible for observation validation.

## Solver transaction ownership

`src/solvers/details/GeneralVoF/State.hh` declares accepted/candidate fields and
arithmetic evidence. `SolverImpl.hh` declares the private coordinator, which
owns a `Resources` instance and the accepted state. `Resources` owns the native
mesh, immutable operation controls, geometry metrics, projection workspace and
momentum workspace. It has no accepted-state pointer. The public solver
implementation forwards operations to the coordinator. Substantial definitions
compile privately, while simple members remain in headers and the
caller-dependent `Execution` collective callback is defined in `.ipp`.

Initialization and advancement prepare a detached state. Remap prepares a
replacement resource owner and detached state. Both finite and Exact preparation
complete their physical checks and final phase observation validation before
returning. Only the coordinator installs the returned accepted snapshot; failed
preparation leaves the original publication and clock unchanged. Final remap
publication swaps the prepared resources and state without allocation or
additional numerical work. Old resources remain alive through those swaps.
Destruction performs no collective work.

`Initialization<Mode>`, `Advance<Mode>` and `Remap<Mode>` compose the narrow
components below. The coordinator selects a mode once per operation; selected
numerical loops call compiled typed implementations directly. Initialization
shares its operation sequence across all three modes. Advance and remap have
separate Exact specializations under `exact/`; finite remap and Exact transfer
retain their different protocols. The old unrestricted Exact context adapter
and aggregate implementation fragments are removed.

## Solver numerical components

The private `GeneralVoF/` directory contains independently compiled components:

| Component | Inputs and responsibility |
| --- | --- |
| `Execution` | Borrowed native mesh/maps; collective failure propagation, reductions and imports |
| `NodeVelocity` | Phase support metadata, cell velocity and byte cap; authenticated canonical node exchange |
| `Curvature<Mode>` | Phase interfaces, native geometry and capillary controls; distributed patch assembly |
| `PhaseStorage<Mode>` | Phase authority and material densities; mixture storage and publication validation |
| `Diagnostics<Mode>` | Phase authority, mass and velocity; physical inventory, momentum and energy reductions |
| `FaceRates` | Phase authority, canonical rate evidence and flow workspace; rates, integrated volumes and topology |
| `AdaptedFaceTransfer` | Source/target geometry and remap evidence; common-refinement face transfer and closure |
| `OwnedFaceTransfer` | Source/target canonical identities and rates; ownership-only face transfer |

These components have no access to the solver coordinator or its accepted-state
pointer. Face transfer borrows pinned source/target views and returns detached
face fields, projection evidence and divergence. Phase storage returns separate
finite mass/error or rational mass results. Exact storage and diagnostics use
explicit specializations compiled in `exact/`; their shared rational reduction
retains the existing codec and storage admission checks.

The architecture check rejects component includes of the coordinator and
rejects whole-state/resource dependencies in the narrow numerical components.

Curvature assembly delegates patch fitting to the existing curvature kernel.
Projection and implicit momentum remain owned by the existing flow workspace
and conservative transport components. The extraction does not create another
numerical backend or broaden their support matrices.

## Compilation and resource contracts

Private `.hh` files declare interfaces and retain simple one-line members.
Heavy templates live in `.tcc` providers under `details/GeneralVoF/`. Their
supported modes are instantiated by `.cc` units under each owning component's
`explicit_instantiation/GeneralVoF/` directory. FVM transport/remap dispatch and
trial methods remain in ordinary `details/GeneralVoF/*.cc` sources. Exact
specialization method definitions remain in `details/GeneralVoF/exact/`; they
are ordinary compiled implementations, not explicit-instantiation statements.
Declaration and provider includes precede
code; only `.ipp` definitions may be included at the end of a header. Public
headers never include compiled providers.

New numerical translation units inherit the owning component's strict floating
point, PGO, PCH and unity-build policy. Internal functions stay hidden inside
their existing library; FVM must not depend on the solver library.

The component CMake files name general-native source-free provider groups
separately from adaptive and common VoF sources. The solver also separates
low-Mach and representation providers. These lists organize existing sources;
they do not create new libraries or alternate numerical backends. FVM providers
retain their existing unconditional registration, while solver VoF providers
retain the `SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF` gate. Explicit instantiations,
strict arithmetic flags, PGO exclusions and PCH/unity settings stay with their
existing translation units.

Splitting a function does not grant another resource budget. Existing admission
scopes, operation counters, packet lifetimes and geometry leases remain in
force. Intermediate records borrow pinned data or move owned buffers; additional
copies or overlapping lifetimes require explicit accounting. Ordinary physical
allowances must not be relabeled as bounded certificates.

The immutable FVM topology owns each retained native tile once. Surface and
frontier ranges, construction controls and finite wall identities refer to that
owner by checked index or canonical face ID. The public owned-incident tile
order is unchanged. Exact wall records retain their original identity payloads
without acquiring the finite construction catalog. Adjacency and complete node
fans borrow full cell keys from one topology-owned intern pool; iteration and
membership use key values, never pointer ordering. Accepted publications pin
the topology for every such borrow.

Retained admission charges tile owners, reference capacities, interned keys,
graph rows and geometry payloads. Temporary growth and encoding buffers retain
their own admission charges while live. Geometry admission deduplicates shared
geometry, provenance-support and typed identity owners across retained roots;
the ledger pins those owners and includes its bookkeeping in admission.
Allocation measurements report their scope and exclusions separately from these
conservative admission estimates; neither metric is process RSS.

The allocation ledger stores a strong pin for each kind-qualified owner address.
Payload and bookkeeping totals belong to the admission scope rather than to
individual map entries. Node charges use the actual entry representation;
admission still precedes publication and a failed scope remains poisoned.

Finite clipping compacts each child through local used flags and an old-to-new
index array over the completed source/crossing vertex domain. Scanning old IDs in
ascending order preserves point numbering. The arrays are reused only between
children of one split and destroyed before final validation or recursive retry.
Ordinary face-distinctness marks are local to one validation call and cleared
along each successfully checked face. Aggregate cap membership and directed
successors similarly use local arrays, preserve the minimum-ID ring start and
winding, and retain duplicate/open/disconnected-ring rejection. Sparse edge and
support lookups remain separate mechanisms.

Scratch admission covers actual capacities, including whole replacement buffers
when growth overlaps an earlier allocation. Reuse and destruction do not refund
cumulative charges. These local indices never allocate by a maximum global ID or
introduce a dense vertex-pair table.

Finite construction history retains shared immutable recipes. Each recipe stores
only its active operation payload: a source geometry owner, a shared clip operation
and side, or an exclusively owned Rebase payload with its parent and full input
halfspace. The large Rebase payload is allocated only for that operation and is
charged once with its recipe owner. Shared clip records retain source-point pins;
the general recipe node has no retained-point vector. Both finite modes and
authenticated replay use the same creation and accounting rules. Construction
packet version 5, root ordering and replayed shells are independent of this private
storage layout.

Finite global point exports are explicit owners. `export_points(ledger, used,
limit)` admits a `DynamicPointExport` or `PointExport<Mode>` into the caller's
budget and returns a handle whose lvalue `points()` view borrows that payload.
The handle or ledger pin must survive the view. `vertex_count()` and numerical
queries need no global export; `local_points()` retains its authority-free
contract. Native planar and Exact point APIs are separate.

Geometry owns only fixed cache metadata: a mutex, weak export roots, and fixed
weak provenance slots. Actual dynamic and typed point buffers, provenance and
supports are allocated on request and admitted before cache publication. Typed
exports retain the canonical dynamic export. Independent ledgers can share one
published payload; a ledger itself is not shared across threads. Geometry-only
admission ignores weak cache entries and has the same cost before and after an
export. Export-specific resource failures occur at the explicit request, while
malformed geometry, support and coordinate-range checks remain at construction.

After all export handles and ledger pins expire, later queries rebuild a payload.
Any individual point or descendant provenance still alive is reused through its
weak vertex slot, preserving identity even for uncertain bounded coordinates.
Failed admission publishes neither new roots nor provenance slots and preserves
ledger poisoning. Scratch spending and retained owners share the caller's limit;
cumulative charges are never refunded when an owner expires.

Numerical rebases share the original export control and retain one flattened
canonical export-source dependency. Deferred export therefore uses the original
frame, coordinates, errors and sealing policy. The ledger charges that strong
dependency. Certificate-only copies share the existing control and dependency;
physical translations and changed shells detach. These edges remain acyclic.
