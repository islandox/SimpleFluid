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

Transport authenticates controls, constructs departure fragments, authenticates
donor allocations, exchanges amounts and validates the candidate. Remap agrees
the schema and target, authenticates overlap requests, transfers contributions,
certifies recipient coverage and installs phase inventories and their evidence
together. A source-pinned trial remains responsible for observation validation.

## Solver transaction ownership

`src/solvers/details/GeneralVoF/State.hh` declares accepted/candidate fields and
arithmetic evidence. `SolverImpl.hh` declares the private coordinator and its
mesh-bound resources; the public solver implementation forwards operations to it.
Substantial definitions compile privately, while simple members remain in the
header and the caller-dependent collective callback is defined in `.ipp`.

Initialization and advancement prepare a detached state. Remap prepares a
replacement resource owner and detached state. Both finite and Exact preparation
complete their physical checks and final phase observation validation before
returning. Only the coordinator installs the returned accepted snapshot; failed
preparation leaves the original publication and clock unchanged. Destruction
performs no collective work.

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

Curvature assembly delegates patch fitting to the existing curvature kernel.
Projection and implicit momentum remain owned by the existing flow workspace
and conservative transport components. The extraction does not create another
numerical backend or broaden their support matrices.

## Compilation and resource contracts

Private `.hh` files declare interfaces and retain simple one-line members.
Heavy templates live in `.tcc` providers included from owning `.cc` files, which
instantiate the supported modes. Declaration and provider includes precede
code; only `.ipp` definitions may be included at the end of a header. Public
headers never include compiled providers.

New numerical translation units inherit the owning component's strict floating
point, PGO, PCH and unity-build policy. Internal functions stay hidden inside
their existing library; FVM must not depend on the solver library.

Splitting a function does not grant another resource budget. Existing admission
scopes, operation counters, packet lifetimes and geometry leases remain in
force. Intermediate records borrow pinned data or move owned buffers; additional
copies or overlapping lifetimes require explicit accounting. Ordinary physical
allowances must not be relabeled as bounded certificates.
