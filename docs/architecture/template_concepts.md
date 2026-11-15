# Template concepts

SimpleFluid uses structural C++23 concepts at region, field-storage and equation
boundaries. A custom type can satisfy a contract without inheriting from a
library type. Concepts check declarations and return types; they do not prove
conservation, map compatibility, provider lifetime, or runtime support.

| Concept | Header | Contract and use |
| --- | --- | --- |
| `Meshes::GeometryProvider<G>` | `geometry/mesh/RegionProviderConcepts.hh` | Const region metrics, node coordinates, ordinal layout, storage identity and geometry epoch; constrains `IsoRegion` geometry. |
| `Meshes::TopologyProvider<T>` | `geometry/mesh/RegionProviderConcepts.hh` | Const owner-oriented region incidence, index pack, cell types, ordinal layout and storage identity; constrains `IsoRegion` topology. Incidence ranges must be sized and randomly accessible, with indexed elements convertible to ordinal values. |
| `MeshType<M>` | `geometry/MeshConcepts.hh` | Const finite-volume queries using local integral ordinals: counts, cell-face traversal, owner/neighbor and physical metrics; constrains field traits/storage, FVM operators and caches, transport packs/factories, physical equations and solver mesh parameters. |
| `FieldType<F>` | `fields/FieldConcepts.hh` | Value and supported location types, identity, and a stable string reference; constrains stored-field descriptors, generic equation storage, and `Problem` field registration/retrieval. |
| `FVM::EquationTermType<T, Pack>` | `FVM/EquationOperators.hh` | Implicit conversion to `FVM::OperatorVariant<Pack>`; constrains forwarding `Equation::add_lhs` and `add_rhs` overloads. |

The provider concepts retain the ordinal-region contract previously declared
in `RegionProviders.hh`, which still includes them. They do not describe the
borrowed `MeshGeometryHandle` and `MeshTopologyHandle` views. `IsoRegion` still
checks matching layouts at construction and checks identity, layout and epoch
at static validation. Ordinal incidence ranges with unrelated element types
are rejected by the topology concept.

`MeshType` covers the common interface exposed by native `MeshHandle`,
`Meshes::PartitionedMesh` and `SolidSubdomain`. Raw geometry with
family-specific structured entity IDs retains the existing `MeshClass`
contract. Do not require `MeshType<Derived>` while forming an incomplete CRTP
base. Maps, boundary metadata, execution leases, and geometry epochs remain
additional capabilities checked or used by individual algorithms; a concept
match alone does not establish that every algorithm supports that mesh.

Template declarations, forward declarations, partial specializations and
out-of-line definitions use matching constraints. When a parameter retains the
name `MeshType`, its declaration qualifies the concept:
`template<SimpleFluid::MeshType MeshType>`. Fixed native-family helpers inside
`MeshHandle` instead use the existing `MeshClass` contract.
Rebuild template consumers and their compiled providers together after changing
constraints so declarations and instantiated symbols remain consistent.

Some generic parameters intentionally retain `class` because these concepts do
not describe their inputs:

- Traversal supports native geometry and mapped meshes; the local-ordinal
  `MeshType` contract does not cover every raw geometry representation.
- Epoch and execution-guard helpers preserve their optional-capability fallback.
- Symbolic matrix plans, collective validation and identity checks may require
  only maps, a communicator or object identity.
- `DirectBatchRange` topology/metric parameters are per-entity records; native
  region adapters receive geometry before it implements the region contracts.
- Callbacks, views and geometry records retain their own weaker contracts.
  Historical field spellings now alias native `FieldStored`; field metadata
  constraints remain distinct from storage and implementation-provider needs.

`FieldType` describes field metadata rather than numerical storage. Locations
are `CellLocation`, `FaceLocation`, and `BoundaryFaceLocation`. Descriptors need
neither a default constructor nor a string constructor: callers can pass an
already constructed custom descriptor to `FieldStored`. The mesh-first
convenience constructors still require construction from a string. Storage
separately requires its value representation to match the Tpetra scalar pack.

Equation terms use a closed variant because assembly dispatches on its exact
alternatives. Transient, convection, diffusion, source, gradient and divergence
operators qualify, as do existing custom implicit conversion adapters. A term
from another pack or an unrelated type with similar members does not qualify.
Adding an assembly operation requires extending the variant and its assembly
and validation branches. The by-value variant overloads remain available,
including brace initialization and explicitly typed member pointers. Both
forwarding and variant calls retain runtime coefficient/provider validation.

Concepts do not change the implementation-provider boundary: supported default
specializations link to the compiled providers. A custom descriptor, mesh or
pack may still require the documented `.tcc` implementation at its own
instantiation site. See [mesh traversal](../mesh_traverse.md) and
[the Trilinos backend](../trilinos_backend.md).
