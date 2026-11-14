# Typed transport assembly

`FVM::TransportSystemFactory<Pack, MeshType>` is the stateless entry point for
scalar, vector, weighted-scalar, physical-temperature, and physical-momentum
assembly. Include `FVM/TransportSystemFactory.hh`; the umbrella
`FVM/Operators.hh` also exposes it. `FVM/TransportAssemblyPacks.hh` defines the
input groups independently of the factory. `TransportSystem<Pack>` and
`VectorTransportSystem<Pack>` retain their matrix/RHS layouts.

The factory receives four small packs by value:

```cpp
Factory::assemble(terms, boundary, options, context = {});
Factory::refresh_temperature_rhs(terms, boundary, options, context);
```

The terms determine the family; the factory exposes corresponding aliases such
as `Factory::WeightedScalarTerms`, `Factory::WeightedScalarBoundary`,
`Factory::WeightedScalarOptions`, and `Factory::WeightedScalarContext`.
Concrete overloads preserve the supported combinations. Mesh identity comes
from `terms.history.old_values`; `MeshFieldTraits<Pack, MeshType>` selects the
legacy or stored field family. Assembly does not adapt a native field to a
legacy mesh or replace a caller's mesh object.

## Ownership and collective execution

Required fields are borrowed through references. Optional history, correction,
coefficient caches, geometry, ALE state, symbolic plans, and gradient workspaces
are borrowed through pointers. Callbacks and matrix/RHS reference-counted
handles are held by value. All referenced objects and callback captures must
remain valid for the entire call. The factory consumes the packs synchronously
and retains none of them; the returned system owns its matrix/RHS handles.
A supplied matrix or RHS may therefore alias the returned storage.

Pack construction binds references and constructs callback objects. It does
not evaluate source, condition, value, sink, constraint, or selector callbacks,
validate numerical inputs, or enter MPI collectives. Validation and callback
materialization stay inside the appropriate assembly path, preserving its
callback order and rank-coherent failure handling. Caller expressions used to
construct a pack should likewise avoid rank-local validation that could throw
before peers enter assembly.

Every rank, including empty partitions, must select the same family and
compatible policy/optional-field categories and participate in the established
collectives. Fields, caches, correction histories, and workspaces retain their
existing exact mesh-identity and map requirements. Geometry epochs, execution
leases, cache invalidation, boundary-face ownership, and owner-oriented flux
signs remain part of assembly's contract.

## Compiled backends and families

The following overloads have individual compiled providers for
`DefaultTpetraTypes`. A listed entry is an assembly/linkage boundary, not
qualification of every geometry, physics combination, or MPI decomposition
represented by that backend.

| Factory member and boundary contract | `Mesh` | `MeshHandle` | Cartesian `PartitionedMesh` | `SolidSubdomain` |
| --- | --- | --- | --- | --- |
| Scalar `assemble`, `ScalarBoundary` | Yes | Yes | Yes | Yes |
| Vector `assemble`, `BasicVectorBoundary` | Yes | Unavailable | Unavailable | Unavailable |
| Vector `assemble`, `VelocityBoundary` | Yes | Yes | Yes | Custom provider |
| Weighted-scalar `assemble` | Yes | Yes | Yes | Custom provider |
| Temperature `assemble` | Yes | Yes | Yes | Yes |
| `refresh_temperature_rhs` | Yes | Yes | Yes | Yes |
| Momentum `assemble` | Yes | Yes | Yes | Custom provider |

Here Cartesian means
`Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>`. The complete
basic-vector condition/value overload is constrained to `Mesh<Pack>` even when
an implementation provider is supplied. Other unlisted mesh/type-pack
combinations require their own providers and must satisfy the existing
geometry and numerical support contracts.

The public packs also restrict resources by family and backend:

| Family | Policies | Context beyond optional matrix storage |
| --- | --- | --- |
| Scalar | Optional non-orthogonal correction | Geometry cache |
| Basic vector with full conditions | Two-point diffusion | None |
| Vector with velocity selector | Non-orthogonal treatment | Geometry; native ALE |
| Weighted scalar | Scalar discretization, non-orthogonal treatment, coefficient interpolation | Geometry; native ALE, symbolic plan, cached scalar RHS |
| Temperature | Non-orthogonal treatment, coefficient interpolation; native scalar discretization; legacy assembly mode | Geometry; native ALE |
| Momentum | Non-orthogonal treatment, coefficient interpolation, assembly mode | Geometry; native ALE and tensor gradient workspace |

All contexts hold `cached_matrix` by value. Native resources are absent from
legacy packs. Temperature does not expose the weighted-scalar symbolic-plan or
cached-RHS resources. The native momentum boundary additionally exposes its
existing optional `zero_gradient_stress` selector; the legacy boundary does not.

## Terms, boundaries, and defaults

Scalar and vector histories group the old field and time step. Weighted scalar
and native temperature histories also permit an `older_values` field. Basic
scalar/vector `coefficients.diffusivity` is a constant. Weighted coefficients
retain independent current `storage_weight`, `advection_weight`, and
`diffusivity` fields, plus optional accepted-old `old_storage_weight`.
`WeightedScalarTerms::sources` contains the explicit per-volume source and
implicit sink; `constraints.fixed_cell_value` imposes exact identity rows.
Missing explicit sources default to zero, while absent sinks and constraints
remain disabled.

Temperature `material` contains density, specific heat capacity, and thermal
conductivity; `power_density` is a volumetric heat source. Current storage and
advection use the product `rho * cp`. Native accepted-old `old_density` and
`old_specific_heat_capacity` fall back to their respective current fields
independently. Weighted accepted-old storage falls back to current storage
when its pointer is null. Momentum retains dynamic viscosity and reference
density in `material`, with its acceleration source supplied separately.
These groups preserve coefficient units and do not merge independent physical
properties into a user-materialized product field.

`ScalarBoundary` separates scalar condition and value callbacks.
`WeightedScalarBoundary` and `TemperatureBoundary` add an optional boundary
coefficient cache. Weighted-scalar assembly supports Dirichlet and Neumann
conditions and rejects Robin. Physical temperature also supports the existing
normalized Robin law
`dT/dn = robin_coefficient * (condition.value - T_face)`. Basic scalar/vector
assembly retains its existing boundary restrictions; packing the inputs adds
no Robin support.

`BasicVectorBoundary` supplies full vector condition/value callbacks for the
legacy basic-vector path. `VelocityBoundary` instead supplies velocity values
and a selector for boundary diffusion, with selectable non-orthogonal
correction. Momentum has the velocity-selector contract plus boundary viscosity
and its supported stress selector. These overloads retain separate boundary
semantics. A migrated value-only basic caller must supply an explicit
Dirichlet condition; it must not choose the velocity-selector overload merely
to omit a condition callback.

Basic scalar options default to an absent `correction`, which selects the
existing two-point assembly. A present `TransportCorrection<Field>` selects
non-orthogonal assembly; its default treatment is `Explicit`, with an optional
lagged field. Thus absent correction and explicit correction are distinct
choices. Weighted scalar, temperature, selector-vector, and momentum options
always carry a correction group whose default treatment is `Explicit`.

Where exposed, discretization defaults remain Backward Euler, first-order
upwind, and transient assembly. Coefficient interpolation defaults to harmonic.
Weighted scalar on both backends and native temperature support constant-step
BDF2 and bounded linear-upwind convection. BDF2 requires an older field on the
same mesh and uses the current storage coefficient for the complete derivative.
A distinct accepted-old coefficient/property selection requires Backward Euler.
Bounded linear upwind retains the implicit upwind matrix and adds its existing
limited conservative deferred correction to the RHS. These scalar schemes do
not extend basic scalar, vector, legacy temperature, or momentum support.

Native ALE accepts a validated active control-volume state. Storage uses
conservative accepted-old/new volumes and the corresponding old/current
properties. Physical temperature and vector/momentum families consume the
synchronized owner-oriented mesh-relative volumetric flux. Weighted scalar
retains the caller's dimensionally consistent flux and advection coefficients:
`VariableDensityMomentumEquation`, for example, supplies relative mass flux in
kg/s with unit advection weight and density storage; density must not multiply
that mass flux again. Absolute pressure/continuity flux remains separate. ALE
retains its Backward-Euler-only restriction and the existing geometry, time-step,
and GCL checks. Legacy contexts do not expose ALE. See the
[planar ALE model](../modeling/planar_free_surface_volume_budget.md).

## Storage reuse and temperature RHS refresh

`assemble()` rebuilds numerical matrix values on every call, including when a
compatible matrix allocation/graph is supplied in `context.cached_matrix`.
Native weighted scalar may additionally reuse a symbolic plan and RHS storage;
the RHS must use the exact owned map and its values are reset. Reusing geometry
or allocation does not assert unchanged coefficients, time step, boundary
conditions, fluxes, or geometry epoch. Graph and identity validation remains in
the owning assembly path.

`refresh_temperature_rhs()` requires an existing operator in
`context.cached_matrix` and preserves its numerical values while recomputing
the complete temperature RHS. Native refresh retains the physical-temperature
restriction to Backward Euler and first-order upwind; advanced scalar schemes
require `assemble()`. Steady/transient mode remains the caller's existing
operator selection. Refresh does not prove that the supplied matrix represents
the requested operator. The equation retains responsibility for prepared
operator eligibility, cache ownership, invalidation, and failure cleanup, and
must reassemble whenever the matrix would change.

## Grouped examples

For legacy or native scalar fields selected by `MeshType`, two-point scalar
assembly with an explicit Dirichlet contract is:

```cpp
using Factory = FVM::TransportSystemFactory<Pack, MeshType>;
const auto condition = [](int, size_t) {
    return BoundaryCondition{BoundaryConditionType::Dirichlet, 0.0};
};
const auto boundary_value = [](int, size_t) { return 0.0; };
const auto system = Factory::assemble(
    typename Factory::ScalarTerms{
        .history = {.old_values = old_values, .time_step = dt},
        .face_fluxes = flux,
        .coefficients = {.diffusivity = diffusivity}},
    {.condition = condition, .value = boundary_value},
    {});
```

An explicit non-orthogonal scalar selection uses
`Factory::ScalarOptions{.correction = FVM::TransportCorrection<ScalarField>{
.treatment = FVM::NonOrthogonalTreatment::Explicit, .lagged_field = &lagged}}`
in place of the empty options pack. `ScalarField` is the transported field's
actual type.

Weighted assembly groups history, independent coefficients, sources,
constraints, boundary coefficients, policies, and reusable storage:

```cpp
const auto weighted = Factory::assemble(
    typename Factory::WeightedScalarTerms{
        .history = {.old_values = old_values, .time_step = dt},
        .face_fluxes = flux,
        .coefficients = {.storage_weight = storage,
            .advection_weight = advection,
            .diffusivity = diffusion,
            .old_storage_weight = &accepted_storage},
        .sources = {.source = source, .implicit_sink = sink},
        .constraints = {.fixed_cell_value = fixed_value}},
    {.condition = condition, .value = boundary_value,
        .coefficients = boundary_diffusion},
    {.correction = {.treatment = FVM::NonOrthogonalTreatment::Explicit,
         .lagged_field = &lagged},
        .coefficient_interpolation = FVM::FaceCoefficientInterpolation::Harmonic},
    {.cached_matrix = matrix, .geometry_cache = &geometry});
```

On `MeshHandle`, temperature can retain accepted-old properties independently:

```cpp
using Thermal = FVM::TransportSystemFactory<Pack, MeshHandle<Pack>>;
typename Thermal::TemperatureTerms terms{
    .history = {.old_values = temperature, .time_step = dt},
    .face_fluxes = relative_flux,
    .material = {.density = density,
        .specific_heat_capacity = cp,
        .thermal_conductivity = conductivity,
        .old_density = &accepted_density}, // Accepted-old cp falls back to cp.
    .power_density = heat_source};
typename Thermal::TemperatureBoundary boundary{
    .condition = thermal_condition, .value = thermal_value};
typename Thermal::TemperatureOptions options{};
typename Thermal::TemperatureContext context{
    .cached_matrix = prepared_matrix, .geometry_cache = &geometry, .ale = &ale};
const auto system = Thermal::assemble(terms, boundary, options, context);
// Once the equation has established that its operator is unchanged:
context.cached_matrix = system.matrix;
const auto refreshed = Thermal::refresh_temperature_rhs(terms, boundary, options, context);
```

## Providers, exports, and source migration

Ordinary consumers include the declaration header and link the FVM library.
Default providers under `src/FVM/explicit_instantiation/` instantiate supported
factory member overloads individually. Geometry-cache providers stay separate.
Public entry points retain targeted export annotations, including the static
Cartesian visibility handling. The declaration-only factory header does not
enumerate redundant `extern template` declarations. `TransportSystem.tcc` retains
the default legacy weighted-scalar suppression needed when other implementation
providers reach that overload, including under ThinLTO. Private assembly helpers
remain local to their owning image; finding a single `assemble` export does not
establish all families.

A custom mesh or type pack supplies the required overload definitions by
including `FVM/TransportSystemFactory.tcc` in a provider translation unit and
explicitly instantiating the overloads it needs. Do not instantiate the whole
factory class to claim unsupported families. For example, with a named
`CustomMesh` and matching fields:

```cpp
#include "CustomMesh.hh"
#include "FVM/TransportSystemFactory.tcc"
#include "fields/FieldStored.tcc" // If custom storage lacks compiled providers.

namespace SimpleFluid::FVM {
using CustomFactory = TransportSystemFactory<DefaultTpetraTypes, CustomMesh>;
template TransportSystem<DefaultTpetraTypes> CustomFactory::assemble(
    CustomFactory::WeightedScalarTerms,
    CustomFactory::WeightedScalarBoundary,
    CustomFactory::WeightedScalarOptions,
    CustomFactory::WeightedScalarContext);
}
```

Custom assembly/cache paths that construct traversal plans also need their
mesh's provider through `geometry/MeshTraverse.tcc`; custom ALE validation uses
`FVM/ALEControlVolumeState.tcc`. Keep providers for translation-unit-local mesh
types in the translation unit that defines the type. Named reusable custom
types can place these definitions in separate provider units. Public headers
and `.ipp` files must not pull heavy factory implementations into consumers.

This migration intentionally breaks source and ABI compatibility. The old
transport free functions and flat weighted request aliases are removed.
Downstream consumers must migrate and rebuild; unchanged result layouts do not
make old function symbols or pack layouts ABI-compatible.

| Previous source form | Factory form |
| --- | --- |
| Scalar `transport_system(...)` | `assemble(ScalarTerms, ScalarBoundary, ScalarOptions, ScalarContext)` |
| Vector `transport_system(...)` | Legacy `assemble(VectorTerms, BasicVectorBoundary, BasicVectorOptions, BasicVectorContext)` |
| Scalar `non_orthogonal_transport_system(...)` | Scalar `assemble` with a present correction group |
| Vector `non_orthogonal_transport_system(...)` | `assemble(VectorTerms, VelocityBoundary, VectorOptions, VectorContext)` |
| `weighted_scalar_transport_system(request)` or positional overload | `assemble(WeightedScalarTerms, WeightedScalarBoundary, WeightedScalarOptions, WeightedScalarContext)` |
| `physical_temperature_transport_system(...)` | `assemble(TemperatureTerms, TemperatureBoundary, TemperatureOptions, TemperatureContext)` |
| Physical temperature with `rhs_only = true` | `refresh_temperature_rhs` with an existing matrix |
| `physical_momentum_transport_system(...)` | `assemble(MomentumTerms, MomentumBoundary, MomentumOptions, MomentumContext)` |

Move time/history fields into `history`, coefficients into `coefficients` or
`material`, callbacks into the appropriate boundary/source group, correction
selection and lagged field together into `options.correction`, and cache/ALE
resources into `context`. Preserve each old non-orthogonal selection explicitly.
Value-only basic calls add a Dirichlet condition callback, and omitted explicit
sources use the zero default. A former native temperature `AssemblyMode` choice
moves to `options.discretization.mode`; legacy temperature and momentum use
`options.mode`.
