# Mesh-to-mesh cell-field transfer

`MeshToMeshTransfer<Pack>` in `fields/MeshToMeshTransfer.hh` builds a reusable
mapping between two `MeshHandle<Pack>` objects. It transfers scalar, vector,
and tensor cell fields, or matching columns of owned Tpetra multivectors.
This is a spatial transfer utility for future coupling; the caller owns the
coupling schedule and the physical meaning of the transferred quantities.

## Choosing a method

| `MeshToMeshTransferMethod` | Definition | Supported geometry | Numerical contract |
| --- | --- | --- | --- |
| `NearestCell` | Copy the value at the closest source-cell centroid. | Every `MeshHandle` family. | Preserves constants and donor bounds; discontinuous and generally not conservative. |
| `InverseDistance` | Average the nearest source-cell centroids with normalized inverse-square distance weights. | Every `MeshHandle` family. | Preserves constants and donor bounds; generally neither linear-exact nor conservative. |
| `ConservativeCellAverage` | Build exact cell-overlap volumes for conservative projection. | Two native orthogonal Cartesian handles, or two coaxial native cylindrical handles, including independent partitions and reordered cells. | Preserves fully covered constants and inventories; partial transfers explicitly report uncovered inventory. |

Nearest and inverse-distance methods use Euclidean distances between physical
cell centroids. A centroid match returns its donor value directly; equal-distance
donors are ordered deterministically by their source global IDs. These methods
can extrapolate beyond the source domain. `max_distance` limits the distance to
the nearest donor; it does not require every inverse-distance donor to lie
within that radius or establish that the target cell lies inside the source
domain. With the default infinite cutoff, extrapolation is unrestricted.
Centroid distance does not account for region or material labels, mesh
connectivity, or an intervening wall. A nearby donor across such a boundary is
eligible; callers must choose source and target domains appropriate for the
coupled quantity.

For inverse-distance interpolation at target centroid \(x_j\), the weights are

\[
w_{ji}=\frac{\|x_j-x_i\|^{-2}}
              {\sum_{k\in N_j}\|x_j-x_k\|^{-2}},
\qquad q_j=\sum_{i\in N_j}w_{ji}q_i,
\]

where \(N_j\) contains up to `neighbors` source cells. Exact centroid matches
are handled separately, without dividing by zero.

For conservative cell averages, with source cell \(S_i\) and target cell
\(T_j\),

\[
q_j=\frac{1}{|T_j|}\sum_i |S_i\cap T_j|q_i.
\]

Construction requires complete coverage of both source and target cells by
default. `coverage_mode = MeshToMeshCoverageMode::AllowPartial` explicitly
permits different domains, including disjoint meshes. A partial transfer must
use `project()` and its conservation report; the older `apply()` entry point
rejects partial mode. Significant excess coverage is rejected in either mode.
No overlap weights are renormalized to conceal uncovered volume.

The map uses piecewise-constant source averages and does not reconstruct a
linear profile inside a cell. Cartesian/cylindrical cross-family intersections,
general unstructured conservative intersections, and multi-region conservative
intersections are not implemented. Interpolation still supports all handle
families.

### Cylindrical geometry and angular measure

For coaxial annular sectors, each overlap volume is exactly

\[
V_{ji}=\tfrac12(r_{hi}-r_{lo})(r_{hi}+r_{lo})
       \,\Delta\theta_{ji}\,\Delta z_{ji}.
\]

Radii and axial intervals are intersected directly; azimuthal intervals are
intersected modulo `2*pi`, including different seam locations, negative angles,
and sectors crossing the seam. Coordinates use the common global Z axis,
origin and length units of the native mesh; angles are radians. The existing
native `OrthogonalCylindrial3D` requires positive inner radius (the TRACY R100
value `0.03815 m` is supported). A full annulus volume is
`pi*(r_outer*r_outer-r_inner*r_inner)*height`; a sector uses its actual angular
span instead of `2*pi`. There is **no implicit symmetry multiplier**.

Mapping an open sector to a full annulus therefore fails in `RequireFull` mode.
`AllowPartial` transfers only their physical intersection and reports the
remaining full-annulus volume. It does not extrapolate an axisymmetric field
around the rest of the circle. A consumer choosing rotational symmetry must
represent that symmetry explicitly before transfer.

A native full annulus needs at least two azimuthal cells. An external
axisymmetric neutron cell covering the whole annulus can be represented by
multiple azimuthal slices: replicate its intensive value, but split its
extensive inventory in proportion to slice volume. On return, sum extensive
slice inventories or volume-average intensive slice values. Do not copy the
whole-annulus inventory into every slice or apply another `2*pi` factor.

The native mesh tolerates angular spans marginally above `2*pi` due to
roundoff. Overlap geometry counts the first physical turn only; native cell
volumes are retained. Any excess appears in the coverage residual rather than
being duplicated or absorbed into normalized weights. All reported uncovered
volumes are signed `cell_volume - covered_volume` residuals; tiny negative
roundoff is also retained. `coverage_tolerance` controls acceptance, not clipping.

## Options and units

| `MeshToMeshTransferOptions` member | Default | Meaning |
| --- | --- | --- |
| `method` | `InverseDistance` | Mapping method. |
| `neighbors` | `8` | Maximum donor count for inverse-distance interpolation; an integer in `[1, INT_MAX]`. |
| `max_distance` | Positive infinity | Nonnegative maximum accepted nearest-donor distance for centroid interpolation, in mesh coordinate units; positive infinity is allowed. |
| `coverage_tolerance` | `1e-10` | Relative cell-volume tolerance for conservative coverage checks, in `[0, 1e-6]`. |
| `coverage_mode` | `RequireFull` | Require both meshes to be fully covered, or explicitly select `AllowPartial` for conservative projection. |

Source and target coordinates must use the same coordinate frame and units.
Transfer preserves the field's units; it performs no unit conversion. Vector
and tensor components are transferred independently in their existing basis,
without rotation into a local cylindrical or other coordinate basis.

## Field API

Use existing source and target handles, after initializing the usual
Tpetra/Kokkos runtime:

```cpp
#include "fields/MeshToMeshTransfer.hh"

using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;

void transfer_temperature(
    SP<const MeshHandle<Pack>> source_mesh,
    SP<const MeshHandle<Pack>> target_mesh)
{
    ScalarCellFieldStored<Pack> source(source_mesh, "temperatureSource");
    ScalarCellFieldStored<Pack> target(target_mesh, "temperatureTarget");

    // Populate owned source values from the source solver.
    for (size_t cell = 0; cell < source_mesh->num_owned_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        source.set_owned_value(lid, 300.0);
    }

    MeshToMeshTransferOptions options;
    options.method = MeshToMeshTransferMethod::InverseDistance;
    options.neighbors = 8;
    options.max_distance = 0.1; // Mesh coordinate units.
    MeshToMeshTransfer<Pack> transfer(source_mesh, target_mesh, options);
    transfer.apply(source, target);

    // Reuse transfer after updating the source field on the same geometry.
    // apply() synchronizes target ghost values.
}
```

The same transfer object accepts matching `VectorCellFieldStored<Pack>` or
`TensorCellFieldStored<Pack>` fields on those handles. Source ghost values need
not be synchronized: transfer reads owned source values and imports remote
donors through Tpetra. Target owned values are replaced and target ghosts are
synchronized. Field names, boundary conditions, solver histories, and material
metadata are outside this operation.

## Conservative quantities and coverage

`project()` requires an explicit `MeshToMeshQuantity`, using the same
representation for both its input and output:

| Quantity | Input and output | Target formula |
| --- | --- | --- |
| `Intensive` | Cell averages or densities, such as J/m³ or mol/m³. | `sum(overlap_volume * source_value) / target_cell_volume` |
| `Extensive` | Integrated cell inventories, such as J or mol. | `sum(overlap_volume * source_inventory / source_cell_volume)` |

With partial coverage, intensive output is the contribution averaged over the
**entire target cell**, not over just its covered part. Uncovered target portions
contribute zero to this transfer. A constant source is preserved only where a
target cell is fully covered; elsewhere the result is that constant times the
target covered-volume fraction. This value is not a complete thermodynamic
state for an unfilled target cell. TF owns occupancy, headspace state and any
other contribution to that target cell.

```cpp
MeshToMeshTransferOptions options;
options.method = MeshToMeshTransferMethod::ConservativeCellAverage;
options.coverage_mode = MeshToMeshCoverageMode::AllowPartial;
MeshToMeshTransfer<Pack> projection(source_mesh, target_mesh, options);

// Inputs and outputs are already cell inventories; no caller volume factors.
auto report = projection.project(source_joules, target_joules,
                                 MeshToMeshQuantity::Extensive);
const auto& coverage = projection.coverage();
// Retain/account for report.uncovered_source_integral[0] in the caller.
```

The existing `apply(source, target)` remains available for conservative
**full-domain intensive** transfer. For diagnostics or any extensive quantity,
use `project()` instead. Both field overloads refresh target ghosts. Raw
multivector overloads support one or more independent component columns,
including noncontiguous column subviews, without changing unselected columns.
Every column in one call uses the same quantity representation; different
columns may have different physical units.

`coverage()` returns a borrowed `const MeshToMeshCoverage&` owned by the
transfer. Its vectors follow the corresponding owned-cell map order:

- `source_cell_volumes`, `source_covered_volumes`, `source_uncovered_volumes`;
- `target_cell_volumes`, `target_covered_volumes`, `target_uncovered_volumes`.

Global scalar totals are `source_volume`, `target_volume`, `overlap_volume`,
`uncovered_source_volume`, and `uncovered_target_volume`. Only owned cells enter
reductions; ghost entries are never counted. Calling `coverage()` collectively
validates geometry epochs and maps before returning the construction snapshot.
Keeping a reference or copying this snapshot does not make it follow mesh
motion. Its lifetime is bounded by the transfer object.

`MeshToMeshConservationReport` contains globally reduced vectors with one entry
per field component, identical on all ranks:

- `source_integral`: complete source inventory, or integral of source averages;
- `transferred_integral`: source inventory supported by the overlap;
- `uncovered_source_integral`: source inventory outside the overlap;
- `target_integral`: integral/inventory of the values actually returned;
- `transfer_error = target_integral - transferred_integral`;
- `conservation_error = target_integral + uncovered_source_integral - source_integral`.

All report entries use integrated units. Numerical errors remain visible;
reports never correct, rescale or clip input/output values. Signed finite field
values are supported. Nonfinite input, output or report entries cause a
collective failure before target values are overwritten. Partial transfer does
not consume or erase the source: the caller retains its uncovered inventory.
For cellwise retained extensive inventory, use
`source_inventory[i] * source_uncovered_volumes[i] / source_cell_volumes[i]`.
For cell averages/densities, use
`source_value[i] * source_uncovered_volumes[i]`.

Conservation of a phase inventory requires the caller to supply the actual
phase inventory or a density that already includes the relevant phase fraction.
The library neither calculates pool/headspace occupancy nor closes coupled
mass, fissile species, energy or hydrogen budgets. Those remain TF physics.

## Raw multivectors and lifetime

`apply(const Pack::multi_vector_type& source,
Pack::multi_vector_type& target)` transfers equally many columns. The input
and output maps must match the source and target owned-cell maps respectively.
This overload has no field or ghost-storage side effects, so an external
coupling adapter is responsible for updating its own halo or field views.
Source and target storage may alias: application computes and validates a
temporary result before replacing the target values.

The transfer retains its two mesh handles and records their geometry epochs.
The field overload requires the exact handles used at construction. An apply
after either geometry epoch changes fails; rebuild the transfer after mesh
motion. Repartitioning or replacing either mesh also requires a new transfer.
The operation rejects non-finite field values. Trial motion and geometry
rollback both invalidate a plan, even when restored coordinates equal the
original coordinates. Reconstruct explicitly after either operation. Stored
source and target map snapshots are checked as well as both geometry epochs.

Construction uses one fixed collective context, independent of whether a
rank has a null source or target handle. The existing constructor
`MeshToMeshTransfer(source, target, options)` is collective on
`Tpetra::getDefaultComm()`. Callers participating in only a subgroup **must**
use the explicit-context overload:

```cpp
MeshToMeshTransfer<Pack> transfer(*construction_comm,
                                source_mesh, target_mesh, options);
```

Every participant must supply the same construction communicator context.
Distinct `MPI_CONGRUENT` duplicates are not interchangeable rank by rank:
MPI collectives on them do not match. Each endpoint must likewise represent
one coherently constructed distributed mesh with a consistent communicator
context across its participating ranks. The source context, target context
and construction context may be three distinct congruent communicators;
their groups and rank ordering must match. Source and target decompositions
may differ. Missing handles and incompatible communicator groups are rejected
collectively on the fixed construction context before entering endpoint
collectives. The explicit communicator reference need only live through
construction; the retained meshes/maps own their endpoint communicators.

`coverage()`, `apply()` and `project()` are collective on the captured source
communicator. Every rank must participate in the same operation with consistent
options and quantity representations, even when it owns no source or target
cells. Raw multivectors must be coherent distributed objects with consistent
component counts across the participant group. Null multivector maps, including
those on ranks excluded by Tpetra `replaceMap()`, are rejected collectively
before communicator access or target mutation. Empty but non-null cell maps
are supported. Intercommunicator exchange and coupling separately launched
solvers are not implemented.

For very coarse native Cartesian or cylindrical grids, set
`MeshHandle<Pack>::DistributionOptions::allow_empty_partitions = true` on all
ranks. This opt-in allows ranks beyond the available nonempty axis slabs to
have empty owned and overlap maps; the existing default still rejects that
layout. It supports independently assigned source/target slab partitions on
the same communicator. Empty-rank fields and transfer calls remain collective.
This does not establish empty-rank support for unrelated solver paths.

## Storage, build, and current limits

Construction temporarily gathers owned source-cell geometry on every rank and
searches it for each locally owned target cell. Its geometry memory is
\(O(N_s)\) per rank, with \(O(N_sN_{t,\mathrm{local}})\) search work before
donor ordering. The reusable result is a sparse Tpetra transfer matrix; repeated
application imports required donor values rather than gathering the whole
source field. Large production meshes may need a distributed spatial search
before construction is affordable.

Link the `SimpleFluid::Fields` CMake target and include the public header.
`DefaultTpetraTypes` uses the library's explicit instantiation. A custom type
pack must deliberately include `fields/MeshToMeshTransfer.tcc` and provide its
corresponding compile/link support, following the other explicit template
instantiations in the project.

The current API transfers cell data. Face flux transfer, boundary mortar
mapping, geometric phase occupancy, high-order reconstruction, mixed-coordinate
intersections, and arbitrary polyhedral conservative intersections remain
separate work.
