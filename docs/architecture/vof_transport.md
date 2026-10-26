# Floating-point VoF transport and state ownership

The opt-in fixed-mesh face-stream redesign is documented in
[conservative face transport](conservative_face_vof.md). It separates donor-cell
admission, measure-only PLIC candidates, shared limiting, and reusable momentum
from the legacy swept partition route described below. The default has not
switched.


Floating-point arithmetic is the primary choice for VoF. The general-native
backend uses `ArithmeticMode::Ordinary` by default; Cartesian VoF retains
its specialized FP reconstruction and directional sweeps. Conservation,
positivity, physical tolerances and atomic acceptance apply throughout.
Optional rational-path contracts live in [ExactVoF](exact_vof.md),
[ExactTransport](exact_transport.md) and [ExactGeometry](exact_geometry.md).

The geometric fraction operator and the momentum operator have distinct time
integration. The Cartesian fraction operator uses directional PLIC sweeps with a frozen
Weymouth–Yue indicator. Momentum uses a conservative implicit upwind equation
on the integrated mass ledger produced by those sweeps. Reusing the swept
volumes with explicit accepted-old donor velocities is incompatible: material
received in one direction can leave in the next direction with a different
velocity from the original cell.

## Conservation requirements and arithmetic choices

For an extensive phase volume `q`, a source-free face-ledger update has the form
`q_new[c] = q_old[c] - sum_f s[c,f] * dV[f]`. One canonical `dV[f]` is used by
both incident cells with opposite incidence signs. Coarse/fine cells consume the
same canonical subfaces exactly once. This gives algebraic cancellation of
internal transfers; floating-point cell updates and reductions still require
controlled accumulation error. It does not imply bitwise equality of independently
summed global totals.

The essential requirements are conservative phase accounting, nonnegative
capacity-limited transfers, compatible phase/total-volume fluxes, preservation
of constant states, and accuracy of interface transport. The coupled flow solver
also requires mass-compatible momentum and its original physical-residual gates.
These properties do not require arbitrary-precision rational coordinates,
volumes or first moments as a general rule. Basic PLIC VoF does not evolve first
moments; methods using centroids or higher-order remapping need accurate moments,
but not necessarily rational ones.

The general-native default uses ordinary FP geometry and compensated numerical
ledgers. `GeneralVoFOptions::arithmetic_mode` selects `Ordinary` (default),
`BoundedCertification`, or `Exact`; the latter two are explicit opt-ins under the [FP geometry contract](general_amr.md#floating-point-geometry-and-arithmetic).
The replacement is implemented with focused component coverage; full coupled
flow qualification remains open. The directional Cartesian roundoff rule below
is not a blanket allowance for the general-mesh backend.

Configuration remains a runtime choice. Operation entry points dispatch into
compiled `Ordinary` or `BoundedCertification` template kernels; scalar arithmetic,
compensated sums, reconstruction, transport/remap, projection and momentum use
fixed-mode helpers and `if constexpr` inside those kernels. `Exact` continues
through its separate rational backend. External value tags, packet provenance,
resource limits and collective mode agreement remain runtime checks, including
on empty owners. No mode silently falls back to another arithmetic policy.

The private `Arithmetic<Mode>` and `SumFor<Mode>` helpers are implementation
interfaces, not template parameters on the public VoF API. Inferred external
values retain their checked runtime adapters. Ordinary geometry has dedicated
scalar numerical providers and retains one `long double` per coordinate, without
interval endpoint accessors. Separate certified providers retain two endpoints
and a nominal estimate and own interval propagation and certification retries.
Vertices, planes, integral caches and planar geometry use the selected scalar
storage behind checked runtime adapters. Mode checks precede typed access,
and allocation accounting uses the actual retained types. Spatial bounding boxes
still retain both coordinate extrema. Heavy template definitions remain in
compiled `.tcc` providers with explicit instantiations for the supported modes.

PLIC reconstruction, convex geometry, planar geometry and their private helpers
live under `src/geometry/PLIC`. Shared headers use neutral filenames such as
`ConvexGeometry.hh`; mode-specific providers live in `details/ordinary/`,
`details/bounded/`, and `details/exact/`, with neutral filenames within each
directory. Shared finite reconstruction and arithmetic helpers live directly in
`details/`. Include paths use `geometry/PLIC/`. Reconstruction and phase arithmetic retain their existing
`SimpleFluidFVM` compilation and export ownership.

The typed geometry API lives in `SimpleFluid::PLIC`.
`ConvexCellGeometry<Mode>`, `Point<Mode>`, `GeometryValue<Mode>` and typed
clip/moment results carry arithmetic policy in their types. Ordinary values
store one scalar and expose no enclosure queries or error field. Bounded values
retain their error enclosure. The exact cell specialization retains rational
geometry; its published `CertifiedGeometryValue<Exact>` is a floating-point
view with conversion uncertainty and an independent nonzero marker, not the
rational scalar itself. `GeometryValue<Mode>` aliases the corresponding published
value type. Binary64 publication rejects overflow and nonzero disappearance;
its error query includes additional narrowing error. In Ordinary that query
bounds only narrowing of the stored scalar, not untracked construction error.

VoF selects its finite arithmetic mode once at each operation entry, including
source construction, reconstruction, transport and remap. Cell registries, swept
pieces, donor partitions, point trajectories and solver face overlaps then use
`ConvexCellGeometry<Mode>`, `ConvexPartition<Mode>` and `Point<Mode>`. Typed
geometry methods call their selected compiled kernels directly, including nested
clipping and partition work; they do not re-enter runtime backend selection.
`certify_fragment_partition()` preserves transport's complementary-split and
donor-batch proofs; native AMR retains its separate shell partition proof.
Reconstruction uses `PLIC::PolyhedralReconstruction<Mode>` and matching
`PolyhedralPhaseSample<Mode>` throughout offset and normal fitting. Its common
`validate`, `solve_offset` and `reconstruct` entries include an explicit Exact
specialization; the finite modes additionally provide `assess_resolution`.
There is no separate `VoF::Exact` reconstruction class or compatibility alias. Runtime state, packet and diagnostic records retain checked `Dynamic*`
representations where an external boundary requires them. The shared runtime
geometry handle exposes storage metadata only; numerical operations require a
typed handle. Untyped geometry getters and reconstruction entries have been
removed. Conversion checks the active mode; cross-mode geometry arguments are
incompatible at the typed API. `GeometryStatistics::runtime_dispatches` counts
the remaining runtime value-query selections separately from arithmetic work.
Typed execution requires zero such selections in each library's instrumentation
domain. Metadata such
as `PlaneAuthority`, `GeometryPacketContext` and resource caps remains independent
of scalar representation. Native mesh/AMR clients use the ordinary PLIC types
directly. The former `Meshes` geometry aliases and public `Exact*` geometry
headers are removed; consumers must migrate and rebuild.

`PlanarGeometry.hh`, `PolyhedralComplex.hh`, `ExtrudedConstruction.hh`,
`GeometryMeasure.hh` and `SweptGeometry.hh` are public PLIC boundaries. Backend
specializations live under `details/ordinary/`, `details/bounded/`, or
`details/exact/`. Native source preparation, packets and AMR no longer construct
rational geometry as a prerequisite. Exact algorithms explicitly import the
represented ordinary source through the Exact specialization when that route is
selected. No arithmetic-mode conversion is an automatic failure fallback.

Native AMR consumes Ordinary cells, planar faces, extrusions and complexes.
BoundedCertification applies to the finite convex and planar-overlap kernels;
it is not a second native hierarchy implementation. Exact swept geometry is
available through `SweptFace<Exact>` and its related public interfaces; finite
sweep transport remains FVM-owned. Unsupported family/mode combinations do not
silently select another backend.

Include `geometry/PLIC/ConvexGeometry.hh` for the finite typed API:

```cpp
using Cell = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>;
const auto cell = Cell::box({0, 0, 0}, {1, 1, 1});
const auto cut = cell->clip({{1, 0, 0}, {0, 0, 0}, .25});
const auto volume = cut.inside->volume(); // GeometryValue<Ordinary>; scalar only.
```

Simple one-line member queries and forwarding functions remain in the public
headers. Conversion, allocation, and resource-accounting implementations remain
in compiled providers with explicit instantiations for supported finite modes.

Geometry rejection diagnostics identify `Ordinary FP geometry` or
`Bounded FP geometry` when the arithmetic mode is known. Ordinary construction
requires a strictly positive computed signed volume and reports its scalar value
on failure; it does not apply an interval positivity certificate. Certification
retains its lower-bound and construction-proof checks. A diagnostic without an
established mode uses the neutral `Finite FP geometry` prefix.

Runtime value/error views and source-provenance snapshots retain their checked
interfaces. Version 3 numerical packets retain their two-endpoint encoding for
compatibility: Ordinary writes its scalar twice, verifies equal endpoints when
decoding, and stores one scalar in memory. Packet encoding does not introduce
interval arithmetic or certified bounds into Ordinary execution.

Physical interface resolution is a separate admission decision. The mandatory
arbitrary subcell-phase fitting case has been withdrawn. The general
backend reports insufficient interface support according to
[mesh/stencil resolution](general_amr.md#resolution-before-additional-precision),
without silently deleting its accepted phase amount. Small numerical inventories
still require conservative accounting; they do not prove a resolved interface.
The implemented support/model screen emits actionable refinement requests; it
is a necessary admission check, not a complete thickness/curvature convergence
test. A caller must refine or reinitialize and retry; no adaptation controller
runs inside a physical trial.

Under the [finite arithmetic policy](general_amr.md#primary-floating-point-policy),
ordinary arithmetic accepts represented scalar results against the configured
physical tolerances, without interval-width admission. Finite values, actual
phase nonnegativity, conservation, topology, solver residuals and resource caps
remain checked. `BoundedCertification` additionally propagates enclosures and
rejects unresolved certification. Source import consumes represented native
coordinates and retained support identities. Bounded certification encloses
operations on those represented inputs; it does not retrospectively certify
untracked ordinary construction error. Optional rational conversion is described in
[ExactGeometry](exact_geometry.md#native-authority-and-the-fp-import-boundary).
Small intermediate contributions still enter conservative
aggregate accounting and error bounds. Removing exact per-fragment values does
not authorize a phase cutoff, mass deletion, looser physical tolerances or larger
resource caps.

Ordinary edge intersections may recover an original source point when its sealed
identity belongs to every retained support and three of those original supports
are independently determined. A scalar rank filter accounts for normal
construction and determinant roundoff; it guards identity reuse without changing
volume or solver tolerances. Coordinate proximity and small residuals do not
establish identity. Missing or ambiguous evidence leaves the nominal construction
unchanged. Packet coordinates cannot create source identities; a decoded native
plane can recover rank evidence only from its pinned source registry after the
existing authority checks and an identical or opposite normal match.
Two triangular supports with the same three distinct live source identities and
proved noncollinear defining points establish the same unoriented plane. This
preserves zero incidence across opposite windings without inferring an inside or
outside orientation from the identity proof. Rounded normals alone, expired
tokens and coordinate look-alikes cannot establish this equivalence.

For ordinary general-native transport, each canonical face receives a fixed
construction allowance from its incident cells: `B_f = tau * min(V_L/N_L,
V_R/N_R)`, using the single incident cell at a physical boundary. Here `tau`
is the configured volume tolerance, `N` counts all incident canonical subfaces,
and `V` is the smaller of the native and represented geometry volumes. Cell
owners supply these quantities; both sides agree on the same face allowance.
The actual signed sum of stored swept-piece volumes is compared with the
unchanged canonical control. This avoids imposing a relative accuracy target
on a face flux that can approach zero.

The trial records each locally retained face's signed construction difference
and the owned-cell signed and absolute totals. Shared face records repeat across
ranks and must be deduplicated by canonical face identity for global summaries.
The canonical cell residual plus its incident
signed construction differences must satisfy the original cell-volume
allowance; the two residuals do not receive separate additive budgets. These
nominal diagnostics are not uncertainty bounds. Donor availability, recipient
occupancy, independent phase conservation and momentum checks still apply.

## Implicit momentum contract

For each velocity component, the private `ConservativeVoFTransport` component
solves

$$
M^\mathrm{new}_c u^\mathrm{new}_c + \sum_f s_{c,f} F_f u^\mathrm{new}_{\mathrm{upwind}(f)} = P^\mathrm{old}_c + S_c.
$$

$M^\mathrm{new}$ is the final extensive mixture mass in kg. $F$ is the signed integrated
mass flux in kg on one canonical face. Incidence signs account for coarse/fine
subfaces exactly once. Incompressible transport forms these quantities from
constant phase densities and the geometric phase-volume ledger. Low-Mach
transport forms them from the final liquid/species/population inventories and
their actual mass fluxes. It never substitutes geometric gas volume for a
species molar flux.

With no physical sources, $M^\mathrm{new} + \nabla\cdot F = M^\mathrm{old}$. The implicit upwind matrix
has diagonal $M^\mathrm{new} + \mathrm{outgoing}(F)$ and nonpositive incoming entries. Its row
sum is positive $M^\mathrm{old}$. The inverse is nonnegative, constants are preserved,
and the transported passive velocity lies within its old bounds up to the
linear residual and floating-point accumulation. Summing rows cancels every
internal momentum flux, preserving total momentum. No gas- or liquid-specific
velocity division is needed at vanishing phase fractions.

The directional compression terms cancel in the divergence-free limit. For
physical volume sources, low-Mach phase storage includes both the directional
correction and $\Delta G - \chi(\Delta G+\Delta C)$ before species and mass recovery.
Physical momentum sources remain integrated physical impulses. Neither a
source momentum nor a geometric compression velocity is invented from a
vanishing phase fraction. A general velocity maximum principle is not asserted
when pressure, capillarity, gravity, phase sources, or drift impulses act.

The low-Mach velocity is the continuous/liquid velocity. Dispersed drift adds
physical momentum $\sum_k W_\mathrm{H_2} n_k w_k$. Its face impulse is included separately;
the new cell drift momentum is removed from the velocity equation and restored
to the conservative physical ledger. Viscosity, gravity, pressure and capillary
impulses follow transport. The pressure/divergence and extensive physical gates
remain independent of the transport linear solver.

The method is first order and introduces implicit upwind diffusion. Its cost
includes three momentum solves per transport update. Geometric PLIC interface
accuracy does not imply second-order velocity or species transport. Matched
temporal/spatial convergence studies remain necessary for application accuracy.

## Directional phase budgets at floating-point resolution

PLIC offsets cannot geometrically represent every positive microscopic fraction.
Each directional sweep checks its donor's outgoing gas and complementary carrier
against their actual extensive inventories. An excess greater than the existing
`2e-12 * cell_volume` fraction-roundoff allowance rejects. A smaller excess is
corrected on the shared canonical face partition, retaining total face volume
and recomputing the complement. Owned donor budgets are rechecked collectively;
coarse/fine cells sum canonical subfaces without additional area weights.

The frozen-WY minority-phase storage uses a long-double `(old - outgoing) +
incoming` update. No cell fraction, velocity or extensive inventory is clipped.
A corrected flux can differ substantially relative to a microscopic phase whose
PLIC offset is below geometric precision, while the absolute adjustment remains
bounded by the roundoff allowance and unexported positive inventory stays in its
donor. This is conservative roundoff repair, not a model for subcell bubble shape.

## Geometry and numerical operator reuse

Immutable adaptive geometry is shared only for the same retained hierarchy
identity, whose topology, partition and native ownership layout are frozen.
Every rank must observe a cache hit before collective construction is skipped.
The metadata includes maps, import plans and flattened reconstruction ranges.
A regular full-cell donor stores an ordinal; coarse/fine intersections retain
explicit geometric overlap records. Each `AdaptiveVoF` instance owns its own
import/export multivectors, and every fraction-dependent result is call-local.
Reinitialization can reuse geometry; replacing the hierarchy rebuilds it.

PLIC Youngs gradients need one fraction layer. Height functions need three
axial layers and one transverse layer. A radius-two PLIC-centroid fit needs
one further layer for its plane gradients, giving a curvature radius of three.
Stored patches therefore use `block_cells + 6`; plain reconstruction samples
`block_cells + 2` and computes only requested owned planes. Curvature computes
only requested owned results and their consumed centroid neighborhoods. The
refinement policy still protects at least four cells, including corner and
periodic neighborhoods. Closed spatial-directory queries and exact original
overlap predicates replace exhaustive block/image candidate scans. Guards merge
only when their rectangular union is exact.

The conservative transport component retains a full native neighbor graph and
linear/import work vectors. An exact collective comparison of final capacities
and signed carrier fluxes identifies identical numerical operators. Source/RHS
values and positivity requirements remain per-quantity checks. Low-Mach liquid,
dissolved gas and donor equations form one adjacent RHS group; resolved hydrogen
and inert form another; each population's number and mole equations share its
carrier-plus-slip group. Momentum components share final mass storage and mass
flux. No operator is reused merely because two quantities have the same graph.
Changing coefficients replaces numerical values and explicitly invalidates
preconditioners. A failed refresh cannot publish a valid cache key.

Passive inventories use the same transport boundary with optional left row
equilibration. Each positive physical row is divided directly by its diagonal;
forming a reciprocal could overflow for a subnormal carrier. The inner relative
target is the larger of the requested physical tolerance and 64 binary64 epsilons.
Belos must satisfy its ordinary true-residual contract for that scaled solve.
A compensated residual of the actual amount and canonical face-flux ledger must
then meet the original unscaled physical tolerance. A global maximum-diagonal
bound is not a mandatory inner target: it can be unattainable despite an accurate
physical solution, or underflow to zero for a subnormal carrier.

If the physical gate fails, up to four zero-start corrections solve the same
equilibrated operator with the original physical defect divided by each row
diagonal. A collective power-of-two RHS normalization keeps correction norms in
range; concentration additions retain compensation across corrections. The
initial solve and corrections share one iteration budget, including any internal
Belos refinement. Exhaustion rejects the candidate. The original physical gate,
final positivity and global extensive ledger remain independent and no inventory
is clipped. Changing scaling policy invalidates numerical cache reuse.

The separate `VoFProjectionWorkspace` retains pressure graph, RHS, solution and
linear-solver storage. It refreshes numerical mobility and susceptibility on
every assembly and invalidates preconditioners. Nonorthogonal corrections and
the independent physical divergence acceptance remain in solver orchestration.
These workspaces retain no canonical physical inventory or phase-pressure history.

## Canonical state, views, caches, and history

| Storage | Role and lifetime |
|---|---|
| Incompressible fraction and velocity | Accepted physical state; mixture mass and momentum are derived using fixed phase densities and cell volumes. |
| Low-Mach liquid mass, gas species, dissolved/donor moles, population number/moles, internal energy, physical momentum | Canonical extensive inventories in private candidate or immutable accepted snapshots. |
| Thermodynamic pressure and phase-pressure history | Physical state/history needed for caloric recovery, pressure work, conservative representation changes and replay. |
| Temperature, densities, phase fractions, population radii/volumes | Derived EOS or moment recovery; rebuilt consistently from canonical inventories. |
| Native `FieldStored` output | Published view of an accepted snapshot. Every owned value is assigned before the final ghost synchronization. |
| Hierarchy maps, import plans, geometric patch metadata | Immutable geometry tied to the hierarchy identity, topology/partition generations and native layout. Regrid replaces that identity. |
| Reconstruction, curvature, linear algebra vectors and numerical operator values | Mutable context workspace. No retained preview or accepted snapshot owns aliases into this storage. |
| Trial flux ledger | Immutable candidate evidence, including phase, species, population, energy and physical momentum fluxes. |

A hierarchy identity is not a population-schema identity. Controllers resolve
required population names collectively at the source-preparation boundary.
Reordering a valid schema preserves semantic destinations; removing a required
name rejects before indexing or publishing. Solver reinitialization invalidates
older trials even when hierarchy and field dimensions are unchanged.

ConstantSlip is a liquid-carrier closure. It applies in currently population-free
liquid so interpolation does not halve its speed at a populated/empty front.
Pure resolved-gas cells retain zero drift. Radius-dependent closures evaluate
only paired positive moments: an empty cell has zero radius and volume. At a
front, the face uses the evaluated populated-side drift; both populated sides
use distance interpolation. Source-created populations acquire an evaluated
closure during source preparation; populations received by transport acquire
their own local closure on subsequent preparation. The timestep selector and
transport use the same face rule. Support uses the paired post-source
number and mole moments, including both births and complete depletion.

See the [durable verification contracts](../verification/vof_solver_contracts.md)
for reproduction commands and the scope of numerical assertions.

A curved high-density-contrast interface lying directly on a coarse/fine seam
still requires capillary qualification. The opt-in disabled curved-seam test
retains its strict parasitic-velocity criterion. The active curved moving-interface
protocol uses density ratio ten, while the high-contrast static protocol protects
a uniformly fine interface neighborhood. Neither qualifies a curved high-contrast
interface directly on the refinement seam. No pressure or velocity gate is relaxed
to claim it.
