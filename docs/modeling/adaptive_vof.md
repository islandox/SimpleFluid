# Adaptive Cartesian volume of fluid

`AdaptiveVoFSolver` extends the source-free, isothermal two-phase model in the
[uniform VoF guide](vof.md) to distributed dyadic Cartesian blocks. It uses the
native `MultiRegionMesh`, `MeshHandle`, and `FieldStored` path. The adaptive
backend has its own immutable mesh snapshots; the original uniform solver's
serial contract remains unchanged.

## Hierarchy and face ownership

`Meshes::AMRGrid` defines a Cartesian box, the number of root blocks, even cell
counts per block, periodic directions, and a maximum level. A block at level
`L` spans one root-block width divided by `2^L`. Refinement replaces one block
with eight children; coarsening requires the complete sibling group. Persistent
`AMRBlock` keys identify spatial blocks independently of leaf ordinals and MPI
ownership. Cell and face ordinals must be reacquired after regridding.

The hierarchy enforces a maximum level difference of one between touching
blocks, including edges and corners. Opposite periodic face partners have equal
levels. Periodic refinement therefore can refine the opposite boundary too.
Balance closure can undo a requested coarsening. Native canonical fine subfaces
tile each coarse face exactly; coarse cells visit all of those subfaces.
Integrated fluxes are summed with their incidence signs, without multiplying
them by area fractions a second time.

Hierarchy and interface descriptions are replicated compact metadata. Solution
fields have distributed owned maps and sparse halos. Reconstruction builds
temporary, padded uniform patches only for blocks needed by the rank; it does
not construct a domain-sized grid at the finest level. An immutable block
directory restricts overlap searches to intersecting dyadic branches. There is
no large-scale partition/load-balance performance claim.

## Interface geometry and transport

`VoF::AdaptiveVoF` reconstructs conservative PLIC planes on local block patches.
Neighbor samples restrict finer cell volumes and extend coarser cells. Curvature
uses geometric coarse-cell prolongation into its virtual stencil, followed by
the height-function and PLIC surface-fit methods of the uniform backend.
Physical wall samples use even scalar reflection; this is not a contact-angle
or wetting model.

Each owned canonical face computes one signed swept gas volume by intersecting
the donor PLIC plane with its subface prism. Sparse face exchange gives both
incident cells the same flux. Liquid takes the complementary swept volume.
Directional Weymouth--Yue compression uses a frozen indicator; its sum vanishes
under the accepted discrete continuity constraint. The geometric update rejects
material bound violations, excessive Courant number, and nonzero wall flux.
It retains roundoff-sized fraction excursions explicitly.

All levels advance with one shared timestep. The coarse cell directly uses the
fine subface fluxes, so there is no independent coarse flux to reconcile. Level
subcycling is not implemented; introducing it requires time-integrated flux
registers and conservative synchronization before acceptance.

For algorithm background, see the reference implementations of
[geometric fraction refinement](https://basilisk.fr/src/fractions.h) and
[split conservative transport](https://basilisk.fr/src/vof.h). These references
do not qualify this implementation's numerical accuracy.

## Flow and transported amounts

Momentum uses the same gas and liquid face masses as the geometric update.
Pressure and capillary forces share the same face mobility and reconstructed
normal gradient. At a coarse/fine subface, adjacent cell centers have different
transverse coordinates: least-squares gradients align the two values to the
subface before their normal difference is taken. A symmetric integrated
two-point pressure matrix supplies defect corrections; acceptance checks the
actual corrected face divergence, including the pressure gauge cell.

The mobility is `d/(rho_o*d_o + rho_n*d_n)`, where the distances are measured
normally to the subface. Density-weighted face pressure supplies a shared Gauss
momentum impulse. These choices recover the uniform solver's force stencil at
equal spacing and avoid assigning the full liquid-side impulse to a light gas
cell. Explicit symmetric viscous stress, gravity, and capillarity retain the
uniform backend's material assumptions. Advection, individual subface swept
width, capillary and viscous restrictions all constrain the common timestep.

Registered passive amounts travel with their selected carrier's face flux.
An implicit donor-concentration equation uses final carrier storage plus
outgoing flux on the diagonal. This handles phase flux passing through cells
with little initial phase volume while retaining nonnegative amounts. The
solver checks finite values, positivity and global conservation before
publication. Signed amounts use a conservation scale formed from absolute
amounts, so cancellation to zero does not weaken or make the budget singular.
This passive scheme and upwind momentum are first order; geometric VoF alone
does not make the complete flow or species discretization second order.

## Conservative remapping

Regridding intersects old and new leaf cells. Refined gas occupancy comes from
the old PLIC geometry; restriction sums gas volumes. A sparse reverse exchange
closes each donor's roundoff-sized geometric residual across receiving ranks.
The correction uses the donor's available gas or liquid capacity and preserves
even tiny positive inventories. A material geometry mismatch rejects the
candidate; no post-remap inventory clipping hides a lost budget.

`ConservedQuantity::amount` is an extensive amount per owned cell. Its declared
`InventoryCarrier` controls how that amount divides among receiver cells:

| Carrier | Remap weight | Example |
|---|---|---|
| `BulkVolume` | intersected control volume | A stored bulk extensive tracer |
| `GasVolume` | intersected PLIC gas volume | Resolved gas moles at uniform intradonor concentration |
| `LiquidVolume` | complementary liquid volume | Dissolved storage, dispersed number or moles carried in liquid |
| `MixtureMass` | `rho_l*V_l + rho_g*V_g` | Momentum, or a mixture-specific stored energy |

Momentum components permit signed amounts. Positive species/population/thermal
amounts use the nonnegative contract. A nonzero inventory with no donor carrier
is rejected. Every amount and gas volume must pass a global extensive
conservation check independent of the flow pressure tolerance. The carrier must
match the physical storage definition; naming a scalar "energy" does not add a
temperature equation, EOS, heat transfer, or reacting kinetics.

## Refinement and coarsening policy

`refinement_plan()` targets 32 cells across the supplied minimum resolved
diameter by default. For a 10 mm diameter this gives 0.3125 mm; choosing 24 gives
about 0.4167 mm. These are initial hydrodynamic resolution targets. Bubble shape,
rise velocity, wake and mass-transfer convergence require separate studies.

The policy protects every mixed cell and detected pure-phase jump, with at
least four cells of reconstruction stencil and a motion buffer formed from the
global maximum face speed times the forecast horizon. Large dimensionless
curvature can trigger further refinement for thin necks. Overlapping interface
bands also protect narrow intervening films and nearby walls. An externally
supplied dimensionless wake indicator tags wakes; explicit promotion boxes
reserve fine neighborhoods for later representation changes. These are mesh
criteria, not breakup, film-drainage or bubble-birth models.

Coarsening proposals contain only complete uniform pure-phase siblings outside
protected neighborhoods. Thus the policy cannot merge distinct gas components
or erase resolved interfaces by coarsening. Explicit solver coarsening requests
must pass that protection policy too. If an interface remains above its diameter
target at the configured maximum level, `RefinementPlan::resolution_limited`
reports that limitation instead of claiming the resolution target was met.

A plan retains its source hierarchy. `regrid(plan)` rejects a plan from a
replaced topology; regenerate plans after advancing the state. The explicit
`regrid(refine, coarsen, protection, hints)` overload can revalidate forecast,
wake and promotion guards for a manual request. Its default protection checks
the current interface and stencil neighborhood; it does not remember hints
from an earlier query.

## Atomic state and replay

State-changing operations and timestep queries are collective. Cell-array input
uses owned-cell order. Face arrays use native overlap order, canonical
owner-outward signs, and authoritative owned-face values; exchange replaces the
ghost entries. Rank-divergent control input is rejected collectively.

`regrid()` prepares a new immutable hierarchy, remaps the accepted state,
reconstructs geometry, builds new maps/halos/graphs and native fields, and
projects the new face flux before publishing the replacement. A rejected
candidate leaves time, diagnostics, inventories, fields and hierarchy unchanged.
Topology and partition generations advance with a changed leaf set. External
mesh transfers and caches must capture the hierarchy identity and these
generations and rebuild after a change. Borrowed array/field references must be
reacquired after a successful state change or restore. A separately retained
mesh shared pointer keeps the old immutable mesh alive; it does not follow the
replacement mesh.

One explicit checkpoint can retain the accepted snapshot for replay. Topology
changes are forbidden until that checkpoint is released. Restoring it restores
the flow and passive inventories together and permits another trial. Coupled
controllers should regrid only at accepted coupling boundaries, then acquire
their next checkpoint. The opt-in [reacting low-Mach controller](reacting_vof.md)
includes its physical inventories and cumulative ledgers in that transaction.
An accepted representation conversion retains phase-pressure histories until
the next physical step; regridding waits until that relaxation succeeds.

## Scope and verification

Cartesian blocks, periodic directions and impermeable slip walls are the
supported geometry. Prism subdivision, curved vessel boundaries, contact angles,
and moving mesh/ALE retain their own roadmap gates.
[Bidirectional representation trials](bubble_representation.md) have separate
birth, resolution, component and thermodynamic-history guards.
Physical gas growth and dissolved transfer use
the separate [reacting low-Mach mode](reacting_vof.md); the source-free path
and its passive inventories do not enable those processes.

Focused verification covers hierarchy coverage, conservative transfer,
coarse/fine interface passage, distributed ownership, pressure/capillary
behavior and rejected/replayed transactions. Measured evidence belongs in the
[Stage 2 verification record](../verification/vof_stage2_20261019.md), separately
from this stable model and API description.
