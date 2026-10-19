# Resolved and dispersed bubble representation changes

Representation changes belong to the external radiolytic controller. They
move existing gas between the resolved VoF field and dispersed molar-pivot
populations; they do not produce hydrogen, dissolve gas, vent a reservoir, or
advance physical time. The physical models remain those described in the
[radiolytic coupling guide](radiolytic_vof.md).

`RadiolyticVoFOptions::representation` enables the conversion policy.
`request_births` registers caller seeds, `inspect_representation` returns
eligibility and refinement requirements, and `preview_representation` prepares
a fixed-clock candidate. The existing controller `commit`, `checkpoint` and
`restore` operations own both that candidate and its registry. `birth_registry`
and `conversion_history` expose accepted metadata; a trial's `representation`
accessor exposes only its proposed batch.
Fixed-clock commits leave the last physical-step statistics unchanged; they
publish their own conversion history and diagnostics without incrementing
production, heat-input or escape ledgers.

Apply requested refinement before acquiring the next checkpoint, then inspect
again on the new hierarchy. After a committed conversion,
`pending_representation_relaxation()` identifies the pressure-history state.
Another conversion or regrid must wait for a successful physical step.
Registered generic passive scalars are rejected during conversion because
their names and geometric carriers do not define a destination representation;
the typed physical inventories have explicit transfer rules below.

## Birth localization and eligibility

A population density does not specify where an individual bubble should be
inserted. The application supplies persistent birth identities, spatial seed
positions, a bounded donor neighborhood, and an eligible population. This is
an explicit localization closure. It must be calibrated along with the
physical size distribution rather than inferred from a void-fraction cutoff.

A promotion consumes one complete bubble from the selected molar pivot.
Fractional cell populations may contribute to that one bubble, but the
assembled donor number must reach one and its molar amount must agree with
the selected pivot. Donors cannot be consumed twice by competing seeds.
Accepted events in one batch have disjoint donor/insertion or demotion-host
supports. Overlapping requests are deferred, and displaced carrier stays
within its event's support rather than being exchanged between unrelated
births. The sample-cell limit bounds the entire sparse planning workspace.
Pending requests retain their identities across rejected trials and mesh
adaptation; a reservation is valid only for the state and topology from which
it was prepared.

Promotion requires both physical size and spatial resolution. The roadmap's
initial 10 mm promotion and 4 mm demotion diameters provide hysteresis; they
are configurable starting values, not universal bubble-transition physics.
At 32 cells across 10 mm, the local mesh width is at most 0.3125 mm. The
insertion region and reconstruction neighborhood must meet the configured
resolution requirement before any donor is removed. An insufficient maximum
refinement level is a reported limitation, not permission to insert an
under-resolved bubble.
The current promotion policy also requires that complete region to be at
`max_level`, even when a coarser level already meets the cells-per-diameter
target. The birth geometry uses local planar cuts of a squared-distance
level set, with its radius parameter inverted to preserve the transferred
total gas volume and native cell capacities. Its finite-grid shape and PLIC
area need not equal those of an analytic sphere; curvature remains computed
from the resulting VoF geometry.

## Complete resolved components

Demotion acts on an entire detached resolved-gas component. Component
connectivity must follow shared gas traces on the canonical Cartesian
coarse/fine and periodic faces. Merely having two positive cell fractions
does not prove that their reconstructed gas regions connect.

A component connected to a wall, a designated headspace, or a larger gas
structure must remain resolved. An attached thin ligament is part of its
parent component and is not independently removed because its local width is
small. A connected-component label describes the current topology; it must
not be presented as a persistent physical identity through arbitrary
advection, breakup, coalescence, or regridding.

Originating birth IDs are attached only to components verified at a committed
promotion. A physical step or regrid invalidates that spatial association;
the origin list is then empty rather than inferred from a nearby seed.
The persistent birth records and accepted event IDs remain available. This
registry does not implement an advected material-lineage tracker.

The demotion resolution check uses the largest actual cell width in the
component, not a finer configured level that has not been created. Periodic
connectivity is retained, but insertion/redistribution stencils currently
must fit within the domain's coordinate box. A component or requested stencil
that wraps a periodic seam is explicitly deferred; it is not split or treated
as wall-attached.

The dispersed model currently stores hydrogen only. Components containing
inert gas remain resolved. Converting them would require a dispersed inert
species equation and corresponding caloric, kinetic, and transfer contracts.

For a pure-H2 component with total amount `n` and represented number one,
demotion to adjacent molar pivots `xi_i <= n <= xi_j` uses

```
N_i = (xi_j-n)/(xi_j-xi_i),
N_j = (n-xi_i)/(xi_j-xi_i),
N_i + N_j = 1,
xi_i N_i + xi_j N_j = n.
```

An exact pivot needs only that bin. Out-of-range components remain resolved;
silently discarding moles or changing the represented number is not an
admissible out-of-grid policy.

## Conservative displacement

Inserting resolved gas concentrates the volume previously distributed among
donor cells. Removing a resolved component does the reverse. The material
volume vacated by gas must be filled, and the material displaced by gas must
be moved into available capacity. This is a conservative redistribution, not
a change to liquid density or a rescaling of stored concentrations.

The remaining carrier packet includes true liquid mass, dissolved hydrogen,
chemically bound donor stock, every remaining dispersed population, internal
energy, physical momentum, and the associated thermodynamic histories. A
new pure resolved-gas cell cannot retain an unrelated dispersed population
or hidden dissolved stock. Each source packet is debited once and every
destination must remain within its physical volume capacity.

The conversion budget preserves total liquid mass, dissolved and donor
amounts, gas species, internal energy, and physical momentum. Physical
momentum is the same quantity used by the flow core,

```
M = m_total u + sum_k(m_k w_k).
```

Changing phase representation can change the continuous velocity because
the dispersed drift contribution changes. Resetting that contribution while
retaining the old velocity would create or destroy momentum.

## Pressure history and subsequent physical relaxation

A dispersed bubble initially has pressure `p_b=p_l+2 sigma/r`. A newly
reconstructed VoF surface generally has a different discrete curvature.
Replacing the old pressure by `p_l+sigma*kappa_new` during conversion would
change gas volume or energy at zero physical time.

The representation transaction therefore retains the transferred material
volumes, phase-pressure histories and phase internal energies. It computes
the new geometry for the next physical step. That next step performs the
ordinary EOS, pressure-work and capillary relaxation, subject to the original
local/global volume, energy and momentum acceptance checks. Conversion itself
has no fictitious timestep, heat source, or pressure-work term.

The single-temperature cell model still constrains which parcel mixtures
can be represented. A mixture that cannot satisfy its carried phase volumes,
energy and thermodynamic domain must reject without publishing. Pressure
history must not be discarded by regridding, another representation change,
or an unrelated initializer before its declared relaxation operation.

For coupled continuation, a pending state first supplies a private physical
predictor using the requested timestep and net heat. Its EOS-consistent
iterate initializes the chemistry/flow correctors. Every corrector still
replays the same accepted inventories for that timestep, and only the final
coupled trial commits. The predictor advances neither the accepted clock nor
the cumulative heat or production ledger.

The native geometric fraction and the typed EOS volume fraction can differ
by a bounded occupancy-roundoff remainder. An exactly absent carrier gives
native `f=1` without changing the retained gas volume or inventory. Interface
geometry uses the solver's native `gas_fraction()` field; typed fractions
describe the material-volume ledger. Split parcel quotas use the same
compensated summation as the core, and true donor overdraw still rejects.

Geometric surface area and represented bubble number are diagnostics of the
conversion. For constant surface tension, report the associated surface-energy
change `sigma*(A_after-A_before)` separately. Internal-energy and momentum
conservation do not establish conservation of total kinetic, surface and
chemical energy. A physical model for unresolved breakup, coalescence or
mixing heat remains an application responsibility.

## Transactions and qualification

Prepared transfers are private until accepted. Commit must validate the
originating controller, solver state, topology and revision; another commit,
restore, or regrid makes an old plan stale. Checkpoints include accepted
birth requests, their registry state and cumulative conversion events along
with the flow inventories. Replaying a rejected physical interval must not
consume another bubble or duplicate an event.

Focused verification separates conservation and selection rules from
physical bubble predictions. It checks complete-bubble budgets, donor
competition, distributed ownership, refinement deferral, hysteresis,
headspace/attachment exclusions, pressure-history continuation, and
rollback/replay. Configurable low resolution thresholds used in small
algorithm tests do not qualify a 10 mm bubble at that resolution.

The [Hysing benchmark](https://www.karlin.mff.cuni.cz/~hron/NMMO403/1934_ftp.pdf)
provides two-dimensional shape, centroid and rise-speed histories with
specified wall conditions. Its no-slip horizontal walls differ from the
current adaptive backend's impermeable slip walls. A short three-dimensional
translation or capillary test is not that benchmark. Matching its boundary
conditions and completing centimetric rise, shape, wake, mass-transfer and
threshold-convergence studies remain separate qualification work.

See the tracked [VoF verification contracts](../verification/vof_solver_contracts.md)
for reproducible numerical assertions and outstanding qualification limits.
