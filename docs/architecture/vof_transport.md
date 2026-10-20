# Conservative VoF transport and state ownership

The geometric fraction operator and the momentum operator have distinct time
integration. The fraction operator uses directional PLIC sweeps with a frozen
Weymouth–Yue indicator. Momentum uses a conservative implicit upwind equation
on the integrated mass ledger produced by those sweeps. Reusing the swept
volumes with explicit accepted-old donor velocities is incompatible: material
received in one direction can leave in the next direction with a different
velocity from the original cell.

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
forming a reciprocal could overflow for a subnormal carrier. The scaled solve
is tightened using the physical diagonal/RHS norm bound, with independently
scaled norms so microscopic carriers cannot overflow a squared norm. A compensated
residual on the original unscaled equation must still meet the original
convergence tolerance. Positivity and the global extensive ledger remain
independent. Changing scaling policy invalidates numerical cache reuse.

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
use distance interpolation. Newly produced or transported populations acquire
their own evaluated closure on subsequent preparation. The timestep selector
and transport use the same face rule. Support uses the paired post-source
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
