# QBMM family kernel and native integration contract

The contract below binds phases Q0 through Q6 of [SF-QBMM-CODE-001](qbmm_reacting_bubbles.md)
to the SimpleFluid source baseline `0029a93f`. The physical assumptions and moment definitions
are in [SF-QBMM-MODEL-001](../modeling/qbmm_reacting_bubbles.md). Q1 provides local numerical
operations. Q2 supplies native state ownership, Q3 couples local population sources to the native corrector,
Q4 transports each family on the fixed mesh, and Q5 exposes endpoint reporting and the coupled
numerical release gate. Q6 adds explicitly supplied common family slip, body acceleration and
application-declared constitutive ranges. None of these options changes the defaults.

## Source alignment

| Source contract | Binding for QBMM |
| --- | --- |
| [ReactionMechanism.hh](../../src/equations/ReactionMechanism.hh) | In `SimpleFluid::Reacting`, `SpeciesId` is a 32-bit unsigned ID. Storage uses `SpeciesCatalog::index(id)` and catalog order. The 64-bit fingerprint covers revision, ordered IDs, names, molar masses, element counts and allowed phases. A matching revision or vector length alone cannot establish compatibility. |
| `CellSpeciesState` and `SpeciesPopulationState` | `amounts` is phase-major. Its base `Dispersed` row and each named `dispersedPopulations[i].amounts` are distinct stocks. Bind the named row selected by its string `id`. `phase_amount()` includes multiple stocks and is unsuitable for a family. `CellSpeciesState` owns moles. It has no population-number member. The design term `PopulationId` denotes the existing stable string identity. |
| [ReactingThermodynamics.hh](../../src/equations/ReactingThermodynamics.hh) | `PhaseThermalState::moles` is the native phase stock. `MixtureThermodynamics::evaluate(T,p,mole_fractions)` returns mass-specific properties including density. Q1 consumes the common-pressure gas specific volume from this EOS when the family is present. The host owns occupancy, internal energy, temperature and pressure work. |
| [BubblePopulationKinetics.hh](../../src/equations/BubblePopulationKinetics.hh) | The existing namespace is `SimpleFluid::BubbleGasPhysics`. `PopulationNumberState::population` and `PopulationNumberEvent::population` are strings. `SpeciesPhaseTransfer` names exact donor and receiver compartments. The count/material packet has no higher moments. Legacy `MolarPivotGrid` uses mol per bubble and needs an explicit composition-dependent unit conversion. |
| [LowMachMixtureSolver.hh](../../src/solvers/LowMachMixtureSolver.hh) | `SimpleFluid::Reacting::LowMachMixtureSolver::Options::phase_roles` is EOS-indexed. The constructor checks EOS kind, a shared catalog object, compatible energy conventions and unique nonempty dispersed identities. Omitting phase roles creates bulk roles. `PhaseRoleDescriptor::identity` must match the named population and the role must be `PhaseRole::DispersedGas`. Only that role has an engaged `CellState::phase_numbers[p]`, including engaged zero for absence. |
| [ReactingMixtureCoupler.hh](../../src/solvers/ReactingMixtureCoupler.hh) | `MixturePhaseBinding` maps physical phases to chemical compartments. A dispersed role requires the same named `SpeciesPhase::Dispersed` binding. The physical liquid can bind both Liquid and Dissolved. Q2 maps each family uniquely to one dispersed EOS phase, including on empty owners. |
| Equations build and visibility | Compiled equation sources belong to `SIMPLEFLUID_EQUATION_SOURCES` and the physical `SimpleFluidSolvers` DSO. Q1 uses `SIMPLEFLUID_EQUATIONS_EXPORT` and the logical `SimpleFluid::Equations` target. The serial test registration discovers individual GoogleTests in the equations test directory. The pure kernel includes no solver, mesh or MPI API. |

## Q1 arithmetic

The family owns its reference mass in kg/bubble. Moment blocks and positive atom measures are
temporary values. A cell view borrows named moles, count and the two owned shape scalars, checks
its catalog and family stamps, and derives the mass coordinate from the species inventory.
An exact all-zero block is the only vacuum. The positive coordinate has no minimum bubble mass
or minimum statistical weight.

The family constructor takes the caller's catalog fingerprint and an ordered list of
`(SpeciesId, mass_fraction)` pairs. Normalization and each species mass relation use a relative
bound of 64 binary64 epsilons. A zero fraction requires exactly zero moles, including in a signed
transfer. The kernel uses the fractions as supplied. The family fingerprint also covers its name,
coordinate, reference mass, fractions and closure policy versions. Dispersed role and family
name are checked on the host descriptor at construction.

Migration from a molar representation requires the declared fixed composition.
The mixture molar mass is `1 / sum(Y_s/W_s)`. The molar-pivot helper accepts extensive counts
and mol/bubble, including legacy bins with zero count. For the existing
`BubbleGasPhysics::UpperBubblePopulation::number_density`, multiply each bin's number per bulk
cubic metre by the bulk cell volume to obtain its extensive count. The mass-pivot helper takes active atoms
in units of the family's reference mass. Both conversions reconcile the result with the supplied
count and material. The explicitly monodisperse initializer derives both missing shape moments.
Changing only the reference mass requires compatible source and destination schemas and multiplies
each order by the corresponding power of the old-to-new reference-mass ratio.

The inverter rescales count and mass by binary exponents before forming the two Hankel determinants.
It uses a compensated difference of products and records each signed determinant divided by the
sum of its two product magnitudes. For a near-singular count Hankel matrix, the input must agree
with both monodisperse shape moments within 32 binary64 epsilons relative to each input moment.
The result reports rank one and the measured residuals. An exact shifted determinant of zero
with positive count determinant identifies massless support at resolved rank two. Inside the
rank-one uncertainty band this can also be caused by rounding a positive singleton. In that
ambiguous case the singleton must retain the entire input count and reproduce all four binary64
inputs exactly on rounding. A positive-definite full-rank input
whose shifted determinant margin falls within the same uncertainty band is rejected as unresolved.

The two-node path obtains the larger eigenvalue of the symmetric Jacobi matrix with a hypotenuse
and the smaller one from its determinant divided by the larger eigenvalue. The squared components
of a Jacobi rotation give both extensive weights independently. All four moments reconstructed
from the **returned binary64 atoms** must lie within 128 binary64 epsilons relative to their own
input magnitudes. A failed inversion exposes its status and zero active atoms. No acceptance
test uses an absolute SI floor.

The geometry view takes the EOS specific volume at common temperature, pressure and composition.
It returns diameter moments per bulk volume, number density, total mass, spherical-equivalent
atom sizes, area per bulk volume and `d32 = 6 M v / A`. For absence the active count and densities
are zero and size extrema and `d32` are disengaged. Q1 constructs this view on demand. Any future
cache needs the family schema, state generation, geometry and material revisions, and the
pressure/temperature evaluation stamp.

## Local event receipts

A substep operates on one named family's pre-material state. Its cell view borrows that
family's catalog-order moles and count. A resize supplies the final positive scaled mass and a
paired species packet. Reaching zero uses a disappearance request. Each birth atom has a separate
species allocation. Coalescence with equal donor indices reserves the parent count twice.
Breakup specifies daughter multiplicities per parent. Every material-changing allocation checks
composition and mass before receipts accumulate, so opposite selective transfers cannot cancel
in the check.

The kernel reserves all internal species donors and original atom counts before adding products.
Donor margins and endpoint receipts use the reacting code's compensated long-double summation.
Each original stock seeds its accumulator. A negative compensated remainder rejects the batch,
including when a small prior withdrawal is below the precision of the rounded gross count.
The checks and signed diagnostics use the retained event contributions. For a count depletion
larger than half the original H0, the kernel evaluates the remaining count from reserved donor
remainders and positive products. This avoids cancellation when the separately rounded
quadrature weights sum a few ulps away from the original count. The raw donor and product
receipts remain available. Complete disappearance still requires exact material exhaustion.
An external exchange names the reservoir and supplies its integrated energy and momentum handoff.
A baseline internal exchange names a liquid or dissolved compartment on the physical liquid carrier. The native host must bind its
energy and momentum treatment when it integrates the receipt. The kernel can reject the complete
batch with a donor shortfall. The caller can then propose a smaller substep.

The result owns its candidate endpoint. Each accepted receipt retains provider revision, substep,
raw donor and product atoms, and directed material transfers. Gross moment arrays record all atomic
removals and insertions. For example, growth removes and reinserts the same count, and its signed
count change is zero. The controller uses the event kind when reporting physical births and disappearances.
In a near-cancelling batch the raw entries and the validated endpoint are needed for replay.

## Q2 native state and transactions

`LowMachMixtureSolver::Options::qbmm_families` is empty on the count-only path. To enable QBMM,
supply one slot per physical EOS phase. A null pointer leaves that phase on its existing path.
Each enabled slot holds a `shared_ptr<const QbmmPopulationSchema>` whose dispersed name and
ordered catalog match the corresponding gas EOS. The constructor verifies unique role identities,
phase order, binding presence, family fingerprints and reference masses collectively. Every
rank supplies the same configuration and state stamps, including ranks with zero owned cells.

`State::qbmm_families` carries the catalog fingerprint, family fingerprint and reference mass per
bound phase. In each owned cell, the corresponding `qbmm_shape` slot owns H2 and H3. The count
stays in `phase_numbers` and the material stays in `phases[p].moles`. An absent bound family has
engaged zero count and shape. An unbound phase has a disengaged shape. Initialization verifies
the schema stamps, fixed composition, presence and inversion of each block. The default velocity
mode requires exactly zero drift; equilibrium temperature is mandatory in both modes. The native linear
solver controls, including its stopping tolerance and iteration limit, must agree across ranks.

Count-only input needs an explicit call to `initialize_qbmm`. Each owned cell and bound phase
specifies a monodisperse assumption or a list of positive atoms that matches the existing moles
and count. The method builds a candidate and enters the normal collective initialization path
only after migration succeeds everywhere. Existing shape-bearing states use `initialize`.

The native preview supports a moving population on a fixed orthogonal mesh; its zero-slip path
does not require a supplied velocity model. A local source changing a bound family's material, count or shape requires a coherent Q3 packet
from the reacting controller. Direct fixed energy inputs and material packets for unbound phases
pass through the native acceptance gates. Only Q6's explicit common phase-slip mode
extends this baseline; ALE, node slip and higher-order moment advection remain excluded.

The transport snapshot owns pre-source shape, count, moles and the family stamps alongside
the catalog stamp. A trial owns its candidate, the exact integrated source packets, transport
snapshot and phase-indexed inversion statuses. Successful and failed previews can coexist with
an earlier valid candidate from the same accepted generation. Commit uses the retained candidate;
changing the accepted generation by commit, initialize or restore invalidates older trials.

### Checkpoint and restart

`checkpoint()` retains the accepted state, its generation, native diagnostics and owned face-flux
history. Fixed-mesh `restore()` requires the creating solver and matching geometry. It checks
the complete moment state collectively. The live generation advances on restore, so restoring
an earlier time or revision cannot resurrect a stale trial. Publication keeps public field
objects stable. Reacquire borrowed host or device storage views after a QBMM publication. All ranks must select the same
generation for checkpoint restore and the same preview for commit.

To migrate an existing count-only solver's private checkpoint, call `restore_qbmm_checkpoint`
with its creating solver and the explicit per-cell initialization choices. Both solvers must
use the same live fixed mesh and compatible phase/catalog/EOS bindings. This path carries
the checkpoint's accepted clock, diagnostics and face-flux history across the conversion.
If the count-only checkpoint records a body acceleration, migrate through the
receiving reacting controller's `restore_qbmm_checkpoint` with the same acceleration.
Migration cannot silently change or drop the force associated with that history.

`serialize_qbmm_restart()` and `restore_qbmm_restart()` are collective. Each rank writes or reads
its own versioned little-endian byte buffer, including ranks with no cells. The archive stores
the ordered catalog and family stamps, phase roles and EOS metadata, canonical cell state,
clock, revision, generation, diagnostics and accepted face flux. The native backward-Euler step
uses this accepted state and flux as its history. The reader validates payload lengths, checksums
across the local and collective image, ordered geometry and partition fingerprints, material,
inversion, and the gas/liquid EOS energy and volume at the stored point before publication.
A new solver with the same configuration and partition may restore the image.

The application supplies the matching material and constitutive models on restart. The archive
checks declared EOS metadata and evaluates the stored thermodynamic point; a C++ callback's
implementation has no portable model identifier. A restart across different MPI partitions or
a change in coordinates requires an explicit conversion outside this format.

Version 4 stores the active or last accepted reacting-controller fingerprint and the physical
material balances. The fingerprint covers declared source/provider presence, identities and
revisions, reaction stoichiometry, phase bindings, integration controls, ordered face and
interphase stages, and thermal-transfer law metadata. After restoring a coupled image, attach
the matching controller before advancing. A fresh native owner can report the restored
canonical state before attachment. Constitutive functions supplied under an existing
identity/revision must retain the same behavior.

Versions 1–3 remain readable. Version 2 added the signed shape-source residual, and version 3
added the normalized shape-transport residual. Earlier images lack an active-controller stamp
and independent element balances. Their caller supplies compatible physics. An in-memory
checkpoint retains its interval receipts. Portable restart restores the canonical state and
archived balance diagnostics; raw prior-interval receipts have explicit unavailable status. The separate convergence record
preserves the corrector/source evaluation counts and any archived gross and shape residuals.
A missing historical shape diagnostic is disengaged, including after re-encoding an older image.

## Q3 coupled local sources

Configure `ReactingMixtureOptions::qbmm_population` with an immutable `MixtureQbmmProvider`.
Its identity and revision match each participating family's source policy and its fingerprint
matches the species catalog. On each native corrector the controller inverts the transported
family **before** evaluating chemistry. It retains those atoms through the ordered chemistry,
face and generic interphase stages. These stages may act on other compartments. Reaction
stoichiometry, species diffusion bindings, generic material slots and external chemistry transfers
touching a bound family require the paired route. A coexisting legacy number law declares
`populations` explicitly and can emit events only for those unbound dispersed roles.

The MPI-free law receives the family's quadrature, actual material predecessor, extensive
species stocks, per-phase EOS volumes in cubic metres, trial phase pressures and temperatures,
accepted time, integrated interval and source-stage metadata. An absent phase has zero physical
volume. The controller passes compensated material inventories from prior accepted stages into donor
reservation, keeping both the high part and its correction. Each law returns integrated Q1 birth, resize, disappearance, coalescence or breakup
operations. Allocation across families uses the same pre-event donor inventories. Original gross
withdrawals must fit before any incoming credit. An omitted law passes through the transported family stocks
and permits independent heat and unrelated chemistry.

Internal material transfer declares one simultaneous final material stage with unique directed
`(species, donor phase, recipient phase)` slots and the donor's partial-molar enthalpy law.
A donor used by both a generic material stage and the QBMM stage must have matching
partial-enthalpy metadata and validity bounds. The controller records raw species movements once. The frozen pressure-conditioned plan uses
the pre-transfer donor enthalpy at each pressure evaluation and retains the opposing energy
entries. External reservoir energy and impulse are signed cell-side inputs. A reservoir transfer
can coexist with internal transfers and earlier generic thermal stages. The flow owner supplies
pressure work. Exact disappearance must satisfy the native zero-energy, zero-volume endpoint.

The native solver verifies the constructive endpoint against the original atom and material
receipts. It checks replay of the signed integrated species, number and shape packets at the
endpoint's scale, including exact zero presence. Unrepresentable packet replay rejects the
preview. `maximum_shape_source_residual` compares both signed canonical shape changes and the
integrated event changes before endpoint rounding. Each bound phase contributes eighteen nonnegative
gross convergence entries, independent of quadrature rank: birth count, disappearance count,
coalescence pairs, breakup parents, incoming mass, outgoing mass, separate production and
consumption for H2 and H3, and positive/negative magnitudes for reservoir energy and each of
the three reservoir impulse components. Gross shape terms include all removed and created atoms. The corrector
requires at least two evaluations even when net source is zero.

`ReactingMixtureCoupler::Trial::local_qbmm_sources()` exposes raw accepted receipts and their
canonical family endpoints in owned-cell and physical-phase order. The count-only receipt records
only its own provider's events from its actual predecessor; diagnostics combine the two providers. Constitutive EOS and
partial-enthalpy evaluations may occur inside pressure roots; kinetic callbacks run only in
the source stage. Commit and restore publish retained state without invoking those laws.

## Q4 native transport

For each bound physical phase the flow owner assembles one weighted-scalar backward-Euler,
first-order upwind operator on the fixed native mesh. The components are catalog-order species
moles, the native count H0, H2, H3, and signed phase internal energy. Storage and advection
weights are one. Each face transports the same phase volume flux times the implicit upwind
extensive stock divided by the donor's **bulk cell volume**. Closed boundary flux is zero.
The mass-weighted transported species determine H1. The native momentum equation consumes
physical mass flux, while energy and material volume retain their pressure-work accounting.

The positive iteration starts every component from the accepted state and advances all components
through the same number of Jacobi sweeps. At zero phase flux the accepted extensive stocks solve
the identity map directly. At nonzero flux a represented endpoint must pass each component's
physical face balance, the true residual of the common stored matrix, and a closed-domain
inventory check at arithmetic roundoff. The matrix residual uses the configured linear tolerance.
The local face residual uses the conservation tolerance and the magnitudes of the old stock,
new stock and integrated face packets. The closed transport budget uses
`(64 + 2 * ranks) * epsilon` times the sum of old and new stock magnitudes. These checks have
no absolute stock floor. The solver also closes the budget for H1 recomputed from each cell's
species. A cell reached from upstream must carry a consistent material/count/shape block. Unreached vacuum stays exactly zero. The solver checks fixed
composition and four-moment inversion collectively before source evaluation and again on the
candidate. Exhausting the linear iteration budget rejects the preview.

`Trial::owned_phase_transport()[face].qbmm_flux[phase]` retains the actual owner-oriented
species, H0, H2, H3, and energy rates used in the balance. Multiply a rate by the trial time
step for its integrated face packet. With QBMM enabled, only bound physical phases have an
engaged receipt. Every signed species receipt must meet its family's fixed composition rule.
A nonzero subnormal donor concentration, face rate, history term or sweep result must agree
with its extended-precision value to within one binary64 epsilon in relative terms.
The reacting controller exposes the same face span through its trial.
The pre-source endpoint remains available in `Trial::transport_snapshot()`. Each source
corrector reconstructs its family there and compares successive transported shape
moments at their own scales. Phase concentrations use the current physical EOS volume.
`advected_material_volume` supplies the continuity and pressure-work budgets.

## Q5 reporting and numerical release gate

On either the native flow or its reacting controller, call `qbmm_report()` for accepted state
or `qbmm_report(trial)` for a retained candidate. All ranks call the same overload and select
the same trial, including ranks with no cells. Both calls return an owned snapshot. Each report
labels the endpoint kind, origin, time, revision, live generation, geometry epoch, species
catalog and controller fingerprint when known. A trial also identifies its preview and source
interval. Foreign, stale or cross-rank-mismatched selections reject collectively. An inversion failure
identifies the family and status at the rejecting owner without publishing a failed candidate. A report
already returned continues to describe its original endpoint after another preview, commit
or restore. Reporting evaluates constitutive properties at the retained absolute pressure and
temperature. It does not run kinetic or population callbacks.

| Report data | Units and ownership |
| --- | --- |
| `species_ids`, `cells[c].families[p].species_moles`, `moments` | Ordered mol/cell and extensive H0–H3 in the family's scaled mass coordinate. H1 is reconstructed from species. The cell report includes the native geometry cell ID and bulk volume in m³. |
| `mass_moment_density_si[k]` | `mass_scale_kg^k * Hk / bulk_volume`. For orders 0–3 the units are count/m³, kg/m³, kg²/m³ and kg³/m³, respectively, with count dimensionless. |
| Species concentrations and mass densities | The bulk denominator is the control volume. Phase concentration and phase density use the occupied family EOS volume. On absence those phase-normalized fields and the phase temperature are disengaged. |
| Native thermodynamic state | Family gas fraction, volume and phase mass refer to the identified physical gas. Total gas fraction also includes background gas, while mixture density includes all phases. The common cell temperature remains available when only the host liquid is present. |
| Quadrature and geometry | Status, active rank, Jacobi conditioning and reconstruction residuals accompany each cell. Common-pressure equivalent-sphere radii/diameters, size extrema, area per bulk m³, and Sauter diameter follow from that measure. Exact absence has zero area and no size extrema or d32. |
| Collective family diagnostics | The EOS-indexed family stamp identifies the coordinate scale. Counts by inversion status, minimum occupied Hankel margins and per-order maximum reconstruction residuals use all owned cells. |
| Interval and balances | Local raw Q1 receipts and the fixed-layout gross corrector record belong to the accepted or selected trial interval. The report identifies the source stage, pre-transport generation/time and residuals. Global species defects include reaction conversion. Mass and declared-element defects in the coupled report charge external reservoir material. The energy defect includes applied heat and native pressure work, in joules. |

The physical mass and element gates compare the change of each global inventory with its
declared external input. Each gate uses the configured conservation tolerance times the largest
of the old inventory, new inventory and gross external exchange, plus a binary64 roundoff
allowance on those operands. A species reaction budget and a closed elemental balance thus
have different meanings for a mole-changing reaction.

`closure_evaluated` records whether occupied-family geometry was evaluated.
`closure_validated` is false throughout this baseline. Numerical realizability, a large gas
fraction and an energy-balanced timestep give no empirical validity range for a closure.

### Capability matrix

| Selection | Baseline contract and gate |
| --- | --- |
| Native mesh and boundaries | Closed, fixed, supported orthogonal native geometry; common-pressure and equilibrium-temperature EOS. QBMM planar ALE rejects before opening a motion trial. Open-reservoir face transport, moving-mesh remap and interface-resolving evolution have no native QBMM entry point. |
| Velocity and order | `SharedCarrier` (zero slip) by default. `SuppliedPhaseSlip` admits a finite prescribed phase velocity relative to the carrier, common to every node in that phase. Both modes require transient backward Euler and first-order upwind. Node slip, independent momentum, steady assembly, BDF2 and higher-order advection reject at construction. |
| Gravity and applicability | Fixed grid, optional explicit SI acceleration. EOS density alone does not activate gravity. Reported validity ranges belong to supplied laws; satisfying them is not empirical validation. |
| Family schema | One named fixed-composition family per dispersed gas phase; catalog, family stamp, reference scale and phase ordering agree on all ranks. Variable composition, cross-family coalescence and unmigrated count-only state reject. |
| Local material and topology | Constructive paired birth, growth, shrinkage, disappearance, coalescence and breakup; transfers may involve the liquid carrier or a declared external reservoir. Declaring either direction of a family-to-background-gas material slot rejects. Generic chemistry, face transport and interphase laws can act on unbound compartments. A legacy number law must name only unbound populations. Selective or unpaired family material returned by a law rejects the preview. |
| Thermal and pool closures | Common temperature and pressure, zero surface tension. Separate phase or node temperatures, bulk-gas conversion and capillary/surface-energy closures reject. A separate background gas may carry its own inventory. |
| Transactions and restart | All owners use a common numerical configuration and controller selection. Commit publishes one retained trial. Rejection preserves accepted canonical state, histories and other compatible previews. Portable restart uses the same geometry, partition, bindings and supplied physics. |

The release suite follows circulating polydisperse gas across partition faces into exact vacuum,
combines it with population events, unrelated mole-changing chemistry and heat, and checks the
physical budgets and reporting against independent atom, EOS and time-integration calculations.
It compares uninterrupted continuation with an owner checkpoint and a portable restart into a
fresh solver. A separate one-cell configuration exercises collective failure with empty owners.
The refinement case varies the flow step and the chemistry substep independently.

## Q6 supplied slip, force and constitutive contracts

The default `Velocity::SharedCarrier` (also named `ZeroSlip`) requires zero drift.
Set `Options::qbmm_model.velocity` to `QbmmModelOptions::Velocity::SuppliedPhaseSlip`
and initialize `CellState::drift_velocity` in m/s, in physical phase order. The field is
prescribed in space, not a second momentum equation or a terminal-rise prediction. The same
carrier-plus-drift face flux selects the upwind donor for each family species, H0, H2, H3
and signed internal energy; H1 is the mass of those species divided by the reference mass.
Each phase chooses its own donor. In counterflow their material fractions can sum to less
or more than one. They are **not** normalized. The material-volume packet, not an EOS donor
volume, supplies `-p*(V_final - V_adv)`. Momentum convection consumes the summed physical
mass rates in kg/s with unit flux weight. Each family's face receipt retains mass, material
volume and donor-drift momentum rates alongside its scalar rates.

The face-packet ABI uses binary64 scalars and three-component double vectors, including
`derived_mass_flux` (kg/s) and `derived_drift_momentum_flux` (kg·m/s²). A finite
species flux can have a mass equivalent below the smallest representable double: a
rounded zero in this diagnostic does not erase the species packet. The containing
`PhaseFaceTransport` records the actual mixture mass and drift-momentum fluxes supplied
to the native momentum equation. Donor products retain a separate binary exponent and
use compensated floating-point summation. Applying a drift or an EOS density before
the final conversion can preserve a representable momentum or material-volume rate
even when the intermediate phase mass rate would round to zero.

Cell reports separate `carrier_velocity` from `mass_weighted_drift = sum(M_j*w_j)/sum(M_j)`
and `barycentric_velocity = sum(M_j*(u_c + w_j))/sum(M_j)`. The latter uses accumulated
absolute phase momentum, not addition of the two rounded report fields. Neither diagnostic
replaces native carrier storage. `ReactingMixtureOptions::body_acceleration` is an optional spatially uniform
vector in m/s². Each source-corrector evaluation uses the post-transfer physical mixture mass
and the actual trial duration to add **one** integrated `M*g*dt` to `CellSources::momentum`.
The three absolute impulse components are separate gross convergence channels; cancellation
against reservoir impulses cannot conceal an unconverged body force. There is no heat packet
for gravity, and neither commit nor restore re-integrates any force or population law.
On a fixed grid the body-forced pressure reconstruction takes the least-squares gradient
of interior pressure differences after subtracting the hydrostatic difference
`0.5*(rho_owner + rho_neighbor)*g·dx`, then adds the local `rho*g`. The pressure
correction uses the corresponding interior stencil without a fictitious zero-gradient
wall sample. This balances stratified as well as constant-density columns without
adding a second force. The mechanical pressure reference is unchanged.

The source corrector keeps positive and negative Fourier heat as separate gross
channels. For `Q = dt*G*(T_owner - T_neighbor)`, two rounded absolute temperatures
can be nearly equal. The canonical Fourier adapter uses the midpoint to the preceding
and following representable value of each temperature, including their different
distances at powers of two. The corrector matches signed heat packets by canonical
face before comparing their intervals. An unchanged face cannot lend its uncertainty
to a neighbor. Any gap between successive intervals must pass the original relative
tolerance at that face's own scale. The solver reports the maximum
comparison uncertainty used, in joules. Identical packets need no allowance. At the
subnormal limit the solver falls back to strict comparison rather than crediting a
rounded-up bound. This is not an absolute source tolerance: it does
not change either physical heat packet, authorize offsets on other gross channels, or
replace the signed energy, caloric or pressure-work checks.

`QbmmClosureVersions::laws` owns ordered, immutable law identities, revisions, units, input
bases and inclusive SI ranges. Bounds and gating policy enter family compatibility. At a
source evaluation `QbmmLawInputs` carries the actual timestep, trial thermodynamics, active
quadrature, equivalent-sphere geometry and physical phase concentrations. Family-gas and
carrier-liquid concentrations use their respective EOS volumes; absent phases have no
concentration or temperature. `ReportOnly` reports an out-of-range or missing input;
`RejectOutside` can fence the evaluation before the law runs. Hard slip and transport-property
ranges also apply at native preview entry and to the final candidate, even without a source
provider. Initialization and restore do not invent a duration to evaluate an interval gate.
Per-family reports distinguish
applicability from realizability and from `closure_validated`. No built-in correlation,
chemical identity or range threshold establishes model validity. Supplied rate adapters
produce integrated events; native material and thermal owners still account for their packets.

Portable version 5 stores the velocity mode, even if all stored drifts are zero. A known
Fourier comparison bound is retained with its source-corrector diagnostics; missing
historical bounds remain disengaged rather than being reported as measured zeros. The controller
signature distinguishes a disabled acceleration from an explicitly supplied vector and binds its
components. Older streams do not opt into new slip or law metadata. Restart remains restricted
to the same partition and matching supplied physics, not arbitrary repartitioning.
The default-only restriction on a legacy image survives subsequent v5 re-encoding.

### Numerical application scope

The counterflow benchmark compares the native face packets and transported extensive
stocks against a separate backward-Euler solve, including transport into vacuum. Closed
vertical columns test the mechanical pressure gradient, cell velocity and flux at rest;
an initially unbalanced column also tests that the force actually builds pressure.
The heated case is a closed unit-depth cavity with opposing wall powers deposited in
the adjacent control volumes, internal Fourier heat transfer and a synthetic polydisperse
population. Gravity-off and reversed-heating controls distinguish buoyancy from slip or
imposed motion. This is not a no-slip, prescribed-wall-temperature Nusselt benchmark.
Flow and thermal sensitivity use a common physical horizon: vary grid spacing at fixed
timestep, then timestep at fixed mesh. Record area and Sauter diameter on those same
trajectories; a small nonmonotone change does not establish a grid-convergence order.

The homogeneous population comparison refines the bin spacing and the timestep separately
and compares the fractional area moment against a two-node population initialized with
the same H0–H3. A separate constant-kernel test checks the bin reference against a discrete
analytic solution. These tests quantify closure error, not empirical validity of a
particular growth, collision, breakup, slip or high-void correlation.
