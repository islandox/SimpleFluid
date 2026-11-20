# QBMM family kernel and native integration contract

The contract below extends the Q0 through Q7c implementation of [SF-QBMM-CODE-001](qbmm_reacting_bubbles.md)
to the SimpleFluid source baseline `0029a93f`. The physical assumptions and moment definitions
are in [SF-QBMM-MODEL-001](../modeling/qbmm_reacting_bubbles.md). Q1 provides local numerical
operations. Q2 supplies native state ownership, Q3 couples local population sources to the native corrector,
Q4 transports each family on the fixed mesh, and Q5 exposes endpoint reporting and the coupled
numerical release gate. Q6 adds explicitly supplied common family slip, body acceleration and
application-declared constitutive ranges. Q7a extends the native transport and reacting controller
to prescribed planar ALE with the same two velocity modes. Q7b adds first-order kinetic transport
with node-dependent slip on a fixed, closed Cartesian mesh. Q7c adds common-block linear
reconstruction and SSPRK(2,2) for shared-carrier families on fixed orthogonal meshes.
Q8a couples Fourier contacts inside the thermal corrector. Q8b admits supplied uniform
acceleration for fixed closed Cartesian kinetic families. Q8c extends fixed shared-carrier
SSPRK(2,2) to gravity on Cartesian and qualified cylindrical meshes.

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
pass through the native acceptance gates. Q6 adds explicit common phase slip. Q7a extends both
velocity modes to prescribed planar ALE.

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

The public moment diagnostics use binary64. An unavailable SI mass-moment density or
Jacobi coefficient is reported as quiet NaN if its magnitude cannot be represented.
Exact vacuum densities are zero. The inversion and conservation gates evaluate their
scaled operands before narrowing diagnostics. Rebuild C++ consumers against these report
and callback layouts when upgrading to Q8c. The interval report preserves original event
donors, products and material packets. Its derived event-moment summaries use binary64;
the source corrector retains the original precision for replay and convergence.

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
| Interval and balances | Local Q1 event reports and the fixed-layout gross corrector record belong to the accepted or selected trial interval. The report identifies the source stage, pre-transport generation/time and residuals. Global species defects include reaction conversion. Mass and declared-element defects in the coupled report charge external reservoir material. The energy defect includes applied heat and native pressure work, in joules. |

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
| Native mesh and boundaries | Closed, supported orthogonal native geometry with common pressure and equilibrium temperature. Prescribed fixed-topology planar ALE supports Cartesian X/Y/Z, cylindrical axial Z on a nonperiodic sector, and `SemiStructuredXY_Z` axial Z. Moving geometry requires nonperiodic topology, zero mixture viscosity and native slip walls. General remap, AMR and open-reservoir face transport are outside this capability. |
| Velocity and order | `NativeImplicit` admits `SharedCarrier` (default) and `SuppliedPhaseSlip`. Fixed closed Cartesian meshes also admit `NodeDependentSlip` with `KineticEuler`. Fixed closed orthogonal Cartesian, native cylindrical and cylindrical `SemiStructuredXY_Z` meshes admit `SharedCarrier` with `RealizableLinearSSPRK2`. The scalar background uses transient backward Euler and upwind. Independent phase momentum, steady assembly, BDF2 and ordinary scalar high-order limiters reject. |
| Gravity and applicability | Fixed native implicit transport, fixed closed Cartesian kinetic Euler, and fixed shared-carrier SSPRK2 on Cartesian or qualified cylindrical meshes admit optional SI acceleration. A supplied zero vector selects the forced policy. Planar ALE requires absent acceleration. Applicability ranges come from the supplied laws. |
| Family schema | One named fixed-composition family per dispersed gas phase; catalog, family stamp, reference scale and phase ordering agree on all ranks. Variable composition, cross-family coalescence and unmigrated count-only state reject. |
| Local material and topology | Constructive paired birth, growth, shrinkage, disappearance, coalescence and breakup; transfers may involve the liquid carrier or a declared external reservoir. Declaring either direction of a family-to-background-gas material slot rejects. Generic chemistry, face transport and interphase laws can act on unbound compartments. A legacy number law must name only unbound populations. Selective or unpaired family material returned by a law rejects the preview. |
| Thermal and pool closures | Common temperature and pressure, zero surface tension. Separate phase or node temperatures, bulk-gas conversion and capillary/surface-energy closures reject. A separate background gas may carry its own inventory. |
| Transactions and restart | All owners use a common numerical configuration and controller selection. Commit publishes one retained trial. Rejection preserves accepted canonical state, histories and other compatible previews. Fixed-grid restart supports versions 1–5 under their original modes, version 7 for unforced kinetic Euler, version 8 for unforced realizable linear SSPRK2, and tagged version 9 for kinetic Euler or realizable linear SSPRK2 with supplied acceleration. Moving restart uses version 6 and binds the original and accepted geometry, motion controls, partition, target policy and supplied physics. |

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

### Q8a heated-cavity corrector

Each live source evaluation records the normalized gross residual and flux change.
The first evaluation has no predecessor. Subsequent entries identify the winning
stage, gross slot, canonical geometry cell and, for Fourier contact, the ordered
face stage and canonical geometry face. The witness retains both packets, their
directions, the comparison scale, relative tolerance and each packet's lower and
upper interval radii. Heat entering the named cell is positive. Equal residuals
use the smallest geometry cell ID, then stage kind, stage index, gross slot and face
ID. All ranks receive the same winning operands. A capped solve exposes the trace
through `SourceCorrectorExhausted`. Accepted trials expose it through flow diagnostics.
The live trace is omitted from portable restart v5–v9.

The original lagged 16×16 corrector rejected its first 1/128 s step at evaluation
80. Ordered face stage 0, gross slot 48, cell 149, face 159 carried heat out of the
cell. The packet changed from −4.3210916714997438e−5 to −4.3210922569869581e−5 J.
The larger magnitude set the scale. Each packet had 2.8421709430404007e−14 J
of uncertainty in each direction. At relative tolerance 1e−8 the uncovered gap
gave 13.41797022 times the allowed change. The signed packet increments alternated
over the final evaluations. At evaluation 80 the cell had 4.2362546163 J/K of
fixed-pressure caloric response and `dt * sum(G) / C = 0.47211515`.

| Single change to the lagged 16×16 case | Outcome | Winning gross residual on first step |
| --- | --- | ---: |
| Zero Fourier conductance | All 16 steps accepted in five evaluations each | 0.0186524 (QBMM slot 68) |
| Gravity off | First step exhausted | 31493.6171 (Fourier face 875) |
| Zero prescribed slip | First step exhausted | 3.94964718 (Fourier face 74) |
| Population events off | First step exhausted | 13.42438990 (Fourier face 108) |

After an unsuccessful outer evaluation the host solves
\((C+\Delta t L_G)\delta T=C(T_{\rm recovered}-T_{\rm guess})\) for the next
temperature guess. Each ordered Fourier stage evaluates its conductance once per
canonical face using the actual predecessor composition and the stage temperature.
The graph sums these conductances. The last active incident stage on each row supplies
its temperature operand and occupied fixed-pressure capacity
\(C=\sum_a m_a[(\partial u_a/\partial T)_p-p_a(\partial\rho_a/\partial T)_p/\rho_a^2]\).
Equilibrium recovery sums all occupied phases. Separate-temperature recovery uses
the selected phase and its own contact graph. The correction respects the temperature
bounds of the incident stages and recovered phases.
The first physical recovery must succeed within the supplied EOS domains before
the block can supply a correction. If the block or its bounded update is
unavailable, the next evaluation uses the recovered temperature.

The next outer evaluation restarts from accepted inventories and recomputes transport,
conductances, signed face packets, EOS states, pressure work and every physical
acceptance gate. The linear solve contributes only a temperature guess. The final
receipt books one signed pair \((-Q_f,+Q_f)\) per canonical contact, with the same
asymmetric operand intervals and strict subnormal comparison as the original corrector.

The two-cell oracle uses 128 and 384 J/K, \(\Delta t G=384\) J/K and a 5 K
initial difference. Its implicit contact transfers 384 J, reverses to −384 J
when the initial temperatures are swapped, and gives exactly zero for equal
temperatures. A separate ordered-stage oracle checks a contact after diffusion
using composition-dependent liquid capacity and an ideal-gas contribution of
\(n(C_v+R)\) to the equilibrium response.

The manufactured cavities below each completed 16 steps of 1/128 s through
1/8 s with the 80-evaluation cap. Each evaluation count is the maximum for an
accepted step. Residual and conservation columns give the largest value across
accepted steps and all three rank counts. The gross residual limit is 1.

| Grid | Evaluations, serial / MPI-2 / MPI-4 | Gross residual | Absolute mass defect (kg) | Absolute internal-energy defect (J) |
| --- | --- | ---: | ---: | ---: |
| 2×2 | 6 / 6 / 6 | 0.0976422 | 4.44e−16 | 7.06e−11 |
| 4×4 | 7 / 7 / 7 | 0.882228 | 6.66e−16 | 6.89e−11 |
| 8×8 | 8 / 8 / 8 | 0.803010 | 1.55e−15 | 6.52e−11 |
| 16×16 | 11 / 11 / 12 | 0.983561 | 2.44e−15 | 6.16e−11 |
| 32×32 | 14 / 14 / 13 | 0.722382 | 1.42e−14 | 6.66e−11 |

The maximum absolute sum of opposite face packets was zero at every resolution.
At fixed 4×4 mesh the separate timestep study reached the same horizon with steps
of 1/64, 1/128 and 1/256 s. The serial thermal dipole was 0.6779813, 0.6794844
and 0.6802403 K·m, respectively. The circulation and thermal-dipole refinement
checks passed on one, two and four ranks.

## Q7a prescribed planar ALE

For each bound phase the unknowns are extensive species moles, H0, H2, H3 and signed internal
energy. Let \(w_f\) be the certified swept volume rate, oriented from the face owner. A cell uses
the same signed face divergence \(D_c\) for every channel. The pre-source stock satisfies

\[
Q_c^* - Q_c^n + \Delta t\,D_c\!\left[
(\phi_{\rm carrier}^{\rm abs}+\phi_{p}^{\rm slip}-w)
(Q^*/V^{n+1})_{\rm up}\right]=0 .
\]

The native factory assembles the history with \(Q^n/V^n\) and coefficient \(V^n/\Delta t\).
The new concentration diagonal contains \(V^{n+1}/\Delta t\). All channels advance through the
same number of positive extensive sweeps, with diagonal
\(1+\Delta t\sum_f[\sigma_{cf}\phi_{p,f}^{\rm rel}]_+/V_c^{n+1}\).
The species and moment packets already carry their donor's bulk concentration. Each phase selects
its donor using its own relative flux. Counterflow therefore preserves the individual upwind
fractions even when their sum differs from one. Acceptance checks the stored operator residual,
the cell face balances, all closed inventories, species-derived H1, fixed composition, exact
absence and four-moment inversion. Transport-law gates use the accepted volume at entry and
the trial volume at the endpoint.

The motion certificate verifies \(D_c w=(V_c^{n+1}-V_c^n)/\Delta t\). At a closed moving wall
the absolute carrier flux equals the sweep rate, the wall-normal drift is zero and the relative
phase transport vanishes. Each phase carries its own advected material volume
\(W^{\rm adv}_{p,c}=W^n_{p,c}-\Delta t D_c F^V_p\). The flow owner charges pressure work
\(-p_{p,c}^{\rm abs}(W^{\rm final}_{p,c}-W^{\rm adv}_{p,c})\). With \(\theta\) the sum of
the upwind phase material fractions, the absolute pressure projection uses the full target

\[
\frac{\sum_p (W^{\rm final}_{p,c}-W^{\rm adv}_{p,c})}{\Delta t}
+\frac{\sum_p W^n_{p,c}-V_c^{n+1}}{\Delta t}+D_c(\theta w).
\]

A direct `preview_ale` accepts supported native inputs. The reacting controller evaluates the
target once per preview and invokes its private coherent source path on live trial geometry.
Population laws receive the trial location, volume and epoch. Paired population allocations,
controller ownership and transport-first source convergence use the Q3 rules.

Preview returns with accepted geometry and absolute face flux restored. Its owned certificate
identifies the creating solver and motion authority, local maps, old and new volumes, swept rates
and target. These lifetime identities survive reuse of a former object address.
Trial reports use the retained bulk volumes and geometry epoch. Another preview or rollback
can change the raw mesh epoch while the accepted generation and the earlier trial remain valid.
Commit requires collective agreement on the selected preview and exact restaging of its
certificate. It prepares the field imports and publication before accepting the motion and
publishes the retained state, absolute flux, diagnostics and interval receipts together.
Target, thermodynamic and population laws run during preview. Commit replays their retained results.

A motion checkpoint preserves the accepted geometry and native QBMM publication, including
the controller fingerprint and available interval receipts. The version 6 portable image
also binds the original reference map, motion axis, deformation start, displacement and
quality limits, GCL tolerances, accepted surface and deformed edges. For an accepted displacement,
the image records the prior surface and timestep. The reader reconstructs the previous cap
position and checks the absolute exterior flux against the resulting wall sweep, including zero
flux at stationary walls. Before publishing, it checks the receiving EOS and the effective
mixture viscosity on the restored geometry. After restore, the motion object exposes equal old and
new control volumes at the restored surface, a zero timestep and zero pending mesh sweep. The solver
retains the absolute flux from its last accepted physical interval. Fixed-grid controller history and a moving controller’s motion authority
are distinguished at checkpoint and restart. A newly attached moving controller can export a
fresh initial condition. After an interval produced on a fixed grid, its first accepted moving
step establishes the moving history for export. The image retains per-rank
canonical state, clock, solver diagnostics and absolute face flux, including moving-wall history.
A controller-written v6 image also preserves the accepted corrector count, reaction extents,
heat and transfer totals, and per-phase population births, deaths and number changes. Prior raw
interval receipts remain unavailable on portable restore.
Direct callers use `serialize_qbmm_restart(motion)` and `restore_qbmm_restart(bytes, motion)`
for numeric targets. Both overloads also accept a named target identity and positive revision.
The reacting controller supplies that pair through its own `serialize_qbmm_restart()` and
`restore_qbmm_restart(bytes)`. Each rank restores its own image with matching owned and overlap
maps and the same target policy. A new motion authority is constructed
on the original reference mesh so that its next displacement uses the same deformation map.

## Q7b node-dependent slip on a fixed Cartesian mesh

Select `Velocity::NodeDependentSlip` and `FamilyTransport::KineticEuler` together.
`Options::qbmm_node_slip_laws` follows physical EOS order. Each bound family supplies one
immutable `QbmmNodeSlipLaw` matching its schema declaration, identity, revision, units, bases
and inclusive SI ranges. Every unbound law slot is empty. Omit the initial
`CellState::drift_velocity` for all-kinetic states. A mixed layout can supply ordinary phase
drifts in physical order with a zero placeholder at every kinetic slot. The law supplies the carrier-relative vector in m/s for one active atom.
Its input includes physical mass per bubble, equivalent spherical diameter, donor pressure and
temperature, and separate family-gas and carrier-liquid concentrations in their EOS volumes.
The law runs deterministically on the donor owner and makes no MPI calls. It takes no numerical
timestep. Slip-speed applicability uses the returned vector. Interval applicability uses the
substep actually taken. An out-of-range `ReportOnly` observation is retained. A failed
`RejectOutside` gate rejects the entire preview collectively.

### Transport, conservation and pressure work

Use the absolute carrier flux frozen for the current corrector and the face area vector directed
from the canonical owner cell \(L\) toward its neighbor \(R\). For an atom in donor cell \(d\),

\[
 g_{d,a}=\phi_{c,f}^{abs}+\mathbf u_{rel,d,a}\cdot\mathbf A_f,\qquad
 q_{L,a}=\max(g_{L,a},0),\quad q_{R,a}=\min(g_{R,a},0).
\]

Quadrature indices belong to a single donor reconstruction. Every active direction transfers a
signed count \(b=h q_{d,a}w_{d,a}/V_d\) using donor **bulk** volume, and carries all these
extensive quantities together.

| Channel | Signed integrated transfer |
| --- | --- |
| Moments of scaled mass \(\xi_a=x_a/x_*\) | \(b(1,\xi_a,\xi_a^2,\xi_a^3)\) |
| Moles of species \(s\) and physical mass | \(bY_s x_a/W_s\) and \(b x_a\), with species authoritative |
| Internal energy and material work volume | \(b x_a U_d/M_d\) and \(b x_a/\rho_{p,d}^{EOS}\) |
| Physical momentum | \(b x_a(\mathbf u_{c,d}+\mathbf u_{rel,d,a})\) |

Coincident presence applies to each directed packet. Opposite streams can cancel any subset of
the signed face moments. Recover the thermodynamic trace of an arriving phase before using it
as a donor. Closed boundaries transfer zero material. The source stage sees the final transported
stocks once per corrector. Interphase exchange and diffusion own their declared enthalpy terms.
The native work term uses \(-p(W^{final}-W^{adv})\), where \(W^{adv}\) starts from accepted
material volume and includes the signed volume transfers.

Each substep exchanges owner traces on the one-ring halo, forms canonical face rates and
returns all outgoing reservations to donor owners. The maximum across owners, families and atoms
selects one duration \(h\) satisfying

\[
 C_{c,a}=\sum_{f\in c}h[\sigma_{cf}g_{c,a,f}]_+/V_c
 \le \mathtt{kinetic\_cfl}\le 1 .
\]

Owners use the actual fractions to check the positive remainder \((1-C_{c,a})w_{c,a}\).
Canonical packet incidences update cell stocks before moment inversion and the next owner-law
evaluation. Zero transfer preserves the original stocks. Full evacuation produces exact absence.
An empty owner enters each exchange, reduction and failure fence. Rejection on invalid arithmetic,
lost time progress or exhausted `maximum_kinetic_substeps` preserves the accepted state and
previously retained valid trials.

For the pressure linearization, each active donor branch contributes
\(\alpha=w_a x_a/(V_d\rho_{p,d}^{EOS})\) to \(\theta\) and
\(\alpha\,\mathbf u_{rel,d,a}\cdot\mathbf A_f\) to \(S\). At an exact zero of \(g\), use
one half of each tied branch in both coefficients. With the corrector carrier flux fixed, the
substep-time averages satisfy \(\bar F^V=\bar\theta\phi_c+\bar S\) against the direct material
volume packet. When the binary64 derivative loses precision at the subnormal scale,
`pressure_rounding_flux` records the coefficient representation correction at the frozen
carrier flux. The receipt retains the independently accumulated physical slip coefficient.
The pressure solve evaluates \(F^V(\phi)=F^V(\phi_0)+\bar\theta(\phi-\phi_0)\) using the
direct all-phase rate at the frozen flux \(\phi_0\). This form preserves a small residual when
normal carrier and slip terms nearly cancel. A zero represented derivative retains the
prescribed rate and requires positive EOS susceptibility on every cell for pressure anchoring.

### Momentum, receipts and continuation

Summing both streams across the kinetic and ordinary phases gives nonnegative mass rates
\(F^+,F^-\). The carrier operator transports \(F^+\mathbf u_L-F^-\mathbf u_R\). Net upwinding
plus the symmetric two-cell coefficient \(\min(F^+,F^-)\) supplies both contributions, including
at zero net mass flux. The drift flux uses each atom's own donor slip. For kinetic phases the
accepted and trial momentum storage includes \(\mathbf D_p=\sum_a w_ax_a\mathbf u_{rel,a}\).
The endpoint law runs after sources and thermal recovery, outside the pressure root. The
barycentric velocity is \((M_{tot}\mathbf u_c+\sum_p\mathbf D_p)/M_{tot}\).

In `PhaseFaceTransport`, an engaged `qbmm_flux[p]->kinetic` selects the directional representation.
Its forward and backward rates retain the canonical face sign. The legacy phase-volume-flux
slots are zero placeholders for kinetic families. The receipt records the actual pressure
coefficients and the mixture momentum inputs. The directional `carrier_momentum_flux`
uses the frozen transport trace. The mixture's `implicit_carrier_momentum_flux` uses the
solved owner and neighbor velocities before the separate pressure correction. `qbmm_report()` uses the retained endpoint
closure for drift and barycentric velocity, and reports the final corrector's subcycle count,
maximum used CFL, atom applicability and directional face evidence when the interval is available.

A checkpoint retains the endpoint closure and historical interval. A portable v7 kinetic image
binds the numerical method, velocity mode, law metadata, CFL and finite subcycle limit, and
retains endpoint node velocities and drift momentum for old momentum storage on continuation.
Commit, restore and reporting use the retained slip closure without invoking its evaluator.
Portable restore uses the original partition and reports prior raw interval packets as unavailable.
A fresh native flow can restore an image from a reacting controller and report immediately.
Attaching the matching controller restores its accepted diagnostics before the next preview.
The archive binds the endpoint EOS volumes used for applicability and the corresponding
retained velocities. Hard instantaneous ranges are checked on these operands during restore.
Earlier portable versions require their original model selections. Kinetic transport rejects
moving and cylindrical meshes, SSPRK2, BDF2, separate temperatures, variable composition,
bulk-gas conversion and capillarity.

### Q8b fixed Cartesian kinetic gravity

A reacting controller may supply a uniform `body_acceleration` in m/s² with
`NodeDependentSlip + KineticEuler`. A prescribed node law supplies its own relative
velocity. A gravity-dependent law sets `required_body_acceleration_m_per_s2` at
construction and receives that same acceleration in its input, including during
initialization of old drift momentum. The receiving controller must match the declared
dependency on every rank. A supplied zero vector selects the forced pressure and restart
policies. With acceleration absent, kinetic transport retains the v7 policy.

For the face from owner \(L\) to neighbor \(R\), use the owner-oriented area vector \(A_f\),
cell-center displacement \(d_f\), mean post-material density \(\bar\rho_f=(\rho_L+\rho_R)/2\)
and the same pressure residual in both reconstructions,

\[
r_f=\pi_R-\pi_L-\bar\rho_f g\cdot d_f,\qquad
G_c\pi=\rho_c g+\operatorname{LS}_{\rm interior,c}(r).
\]

The provisional absolute face rate is
\[
\tilde\phi_f=\overline{\tilde u}\cdot A_f
+\Delta t\,\overline{(G\pi-\rho g)/\rho}\cdot A_f
-\frac{\Delta t\,|A_f|}{\bar\rho_f |d_f|}r_f .
\]
Static walls have zero rate and supply no gradient sample. The cell pressure correction
uses `CellPressureGradient::Interior`. The directly accumulated volume rate anchors
\(F^V_f(\phi)=F^V_{f,0}+\bar\theta_f(\phi-\phi_{f,0})\). Exact ties use the symmetric
half derivative. The operator checks nonnegative mobility, EOS anchoring where mobility
vanishes, and the final corrected physical material continuity.

In the unconditioned caloric solve, a forced kinetic candidate retains the transported
temperature when it exactly satisfies the recovered enthalpy equation at the current
pressure. Otherwise the existing bracketed inversion and all physical gates apply.
This preserves an exact stationary volume without charging a kinetic withdrawal for
roundoff introduced by thermal bisection.

For a stationary one-dimensional Cartesian column with axial gravity, an absolute
pressure rounded to binary64 can leave a small apparent edge imbalance. The column
check integrates the trapezoidal head from actual densities and requires one common
pressure datum consistent with every cell's asymmetric rounding interval, including
ties-to-even. This applies only to mechanical pressures recovered from those same
absolute pressures, with unchanged physical inventories, zero material transport,
zero volume target, zero slip and no other impulse. Independent higher-precision
mechanical data and resolvable column-head errors retain their driving residual.
Multidimensional heads use the direct interior residual.

The source stage books one post-material \(M_c g\Delta t\) per corrector candidate.
The carrier predictor includes old-to-new node drift momentum and both directional
mass rates. The solved predictor velocity enters the implicit carrier face receipt.
Projection supplies the final cell velocity and its pressure impulse. Body and other
external impulses are retained separately in kg m/s before their sum enters the physical
momentum equation. In the balanced predictor the hydrostatic pressure contribution cancels
the matching booked body impulse before rounding, preserving a small collinear reservoir
impulse. The force creates no caloric source.

Portable v9 uses little-endian 64-bit words. Magic and version are followed by the payload
type (3 for fixed kinetic), force presence, three SI components, the balanced pressure
and thermal revisions, advancement controls, the physical-provider fingerprint, and the
accepted split impulse when available. An advanced image also carries the interval start,
duration and rank-local body, other and total impulse operands. Restore checks the
clock, the local body impulse against the canonical post-material species mass, and the
three independently reduced totals before publishing the endpoint. The common
state header checks the rank-local mesh and partition, ordered phase roles, declared EOS
schema, species catalog, family stamps, law and controller signatures, and the kinetic
CFL and substep policy. It retains the v7 endpoint node velocities, drift momentum,
EOS volumes, accepted face rates and interval summaries. A receiving flow can report
the accepted endpoint before attachment. Continuation requires the matching controller.
Owner checkpoints retain raw interval receipts. Portable history exposes only the saved
summaries. Unforced kinetic and linear archives, planar archives, and fixed implicit archives keep their
original wire versions and decoders.

For a built-in ideal gas, v9 binds the gas constant and every species' heat capacity,
reference temperature and reference energy. For an opaque EOS, liquid property law or
supplied gas transport callback, declare a `ThermalLawIdentity` in the corresponding
physical-phase slot of `Options::qbmm_eos_law_identities`. A supplied mixture
viscosity law also needs `Options::qbmm_mixture_viscosity_identity`. Each declaration
contains a name, revision and matching catalog fingerprint. Update its revision when
the callback or its parameters change. These bindings include unoccupied phases.
A forced accepted endpoint retains its controller signature and aggregate diagnostics
across detachment, reattachment and native re-encoding.

The serial, MPI-2 and MPI-4 column oracles integrate the hydrostatic head from the
actual ideal-gas and liquid inventories. Three accepted steps gave zero maximum
speed, absolute face rate and edge defect for the stratified and reversed columns
and the nonaxial constant-density head with dyadic data. With axial gravity of
−0.981 m/s², speed and face rate remained zero and the maximum raw edge defect
was 5.33×10⁻¹⁵ Pa. Opposite 0.001 Pa perturbations produced opposite vertical
velocities of magnitude 4.74×10⁻⁶ m/s. The flat-pressure stratified starts gave mean
vertical velocities of −0.014677 and +0.014684 m/s on the first step.
For weights 2 and 1 at scaled masses 1 and 3, the isolated-face test used
\(s=1/8\) m/s and \(A/V=4\) m⁻¹. The four upward rates were 2, 4, 10 and 28 per second.

The four-step closed plume transports two sizes across a rank face. The table gives
the serial values. The same assertions ran on two and four ranks.

| Quantity at 1/16 s | Prescribed slip, force absent | Prescribed slip with gravity |
| --- | ---: | ---: |
| Fine / coarse number centroids (m) | 0.504762 / 0.512477 | 0.504153 / 0.511876 |
| Gas mass centroid (m) | 0.509391 | 0.508787 |
| Fine / coarse mean rates into the top layer (s⁻¹) | 0.842330 / 1.223470 | 0.817766 / 1.210114 |
| Gas retained in the top layer (kg) | 0.0176279 | 0.0173754 |
| Integrated liquid return volume (m³) | 0.000861892 | 0.00134738 |
| Bottom minus top pressure (Pa) | −1.40745 | 9.65712 |

A separate two-cell counterflow took four kinetic substeps. The independent donor
replay transferred 0.00774147 kg upward and 0.0140850 kg downward, recovered the
change in drift momentum and checked the carrier momentum row. The full-step
body impulse was −0.4375 kg m/s along the column. MPI-4 used two empty owners.

The forced and explicit-zero restart fixtures each write two successive accepted
states. A separate process reads the first image, advances once, and compares its
canonical state, retained closure, face rates and diagnostics with the second image
on the same partition. Owner-checkpoint replay checks the same continuation in memory.
The rejection cases vary force presence and components, node-law revision, EOS and
viscosity bindings, controller signature, payload type, method, interval duration and
impulse totals. A failure on one owner preserves the accepted state and the previously
valid preview. The v7 continuation check retains absent forcing.

## Q7c realizable linear SSPRK(2,2) on fixed meshes

Select `QbmmModelOptions::FamilyTransport::RealizableLinearSSPRK2` with
`Velocity::SharedCarrier`. The family transport controls are separate from
`QbmmModelOptions::transport`, which keeps its `ScalarTransportDiscretization` type
for the transient backward-Euler upwind background. The fixed mesh supplies oriented face
rates, bulk volumes and physical centroids. Native cylindrical sectors use the volume
centroid of the sector and the surface centroid of each curved face. A polygonal
`SemiStructuredXY_Z` cylinder uses the metrics of its extruded polygons. The native
cylindrical mesh constructor requires a positive inner radius. A closed exterior wall has
zero transport in both Runge–Kutta stages.

### Donor reconstruction and stage acceptance

Let \(Q_i=(n_s,H_0,H_2,H_3,U)_i\) denote extensive family inventory and
\(B_i=Q_i/V_i\). At each stage the donor owner gathers neighboring densities from a
one-ring cell halo. The same geometric stencil acts on each independent component. The
derived slope of \(H_1\) is \(\sum_s W_s\nabla B_{n_s}/x_*\).
For face centroid \(r_f\) and cell centroid \(r_i\), put
\(d_{if}=\nabla B_i\mathbin{\cdot}(r_f-r_i)\). Exterior walls contribute no
fictitious neighbor stocks. An exactly empty donor uses a zero slope.

For each transporting face, the weighted sum of neighbor displacements must reproduce
the donor-to-face displacement to roundoff. This admits one-dimensional and planar
stencils when they determine the active trace. A two-sector annulus with one radial
cell can have both cell centroids on a line that misses their common angular face
centroid. Affine fields that agree on that line can differ at the face. A nonzero flux
across such a face rejects collectively.

For a stage of duration \(h\), define the outgoing fractions and their donor total by

\[
c_{if}=h[\sigma_{if}\phi_f]_+/V_i,\qquad C_i=\sum_f c_{if}.
\]

Owners agree on a common substep with \(C_i\le\mathtt{linear\_cfl}\le1/2\).
The default cap is \(1/2\). Exhausting `maximum_linear_substeps` or losing representable
time progress rejects the preview. Starting at one, the owner halves one common limiter
\(\ell_i\) for all components and all outgoing faces, with zero available as the upwind
fallback. A candidate must certify every outgoing trace **and** the material left behind.

\[
T_{if}=B_i+\ell_i d_{if},\qquad
R_i=B_i-\frac{\sum_f c_{if}\ell_i d_{if}}{1-C_i}.
\]

A certificate checks the species mass relation under the Q1 binary64 policy, exact
zero for excluded species, exact vacuum, the Q1 inversion with positive support, and
the signed energy interval \(e_{\min}m\le B_U\le e_{\max}m\). Here
\(m=\sum_s W_s B_{n_s}\), and the specific-energy bounds come from occupied members
of the donor stencil. In particular, an unoccupied neighbor contributes no zero-valued
specific-energy sample. Negative reference energies are valid.
The bound comparison uses the signed cross-products of extensive energy and
species-derived mass. It remains meaningful for representable energy stocks whose
specific energy would underflow a binary64 division.
Ordering uses the sign of the scaled difference. For independently rounded stocks at
a bound, the allowed cross-product defect is at most two binary64 epsilons times the
sum of the two product magnitudes. There is no absolute energy floor.

The implementation represents a trace as \(V_iT_{if}\) and the retained stock as
\(V_i(1-C_i)R_i\). Both scale factors are strictly positive. Retaining this scale through
the cone test and packet calculation avoids an independent division and rounding of
each moment at a rank-one boundary. Species remain authoritative for the mass moment.
A positive decomposition of a singleton can change its weight while retaining its
support. A slope that would move that support uses the first-order fallback.

Canonical face owners distribute the certified packets needed by incident cells. A
forward-Euler stage combines \((1-C_i)R_i\) with
\(\sum_{f\,\mathrm{in}}h|\phi_f| T_{jf}/V_i\). Each coefficient is
nonnegative even when the frozen carrier flux has nonzero divergence. SSPRK(2,2) applies
this operation twice and takes half the starting stock plus half the second forward-Euler
result. Both stages require certified represented endpoints.

### Physical packets, correctors and restart

Each substep contributes \(h(F^{(1)}+F^{(2)})/2\) to the interval integral. The stages
carry species, count, both shape moments, signed energy, species-derived mass, material EOS
volume, pressure coefficients and momentum. The source snapshot and pressure work
\(-p(W^{final}-W^{adv})\) consume the resulting transport. The corrector freezes its
carrier flux across both stages and evaluates the coupled sources on the transported
state. The spatial accuracy qualification concerns this explicit advection operator.
Whole-step temporal accuracy also depends on the background flow, pressure and source
integration.

For an interior face with zero frozen carrier rate, both one-sided EOS traces contribute
half their material-volume coefficient to the pressure derivative. Each side starts at
the donor's accepted limiter and can reduce its slope to an admissible trace for this
zero-rate face. An unresolved offset uses the stage-center value. These ties carry
zero material, energy and momentum.

`owned_phase_transport()` exposes the final corrector's interval-averaged face rates.
A live trial or its accepted report retains the interval packets. Owner-bound checkpoints
preserve the accepted interval. Portable version 8 binds the method, algorithm revision,
CFL and subcycle cap at an accepted step boundary. A new process resumes from canonical
state and accepted carrier flux on the same partition. It needs a single accepted time
level. Raw historical face receipts follow the existing portable restart availability policy.

The trial diagnostics, report and available interval expose `family_transport`,
`linear_substeps`, `linear_stages`, `maximum_linear_cfl`, `maximum_linear_stage_residual`
and `maximum_linear_step_residual`. The residuals measure the worst cell and channel
balance over the forward-Euler stages and the substep averages, respectively. They are
normalized by the configured conservation tolerance and must be at most one.

### Smooth-interior spatial convergence

The prescribed-transport test rotates two positive atom weights at scaled masses 1 and 2
and a signed energy perturbation through a fixed interval of 0.01. A streamfunction
sets the face-integrated divergence-free carrier flux and tapers it to zero outside the
vortex. All three meshes use one axial layer. The native cylindrical sector has curved
radial faces. The semi-structured sector has straight-sided quadrilaterals in XY.
Tensor-product Gauss quadrature integrates the analytic initial and rotated final
densities over each physical cell, using the polar or bilinear Jacobian where appropriate.
The error is the volume-weighted mean absolute density error. Each reported order is
the logarithm to base two of successive errors.

| Mesh | Channel | 16 × 16 | 32 × 32 | 64 × 64 | 16 → 32 | 32 → 64 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cartesian | H0 | 3.474e-05 | 8.771e-06 | 1.979e-06 | 1.986 | 2.148 |
| Cartesian | H1 | 5.172e-05 | 1.355e-05 | 3.146e-06 | 1.933 | 2.106 |
| Cartesian | H2 | 8.996e-05 | 2.383e-05 | 5.703e-06 | 1.917 | 2.063 |
| Cartesian | H3 | 1.713e-04 | 4.484e-05 | 1.098e-05 | 1.933 | 2.030 |
| Cartesian | species A | 1.293e-05 | 3.387e-06 | 7.866e-07 | 1.933 | 2.106 |
| Cartesian | species B | 1.940e-05 | 5.080e-06 | 1.180e-06 | 1.933 | 2.106 |
| Cartesian | U | 1.696e-05 | 4.133e-06 | 9.271e-07 | 2.037 | 2.156 |
| Cylindrical sector | H0 | 3.501e-05 | 8.777e-06 | 1.923e-06 | 1.996 | 2.190 |
| Cylindrical sector | H1 | 5.222e-05 | 1.361e-05 | 2.972e-06 | 1.940 | 2.195 |
| Cylindrical sector | H2 | 9.023e-05 | 2.389e-05 | 5.211e-06 | 1.917 | 2.197 |
| Cylindrical sector | H3 | 1.705e-04 | 4.487e-05 | 9.844e-06 | 1.926 | 2.188 |
| Cylindrical sector | species A | 1.306e-05 | 3.403e-06 | 7.431e-07 | 1.940 | 2.195 |
| Cylindrical sector | species B | 1.958e-05 | 5.105e-06 | 1.115e-06 | 1.940 | 2.195 |
| Cylindrical sector | U | 1.725e-05 | 4.205e-06 | 9.281e-07 | 2.036 | 2.180 |
| Polar XY/Z extrusion | H0 | 3.510e-05 | 8.790e-06 | 1.930e-06 | 1.998 | 2.187 |
| Polar XY/Z extrusion | H1 | 5.233e-05 | 1.363e-05 | 2.981e-06 | 1.941 | 2.193 |
| Polar XY/Z extrusion | H2 | 9.052e-05 | 2.391e-05 | 5.215e-06 | 1.921 | 2.197 |
| Polar XY/Z extrusion | H3 | 1.706e-04 | 4.487e-05 | 9.833e-06 | 1.927 | 2.190 |
| Polar XY/Z extrusion | species A | 1.308e-05 | 3.406e-06 | 7.452e-07 | 1.941 | 2.193 |
| Polar XY/Z extrusion | species B | 1.962e-05 | 5.110e-06 | 1.118e-06 | 1.941 | 2.193 |
| Polar XY/Z extrusion | U | 1.723e-05 | 4.206e-06 | 9.287e-07 | 2.035 | 2.179 |

The finest-grid test also advances the same interval in two and four equal steps.
For the seven requested channels, the largest difference between the one-step and
two-step solutions is below 2% of the measured one-step spatial error. Each successive
time refinement must change the solution by less than 5% of that error. The two
species have different positive mass fractions and molar masses. Their proportional
errors and the derived first-moment error reflect their fixed composition.


## Q8c fixed shared-carrier SSPRK(2,2) with gravity

A reacting controller can supply one uniform acceleration in m/s² for closed Cartesian grids,
native cylindrical sectors and complete annuli, and qualifying orthogonal polygonal disks or
annuli extruded in Z. The native annulus has a positive inner radius. Its azimuthal seam is an
interior face with one owner and two incident cells. Polygonal disks can include the axis.
A forced XY/Z extrusion must have an edge-connected base with one outer boundary and, for an
annulus, one nested concentric inner boundary. The corners on each wall follow the circle
around its center. A straight facet may have intermediate collinear vertices. This check uses
the boundary geometry independently of patch names.
Each transporting SSPRK2 trace still requires a resolvable face-centroid offset. The forced
method certifies the common reconstruction factor against every interior face the cell could
supply. A small transverse rate can then reverse without adding or removing a constraint on
a large radial or axial packet. Only actual outflow contributes to the donor withdrawal or
CFL. Both donors supply pressure derivatives at an exact tie. A nontransporting, unresolved
offset uses its cell-center trace. The v8 method retains its outgoing-face limiter.
The combination requires shared carrier, fixed geometry and disabled bulk-gas conversion. The cylindrical
kinetic, supplied-slip SSPRK2, and moving-gravity combinations reject.

Pressure reconstruction uses the Q8b edge residual with the actual center displacement in
Cartesian coordinates, including across the seam. The scalar area and center distance set the
pressure mobility. The owner-oriented area vector projects the velocity onto the same face.
In particular, a curved radial sector face has area \(r\,\Delta\theta\,\Delta z\).
The cell reconstruction uses interior differences, with zero carrier flux at a static wall.
The reacting source books one post-material \(M_c g\,\Delta t\) per cell in each private
corrector candidate. Both RK stages and all substeps share the frozen carrier flux. Their
half-weighted species, energy, material-volume, momentum and pressure-derivative packets feed
the outer solve. Replacing a candidate, committing it or restoring it introduces no further
body impulse. An explicit zero acceleration selects the same pressure and thermal policies.

If the unrelaxed occupancy, flux, continuity, thermal, source and transport checks pass but
a Fourier contact dominates the failing gross residual, Q8c takes a quarter step in the next
Fourier temperature correction and in the pressure, carrier-velocity and face-rate guesses.
The next corrector recomputes both RK stages, physical sources, EOS recovery and projection
from the accepted predecessor. Every accepted heat packet uses the two temperatures actually
supplied to its face law. The heated tests also measure the difference obtained by evaluating
the contact again at the recovered endpoint temperatures.

Portable v9 payload **4** and linear revision **1** together identify the forced all-incident
limiter, including for supplied zero acceleration. A bare v8 revision **1** uses the outgoing
limiter. The two policies can produce different face packets on the same inputs. The v9 payload
retains the CFL and substep controls, stage counts, maximum CFL and normalized stage and step residuals. Its forced header binds
the supplied acceleration, balanced pressure and thermal revisions, physical providers,
advancement controls and reacting controller. The endpoint checks the same partition,
geometry, EOS and family bindings as the writer. An accepted forced interval carries the
split impulse, clock and rank-local impulse operands for validation against the endpoint mass.
Owner checkpoints retain the raw receipts. A fresh reader exposes the archived summaries
and requires the matching controller for advancement. V9 payload **3** continues to identify
forced Cartesian kinetic transport. Versions 5 through 8 retain their own layouts, and a
bare v8 linear image binds absent force.

The public report adds `balanced_linear_pressure_policy`, `forced_linear_thermal_policy`,
`linear_cfl_limit` and `linear_substep_limit`. The two revision fields engage for supplied
acceleration, including zero. The Q7c table measures smooth-interior prescribed family advection
with the unforced outgoing-trace limiter. The Q8c tests measure stage replay under gravity
and grid and timestep sensitivity of heated vessels with a backward-Euler background.

### Numerical evidence

The serial, MPI-2 and MPI-4 rest tests advanced three steps on each mesh. The table gives
the largest speed, interior or wall face rate, and trapezoidal pressure defect over the
stratified, reversed, constant-density oblique and absent-gravity rest cases at all three
rank counts. The oblique acceleration was (0.5, −0.25, −1) m/s². Positive and negative
flat-pressure starts accelerated in the corresponding axial direction.

| Mesh | Speed (m/s) | Face rate (m³/s) | Edge defect (Pa) |
| --- | ---: | ---: | ---: |
| Cartesian | 2.61e-15 | 2.48e-15 | 3.53e-13 |
| Native walled sector | 1.11e-14 | 8.21e-16 | 8.29e-13 |
| Native annulus with seam | 5.59e-15 | 3.64e-15 | 7.41e-13 |
| Polygonal disk | 9.37e-15 | 7.46e-15 | 8.46e-13 |
| Polygonal annulus | 4.1e-15 | 4.24e-15 | 6.93e-13 |

The balanced circulating replay used 4 substeps, 8 stages and 6 source evaluations.
The maximum mass-rate discrepancy against the two-stage replay was 2.22e-16 kg/s.
The endpoint body impulse was -5.81818182 kg m/s axially. The rotating native annulus
also transferred family mass across its azimuthal seam on a partition boundary.

Both heated vessels ran to 1/8 s. The table gives serial values on four radial and four axial
cells at the matched 1/64 s timestep, with inner-wall heating and outer-wall cooling. Reversing
the wall powers reversed circulation and the temperature and pressure dipoles. Removing gravity
suppressed vertical circulation while retaining the thermal dipole.

| Vessel | Inner / outer vertical speed (m/s) | Circulation (m²/s) | Thermal dipole (K m) | Pressure dipole (Pa m²) | Gas fraction | Bubble area (m²) | d₃₂ (m) | Inner / outer wall heat (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native sector | 0.001055 / -0.0007561 | 0.00062299 | 0.54897 | 0.0021914 | 0.5 | 30.8522 | 0.109392 | 600 / -600 |
| Polygonal annulus | 0.0006186 / -0.000444 | 0.00027153 | 0.50709 | 0.0028711 | 0.5 | 174.527 | 0.109392 | 3394.11 / -3394.11 |

At fixed timestep, doubling the radial and axial resolution changed circulation by 32.6% in the
native sector and 7.4% in the polygonal annulus. Halving the fine-grid timestep changed
circulation by 5.25% and 15.72%, respectively. The polygonal case needed a further
halving to 1/256 s to meet the 15% sensitivity criterion (11.13% on the finest pair).
These differences use the larger magnitude of each pair as their denominator.

Every interior Fourier contact booked an equal and opposite pair. Across all cases and ranks
the largest discrepancy against the heat computed from the actual face-law operands was
1.39e-17 J. Re-evaluation at recovered endpoint temperatures differed by up to
6.17e-12 J. The largest relative difference was 2.0 on a nearly isothermal contact with
opposite packet signs. The largest difference expressed in units of temperature spacing was
701 ULPs. These maxima are taken independently and compare different evaluation states.
The maximum momentum budget defect was 2.66e-14 kg m/s. Pressure work was reconstructed independently from face-advected
phase volumes, endpoint EOS volumes and phase pressures.

At one, two and four ranks, forced and supplied-zero tag-4 shards resumed in a fresh process
with the uninterrupted state, face fluxes and stage diagnostics. Owner checkpoints retained
the interval receipts. In the one-cell MPI case, a damaged shard on an empty owner rejected
collectively, and the controller could still commit its earlier valid preview.
