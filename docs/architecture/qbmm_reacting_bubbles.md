# QBMM reacting bubbles: code design

**Document:** SF-QBMM-CODE-001  
**Revision:** 0.2  
**Date:** 2026-11-19  
**Status:** Proposed extension to the inspected generic SimpleFluid design; not an implementation report.  
**Companion:** [Modeling specification](../modeling/qbmm_reacting_bubbles.md)

## 1. Purpose and decision

Extend the existing conservative reacting multiphase solver with a quadrature-based representation of dispersed bubble size. Reuse its species inventories, thermodynamics, phase-volume closure, transport, source iteration, advance ownership, and transactional publication. Do not create another flow solver or a parallel inventory authority.

The initial extension is a two-node quadrature method of moments (QMOM), with four scaled mass moments for each explicitly configured fixed-composition bubble family. Existing extensive population count supplies moment zero. Existing extensive species inventories supply moment one through their total mass. The extension stores only moments two and three as additional physical state.

The initial supported calculation has common thermodynamic pressure, common gas velocity within each family, and the current common-temperature caloric closure. No interface geometry is reconstructed. High gas volume fraction is a separate closure qualification, not a property conferred by moment inversion.

The code is general-purpose. Species, reactions, heat input, vaporization, dissolution, injection, and population laws come from application configuration or supplied immutable providers. No species identifier, reaction mechanism, gas yield, boiling law, empirical drag coefficient, or material dataset is built into the generic QBMM layer.

## 2. Source baseline and evidence

This revision was checked against the clean `develop` worktree at commit `0029a93fd5fcb83658daf82dc80e9cd72a490031`. The separate `refactor/vof` worktree at `bf219ff1b8ef09bf8ec72ff53430f45fa4bc9e08` was inspected only to resolve branch scope. Do not combine capabilities from those worktrees into one supported feature set.

The repository already contains the following facilities:

| Facility | Current source status | Consequence for QBMM |
|---|---|---|
| Generic catalog and mechanism | `SpeciesCatalog`, `ReactionMechanism`, immutable schema and conservative reaction receipts | Reuse species identity and conservation checks. |
| Chemical compartments | `CellSpeciesState`, liquid, dissolved, resolved-gas, and named dispersed inventories | Use existing phase bindings; do not add a second gas inventory. |
| Conservative native flow | `Reacting::LowMachMixtureSolver` | Extend this owner, not an invented future backend. |
| Native reacting controller | `Reacting::ReactingMixtureCoupler` | Integrate the population stage into its source corrector. |
| Physical population count | `PhaseRole::DispersedGas` and optional `CellState::phase_numbers` | Alias moment zero to the existing count. |
| Higher moments and inversion | No native QBMM moment state or generic moment transaction was found in the inspected path | These are new work. |
| Thermodynamics | `MixtureThermodynamics`, caloric convention, physical phase bindings and common-pressure closure | Derive phase volume once from accepted material and thermal state. |
| Transactions | Preview, controller lease, commit, checkpoint/restore, fixed/ALE candidates | Add moments explicitly to every owned transaction structure. |
| Generic non-interface transport | Native phase transport with carrier and supplied phase-relative velocities | A dispersed-phase calculation does not require VoF. |
| VoF reacting integration | `ReactingVoFCoupler` excluded from the inspected solver build pending migration | Do not make it a dependency or claim current support. |

The current source is the authority for exact C++ signatures. New names and records below are **proposed interfaces**, even where they are shown in C++ syntax. Source references are collected in Section 18.

The current fixed-grid native path requires supported orthogonal face geometry. Its ALE path has further closed-domain, geometry, wall, and inviscid restrictions. This proposal inherits those restrictions. It does not expand native support to arbitrary nonorthogonal grids, open reservoirs, pressure waves, turbulent interfacial dynamics, or a two-momentum gas–liquid model.

## 3. Scope and capability gates

Deliver the extension in independently reviewable stages.

**Initial capability:** fixed mesh, admitted closed native-flow configuration, common temperature, common pressure, zero supplied phase slip, one fixed-composition family per compatible dispersed physical phase, backward-Euler upwind transport, local constructive population events, and complete restart/rollback support.

A common, explicitly supplied phase-relative velocity may be enabled only after phase material, count, moment, internal-energy, volume, and momentum bookkeeping have passed the corresponding transport checks. Native carrier velocity must not be relabeled as barycentric velocity without proving that identity for the selected relative-velocity closure.

The following remain optional later capabilities:

- Node-dependent size slip with coupled kinetic face packets.
- Higher-order moment advection and temporal integration.
- Variable composition within a population.
- Cross-family merging with different compositions.
- Unresolved bulk-gas conversion and associated exchange/area laws.
- Capillary pressure, surface energy, and independent node temperatures.
- Moment-aware ALE/remap on the native supported geometries.
- Independent phase momentum or interface-resolving methods.

Use explicit capability validation at construction and restore. Unsupported combinations fail before an advance begins; they do not silently select another physical model.

## 4. Ownership and state layout

### 4.1 Immutable family schema

A proposed `QbmmPopulationSchema` contains:

| Entry | Meaning |
|---|---|
| Population identity | Stable named dispersed identity, not a quadrature-node index. |
| Catalog fingerprint | Exact ordered species schema to which coefficients refer. |
| Physical phase binding | Existing compatible dispersed phase, used by the native owner. |
| Coordinate kind | Mass per bubble, SI kg. |
| Reference mass | Positive immutable `mass_scale_kg` used only for conditioning. |
| Composition mode | Initially `FixedMassFractions`. |
| Composition vector | Immutable normalized nonnegative mass fractions in catalog order. |
| Moment order/node limit | Orders 0–3, at most two quadrature atoms. |
| Closure versions | Explicit inversion, source, thermodynamic and transport policy identities. |

The family schema is part of checkpoint and provider compatibility. It is not reconstructed from vector lengths, field names, a guessed mean molar mass, or a default species.

Within a family, all bubbles share the same composition vector. Several families can represent several fixed compositions, but interfamily transport and merging still need explicit material rules. Sharing a physical phase between several families is deferred unless a thermodynamic allocation contract is implemented; independently summing nonlinear mixture-EOS evaluations is not acceptable.

### 4.2 Canonical cell state

For a cell of volume `V` and family mass coordinate `xi`, define extensive moments

\[
H_k=V\int_0^\infty(\xi/\xi_{\rm ref})^k f(\xi)\,d\xi .
\]

Here `f` is number per bulk volume per unit mass. All `H_k` have count-like units because the coordinate is scaled.

| Quantity | Authority |
|---|---|
| Species amounts `n_s`, mol/cell | Existing physical phase material state. |
| Bubble count `H0=N` | Existing engaged dispersed phase number. |
| First mass moment `H1=sum(W_s n_s)/xi_ref` | Derived from authoritative species amounts. |
| `H2,H3` | New owned QBMM shape state. |
| Internal energy and thermal state | Existing native phase energy owner. |
| `p0` and mechanical pressure | Existing native flow owner. |
| Nodes, weights, sizes, area, void fraction | Derived views/caches only. |

Proposed storage is deliberately small:

~~~cpp
// Proposed, not a declaration present in the audited source.
struct QbmmShapeState
{
    double H2;
    double H3;
};

struct QbmmCellView
{
    PopulationId population;
    SpeciesInventoryView species;  // immutable alias
    double number;                 // alias of native phase_numbers
    QbmmShapeState shape;
};
~~~

The mathematical block contains four moments, even though only two new scalars are stored. Every operation must update species, number, and shape compatibly. Computing an independent `H1` field and then overwriting species to match it creates a second owner and is prohibited.

The fixed-composition gate checks `W_s n_s=Y_s M` with one declared roundoff policy. It must reject a material packet that creates an unsupported composition. This restriction applies to the optional fixed-composition QBMM family, not to the existing generic species infrastructure.

### 4.3 Exact absence

A family is absent only when its material, count, and higher moments are all zero. A material-empty family with positive count, positive shape moments with no number, or positive material with no population birth is invalid.

Absent phases have no invented temperature, density, heat capacity, radius, or epsilon inventory. Views should return an explicit absence state, not an apparently valid zero-radius bubble. A caller requiring a liquid concentration must first establish that a physical liquid volume exists.

The physical `BulkGas` role has no bubble number. If an optional bulk-gas pool is later used, its count is disengaged, not represented by a fake bubble with zero or unit number.

## 5. Pure numerical kernel

Implement the inversion and moment algebra in a mesh-independent module. It may use the project’s existing lightweight numerical types but must not access fields, MPI, solver clocks, pressure iteration, or application callbacks.

Proposed file organization, subject to repository conventions:

~~~text
src/equations/QbmmPopulationSchema.hh
src/equations/QbmmMoments.hh
src/equations/QbmmInversion.cc
src/equations/QbmmPopulationEvents.hh
src/equations/QbmmPopulationEvents.cc
src/solvers/LowMachMixtureSolver.*       # explicit owned state/transport extension
src/solvers/ReactingMixtureCoupler.*    # coupled source-stage integration
~~~

Keep physical kernels separate from host orchestration; they may later serve another admitted host without creating another inventory owner.

### 5.1 Inversion result

Return at most two atoms in the dimensionless mass coordinate, with extensive counts as weights:

~~~cpp
// Proposed value types.
struct QbmmAtom { double count; double mass_over_scale; };

struct QbmmQuadrature
{
    std::array<QbmmAtom, 2> atoms;
    unsigned active_atoms;
    InversionStatus status;
    MomentResiduals reconstruction;
    ConditioningReport conditioning;
};
~~~

An empty population returns zero active atoms. A rank-one population returns one. Do not maintain inactive nodes with small positive weights.

The algorithm must:

1. Assemble `H0` and `H1` from canonical state and check finite inputs.
2. Handle exact vacuum and rank-one consistency.
3. Test the two relevant Hankel matrices with scale-aware arithmetic.
4. Construct the symmetric two-by-two Jacobi matrix for a full-rank input.
5. Recover positive weights for active atoms and strictly positive supported mass nodes.
6. Reconstruct all four moments and check residuals against the same declared scales.
7. Return a typed failure if the physical state is not realizable.

Roundoff handling is not permission to project arbitrary data onto the moment cone. In particular, never change canonical species or count to make an inversion succeed. If rank reduction changes higher moments beyond the declared numerical policy, it is a closure approximation requiring an explicit result and error budget.

Coordinate scaling is immutable in a restart. A future change of scale is an explicit conversion `H'_k=H_k(xi_ref/xi'_ref)^k` together with schema migration, not a runtime conditioning toggle.

The Stieltjes moment tests admit support at zero mathematically. The physical gas model does not admit a positive-count, zero-mass bubble: for example, `[2,1,1,1]` represents unit weights at sizes zero and one and must fail the active-gas support gate. Reaching zero mass requires an explicit material/count disappearance event before acceptance. Inversion must not silently discard the zero atom.

### 5.2 Derived physical views

For each atom, `xi_a=xi_ref*x_a` and `n_{a,s}=w_a Y_s xi_a/W_s`. Under the baseline common-pressure fixed-composition EOS, atom volumes are consistent with the same homogeneous physical phase thermodynamics.

A proposed `QbmmGeometryView` exposes physical number density, mass, diameter moments, interfacial area density, and size extrema. It cannot publish a second phase volume. Native phase volume and total void are reconstructed from authoritative material, energy, and pressure.

Spherical-equivalent diameter is a closure for area and rate evaluation; it is not an interface mesh. For the initial thermodynamics, surface tension is omitted from the pressure/energy closure. A portable radius function supporting Laplace pressure must not be inserted into native occupancy without the separate capillary work and energy extension.

Caches must carry state generation, material revision, geometry revision, family schema, and the relevant pressure/temperature evaluation stamp. A cache built for a source predictor is not accepted-state output.

## 6. Conservative population events

### 6.1 Event structure and provenance

Reuse existing species-transfer and reaction receipts as the material foundation. Extend the source pipeline with a proposed typed `QbmmEventReceipt` containing:

- Stable population and family-schema identity.
- Event kind and immutable law/provider identity.
- Raw donor/product atom specification or an equivalent auditable constructive update.
- Gross consumed and created count, not only their difference.
- Associated raw species transfers or reaction extents.
- Changes in `H2,H3` and the implied first-moment mass consistency check.
- Explicit heat/enthalpy policy and momentum handoff where applicable.
- Rejection/limiting reason and substep provenance.

No independent QBMM callback directly mutates a field, accepted ledger, solver time, or global state. The owner accepts or rejects one coherent packet.

Material injection and population nucleation are different inputs. Added gas may grow existing bubbles or create new ones; mass alone does not determine number or birth size. Similarly, a count birth does not supply mass or latent energy. Require the application’s allocation law.

### 6.2 Positive atomic construction

For each local population substep:

1. Reconstruct an admissible temporary atomic measure from a consistent pre-material state, initially the transported snapshot.
2. Evaluate immutable local laws at the current trial thermal/material state and obtain explicit size allocation for each material-changing event.
3. Reserve all competing donor species and donor atom counts together.
4. Remove finite nonnegative donors, retain residual atoms, and add nonnegative product/birth atoms.
5. Sum the resulting four moments and full species receipts.
6. Compress the positive measure to at most two atoms.
7. Check mass, species, elements, count semantics, realizability, and thermal convention.
8. Return owned source-stage data without publication.

Coalescence and breakup preserve the first mass moment when their material is closed. Chemical reactions preserve mass and declared elements, but need not preserve total moles. Complete dissolution has a material removal and an explicit disappearance/count event; it is not performed by clipping a negative radius.

For coalescence between two identical quadrature atoms, the event law must account for pair counting and availability consistently. The continuous number representation does not require integer counts, but the integrated event still cannot consume more than the donor measure.

The constitutive coalescence and breakup laws are supplied inputs. Kernel dimensions, collision efficiency, turbulent dissipation basis, and applicability ranges must be explicit. Do not select empirical coefficients through species names or assumed equipment.

Every source stage touching a QBMM family's species must update its count and shape consistently before a downstream stage observes it. Never invert a hybrid state with newly changed species mass and old higher moments. For example, adding mass to a monodisperse `[1,1,1,1]` block cannot be represented by changing only its first moment to obtain `[1,2,1,1]`. Start from the retained pre-material measure and apply the supplied growth/birth/removal allocation.

An existing chemistry, diffusion, or interphase stage can participate only through a moment-aware adapter that links its raw accepted material receipt to this atomic update exactly once. A receipt whose material debit is already booked must not debit it again. Until that adapter exists, reject independent material-changing stages for the QBMM-bound family while allowing unrelated generic species stages. The initial coherent QBMM stage can supply its own paired transfer receipts to the existing thermal owner.

### 6.3 Relation to current count-only events

The existing native `population_events` callback is evaluated after the material endpoint and returns explicit count changes. It does not determine birth sizes, higher moments, donor-atom budgets, or material redistribution.

Retain that route for existing count-only users. A QBMM-enabled family needs the new coherent material-and-shape stage. Its count cannot also be independently changed by the legacy count-only tail. Capability validation must reject two active event authorities for the same family.

The existing `advance_multispecies_population_exchange` and `MultispeciesPopulationPacket` provide useful conservative material/count contracts. They do not by themselves make a packet moment-realizable. Reuse their validated species/number semantics through an explicit adapter rather than claiming current API already transports `H2,H3`.

## 7. Native transaction extensions

At the inspected revision, native owned records do not contain arbitrary QBMM sidecars. Extend them explicitly:

| Existing record or path | Required new coverage |
|---|---|
| `LowMachMixtureSolver::CellState` | Shape state mapped to each enabled dispersed family. |
| `TransportPhaseInventory` / `TransportSnapshot` | Transported pre-source higher moments and family-schema stamps. |
| `CellSources` | Integrated higher-moment changes tied to the material/count packet. |
| Source-stage provider and receipts | Gross atom/count/mass activity and deterministic family identity. |
| `Trial` | Candidate moment state, diagnostics, retained raw receipts, reconstruction status. |
| `Checkpoint` | Canonical shape state, schema, generations, history needed by enabled integrator. |
| Initialization/restore validation | Material/count/moment consistency and exact absence. |
| Commit and publication | Single atomic publication of all canonical quantities. |
| MPI serialization/halos | Stable family order, optional-state semantics and schema agreement. |

A separately stored map in an application callback is not transactional state. It would escape rollback, checkpoint, ownership, and source convergence and is not an acceptable initial implementation.

Preserve existing semantics: several previews of the same accepted generation may coexist; another successful or failed preview does not automatically invalidate an earlier compatible trial. Commit, initialize, restore, or ownership changes invalidate incompatible candidates according to the native contract.

Commit is a verification/publication operation. It must not call user chemistry, nucleation, EOS, quadrature-law callbacks, or mesh-target providers again. Retain the accepted candidate and its required derived validation data. If pure inversion is deliberately used as a local consistency check, it must not be a hidden physical reevaluation or change candidate state.

Use raw compensated source/stage records for exact transaction comparison. Rounded reports are never reconstructed into authoritative inputs.

## 8. Transport integration

### 8.1 First implementation

Begin with the native backward-Euler first-order upwind transport path, fixed mesh, and zero slip. For one family, apply the same conservative positive transport mapping to:

\[
n_s,\qquad H_0,\qquad H_2,\qquad H_3.
\]

The derived `H1` must follow the same mapping because it is the linear mass-weighted sum of the transported species. Coefficients, time level, phase velocity, boundary values, cell-volume treatment, and donor selection must agree.

This is a block compatibility requirement even if the implementation solves scalar systems. Independent scalar nonnegativity is insufficient. In exact arithmetic, a common nonnegative transport map forms nonnegative combinations of donor measures and preserves realizability. Independent solver tolerances, repairs, higher-order reconstructions, or source insertion can break that argument. Check the final combined state and reject an incompatible endpoint.

Species and moment density use bulk cell volume as denominator for transport: `n_s/V` and `Hk/V`. Do not multiply by phase fraction again when the packet is already an extensive inventory or bulk-volume concentration. Do not divide the number by a liquid-only volume.

Use the project’s current typed native transport construction. The implementation must select the admitted concrete field/mesh specialization before assembling transport, use its existing boundary and geometry contracts, and reuse the same accepted coefficient/operator construction for the moment block. It must not introduce the removed legacy `Mesh` facade.

The current construction seam is `FVM::TransportSystemFactory<Pack, MeshType>::assemble(terms, boundary, options, context)` in `src/FVM/TransportSystemFactory.hh`. Its aliases select `WeightedScalarTerms`, `CoefficientTransportBoundary`, `WeightedScalarAssemblyOptions`, and `WeightedScalarAssemblyContext` from `src/FVM/TransportAssemblyPacks.hh`. Terms group history, face fluxes, coefficients, sources, and constraints. Options carry discretization/correction/interpolation; context carries existing matrix, RHS, geometry, ALE, and symbolic-plan caches. These borrowed assembly inputs are synchronous views, not owned trial history.

For bulk-volume moment densities, match the admitted unit storage/advection weights and actual phase-relative volume flux used for native species/count transport. The retired free `transport_system` and `weighted_scalar_transport_system` entry points are not integration targets. Existing per-scalar repair or solver convergence is still subject to the final moment-block gate.

### 8.2 Volume, energy, and momentum consistency

The current native solver distinguishes advected material-volume packets from an EOS-reconstructed phase volume. Preserve that distinction. A law requesting physical phase concentration uses the current phase EOS volume at the prescribed source stage; old advected volume or cell volume cannot silently replace that denominator. This is distinct from rate normalization: an explicitly declared `SourceVolumeBasis::BulkVolume` remains valid and uses the bulk control volume when converting a volumetric reaction rate into an integrated extent. Honor the declared basis and convert once.

All species, number, and shape transport belong to the same phase material path. Native phase internal energy `U` is advected with that phase. Pressure-conditioned partial enthalpy accompanies the separately declared material-transfer/diffusion stages, and pressure work remains a separate native contribution. Shape moments do not carry extra latent energy or create another copy of any thermal term.

The native momentum path consumes physical mass flux. A mass flux already in kg/s must not be multiplied by density again. If phase slip is later enabled, use the owner’s phase flux and donor rules; counterflow donor fractions need not sum to one and must not be normalized.

### 8.3 Optional node-dependent slip

A size-dependent drift cannot be represented by transporting all moments with a mean diameter velocity. It requires a different closure and coupled face packet.

A future kinetic packet evaluates each upwind donor quadrature atom with its supplied relative velocity and transfers its count, all moment contributions, species, physical mass, phase internal energy, material volume for native pressure-work accounting, and required momentum together. Use pressure-conditioned partial enthalpy for separately declared material exchange; do not add it again to internal-energy advection. Node 0 in one cell has no material identity relation to node 0 in its neighbor.

For an explicit implementation, the sum of all outgoing fractions for every donor atom must not exceed one during the actual substep. Common subcycling and donor accounting belong inside the existing advance owner. A separate particle/population timestepper cannot commit outside the mixture transaction.

The current physical phase slip slot is not a per-node velocity state. Do not register changing quadrature nodes as new persistent physical phases. Define and review the new kinetic flux adapter before enabling this capability.

### 8.4 Higher order and ALE

Existing bounded scalar schemes and BDF2 are not automatically realizable moment schemes. Higher-order spatial/time transport needs its own moment-cone proof, algorithm, and tests. Keep it disabled for the first QBMM release.

For ALE, transport extensive moments with the accepted old and trial new geometry and the actual relative flux. Moment-aware snapshot, rollback, checkpoint, and geometric-conservation coverage must be present before enabling it. For a conservative remap, a common nonnegative overlap map preserves a nonnegative measure; averaging radii or independently interpolating nodes does not.

The native ALE candidate is private and restores accepted geometry after preview. QBMM cannot retain views into discarded trial geometry or publish a new mesh before the native `commit_ale` succeeds.

## 9. Coupled source and pressure iteration

Integrate within the existing `preview_with_sources` and `ReactingMixtureCoupler` orchestration. The source-stage provider is an owner-controlled collective stage. User constitutive laws remain deterministic, immutable, and MPI-free.

A proposed native QBMM trial follows this schedule:

1. Validate owner lease, immutable schema, accepted generation, mesh and enabled capabilities collectively.
2. Form the current flow/thermal trial and perform native material, number, and higher-moment transport from the accepted state.
3. Capture an owned transported snapshot before sources and reconstruct each family's consistent pre-material atomic state.
4. Evaluate local chemistry and explicit transport/exchange stages in the documented native order. Any stage touching a QBMM family must apply a paired material/count/shape update before exposing its endpoint; otherwise reject that configuration.
5. Execute additional coherent QBMM events at their declared position, carrying forward the consistent temporary measure and consuming each raw linked material receipt once.
6. Assemble one integrated source packet and retained gross event record.
7. Recompute phase thermodynamics, volume closure, pressure correction, energy, and momentum through the existing owner.
8. Repeat the shared corrector until native physical residuals and new moment/source residuals pass.
9. Produce a private valid trial; publish only through commit.

Do not add a source-first outer integrator around the native transport-first corrector. Do not call population laws inside pressure root evaluations. Pressure roots evaluate retained source-conditioned thermal/material data according to the existing contract.

Current source convergence requires at least two source evaluations and includes gross reaction/transfer activity. Extend that rule to gross births, deaths, topology activity, transferred mass, and nonnegative shape-production/consumption magnitudes. Put signed `Delta H2` and `Delta H3` in the new moment source channel with its own convergence residual. The existing `SourceStageResult::gross_events` remains a nonnegative fixed-layout channel; it must not receive signed increments. A birth and death of equal count can have zero net number and still require convergence. Likewise, net-zero species transfer can conceal significant exchange and heating.

An intermediate thermodynamic predictor may temporarily overfill its reference cell. It is a trial construction device, not permission to accept an overfilled endpoint. Final pressure/volume, species, energy, population, and realizability gates must all pass independently of Krylov convergence.

## 10. Thermal and constitutive interfaces

Use the supplied `MixtureThermodynamics` and its declared `EnergyConvention`. Preserve the distinction between formation energy represented in caloric properties and explicitly supplied reaction heat. Count each exactly once.

Interphase mass transfer carries the material’s appropriate pressure-conditioned partial enthalpy through the current thermal contract. Native flow owns pressure work. Supplied heat input remains an independent source; a chemistry law driven by that input does not automatically create an additional copy of the same energy.

Initial QBMM uses the current common-temperature path. Existing separate-phase thermal options may remain available to non-QBMM users, but QBMM support for them is a separate acceptance gate. There is initially no node-temperature distribution.

The property/law context should expose explicitly:

| Input | Basis |
|---|---|
| Species amounts | Extensive mol/cell; immutable view. |
| Local rate concentrations | Named physical phase EOS volume, mol/m³ of that phase. |
| Bubble quadrature | Extensive weights and kg/bubble nodes, clearly distinguished from densities. |
| Thermodynamic state | Trial absolute pressure, temperature, composition, and presence. |
| Transport properties | Supplied law/EOS values with SI units and validity range. |
| External forcing | Explicit field/value and unit basis; no implied yield conversion. |
| Time interval | Actual integrated substep, not an assumed unit time. |

The application chooses dissolution, boiling, nucleation, coalescence, breakup, and any future slip law. A boiling law must explicitly link donor material, vapor composition, latent/caloric convention, and bubble allocation. A dissolved species compartment shares the physical solution phase with liquid; it does not create another heat capacity or occupancy.

Natural convection also needs explicit gravity/body-force integration. The inspected native momentum path accepts an integrated cell impulse through `CellSources::momentum`; changing EOS density alone does not activate gravity. Bind the supplied acceleration/force to that admitted source path with a consistent mass and time basis, pressure-reference convention, and source-corrector treatment. Add hydrostatic-balance and heated-circulation cases before claiming a natural-convection application. A separate weak Boussinesq implementation is not evidence that this native configuration has supplied that force.

## 11. Optional bulk-gas pool and high void

An optional unresolved pool may map to the existing generic resolved-gas compartment and native `BulkGas` physical role. The compartment name `Resolved` does not force interface resolution; the selected host and closure determine geometry.

This is an extension to physical modeling, not a mandatory QMOM operation. Define conversion only after a supplied constitutive law, thermodynamic mapping, and qualification case exist. No universal void threshold, pool rate, area law, or return distribution is specified by this design.

A dispersed-to-pool packet transfers the exact species and energy according to the shared owner, removes matching population atoms/moments, and retains count-death provenance. A pool-to-dispersed packet supplies an explicit nonnegative birth distribution with matching material. Converting state must not double-count phase volume or silently destroy bubble surface/kinetic energy if such energies are later modeled.

A nearly gas-filled cell still requires an admitted pressure/thermal/momentum closure and valid material disposition. Nonvolatile material cannot disappear because the liquid volume tends to zero. Use supported physical transport/phase transfer, add a separately designed receiving phase, or reject the unsupported state.

Do not infer venting, connectivity to a headspace, breakup at a free surface, or topological gas cavities from a local scalar void threshold. The current closed native domain does not acquire open-boundary support through a population option.

## 12. Public configuration and reporting

Provide a typed configuration under the existing reacting options. A possible schema is:

~~~text
qbmm:
  enabled: true
  families:
    - population: <existing dispersed identity>
      physicalPhase: <existing compatible binding>
      coordinate: mass
      massScaleKg: <positive supplied conditioning scale>
      composition: <explicit fixed mass fractions by species ID>
      closure: qmom_two_node
  transport: native_upwind_backward_euler
  thermal: common_temperature
  capillarity: disabled
  nodeSlip: disabled
  populationLaws: <explicit immutable provider>
~~~

This is an illustrative contract, not a parser presently implemented in the repository. No numerical scale, material property, source strength, void switch, or closure coefficient is prescribed.

Validation must reject unknown species, mismatched fingerprints, unsupported phase binding, duplicate owners, nonnormalized composition, invalid scale, unsupported transport or energy mode, and conflicting legacy number callbacks.

Expose canonical state and derived diagnostics separately:

- Accepted species inventories, count, `H2,H3`, and derived `H1`.
- Moment densities with explicit bulk-volume units.
- Inversion rank, condition indicators, reconstruction residuals, and failure status.
- Gas volume fraction from native EOS, not from a second independent field.
- Number- and area-related size statistics with absence semantics.
- Raw accepted population receipts and gross source convergence measures.
- Physical inventory/energy/element balance defects and source/transport provenance.

A reporting helper such as `LiquidMassInventory::initializeFromSpecies` remains a follower. Do not turn it into another mass transport equation. Output should distinguish “closure evaluated” from “closure validated for this regime.”

## 13. MPI, restart, and deterministic behavior

Use stable ordered family identities across all ranks. Include the catalog and family fingerprints in serialization and construction preflight. An empty rank must participate in the same scheduled collective validation and failure fences.

Local inversion and constitutive laws do not perform collectives. The owner converts local failures into collective outcomes at agreed barriers. Throwing on one rank before the matching global schedule is entered can deadlock the run and must be prevented by the established validation helpers.

Checkpoints store canonical material, energy, number, and shape state, not cached radii or quadrature alone. They also store model schema, scale, event-law identity/revision, and sufficient time-integration history for the actually enabled method.

A count-only old checkpoint does not contain a unique size distribution. Require one explicit migration choice: an externally specified distribution matching mass/count, or a declared monodisperse closure with `H2=H1²/H0` and `H3=H1³/H0²`. Mark that as initialization of missing information. Never imply the original polydispersity has been recovered.

Restoring an old molar-coordinate population is exact only for its declared fixed composition. Transform with its actual mixture molar mass, update the schema, and check all moments. Variable-composition data require an explicit richer conversion or rejection.

## 14. Acceptance and rejection rules

Use physical gates in addition to algebraic solver success:

| Gate | Required behavior |
|---|---|
| Material conservation | Raw receipts balance species transfers; reactions satisfy declared mass/element rules. |
| Donor budgets | Combined processes cannot consume unavailable original donors. |
| Fixed composition | Every family matches its immutable composition within stated arithmetic policy. |
| Population presence | Exact absence and explicit birth/disappearance semantics. |
| Realizability | All accepted family moment blocks reconstruct a supported nonnegative measure. |
| Volume/pressure | Native physical occupancy and local/global continuity criteria. |
| Energy | Existing caloric/formation convention and thermal residual gates. |
| Source convergence | Net and gross QBMM measures converge within the shared corrector. |
| Transaction identity | Schema, generation, owner, geometry, and retained packet match. |
| Capability | Selected geometry, thermal, transport, slip, pool, and capillarity modes are admitted. |

Reject a candidate without publishing partial species, moment, history, field, or mesh changes. A timestep reduction can address temporal donor/closure errors; it cannot make an unsupported physical model valid.

Do not repair a rejected state by truncating void, renormalizing counterflow fractions, deleting unsupported species, seeding epsilon bubbles, silently changing composition, or relaxing continuity to hide cancellation. The current source’s small-volume-rate cancellation limit remains a numerical limitation to diagnose, not a reason to weaken the new physical gates.

## 15. Verification plan and completion criteria

These are implementation acceptance tests, not claims that the source has already passed them. The supplied arithmetic oracle script checks only standalone design identities.

| Test group | Meaningful acceptance evidence |
|---|---|
| Inversion | Exact vacuum, rank one, two-node reconstruction, badly conditioned and invalid Hankel inputs. |
| Event algebra | Birth, growth, coalescence, breakup and dissolution with independently computed mass/count/moment results. |
| Chemistry | A synthetic mole-changing mass-conserving reaction; no false total-mole invariant. |
| Composition | Fixed-composition events accepted; selective inconsistent transfer rejected. |
| Shared transport | Positive common mapping preserves all moments/species; deliberately mismatched mapping fails the block gate. |
| Native transport | Closed moving population, discrete conservation, phase-number alignment, and final moment realizability. |
| Thermal coupling | Input/reaction/enthalpy/pressure-work accounting, including no double heat source. |
| Gravity and convection | Explicit body-force impulse, hydrostatic balance and heated circulation when a natural-convection application is claimed. |
| Corrector | Gross activity with zero net source cannot falsely converge after one evaluation. |
| Transactions | Failed preview rollback, compatible coexisting previews, stale commit rejection, no callback at commit. |
| Restart | Exact round-trip canonical state; schema/scale mismatch rejected; explicit old-state migration. |
| MPI | One and multiple ranks, empty ranks, one-rank inversion failure without deadlock. |
| Optional features | Separate suites for phase slip, node slip, ALE/remap, pool conversion and capillarity before enablement. |

Mesh and timestep convergence are required for application observables. Also perform a population-closure convergence study: compare two-node results with a finer validated size-distribution reference or higher-order closure. A spatially converged four-moment result can still have a size-distribution closure error.

Qualification requires a suitable experiment or independently validated benchmark for the chosen growth, coalescence, breakup, slip, and high-void closures. Algebraic conservation and a passing unit suite do not validate an empirical bubble model.

## 16. Proposed implementation phases

Phase labels are local to this revised design and do not assert completion in any project TODO.

| Phase | Work | Exit condition |
|---|---|---|
| Q0 — contract and source alignment | Freeze supported branch, schema, state ownership, scopes and callback position | Reviewer agrees exact current/proposed boundary. |
| Q1 — pure moments | Scale-aware inversion, atomic event algebra and independent oracles | Realizable/invalid/rank cases behave as specified. |
| Q2 — owned canonical state | Extend state, snapshot, source, trial, checkpoint, serialization and restore | No material/count/shape split survives rollback or restart. |
| Q3 — conservative local sources | Coupled material/count/shape receipts and finite donors | Closed source/energy balances and gross-event convergence pass. |
| Q4 — native transport | Common upwind backward-Euler block, fixed mesh and zero slip | Conservation and realizability pass native and MPI cases. |
| Q5 — coupled release gate | Shared native corrector, reporting and failure handling | End-to-end baseline passes documented capability matrix. |
| Q6 — supplied phase slip and applications | Qualified phase slip, law interfaces and application benchmarks | Model validity and sensitivity recorded. |
| Q7 — optional size slip/ALE/order | Add each with its own coupled flux or remap contract | Dedicated acceptance evidence, no implied blanket support. |
| Q8 — optional dense-gas/composition/capillarity | Separate physical extensions | New state/energy/closure contracts and validation approved. |

Do not recreate `LowMachMixtureSolver` as Q0. It is an existing integration point. Do not list `ReactingVoFCoupler` restoration as a prerequisite for Q1–Q5.

## 17. Migration from revision 0.1

| Earlier assumption | Revision 0.2 action |
|---|---|
| Application-specific gas production in core | External generic reaction/transfer/source provider. |
| A selected molecular species defines bubble content | Explicit arbitrary fixed-composition family and catalog. |
| Molar bubble coordinate generalized without qualification | Mass coordinate, or explicit fixed-composition coordinate migration. |
| Four independently owned moments plus gas fields | Existing species/count authority; only two new shape scalars. |
| Future mixture backend must be built | Extend current native low-Mach owner and controller. |
| Source-first split around flow | Integrate into current transport-first shared corrector. |
| Count callback can provide the whole QBMM update | Add an owned material/count/shape packet and transaction extension. |
| Radius EOS with Laplace pressure inserted into void | Common-pressure baseline; capillary work/energy deferred. |
| A preset void switch extends validity to gas cavities | Optional qualified pool model; no universal threshold. |
| Ordinary bounded scalar transport ensures moment validity | Common positive block first; specialized higher order later. |
| New preview automatically invalidates all older candidates | Preserve current generation/lease transaction semantics. |
| Fixed count suffices for checkpoint migration | Explicit distribution reconstruction assumption required. |

The original local phase labels, default coefficients, and application-specific terminology are superseded where they conflict with this document. Existing generic source functionality remains available outside this optional QBMM mode.

## 18. Source and literature references

All source links are pinned to the inspected `develop` commit:

1. [README and current branch scope](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/README.md).
2. [Generic reacting species contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/reacting_species_contract.md).
3. [Native low-Mach reacting-flow contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/low_mach_reacting_flow.md).
4. [External host and weak adapter contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/external_host_reacting_flow.md).
5. [LowMachMixtureSolver state and transaction API](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/solvers/LowMachMixtureSolver.hh).
6. [ReactingMixtureCoupler API](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/solvers/ReactingMixtureCoupler.hh).
7. [Population kinetics API](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/equations/BubblePopulationKinetics.hh).
8. [Pressure-conditioned thermal contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/equations/PressureConditionedThermal.hh).
9. [Solver build scope](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/solvers/CMakeLists.txt).
10. Heylmun, Kong, Passalacqua and Fox (2019), “A quadrature-based moment method for polydisperse bubbly flows,” *Computer Physics Communications* 244, 187–204. [DOI: 10.1016/j.cpc.2019.06.005](https://doi.org/10.1016/j.cpc.2019.06.005). Supports the population-moment framework and illustrates that dense bubbly flow needs additional physical closures.
11. Passalacqua, Laurent and Fox (2020), “A second-order realizable scheme for moment advection on unstructured grids,” *Computer Physics Communications* 248, 106993. [DOI: 10.1016/j.cpc.2019.106993](https://doi.org/10.1016/j.cpc.2019.106993). Supports the need for a specifically realizable higher-order transport method.
12. Bryngelson, Colonius and Fox (2020), “QBMMlib: A library of quadrature-based moment methods,” *SoftwareX* 12, 100615. [DOI: 10.1016/j.softx.2020.100615](https://doi.org/10.1016/j.softx.2020.100615). Background for moment inversion and quadrature closure.
13. [Typed transport system factory](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/FVM/TransportSystemFactory.hh) and [assembly packs](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/FVM/TransportAssemblyPacks.hh).
14. [Native momentum and explicit source integration](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/solvers/LowMachMixtureSolver.cc).

The architecture above is a proposed project-specific integration derived from the inspected source contracts. It is not attributed to a paper as an already validated implementation.

