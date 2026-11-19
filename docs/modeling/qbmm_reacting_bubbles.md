# QBMM-compatible reacting bubble model

## Modeling specification for volume-averaged gas–liquid flow

| Item | Value |
| --- | --- |
| Document ID | SF-QBMM-MODEL-001 |
| Revision | 0.2 |
| Date | 2026-11-19 |
| Status | Proposed QBMM extension to the existing generic reacting-flow implementation |
| Source baseline | SimpleFluid `develop`, commit `0029a93fd5fcb83658daf82dc80e9cd72a490031` |
| Companion | [Code design](../architecture/qbmm_reacting_bubbles.md) |
| Supersedes | The application-specific revision 0.1 modeling specification |

**Decision.** Add a two-node, four-moment QMOM population representation to the existing species and low-Mach flow owners. Use bubble mass as the internal coordinate and initially support pure-species or immutable-composition population families. Preserve the existing authoritative species amounts and population counts. Store only the two additional higher moments. Reconstruct gas volume from the supplied thermodynamics and interfacial area from quadrature; do not reconstruct bubble interfaces.

Throughout this document, **[C]** identifies an observed current repository contract, **[P]** a proposed requirement, **[D]** a mathematical consequence, and **[R]** a published method. A proposed equation or test is not an implementation or validation claim. Repository statements refer to the pinned baseline above; the numbered source list identifies the inspected documents.

## 1. Scope and capability boundaries

### 1.1 Physical purpose

The model describes a liquid containing a statistically represented gas population. Applications supply a chemical mechanism, injection or withdrawal, interphase transfer, heat input, and any population-process laws. Suitable uses include gas-producing solution chemistry, prescribed gas injection and dissolution, and prescribed heating with explicitly bound solvent/vapor transfer. Heating and gas production are independent inputs: an energy source does not specify a material source.

The principal outputs are conservative species inventories, phase masses and occupied volumes, temperature, flow, bubble number, and area or size statistics. A quadrature node is an integration point representing many bubbles; it is not a resolved bubble or a tracked cavity. Four moments describe the distribution only through selected integrals. They do not determine its complete shape, tails, gas connectivity, or size–composition correlation. [D, R1–R3]

Three decisions remain separate: the population representation, the flow/thermodynamic backend, and the physical closures. Selecting QMOM changes the first. It does not add a gas momentum equation, establish a high-void drag law, or supply reaction and phase-change kinetics.

### 1.2 Existing foundation and proposed increment

**[C]** `Reacting::SpeciesCatalog`, `ReactionMechanism`, and `CellSpeciesState` already provide supplied chemistry, finite extensive species amounts, and named population counts. `MixtureThermodynamics` supplies phase properties. `LowMachMixtureSolver` and `ReactingMixtureCoupler` already own native conservative species, phase internal energy, momentum, pressure/volume closure, and transport-first source coupling. The initial QMOM path extends this flow core. It does not replace it. [C1–C4]

| Capability | Pinned baseline | Revision 0.2 decision |
| --- | --- | --- |
| Generic species catalogs, mechanisms, finite material donors | Existing | Reuse without special species names |
| Common-pressure native low-Mach flow and phase energy | Existing, with documented backend restrictions | Initial integration owner |
| Explicit native population-number events | Existing | Add compatible higher-moment events |
| Two-node mass-coordinate QMOM | Absent | Proposed initial population backend |
| Pure-species or fixed-composition families | Generic inventories can represent them | Required restriction for initial QMOM |
| Arbitrary evolving bubble composition | Generic species model exists; this is not a size/composition QBMM closure | Keep existing path; reject incompatible initial-QMOM operations |
| Common carrier velocity, zero relative slip | Available flow configuration | Initial QMOM transport configuration |
| Size-dependent slip and drag | Not supplied by this proposal's first increment | Separate transport/momentum/energy qualification |
| Unresolved bulk-gas pool | Inventory/physical-phase binding is possible | Optional extension, separately gated |
| Node Laplace pressure, surface energy, bubble thermal nonequilibrium | Not the current native common-pressure closure | Deferred coupled extension |
| VOF, interface reconstruction, bubble-shape resolution | Outside supported reacting-flow scope | Not required or enabled |
| Higher-order moment transport, general AMR, arbitrary ALE | Not implied by existing lower-level utilities | Independent extensions |

**[P]** The first deliverable uses the native closed, supported mesh/flow configuration, common temperature, and zero slip. The existing generic separate-temperature capability requires independent qualification for this QMOM backend. The existing Boussinesq/reference-volume adapter remains a separate approximation; neither its population callback nor its boiling coverage qualifies this native QMOM integration. [C2–C5]

## 2. State, composition, and authority

### 2.1 Extensive state and dimensions

Let a cell have geometric volume \(V_c\). A named population family \(b\) has species amounts \(n_{bs}\) in mol per cell, molar masses \(W_s\) in kg/mol, and count \(N_b\). Count is an expected number and need not be an integer. Its authoritative mass is

\[
M_b=\sum_s W_s n_{bs}.
\]

Choose a finite positive, immutable mass scale \(x_*\), in kg per bubble, and let \(x>0\) denote physical bubble mass. Define the cell-extensive number measure \(d\mathcal N_b(x)\) and scaled moments

\[
h_{bk}=\int_0^\infty (x/x_*)^k\,d\mathcal N_b(x),\qquad k=0,1,2,3.
\]

All \(h_{bk}\) have count units; scaling prevents high powers of very small SI bubble masses from undermining conditioning. The corresponding bulk-volume moment is

\[
m_{bk}=\frac{x_*^k h_{bk}}{V_c},\qquad [m_{bk}]=\mathrm{kg}^k/\mathrm{m}^3.
\]

**[P] There is one material and number owner.**

| Quantity | Authority |
| --- | --- |
| Species \(n_{bs}\) | Existing canonical species state |
| Count \(h_{b0}=N_b\) | Existing canonical population number |
| First moment \(h_{b1}=M_b/x_*\) | Derived from canonical species; never separately advanced |
| Higher moments \(h_{b2},h_{b3}\) | New canonical extensive state |
| Quadrature nodes/weights, area, diameters | Reconstructed caches |
| Phase volume, temperature, internal energy | Existing thermodynamic/flow ownership; volume and temperature recovered according to that contract |

Do not independently store and transport a second authoritative bubble-mass moment. A predicted first-moment increment is a consistency receipt; the species inventory determines the accepted value. No post-step overwrite can repair an inconsistent material/moment transaction.

**[C]** Liquid and Dissolved are chemical compartments sharing one physical solution phase and EOS. They are not two additive copies of liquid mass. The native dispersed count is extensive; its transport concentration uses \(N_b/V_c\). Adding an extra void-fraction multiplier to a concentration already expressed per bulk cell double counts occupancy. [C1, C2]

### 2.2 Initial fixed-composition family

**[P]** Each family declares immutable mass fractions \(Y_{bs}\ge0\), with \(\sum_sY_{bs}=1\), and enforces

\[
W_s n_{bs}=Y_{bs}M_b.
\]

A node of mass \(x\) contains \(Y_{bs}x/W_s\) moles of species \(s\). Initial conditions, inflow, source events, remap, and restart must satisfy the same composition. Pure gas is the special case with one nonzero fraction. Composition tolerances address declared floating-point error; they do not authorize changing species to fit the family.

Why this restriction matters: two equally massive bubbles, one pure A and one pure B, can have the same size moments and cell-total species amounts as two mixed A/B bubbles. Selective dissolution distinguishes these populations. Assigning the cell-average composition to every quadrature node would impose gas mixing across the population, not recover missing information. [D]

Therefore the initial backend rejects selective transfer, gas reactions changing \(Y_b\), or cross-family coalescence with incompatible compositions. Liquid reactions may produce a compatible gas packet. Different gas families may coexist with explicit compatible physical-phase bindings, but combining their material into a new composition requires a defined receiving representation.

Use one family per compatible dispersed physical-phase binding initially. Multiple families sharing one physical EOS phase need an explicit thermodynamic allocation; independent nonlinear mixture EOS evaluations must not be summed as if they were distinct physical phases. A future conditional/multivariate composition model, or an explicitly declared common-composition approximation, requires its own state, equations, and tests. Generic variable-composition operation remains available through the existing non-QMOM path.

### 2.3 Migration and emptiness

**[P]** A fixed-pivot population converts by summing its actual extensive atoms; a lumped population converts as \(h_k=N(M/(Nx_*))^k\). For fixed composition, a molar coordinate converts exactly using \(W_{\rm mix}^{-1}=\sum_sY_s/W_s\) and \(m_k^{\rm mass}=W_{\rm mix}^k m_k^{\rm mol}\). Never relabel old molar pivots as masses.

An absent family has zero species, count, and both higher moments. Positive mass requires positive count and positive reconstructed masses. A state with positive count and zero mass is not a population of physical bubbles. Tiny positive populations retain their actual inventory; no seed amount, minimum count, or numerical void floor is introduced. Complete disappearance is an explicit material and number event. A disappearing physical phase must also satisfy the existing zero-energy and zero-volume contract. [C1, C4]

## 3. Two-node QMOM and realizability

### 3.1 Representation

**[P, R1–R3]** Reconstruct up to two positive atoms:

\[
h_k=\sum_{a=1}^{n_q}N_a s_a^k,\qquad s_a=x_a/x_*,\quad N_a>0,\quad s_a>0,\quad 0\le n_q\le2.
\]

The method evolves moments and inverts them to quadrature: it is QMOM. DQMOM directly evolves quadrature variables and has a different numerical contract. [R4] Node ordering is a reproducible cache convention, not persistent material identity.

Nonnegative individual moments do not establish realizability. Inspect the Hankel matrices

\[
H_0=\begin{pmatrix}h_0&h_1\\h_1&h_2\end{pmatrix},\qquad
H_1=\begin{pmatrix}h_1&h_2\\h_2&h_3\end{pmatrix}.
\]

For full-rank \(H_0\), a positive semidefinite \(H_1\) is the nonnegative-support condition; strictly positive retained nodes are then checked explicitly. Singular cases require rank/range consistency. Vacuum requires all four moments zero. A rank-one population with \(h_0>0\) requires

\[
h_2=h_1^2/h_0,\qquad h_3=h_1^3/h_0^2,
\]

and has one node \(s=h_1/h_0>0\). A mathematical zero-size atom belongs to the disappearance boundary; the inverter must not silently drop its count. [D]

### 3.2 Full-rank inversion

For a well-conditioned rank-two state, form

\[
a_0=\frac{h_1}{h_0},\quad b_1=\frac{h_2}{h_0}-a_0^2,\quad
D=h_2-2a_0h_1+a_0^2h_0,
\]

\[
a_1=\frac{h_3-2a_0h_2+a_0^2h_1}{D},\qquad
J=\begin{pmatrix}a_0&\sqrt{b_1}\\\sqrt{b_1}&a_1\end{pmatrix}.
\]

The eigenvalues of the symmetric Jacobi matrix \(J\) are \(s_a\). For normalized eigenvectors \(v_a\), \(N_a=h_0(v_{a,1})^2\). Verify reconstruction of all four input moments, positive support, and species-derived first mass after inversion. [D]

**[P]** Use scaled, stable arithmetic with explicit conditioning and rank classification. Residual scales must follow the actual moment terms and configured coordinate scale. An absolute floor of one in unscaled SI moment units can hide the entire population and is prohibited. Near-degenerate states require a documented bounded arithmetic policy or rejection/retry; independently clipping determinants or moments is not a physical repair. Rank reduction may not silently alter authoritative count or mass.

Quadrature exactly reproduces the retained moments within the arithmetic policy. Its estimate of area, nonlinear transfer, or a distribution tail is approximate. Increasing spatial resolution does not remove this population closure error.

## 4. Reactions, transfer, and population events

### 4.1 Chemical conservation and supplied laws

**[C]** For integrated reaction extent \(\Delta\zeta_r\), the mechanism supplies

\[
\Delta n_s=\sum_r\nu_{sr}\Delta\zeta_r.
\]

A closed reaction conserves mass, \(\sum_sW_s\nu_{sr}=0\), and each catalogued elemental inventory. Total moles generally do not remain constant. Reaction rates carry an explicit volume basis; a liquid-based rate integrates over actual liquid volume, while an integrated extent receives neither another volume factor nor another time factor. Competing events share one donor ledger. [C1, C3]

The synthetic reaction \(2A\rightarrow B\), with \(W_A=1\) and \(W_B=2\) in consistent units, conserves mass while reducing total moles by one per reaction extent. At fixed ideal-gas temperature and pressure its gas volume changes. This demonstrates why a general reactive model cannot use total-mole conservation as its material invariant. It also illustrates a composition-changing operation outside the initial fixed-composition QMOM family. [D]

No empirical production yield, nucleation size, breakup kernel, or Henry coefficient is a default. Each enabled law declares its identity/revision, units, state inputs, applicable domain, and reference or calibration. External injection/withdrawal identifies its reservoir and material, energy, and momentum packets. Producing dissolved material does not also produce free bubbles.

### 4.2 Event representation

**[P]** Every population event specifies its material change, number change, higher-moment change, and applicable energy/momentum receipt. An amount of gas alone cannot determine whether it creates bubbles or grows existing bubbles. Existing native material stages and explicit number events remain the owners; the QBMM extension appends matching moment information to the same accepted transaction.

Reconstruct the working atoms from the **consistent transported, pre-material state**, before a chemistry, face-transfer, or interphase stage changes family species. Each family-touching material stage must carry a matching allocation to existing nodes, births, removals, or products. Replay that accepted material receipt once into the private working measure, then obtain its new moments. The material owner has already booked the species debit/credit; shape replay must not apply that material a second time. Native number publication may remain at its existing final stage, but the private staged measure must retain the logically paired count and moments and expose only consistent endpoints to subsequent population laws.

Never invert a changed species-derived \(h_1\) alongside old count and higher moments. For example, adding mass to \([1,1,1,1]\) without updating shape can produce the impossible vector \([1,2,1,1]\). Independent generic material callbacks that touch a QMOM family without a compatible shape event must reject. This requirement also applies to any selective face-species stage; generic source support alone does not qualify a QMOM combination.

For an event adding \(B\) bubbles at mass \(x_b\),

\[
\Delta h_k=B(x_b/x_*)^k,\qquad
\Delta n_s=BY_sx_b/W_s.
\]

The source reservoir loses the same material for internal transfer. A distribution of births is a sum or integral of such positive packets. Both \(B\) and the supplied distribution must be explicit. Existing generic chemistry does not manufacture a bubble number from material production.

For continuous growth \(g(x)=dx/dt\), a bulk number density \(f(x)\) satisfies

\[
\partial_t f+\nabla\!\cdot(\mathbf u_b f)+\partial_x(gf)=\mathcal B+\mathcal C+\mathcal F.
\]

With \(m_k=\int x^k f\,dx\), its interior growth term is

\[
S_k^{\rm grow}=k\int x^{k-1}g(x)f(x)\,dx,
\]

plus the support-boundary term from integration by parts. Zeroth moment stays constant during growth only before complete disappearance. A bubble reaching zero mass is removed through an explicit disappearance event; retaining its number indefinitely or deleting it at an arbitrary positive mass changes the model. [D]

In the fixed-composition mode, every node transfer must satisfy \(\dot n_s=Y_sg/W_s\). A pure species can dissolve or grow using a supplied equilibrium and film law. A multicomponent selective film law normally violates the initial closure and must reject rather than be projected onto fixed fractions. Concentrations use actual liquid volume; fugacity or ideal partial pressure follows the selected material law. An empty liquid phase has no liquid-based reactive or transfer capacity. [C3]

### 4.3 Coalescence and breakup

For \(E\) coalescence events combining masses \(x\) and \(y\),

\[
\Delta h_k=E\left[\left(\frac{x+y}{x_*}\right)^k-\left(\frac{x}{x_*}\right)^k-\left(\frac{y}{x_*}\right)^k\right].
\]

Thus \(\Delta h_0=-E\) and \(\Delta h_1=0\). A same-node pair consumes two parent bubbles per event. Count donor reservation must include this multiplicity. The event conserves each species only when the parent/product composition mapping is explicitly compatible.

For breakup of one parent mass \(x\) into a daughter measure with multiplicities \(b_j\) and masses \(x_j\),

\[
\Delta h_k=E\left[\sum_jb_j(x_j/x_*)^k-(x/x_*)^k\right],\qquad
\sum_jb_jx_j=x.
\]

The mass moment is conserved; bubble number need not be. Neither event conserves area. Any surface-energy or dissipation model is an additional physical choice, not an automatic consequence of the mass balance. The initial native zero-surface-tension approximation does not claim a capillary energy budget. [D, C4]

### 4.4 Constructive source integration

**[P]** An initial realizable local scheme operates on a temporary nonnegative atomic measure: reserve all original outgoing donors, retain nonnegative remainders, add accepted products, form the four moments, then compress by QMOM. Growth moves atoms with explicitly paired species transfer. Births add atoms. Coalescence and breakup remove and add their budgeted atoms.

Subcycling or a bounded integration method is permitted, but actual finite donors must still be checked. Simultaneous incoming material or newly born number cannot finance gross outgoing requests in the same reservation. Ordered stages may use their actual predecessor endpoints, consistent with the existing controller. A failed event rejects or recomputes the trial; it is not clipped afterward to fit an inventory, a moment cone, or a void limit. [C2–C4]

Atomic moment compression preserves the retained moments; it need not preserve an unretained nonlinear rate integral. Integration and compression error therefore require separate convergence checks. Existing reaction machinery is not a general stiff chemistry solver, and choosing QMOM does not add one.

## 5. EOS, area, energy, and phase change

### 5.1 Common-pressure node geometry

**[C, P]** The initial native path uses common absolute phase pressure and the already supplied phase EOS, with zero surface tension. Node radii do not acquire independent Laplace pressures. The flow solver obtains phase volume from species, energy, and pressure; the population module consumes that thermodynamic state. [C1, C2, C4]

For fixed composition and temperature under an extensive common-pressure EOS, let \(v_b(p,T,Y_b)\) be gas specific volume. Then

\[
v_a=x_a v_b,\quad V_b=\sum_aN_av_a=M_bv_b,\quad
\alpha_b=V_b/V_c.
\]

For an ideal mixture,

\[
V_b=\frac{RT}{p}\sum_sn_{bs}.
\]

Consequently gas volume and mixture material density depend on authoritative inventory and thermodynamics, not on higher size moments in this initial closure. The extra moments affect area, transfer rates, and size statistics. Compression or thermal expansion changes node volume without changing its mass coordinate or introducing a fictitious moment source. [D]

Equivalent-spherical geometry supplies

\[
r_a=\left(\frac{3v_a}{4\pi}\right)^{1/3},\quad
A_b=\sum_aN_a4\pi r_a^2,\quad
a_b=A_b/V_c,\quad d_{32}=6V_b/A_b.
\]

This is a closure for area, not interface reconstruction. At high void it may cease to represent actual interfacial area. No assertion about connectivity follows from a large equivalent radius or a quadrature node.

A future \(p_a=p_l+2\sigma/r_a\) model changes EOS volume, pressure derivatives, energy/work, and source coupling. It requires a coherent caloric and capillary extension; adding that formula only inside the radius evaluator is incompatible with the present native pressure model.

### 5.2 Phase energy and thermal ownership

**[C]** Native phase storage is extensive internal energy. Common temperature is the default; separate temperatures require explicit phase-energy allocation. The supplied caloric reference and `EnergyConvention` determine reaction heat. With formation energy included, composition changes must not receive the same heat of reaction again. With sensible storage, the excluded chemical energy has an explicitly supplied owner. Heat used to evaluate a rate is not automatically another thermal source. [C1–C4]

Native interphase stages freeze their material amounts, evaluate true partial-molar enthalpies at pressure-conditioned caloric anchors, and debit/credit the same transferred enthalpy. The solver alone applies

\[
-p\left(V_{\rm final}-V_{\rm adv}\right),
\]

where \(V_{\rm adv}\) is the actual transported physical phase-volume packet. It is neither a donor concentration volume nor the geometric cell volume. Pressure roots may evaluate pure constitutive laws but may not rerun kinetics or number/moment events. A phase that disappears must lose its actual energy consistently; the remaining energy cannot be discarded because its volume is small. [C2, C4]

**[P]** QBMM carries no independent node temperature or node energy in the initial mode. Its material and moment receipts participate in the existing phase-energy transaction. Pure-vapor birth, evaporation, and condensation may be represented only through a supplied compatible solvent/vapor binding and energy law. Vapor and other gas products cannot be silently mixed into one fixed-composition family. No second standalone boiling source may debit the same solvent or add the same latent heat. [C5]

The thermal contract is not a general total-energy claim. Relative kinetic energy, drag heating, interface/surface energy, and unresolved breakup/coalescence dissipation remain excluded or explicitly budgeted by separately qualified extensions.

## 6. Flow, high void, and gas-only cells

### 6.1 Initial velocity and material-volume closure

**[C]** Native velocity is a shared continuous/carrier velocity, with phase drift defined relative to it; it is not automatically a mass-averaged velocity. **[P]** Initial QMOM sets relative slip to zero and transports all family quantities with the existing compatible phase flux. Natural convection requires explicitly supplied gravity/body-force integration through the admitted momentum-source path; the inspected native force assembly does not activate gravity merely because the EOS density changes. Actual mixture mass supplies the material weighting once that force is provided. This approximation transports bubbles with the carrier; it does not predict their terminal rise relative to liquid. [C2, C7]

For a later drift extension, write \(\mathbf u_j=\mathbf u_c+\mathbf w_j\) and \(q_j=M_j/V_c\). The barycentric velocity is

\[
\mathbf u_m=\mathbf u_c+\frac{\sum_jq_j\mathbf w_j}{\rho},\qquad \rho=\sum_jq_j.
\]

A mass-averaged momentum equation would use relative velocities \(\mathbf w_j-\sum_kq_k\mathbf w_k/\rho\), including their drift momentum flux. These relations cannot be inserted into a carrier-velocity solver without defining the momentum and energy owners. Size-dependent slip additionally requires consistent node/species/moment transport. [D]

The native pressure solve closes accepted material volume and local physical continuity using the exact transported volume packet. Do not impose an additional independent void source or clip \(\alpha\) to conceal inconsistent inventory. The familiar ideal-gas conversion \(\dot\alpha=\dot n RT/p\) applies only to its declared fixed-pressure/temperature source approximation; it is not the native continuity equation. [C2, C3]

### 6.2 High-void interpretation

Successful inversion does not establish validity of a dispersed-bubble closure. A cell can remain mathematically realizable as liquid disappears while its area, drag, heat-transfer, or coalescence assumptions become physically inappropriate. Conversely, a finite-volume cell need not resolve a gas interface to represent conservative bulk gas occupancy.

**[P]** Every closure declares its admissible state range, and acceptance distinguishes numerical feasibility from physical applicability. A gas-dominant or gas-only continuation requires the selected EOS, phase-energy disappearance, pressure, transport, and boundary configuration to admit that state. There is no universal void threshold or automatic claim that setting `alpha_max=1` qualifies the physics. Liquid-based chemistry stops when its carrier is absent; absent phases have no invented temperature or heat capacity. [C1–C4]

The application may use the initial population closure within its stated range, stop outside that range, or enable a separately defined unresolved-pool model. The choice must be explicit and included in model uncertainty studies. Mesh refinement cannot supply missing topology physics.

### 6.3 Optional unresolved bulk-gas pool

**[P]** An optional pool stores gas material and phase energy without bubble-size moments. The existing catalog compartment `SpeciesPhase::Resolved`, bound to physical `PhaseRole::BulkGas`, is a possible representation seam. The legacy word `Resolved` does not imply a geometrically resolved interface. The exact binding, source integration, and flow configuration still require qualification before this extension is enabled.

An application supplies the transfer law and validity range. This document prescribes no critical void fraction, relaxation time, area law, or automatic pooling rule. For a fraction of selected quadrature atoms transferred to the pool, remove their species, number, and every moment consistently and add the identical species to the receiving phase through the existing energy transaction. Number loss is loss of dispersed-population information, not material escape.

The pool's exchange area is an independent supplied closure; its volume alone cannot determine area. Return to a dispersed family requires an explicit compatible birth distribution and energy transfer. Lost moments cannot be recovered from pooled mass. Different pool composition cannot enter a fixed-composition family without a compatible partition or richer population model. A pooled cell is an unresolved averaged model, not a prediction of cavity shape or connectivity.

## 7. Numerical transport and accepted-state transactions

### 7.1 Common conservative operator

**[C]** Native transport uses backward Euler, first-order upwind, and componentwise extensive balance checks in addition to the original linear-solver gate. Species, signed phase internal energy, and number use their declared compatible phase fluxes. The returned donor concentrations use the actual new cell volume, including the supported planar-ALE path. [C2]

**[P]** Extend that same transport operator and receipts to \(h_2,h_3\). Count, all family species, and both higher moments must describe one transported nonnegative measure. In exact arithmetic, a shared positive transport operator forms nonnegative combinations of donor measures and preserves realizability. Separate approximate solves still require compatible packet checks, composition checks, and full four-moment reconstruction; scalar nonnegativity alone is insufficient.

Use the existing count/species phase flux exactly once. Do not multiply a bulk-normalized moment by another gas fraction, use the pressure-work volume as its denominator, or transport an independent first moment. Retain the actual moment face packets for budgets and replay.

If a future explicit kinetic method transports node packets, each donor's packet carries all relevant count, species, and moments together. Node indices across neighboring cells need not represent equal sizes and must not be matched as material labels. Every outgoing fraction is nonnegative and their sum over all faces must be at most one for each donor atom, leaving a nonnegative remainder. An explicit positivity/CFL restriction belongs to that alternative scheme. It is not a description of the current implicit native method.

Ordinary independent higher-order scalar limiting does not guarantee a realizable moment vector. Higher-order transport requires a dedicated realizability-preserving reconstruction and its timestep restrictions. [R2]

### 7.2 Transport-first source sequence

**[C, P]** Preserve the native controller sequence:

1. Start every flow/source corrector from the same accepted state and geometry.
2. Transport the authoritative inventories, number, phase energy, and proposed higher moments; validate the actual transport endpoint.
3. Reconstruct the consistent pre-material population measure. Evaluate chemistry against transported donors and the trial thermodynamic state, followed by configured ordered face stages and local interphase stages. Pair every family material change with its staged shape effect, consume each material receipt once, and publish compatible count/moment events at the existing number-event stage.
4. Freeze event amounts for pressure/thermal recovery. Evaluate only allowed constitutive laws inside pressure roots.
5. Check source, transported-state, thermal, gross-event, occupancy, and physical continuity convergence; repeat from the accepted state when necessary.
6. Commit fields, moments, receipts, geometry, clock, and histories together, or restore them all.

The current native source loop requires at least two source evaluations. Add a new signed moment-source channel and residual for higher-moment changes. Keep existing fixed-layout gross-event channels nonnegative: record separate production and consumption magnitudes for shape processes rather than inserting signed \(\Delta h_2\) or \(\Delta h_3\) into them. Equal net sources cannot hide changing forward/reverse reaction turnover, simultaneous birth/death, or coalescence/breakup activity. Event channel dimensions and ordering must remain stable across correctors, including rank changes in quadrature. [C2]

Providers remain deterministic, reentrant, and free of unowned history or MPI operations. Accepted-event receipts bind schema, family composition, coordinate scale, provider identity/revision, time interval, geometry, and source stage. Commit, rollback, and checkpoint replay do not call physical laws again. Failed previews cannot leave accepted higher moments inconsistent with accepted species.

### 7.3 ALE, remap, and precision

**[P]** Existing native prescribed planar motion is a separately tested extension for the new moments. Storage uses accepted-old and trial-new volumes; fluxes are mesh-relative. Source locations and geometry receipts refer to the actual trial geometry. General mesh motion or refinement is not enabled by adding two fields.

For an independently qualified remap, use the same nonnegative physical overlap weights for number, species, energy, and higher moments. A positive combination of measures remains realizable. Averaging node locations or mean diameters is not equivalent. Conserving four moments does not necessarily conserve nonlinear area or a general EOS mapping; check material volume and energy under the selected thermodynamic allocation separately. [D]

Retain physical acceptance gates independently of algebraic tolerances. The pinned native flow documents a binary64 limitation when subtracting nearly equal EOS and advected volumes and dividing by a very small timestep. QBMM does not remove that limitation, and no artificial inventory normalization or relaxed continuity tolerance is introduced here. [C2]

## 8. Spatial resolution, boundaries, and outputs

### 8.1 Mesh and model resolution

The mesh resolves changes in bulk velocity, temperature, composition, and gas occupancy. It need not resolve each bubble surface. A useful spatial study refines the cells until the intended averaged outputs and their gradients converge; it also checks whether the averaging volume and constitutive laws remain physically meaningful. Very small quadrature counts may indicate a statistical averaging limitation even though fractional expected counts are numerically valid.

Use three independent studies: spatial resolution, temporal/source integration resolution, and population representation order or comparison with a more resolved size distribution. Refining the mesh does not correct a two-node area approximation, an inappropriate high-void closure, or omitted relative bubble rise. Quadrature nodes are not bounds on the true distribution tails, so a node-based size/mesh ratio is a diagnostic rather than a rigorous maximum-bubble criterion.

### 8.2 Boundaries and reservoirs

**[P]** Initial native use retains its documented closed-boundary restrictions. Local prescribed injection can enter as an explicit external material/energy/momentum event; the resulting pressure and volume must satisfy the same acceptance gates. This does not imply support for arbitrary open advective boundaries.

Any later outlet or headspace coupling must transfer the actual outgoing species, number/moments where relevant, and energy/momentum receipts once. Reaching a top cell is not automatically gas escape. A bulk-gas inventory is not automatically a geometric headspace, and a prescribed-temperature headspace is not an adiabatic energy model. Existing weak field/headspace coupling has different ownership and cannot qualify the native QMOM route by association. [C6]

### 8.3 Published fields

Publish accepted species amounts or declared concentrations, family number, phase mass/volume, gas fraction, mixture density, temperature, size moments, equivalent-sphere area, and diagnostics of closure applicability. Distinguish phase density \(M_b/V_b\) from bulk contribution \(M_b/V_c\). Preserve the volume basis and time/geometry identity in exported fields.

Downstream models receive actual bulk molar concentration \(n_s/V_c\), or an explicitly requested phase-normalized concentration. They must not multiply by a void factor a second time if the field is already normalized by bulk volume. A bulk-volume **reaction-rate basis** is also valid when explicitly declared; it does not change the physical denominator of a liquid-phase concentration. Exported derived quantities do not become new owners of species or gas volume.

## 9. Verification and physical qualification

### 9.1 Proposed mathematical tests

The following exact arithmetic cases are design oracles, not evidence that a C++ QBMM implementation already exists. Sizes below are scaled masses \(s=x/x_*\).

| Case | Expected \([h_0,h_1,h_2,h_3]\) |
| --- | --- |
| Weights 2 and 3 at sizes 1 and 4 | \([5,14,50,194]\) |
| Weight 5 at size 2 | \([5,10,20,40]\) |
| Birth of one bubble at size 3 | Increment \([1,3,9,27]\) |
| First distribution with every size increased by \(1/2\) | \([5,33/2,261/4,2241/8]\) |
| One coalescence \(1+4\rightarrow5\) | Increment \([-1,0,8,60]\) |
| Reverse breakup \(5\rightarrow1+4\) | Increment \([1,0,-8,-60]\) |

For the first distribution the Jacobi entries are \(a_0=14/5\), \(b_1=54/25\), and \(a_1=11/5\). Test vacuum, positive rank one, two close nodes, ill-conditioned moments, negative-support candidates, and singular inconsistent moments separately.

### 9.2 Required integration evidence

**[P]** Qualification includes:

- Authoritative count/species compatibility through birth, growth, partial dissolution, explicit complete disappearance, coalescence, and breakup.
- Exact fixed-composition checks, rejection of selective/composition-changing events, and the synthetic mass-conserving but mole-changing reaction in the generic chemistry kernel.
- Common-operator transport into exactly empty cells and transport of very small positive stocks, with moment, species, count, and phase-energy budgets.
- Pressure/temperature changes that alter EOS volume but leave mass-coordinate moments unchanged when no transport or material event occurs.
- Finite-donor competition, simultaneous gross death/birth, source-corrector convergence, pressure-root callback discipline, and no source duplication on retry.
- Closed common-pressure thermal/material cases, including absent and arriving phases, using existing physical occupancy and caloric gates.
- Explicit gravity/body-force integration, hydrostatic balance, and heated circulation before claiming a natural-convection application.
- Checkpoint/restart, failed-preview restoration, deterministic family/scale fingerprints, and MPI runs including actual empty owners.
- Separately gated moving-mesh, pooling, slip, or open-boundary tests only when those extensions are implemented.

Conservation checks distinguish species reaction budgets from invariants: reactions can consume one species and produce another; closed total mass and declared elements remain invariant. Internal phase transfer cancels in total species, while external reservoirs and boundaries remain explicit signed terms. Source turnover and number changes require their own gross-event checks.

### 9.3 Physical validation and unresolved closures

Numerical tests establish equations, bookkeeping, and robustness for their supplied cases. Physical validation requires relevant measurements for gas amount, occupied volume, temperature, size/area statistics, and transport under the selected chemistry and flow regime. A zero-slip model cannot be validated as a terminal-rise model; a homogeneous pool cannot establish cavity shape predictions.

Before an application is qualified, supply its EOS/caloric data, reaction/transfer laws, birth distribution, population kernels if enabled, applicable pressure/time/void range, and boundary/reservoir model. Quantify sensitivity to source integration, quadrature, and any pooling or drift closure. Low-Mach operation additionally requires an appropriate pressure-equilibration regime; resolved acoustic shocks or arbitrary compressible transients are not supplied by this backend.

## 10. Sources and interpretation limits

### Current repository contracts

These source paths were inspected at SimpleFluid commit `0029a93fd5fcb83658daf82dc80e9cd72a490031`. The links are pinned to that exact revision, rather than to a moving branch.

- **C1.** [Reacting species contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/reacting_species_contract.md): catalogued extensive species/count ownership, thermodynamics, energy convention, disappearing phases, and distributed acceptance.
- **C2.** [Low-Mach reacting flow](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/low_mach_reacting_flow.md): native physical state, carrier/phase fluxes, implicit transport, transport-first source loop, at-least-two evaluations, pressure/volume gates, and documented precision limits.
- **C3.** [Reacting gas sources](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/modeling/reacting_gas_sources.md): generic mechanisms, units and volume bases, finite donors, material and energy ownership, and independently supplied closures.
- **C4.** [Interphase thermal contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/reacting_interphase_thermal.md): ordered frozen events, true partial-molar enthalpy, solver-owned pressure work, complete transfer, and common-pressure zero-surface-tension restrictions.
- **C5.** [Bubble and boiling ownership](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/modeling/reacting_bubble_boiling.md): separate native/Boussinesq thermal ownership, finite solvent/vapor events, and prohibition on a second independent material/latent owner.
- **C6.** [Species-bound surface/headspace contract](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/docs/architecture/reacting_species_surface.md): existing weak field/headspace and planar-surface ownership; not native QMOM qualification.
- **C7.** [Native low-Mach implementation](https://github.com/islandox/SimpleFluid/blob/0029a93fd5fcb83658daf82dc80e9cd72a490031/src/solvers/LowMachMixtureSolver.cc): momentum force assembly includes explicit cell impulse; no automatic gravity follows from density evaluation.

### Primary method references

- **R1.** Heylmun, J. C.; Kong, B.; Passalacqua, A.; Fox, R. O. (2019). *A quadrature-based moment method for polydisperse bubbly flows.* Computer Physics Communications 244, 187–204. [DOI: 10.1016/j.cpc.2019.06.005](https://doi.org/10.1016/j.cpc.2019.06.005). The publisher record describes size–velocity QBMM coupled to two-fluid equations and additional dense-interaction/phase-disappearance treatment. It does not establish a universal one-velocity high-void model.
- **R2.** Passalacqua, A.; Laurent, F.; Fox, R. O. (2020). *A second-order realizable scheme for moment advection on unstructured grids.* Computer Physics Communications 248, 106993. [DOI: 10.1016/j.cpc.2019.106993](https://doi.org/10.1016/j.cpc.2019.106993). Supports the need for dedicated realizable higher-order moment transport and its restrictions; it does not justify independent ordinary scalar limiters.
- **R3.** Bryngelson, S. H.; Colonius, T.; Fox, R. O. (2020). *QBMMlib: A library of quadrature-based moment methods.* SoftwareX 12, 100615. [DOI: 10.1016/j.softx.2020.100615](https://doi.org/10.1016/j.softx.2020.100615); [author preprint](https://arxiv.org/abs/2008.05063). Supports the population-balance, moment-inversion, and quadrature framework.
- **R4.** Marchisio, D. L.; Fox, R. O. (2005). *Solution of population balance equations using the direct quadrature method of moments.* Journal of Aerosol Science 36, 43–73. [DOI: 10.1016/j.jaerosci.2004.07.009](https://doi.org/10.1016/j.jaerosci.2004.07.009); [author institutional record](https://iris.polito.it/handle/11583/1402888). Defines DQMOM's direct quadrature-variable evolution and distinguishes it from the selected QMOM implementation.

The algebraic identities and exact arithmetic tests in this document are derived requirements. The literature motivates the method family; it does not validate the project's supplied material closures, optional pooling law, or any application-specific parameter set.
