# Pressure-conditioned local interphase transfer

The native low-Mach solver owns phase internal energy and pressure work.
`ReactingMixtureCoupler` owns optional local interphase events after conservative
transport, local chemistry and ordered face sources, and before population-number
events. This path uses finite catalogued material, supplied thermodynamics and
explicit configured transfers. It does not infer vapor, solvent or an inert gas.

## Frozen event ownership

`MixtureInterphaseEventProvider` supplies an immutable ordered
`InterphaseConfiguration` with stable identity, revision and catalog fingerprint.
Each stage declares either signed heat slots between two phases or nonnegative
material slots naming a species, donor and recipient. The kinetic callback returns
integrated amounts in that exact slot order. It receives actual post-face stocks,
transported inventories, the thermodynamic guess, accepted time and the current
canonical `ReactionLocation`; it performs no MPI operations or history writes.

Every material stage reserves all original outgoing events against its predecessor
stocks before admitting incoming credit. Heat stages similarly reserve gross
outgoing sensible capacity and receiving headroom within the supplied caloric
bounds. An empty heat donor has no invented temperature. Successive ordered stages
consume their actual predecessor endpoints. Number events run after the final
material endpoint; disappearance and creation require explicit compatible counts.

The compiled plan retains original static energy channels, declared heat and
frozen material/heat amounts. Pressure evaluations can call pure constitutive
laws, but cannot reintegrate kinetics, chemistry, face fluxes or population events.
Commit and restore invoke none of these laws.

## Energy and pressure recovery

At candidate absolute pressure `p`, each pre-transfer caloric anchor satisfies
`H_pre = U_adv + E_static + p*V_adv`. `V_adv` is the exact transported physical
phase-volume packet, not a donor concentration volume or the geometric cell
volume. Common-temperature recovery records its zero-net caloric redistribution
explicitly. Separate-temperature recovery uses each occupied phase's own energy.

For material transfer, the supplied law returns true partial-molar enthalpies
`h_i = (dH/dn_i)_(T,p,n_other)` and their temperature and pressure derivatives in
catalog order. The evaluator checks the donor Euler identity and both derivative
identities against the actual mixture EOS. A mean-mixture enthalpy is not a
substitute. Every transfer debits and credits the same `dn_i*h_i(T_anchor,p)` and
then recovers the next stage. Complete donor removal therefore removes its actual
pressure-conditioned enthalpy; the final empty phase has exactly zero energy and
temperature throughout the admissible pressure branch.

The full pressure derivative propagates through every anchor and partial-enthalpy
law. The same resulting `dV/dp` drives thermodynamic pressure recovery and local
mechanical susceptibility. Only the flow solver applies
`-p*(V_final-V_adv)`. No second pressure-work or inferred latent-heat term is added.
A phase-change energy offset must be present in the supplied thermodynamics or
have a separately declared physical owner; the energy convention cannot hide it.

Compensated original channels survive static/dynamic cancellation before final
narrowing. Numerical operand uncertainty is reported and must remain a small part
of the unchanged physical energy budget. Temperature inversion resolves the
representable monotone EOS bracket; an additive caloric reference cannot enlarge
its stopping tolerance. An unrepresentable endpoint or heat allocation rejects.

## Receipts and geometry

`integrated_sources()` retains the exact static source packet. The frozen
`conditioned_thermal_receipt()` and `final_thermal_evaluations()` are separate
receipts for the actual pressure-dependent energy application. They retain source,
schema, invocation, generation and geometry identities, staged anchors, active
material-slot constitutive values, paired heat/enthalpy and pressure-work data.
Raw plans and evaluated energy both participate in nonlinear convergence.

Prescribed planar motion uses the existing native ALE transaction. The source
frame and thermal plan refer to actual trial geometry, while phase storage and
advection retain accepted-old/new volumes and relative fluxes. Preview restores
accepted geometry before returning; exact immutable receipts bind the completed
work. An earlier valid trial remains usable after a later rejected preview.
Moving commit and checkpoint replay require the controller's live motion authority.

## Feasibility and limits

Only a provider for the complete ordered thermal plan may certify its caloric
pressure domain. Bare final-phase affine EOS metadata cannot certify intermediate
anchors or pressure-dependent transfer energy. Without a whole-plan certificate,
bounded search reports unresolved feasibility when its evaluation budget is
exhausted. A typed caloric-range failure is distinct from malformed metadata,
invalid derivatives, material overdraw or a real property-law exception.

The implemented model has common phase pressure and zero surface tension. It
uses the native flow's closed orthogonal mesh and prescribed-motion restrictions.
It supports configured local stages and their finite donor removal/creation;
implicit simultaneous interface laws require a separate coupled residual and
Jacobian. A strong heat input can be infeasible for the declared ordered anchors
without proving that every possible coupled physical interface model is infeasible.

If earlier chemistry or face transport has already emptied a phase while leaving
nonzero advected work volume or unallocated energy, this path rejects it. A trace
of the pre-transfer state and associated work is required. General pressure-conditioned
transfer across faces additionally needs a constitutive halo and pressure-Jacobian owner; source
callbacks must not be rerun inside pressure roots to approximate that ownership.

The separate Boussinesq finite-vapor adapter owns its reference-density `cp*T`
approximation and latent accounting. Native phase-energy tests do not qualify that
approximation. Neither path establishes evolving slip momentum, capillary energy
or a combined thermal/kinetic total-energy model. VoF, its bounded floating-point
geometry route and its exact-rational route remain outside this refactor scope.

Focused tests include independent complete-transfer pressure/temperature roots,
nonideal tangents, separate-temperature heat exchange, raw cancellation, original
finite donors, callback-free commit/replay and actual empty MPI owners. A one-cell
moving full-transfer case checks the independent backward-Euler work balance and
retains an earlier trial across a later failure. Numerical manufactured cases are
not physical validation of a supplied constitutive law.
