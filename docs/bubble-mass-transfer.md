# Liquid-side bubble mass transfer

`ReactingGasMode::Species` uses finite catalogued species and explicitly named
dispersed populations. The field adapter's built-in film stage evaluates each
eligible species in each occupied population; there is no preferred H2 slot or
large-bubble-only transfer inventory. An application may instead supply the
whole `population_process` packet and own its constitutive law and event order.
Enabled setup requires an explicit mechanism/catalog, initial material and counts,
mixture thermodynamics and per-species properties; it has no heat-yield fallback.

`ReactingGasOptions::mass_transfer_mode` selects `Hughmark`,
`SphericalDiffusion`, `FengMichaelidesClean` or `FengMichaelidesRigid`. The neutral
database spellings are `hughmark`, `sphericalDiffusion`, `fengMichaelidesClean`
and `fengMichaelidesRigid`. The enum default remains the guarded Hughmark
implementation; it is not a universal recommendation. A database selector cannot
supply the missing catalog, EOS or finite inventories required to enable Species.
Retired yield/kinetics/property selectors are rejected rather than mapped to an
unnamed gas.

Each `ReactingSpeciesProperties` entry identifies a stable species ID and supplies
its own molecular diffusivity or MPI-free `diffusivity_provider`. Zero diffusivity
disables that species' film process. Henry coefficients use
mol/(m3 actual liquid Pa); equilibrium uses the supplied gas-mixture fugacity,
`C_i,eq = H_i*f_i`, or `equilibrium_concentration`. For an ideal mixture this reduces to
`f_i = y_i*p_g`. A nonideal provider cannot silently substitute total or partial
ideal-gas pressure for its equilibrium law.

All dimensional inputs and outputs are SI. The shared evaluator reports
dimensionless diameter-based numbers using the explicitly supplied bubble-liquid
relative speed:

```text
d  = 2 R
Re = rho_l |u_rel| d / mu_l
Sc = mu_l / (rho_l D_m)
Pe = Re Sc = |u_rel| d / D_m
Sh = k_L d / D_m
```

`D_m` is the molecular diffusivity of the selected species in the actual liquid.
It is a film property: it does not implicitly enable a bulk finite-volume
`alpha_l D_m` operator. The field adapter's conservative bulk-mixture-diffusion
selection remains rejected until it has a complete transport/enthalpy owner.
The native low-Mach controller has a separate supplied finite face-source contract;
see [its ownership and support limits](architecture/low_mach_reacting_flow.md).
Turbulent or numerical diffusion is never inserted into interfacial `Sc`.

The shared evaluator accepts an explicit bubble-liquid relative speed. The current
default Species film stage supplies `constant_slip_velocity` for `ConstantSlip`
and zero otherwise. In particular, selecting Celata for population transport does
not automatically feed that computed velocity into the film evaluator. A supplied
population law must own any additional consistent slip/transfer coupling. Comparing
selectors requires the same declared relative speed and physical properties.
The selector changes `k_L` only; it does not independently change drag, velocity,
mixture equilibrium, finite stocks, counts or the supplied event schedule.

For the built-in stage, the integrated request is

```text
dn_i = dt * k_L,i * (4*pi*R^2*N) * (n_i,dissolved/V_liquid - C_i,eq).
```

Positive `dn_i` moves that amount from Dissolved to the named Dispersed bin;
negative `dn_i` reverses the transfer. All original outgoing species/count events
share finite donor limits before incoming credit. No material or count is clipped
to repair overdraw. Count births/deaths are explicit. Only a declared built-in
complete-dissolution stage may issue its own remaining-count death; arbitrary
chemical disappearance or a user packet cannot trigger a generic count reset.
The law and its low/high-Pe branch are evaluated on each actual local state.

## Implemented laws and domains

| Selector | Diameter-based equation | Runtime domain and meaning |
| --- | --- | --- |
| `hughmark` | `Sh = 2 + 0.6 Re^(1/2) Sc^(1/3)` for `Re < 776.06`; `Sh = 2 + 0.27 Re^0.63 Sc^(1/3)` otherwise | Existing behavior, including the strict `0 <= Sc < 250` guard. This is the historical rigid-sphere-like branch; it has no mobile-interface model. Its low-Re, high-Sc use is not qualified. |
| `sphericalDiffusion` | `Sh = 2`, hence `k_L = D_m/R` | Exact steady radial diffusion for a sphere with fixed surface concentration in quiescent, unbounded liquid. Runtime requires `Pe = 0`. In a CFD flow, it is an explicit local diffusion-only approximation, not a finite-Pe high-Sc closure. |
| `fengMichaelidesClean` | Low Pe: `Sh = 2 + Pe/2 + (Pe² ln Pe)/12`; high Pe: `Sh = 1.49 (Pe/2)^0.435` | Inviscid clean/mobile endpoint. Requires `Re < 1`; low branch `0 <= Pe < 1`, high branch `100 <= Pe/2 <= 5000`; rejects the gap. |
| `fengMichaelidesRigid` | Low Pe: `Sh = 2 + Pe/2 + (Pe² ln Pe)/8`; high Pe: `Sh = 1.49 (Pe/2)^0.322` | Rigid/immobilized endpoint, with the same disjoint gates. |

The retained expression and `Sc` guard reproduce the existing
`BubbleGasPhysics::hughmark_sherwood` implementation. Its historical
reference is G. A. Hughmark, “Mass and heat transfer from rigid spheres,”
*AIChE Journal* **13** (1967), 1219–1221,
<https://doi.org/10.1002/aic.690130638>. The original correlation review identified the article through its publisher
preview but could not independently compare its equations there;
the equations above are documented as the retained implementation, not as a
new qualification of Hughmark's original data.

The diffusion reference follows directly from the steady radial equation
`(1/r²) d/dr (r² dC/dr) = 0` with `C(R)=C_s` and `C(infinity)=C_inf`:
`C(r)=C_inf+(C_s-C_inf)R/r`. Its interfacial flux magnitude is
`D_m |C_inf-C_s|/R`, giving `k_L=D_m/R` and diameter-based `Sh=2`.
Zero translational slip alone does not establish a quiescent, unbounded
liquid; shear, nearby bubbles, and boundaries can disturb this ideal radial
field. There is no gas-viscosity ratio in this diffusion-only solution. A transient
bubble-growth/dissolution model such as P. S. Epstein and M. S. Plesset,
“On the Stability of Gas Bubbles in Liquid-Gas Solutions,” *Journal of
Chemical Physics* **18** (1950), 1505–1509,
<https://doi.org/10.1063/1.1747520>, is a separate extension. The CFD
timestep is not a bubble age and no transient term is used here.

The shared evaluator rejects nonfinite/nonphysical dimensional inputs with
`invalid_argument`, and finite states outside a selected model's domain with
`domain_error`. It reports `R`, `d`, `u_rel`, `rho_l`, `mu_l`, `D_m`, `Re`, `Sc`,
`Pe`, `Sh`, `k_L`, the model, and an applicability label. The distributed owner
fences local failures before the next communication;
peers receive collective failure. Callback laws themselves perform no MPI work.

The pure evaluator's `BubbleMassTransferResult` reports the dimensional inputs,
`Re`, `Sc`, `Pe`, `Sh`, `coefficient`, model and applicability label. Those are
constitutive evaluation results, not proof that a kinetic request was accepted.
The field adapter's `population_ledgers()`, `species_step_events()` and accepted
`species_event_receipt()` identify actual per-species/population material events.
An independent test can recover the accepted coefficient from those events, actual
area and the declared equilibrium driving force. Retired characteristic-radius
`K_L` output and five-slot H2 diagnostics are not authoritative general observers.
No aggregate min/max envelope is a substitute for those keyed receipts.

This Boussinesq field adapter does not own extensive phase internal energy.
Selective escape requires a supplied partial-molar-enthalpy trace, but the trace
is not another phase-U debit. The [weak surface adapter](architecture/reacting_species_surface.md)
uses the same accepted events for its vent/headspace, with actual count escape
reported separately. The fixed-grid finite-solvent/vapor `cp*T` adapter owns an
explicit reference-volume/latent approximation and cannot be combined with that
weak supplied-EOS surface/ALE path. Native
[pressure-conditioned interphase transfer](architecture/reacting_interphase_thermal.md)
has a separate phase-energy and pressure-work owner. A film coefficient by itself
qualifies none of these thermal combinations.

## Feng–Michaelides finite-Pe endpoints

Z.-G. Feng and E. E. Michaelides, “Mass and heat transfer from fluid spheres
at low Reynolds numbers,” *Powder Technology* **112** (2000), 63–69,
<https://doi.org/10.1016/S0032-5910(99)00306-X>, p. 67 Eq. (19), fits its
Stokes-flow numerical results with

```text
Sh_d = 1.49 Pe_a^[0.322 + 0.113 sigma/(0.361 + sigma)]
Pe_a = R |u_rel| / D_m = Pe_d/2
sigma = mu_liquid / mu_sphere.
```

Eq. (11), p. 65 defines **radius-based** `Pe_a`; the source's average `Sh`
is **diameter based** by Eq. (A-12), p. 69. `sigma = 0` is the rigid
sphere limit and `sigma -> infinity` is the inviscid gas-bubble limit
(p. 64). The clean/mobile and immobilized labels describe these ideal
hydrodynamic endpoints, not measured surfactant contamination. The scalar
equation and fixed surface concentration boundary condition, Eqs. (10) and
(13), solve transfer in the *external liquid*. Internal circulation changes
the external Stokes velocity field; internal gas resistance, interfacial
kinetics, and transient bubble age are not included. The configured slip and
drag law are held fixed when comparing the endpoints.

Table 1, p. 67 samples `Pe_a = 10, 20, 50, 100, 200, 500, 1000, 2000,
5000` at `sigma = 0, 0.1, 0.25, 0.5, 1, 2, 5, 10, infinity`. For `Pe_a=100`,
the tabulated Stokes results are `Sh_d=6.86` rigid and `10.91` inviscid;
Eq. (19) gives about `6.564` and `11.046`. At `Pe_a=10`, the fit is about
17% low for the rigid endpoint and 11% low for the inviscid endpoint. The
paper describes Eq. (19) as most useful at large Pe and gives no uniform
error bound or finite-Re sweep. Consequently this implementation uses
`100 <= Pe_a <= 5000` and `Re_d < 1` as **conservative operational gates**
inferred from the tabulated fit residuals and Stokes assumption, not as a
source-declared validated Re–Sc rectangle. No separate Sc restriction is
stated by the paper. The published table checks the Stokes scalar solve; it
does not experimentally validate the model at a specific finite Re or Sc.

The finite-range fitted exponents `0.322` and `0.435` are distinct from the
paper's asymptotic `1/3` rigid and `1/2` inviscid powers. Eq. (19) gives
`Sh=0` at `Pe=0` and therefore must not be used there. On p. 64, Eq. (4),
the authors quote the steady limit of their earlier low-Pe work [6,7]:

```text
Sh_d = 2 + Pe_d/2 + [(3 + 2 sigma)/(24(1 + sigma))] Pe_d² ln(Pe_d),
valid as a small-Pe asymptote for Pe_d < 1.
```

The logarithmic coefficient is `1/8` rigid and `1/12` inviscid. At exactly
zero slip, the limit `Pe_d² ln(Pe_d) -> 0` is applied without evaluating
`ln(0)`, giving `Sh=2`. This is a truncated asymptote; the paper supplies
no numerical error bound near `Pe_d=1`. The same endpoint selectors use it
only for `0 <= Pe_d < 1`, and use Eq. (19) only in the operational high-Pe
window. They reject `1 <= Pe_d < 200` rather than blending, falling back to
Hughmark, or substituting `Sh=2` at finite Pe. Result labels distinguish low-Pe
from fit evaluations; accepted material must still be checked from keyed events.
The Reynolds-dependent expression often
quoted under the authors' names belongs to a distinct **2001** paper and is
not used here.

A. Dani, A. Cockx, and P. Guiraud, “Direct Numerical Simulation of Mass
Transfer from Spherical Bubbles: The Effect of Interface Contamination at Low
Reynolds Numbers,” *International Journal of Chemical Reactor Engineering*
(2006), <https://doi.org/10.2202/1542-6580.1304>, compares creeping-flow
clean and fully contaminated bubbles. Its author-posted text includes
Clift-attributed finite-Pe endpoint equations, but the original correlation review
could not check those PDF equations visually. This documentation cleanup has not
reverified the primary-paper images. Its moderate-Re comparisons do not
establish a high-Sc moderate-Re closure, so no `Dani` selector, regime switch,
or extrapolation is implemented. The Feng fit also does not cover moderate Re
or the unsourced intermediate-Pe gap.

The optional explicitly configured micro/large first-order dissolution and
conversion frequencies are separate declared events. Their loss may overlap a
film-driven request; the combined finite donor budget must still pass. They are
not an inferred bubble age, a replacement for the supplied film domain, or a
license to discard another species' residual gas/count inventory.

These correlations use spherical representative bins and supplied constitutive
properties. Native and legacy field backends preserve the caller's mesh identity;
serial/MPI execution, including XY/Z decompositions and empty owners, has separate
map/halo/epoch/failure contracts. Geometry/operator coverage alone does not qualify
every physics combination. This guide records model equations and source
provenance, not experimental validation. VoF and the bounded floating-point and
exact-rational VoF/general-AMR geometry paths remain deferred outside this refactor;
ordinary finite-donor, conservation and roundoff checks remain required.
