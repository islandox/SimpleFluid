# Liquid-side bubble mass transfer

The `sheng2024TwoPopulation` kinetics uses a selected liquid-side coefficient
for the exchange between dissolved H2 and **active large bubbles**. The
selector `bubble_mass_transfer_model` defaults to `legacyHughmark` and also
accepts `sphericalDiffusion`, `fengMichaelidesClean`, and
`fengMichaelidesRigid`. These are separate from the molecular
diffusivity provider (`hydrogen_diffusivity_model` or the C++ callback).

All dimensional inputs and outputs are SI. The shared evaluator reports
dimensionless diameter-based numbers using the bubble-liquid relative speed
supplied by the existing rise law:

```text
d  = 2 R
Re = rho_l |u_rel| d / mu_l
Sc = mu_l / (rho_l D_m)
Pe = Re Sc = |u_rel| d / D_m
Sh = k_L d / D_m
```

`D_m` is the **molecular** dissolved-H2 diffusivity. The bulk finite-volume
operator separately uses `alpha_l D_m`; turbulent or numerical diffusion is
never inserted into the interfacial `Sc`. The selector changes only `k_L`.
It does not change bubble slip, drag, liquid velocity, Henry/Laplace
equilibrium, conversion, lifetime terms, or the conservative inventories.
Comparisons between selectors use the same prescribed or computed slip.
The runtime mode is resolved once before each cell-field sweep (preflight,
kinetics, or derived `K_L`). Each cell still receives its own physical-input
and applicability checks; the Feng low/high Pe branch remains state-dependent.

## Implemented laws and domains

| Selector | Diameter-based equation | Runtime domain and meaning |
| --- | --- | --- |
| `legacyHughmark` | `Sh = 2 + 0.6 Re^(1/2) Sc^(1/3)` for `Re < 776.06`; `Sh = 2 + 0.27 Re^0.63 Sc^(1/3)` otherwise | Existing behavior, including the strict `0 <= Sc < 250` guard. This is the historical rigid-sphere-like branch; it has no mobile-interface model. Its low-Re, high-Sc use is not qualified. |
| `sphericalDiffusion` | `Sh = 2`, hence `k_L = D_m/R` | Exact steady radial diffusion for a sphere with fixed surface concentration in quiescent, unbounded liquid. Runtime requires `Pe = 0`. In a CFD flow, it is an explicit local diffusion-only approximation, not a finite-Pe high-Sc closure. |
| `fengMichaelidesClean` | Low Pe: `Sh = 2 + Pe/2 + (Pe² ln Pe)/12`; high Pe: `Sh = 1.49 (Pe/2)^0.435` | Inviscid clean/mobile endpoint. Requires `Re < 1`; low branch `0 <= Pe < 1`, high branch `100 <= Pe/2 <= 5000`; rejects the gap. |
| `fengMichaelidesRigid` | Low Pe: `Sh = 2 + Pe/2 + (Pe² ln Pe)/8`; high Pe: `Sh = 1.49 (Pe/2)^0.322` | Rigid/immobilized endpoint, with the same disjoint gates. |

The legacy expression and `Sc` guard reproduce the existing
`RadiolyticGasPhysics::hughmark_sherwood` implementation. Its historical
reference is G. A. Hughmark, “Mass and heat transfer from rigid spheres,”
*AIChE Journal* **13** (1967), 1219–1221,
<https://doi.org/10.1002/aic.690130638>. The publisher preview identifies
the article but does not provide its equations for an independent comparison;
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
`Pe`, `Sh`, `k_L`, the model, and an applicability label. The originating
rank's model exception identifies the population, cell, and rank; peers
receive a collective failure before the next communication.

The optional field `K_L` evaluates the selected law at the reconstructed
**characteristic radius** for visualization. The source term uses the actual
large-population radius, so `K_L` is not the exact coefficient of every
population source term. Accepted-step statistics record the independently
minimized and maximized dimensional inputs and outputs from active
large-population kinetic evaluations; trial/reconstruction calls do not enter
that envelope. A cell with neither population has no transfer evaluation;
when only microbubbles are present, the characteristic-radius diagnostic
can still be evaluated, while no large-population kinetic transfer occurs.

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
Hughmark, or substituting `Sh=2` at finite Pe. Accepted-step diagnostics
count the low-Pe and fit evaluations separately, including a mixed-regime
label if both occurred in one step. The Reynolds-dependent expression often
quoted under the authors' names belongs to a distinct **2001** paper and is
not used here.

A. Dani, A. Cockx, and P. Guiraud, “Direct Numerical Simulation of Mass
Transfer from Spherical Bubbles: The Effect of Interface Contamination at Low
Reynolds Numbers,” *International Journal of Chemical Reactor Engineering*
(2006), <https://doi.org/10.2202/1542-6580.1304>, compares creeping-flow
clean and fully contaminated bubbles. Its author-posted text includes
Clift-attributed finite-Pe endpoint equations, but the PDF equations could
not be checked visually in this update. Its moderate-Re comparisons do not
establish a high-Sc moderate-Re closure, so no `Dani` selector, regime switch,
or extrapolation is implemented. The Feng fit also does not cover moderate Re
or the unsourced intermediate-Pe gap.

The population model also includes a large-bubble lifetime dissolution term
when interfacial transfer is negative. That term can overlap with the
`k_L`-driven loss; it is retained as a separate physical-model question.
