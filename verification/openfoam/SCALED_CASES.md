# Tenfold domains with retained cell widths

The convection results below were measured before SST was enabled. Current
convection uses [Menter-1994 SST and resolved walls](bottomHeatedBubblyConvection/SST.md);
those earlier laminar results do not validate the new model.

The default domains are enlarged by ten in x, y and z. Original wall-layer
counts, first-cell heights, stack thicknesses and core spacings are retained;
extra core cells fill the larger domains. Wall layers are not repeated inside
the domain. Both implementations read the same native-factory coordinates.

| Case | Dimensions x × y × z, m | Cells x × y × z | Total |
| --- | --- | --- | ---: |
| Bottom convection, default | 0.2 × 0.02 × 0.2 | 244 × 10 × 324 | 790,560 |
| Bottom convection, fine | 0.2 × 0.02 × 0.2 | 364 × 10 × 484 | 1,761,760 |
| Bottom convection, uniform control | 0.2 × 0.02 × 0.2 | 120 × 10 × 160 | 192,000 |
| Dispersed bubbles | 10 × 10 × 10 | 10 × 10 × 404 | 40,400 |
| Planar ALE, initially | 10 × 10 × 10 | 10 × 10 × 84 | 8,400 |

For example, the default convection first x/z widths remain 0.296/0.222 mm,
and y cells remain 2 mm wide. The domain volume increases by 1000. Local heat
and gas source densities stay fixed, so integrated convection heating becomes
133.333 W and hydrogen production 2.666667e-5 mol/s.

Convection retains dt=0.02 s and end=20 s; ALE retains dt=1 s and 20 heated
steps plus five source-off steps in steady mode. Bubble transport retains
dt=0.01 s. Its transient still ends at 1 s (5% ideal column clearance); steady
mode now runs to 100 s to allow escape to equilibrate in the longer column.
The full steady bubble case consequently requires 10,000 timesteps.

The expanded y direction contains ten solved layers. Field CSVs and figures
select the central-y cell plane, with bounds stated in the manifest. Bubble
profile comparisons select its central x/y column. Global inventories,
sources, energy and cold-wall heat loss include the complete domain. The
convection history now exports `wall_heat_loss_W`, allowing mesh sensitivity
analysis to use full-domain heat loss instead of reconstructing it from a slice.

Global extensive absolute tolerances scale with volume, while intensive field
limits and per-cell ALE GCL/continuity gates are retained. Relative face flux
is reported and bounded per initial domain volume. ALE
requires more accurate local accumulation and tighter level coupling
on the larger domain; see its [case contract](planarALE/README.md). No ALE
physics guards were relaxed. The original no-slip walls remain. Resolved
columns retain weak internal circulation; raw and reference-volume-normalized
relative fluxes are reported explicitly. The reduced reference comparison
accepts uniform energy/volume behavior, not pointwise zero carrier circulation.
IF97 remains optional and required explicitly
by these real-water examples. ALE still uses its supported linear Boussinesq
reference-water closure and conserved liquid-mass sensible energy.

Original inputs are retained in `mesh*_scale1.dat` and
`reference_scale1.properties`. The six routine physical CTests select these
small fixtures explicitly. Normal comparison launchers select the enlarged
defaults. Regeneration and mesh overrides are documented in
[boundary-layer meshes](BOUNDARY_LAYER_MESHES.md).

## Focused validation, 2026-09-09

Validation uses GCC Release with IF97 enabled and OpenCFD OpenFOAM v2606
optimized reference applications. Retained outputs are under
`build/verification/scale-up/` in this worktree.

- Native exporter: all five meshes reproduce their saved coordinates exactly
  at both scales 1 and 10 (10 checks).
- Shared mesh tests: 6 passed; field plotting: 10 passed; comparator: 21 passed;
  performance-runner bookkeeping: 8 passed. These Python checks are not solver
  performance measurements.
- OpenFOAM `checkMesh`: default/fine convection, bubble and ALE all report
  `Mesh OK`, with zero non-orthogonality. Every generated mesh point matches
  the canonical coordinates; the largest discrepancy is 4.27e-14 m.
- Full scaled bubble transient: both solvers accepted 100 steps. Paired
  comparison passes 2,020 axial profile/time samples, with maximum molar
  concentration difference 1.230e-13 mol/m³. The field/error gallery covers
  20,200 sampled-plane cell/time records.
- Full scaled ALE: 20 heated steps and the 25-step source-off case accepted
  by both solvers. SimpleFluid reaches 301.920513032049 K and level
  10.005272189316 m. Its maximum absolute energy residual is 0.002390 J,
  mass residual 1.165e-9 kg and cell GCL residual zero. The maximum paired
  energy difference is 0.004151 J; OpenFOAM's energy residual is at most
  0.004789 J, both within the declared 0.1 J / 0.05 J bounds. The maximum
  relative face flux is 9.535e-9 m³/s (9.535e-12 s⁻¹ per reference volume).
  Cell velocity differs substantially from the affine reference: the maximum
  vector difference is 7.316 mm/s. The velocity gallery is diagnostic; passing
  the energy/volume comparison does not qualify ALE momentum accuracy.
- Six physical CTests pass using the explicit scale-1 fixtures. All 17 focused
  `LiquidMassInventoryTest.*` tests pass, including the new 8,000-cell
  regression against spurious volume from accumulation roundoff.
- The scaled steady bubble startup passes a five-step paired check; it is
  explicitly partial and does not establish steady-state convergence.
- Default scaled convection: both solvers pass a one-step startup comparison
  through 0.02 s, including global source and conservation gates. The gallery
  covers 158,112 sampled-plane cell/time records and explicitly identifies the
  partial scope. It is not a developed-plume result.

The full enlarged convection trajectories, fine-grid flow study and 100-second
steady bubble trajectory have not been completed in this focused update.
Earlier scale-1 convergence and performance tables remain historical. The
larger domain is not dynamically similar: gas EOS pressure remains prescribed,
front/back wall friction is absent, and the ALE comparison remains uniform
thermal expansion against a reduced OpenFOAM energy/volume reference.

Timing files are retained for diagnosis. The bubble transient used 58.17 s in
SimpleFluid and 1.156 s in OpenFOAM; the one-step convection startup used
40.08 s and 36.65 s, respectively. These are single serial loop observations,
including diagnostics and CSV output, with different equation/solver work and
without repeated isolated timing. They do not establish a general speedup,
scaling efficiency, or developed-convection performance comparison.

[Local figure gallery](../../build/verification/scale-up/index.html) and
[validation details with exact changed files](../../build/verification/scale-up/validation.json)
are retained in this worktree. The latter records executable/source hashes,
comparison inputs, sample counts, focused-test totals and incomplete runs.
