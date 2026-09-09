# Water-case distribution and error figures

The case `run_comparison.sh` launchers render the fields after the numerical
comparison succeeds. The run's `figures/index.html` gallery links SVG figures,
matched CSV data, and error statistics. PNG and PDF copies are also written
when `rsvg-convert` (librsvg) is installed. This uses the existing verification
SVG workflow without requiring Matplotlib. Explicitly select formats with
`--formats svg` or `--formats svg png pdf` on the plotting script.

Each gallery contains temperature in Kelvin, pure-liquid density, gas-phase
volume fraction, and liquid velocity magnitude.

| Figure | Content |
| --- | --- |
| `mesh` | Shared reference cell boundaries and wall spacings at t=0 |
| `distribution` | OpenFOAM and SimpleFluid x–z sections at the final matched time, using common color scales |
| `relative` | Final-time relative errors, with undefined values shown in gray |
| `absolute` | Final-time absolute errors, including fields whose relative error is undefined |
| `history_distribution` | Both solvers' full time–height field histories |
| `history_relative` | Relative errors across the full matched history |

These are actual cells from the central-y plane of the solved 3-D mesh.
The enlarged cases have ten y layers; the manifest records the selected layer
and its physical bounds. Global histories include every layer. The x–z figure
displays the selected cells at their physical x/z extents. Histories use
normalized height `z/L(t)` to follow the moving ALE cells. Time bins are centred
on actual output times and do not interpolate intermediate physical states.
The steady gallery includes the approach to equilibrium and its final state.

The `bottomHeatedBubblyConvection` case additionally exports x bounds for a
resolved x–z grid. Both sides solve liquid momentum; its arrows and vector
errors compare solved flow fields. Histories display the central x column,
while the matched CSV and statistics retain every sampled-plane cell and time. Its localized
source produces spatial temperature, density, gas-fraction, and velocity
variations; neither velocity nor gas fraction is prescribed to a uniform field.

## Field ownership and interpretation

- **Dispersed bubbles:** temperature is prescribed and isothermal, density
  comes from the applied IF97 material fields, and gas fraction comes from
  the transported bubble populations and capillary equation of state. The
  plotted liquid velocity is prescribed at 0.1 m/s upward; bubbles have an
  additional prescribed 0.4 m/s upward slip. These velocity panels check case
  inputs; neither solver solves carrier momentum in this fixture.
- **Planar ALE:** SimpleFluid exports cell temperature, actual material
  density, and its solved cell velocity. Gas fraction is explicitly zero
  because this thermal-expansion case contains liquid only. OpenFOAM exports
  cell temperature and the same reference-water linearized density law. Its
  velocity is the affine kinematic reference
  `U_z = z/L_new * (L_new-L_old)/dt`, with zero transverse velocity; it does
  not solve momentum. Velocity errors are diagnostics against that reference,
  not an additional momentum acceptance gate. Actual SimpleFluid velocities
  are never replaced by mesh velocities.

The case manifests define geometry-scaled extensive tolerances and unchanged
intensive field criteria.
The plotter additionally requires complete time/cell coverage, finite fields,
positive temperature/density, bounded gas fraction, contiguous column cells,
matching physical cell extents, and fresh output in the launchers.

## Relative errors and zeros

For temperature, density, and gas fraction, the plotted signed percentage is
`100 (SimpleFluid - OpenFOAM) / abs(OpenFOAM)`. For velocity, it is
`100 ||U_SimpleFluid - U_OpenFOAM|| / ||U_OpenFOAM||`; the full vector prevents
equal speeds with different directions from appearing identical. The
velocity distribution itself shows `||U||` in m/s.

Relative error is **undefined**, including zero/zero, when the reference
magnitude is at or below these fixed denominator floors:

| Quantity | Reference floor |
| --- | --- |
| Temperature | 1e-12 K |
| Liquid density | 1e-12 kg/m³ |
| Gas fraction | 1e-14 |
| Velocity magnitude | 1e-12 m/s |

Undefined values are gray in figures, blank in `matched_fields.csv`, and
counted in `statistics.json`. They are not replaced with zero relative error.
Absolute errors remain available, so an error at zero reference cannot
disappear. Gas-fraction errors near the bubble-free front are masked below
the stated floor. Scalar relative-error color scales are symmetric about zero.

## Replot retained outputs

Both solvers retain `fields.csv`, containing accepted physical time, cell ID,
physical lower/upper z bounds, temperature, density, gas fraction, and all
three velocity components. For example, from the repository root:

```sh
python3 verification/openfoam/plot_water_fields.py \
  --manifest verification/openfoam/planarALE/transient.json \
  --openfoam /path/to/run/openfoam/fields.csv \
  --simplefluid /path/to/run/simplefluid/fields.csv \
  --output-directory /path/to/run/figures
```

`spatial_output` in each run's `manifest.json` specifies the expected cell IDs and
cross-section width. The main manifest supplies the exact physical output
times. Generating figures alone does not establish that the separate solver
acceptance checks passed.
For a selected non-default mesh, pass the retained run manifest rather than
the checked-in default manifest. Explicit `mesh_edges_m` are checked against
the output geometry before plotting; ALE permits its measured axial stretch.
