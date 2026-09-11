# Region-native CFD execution qualification

This pass implements structured-interface inverse indexing, region-native CFD
transport/gradient execution, and stored-field bulk gradient views. It retains
the canonical field layout, numerical formulas, summation order, interface
orientation, geometry leases, and existing MPI ownership/halo behavior.
Coupled GMRES/restart, Schur preconditioning, and block communication policies
were not changed. The verification controls committed as `ff42796` were retained
unchanged; the optimization is a working-tree change on that source base.

## Implementation

- Structured removed patches cache their origin, extents, strides and count.
  Regions whose removals consist of complete structured sides use direct
  quotient/remainder selection, including multiple sides and empty orientation
  blocks. Partial, explicit and coarse/fine mappings retain the general inverse
  search, with cached structured counting. No global face-to-native array is
  added. In the scalar measurements, additional metadata was 336 bytes per rank
  at both tested sizes, with zero compatibility connectivity.
- Weighted scalar transport (including SST equations), its explicit
  nonorthogonal correction, physical momentum row assembly, transport-cache
  construction and pressure-face geometry use existing region execution views
  and temporary resolved faces. Existing cached face IDs and incidence order
  are reused; no additional retained face-metric table is introduced.
- Stored scalar/vector least-squares and Gauss gradients acquire bulk views.
  Cached centers still read authoritative owned values; neighbor/stencil/Gauss
  reads retain their overlap semantics. Output overlap is not implicitly
  published. Region Gauss kernels use transformed geometry and periodic images.
  The native `CellOperators.hh` paths already used bulk views and were retained.

Momentum transpose-stress and explicit momentum correction loops, primitive
face interpolation and several other generic consumers remain future work.

## CFD windows

The workload is the supplied SST/bubbly-convection case: two MPI ranks, three
fresh sequential repeats of 10 steps at 0.02 seconds, with the same meshes,
properties, PCG/DIC pressure solver and BiCGStab/SGS transport. Times are the
slowest-rank loop times, including diagnostics/output and excluding initialization
and merging. These initial 0.2-second windows do not qualify the complete
20-second schedule or developed flow.

All 18 principal candidate executions passed the unchanged OpenFOAM comparison
and conservation gates. Every exported CSV numeric value and the flow/gas
solve and iteration counts matched the retained SimpleFluid outputs exactly.
The new smoke and two separate profiled executions passed the same checks.

| Cells | Retained two-region median, s | Current native median, s | Current one-region median, s | Current two-region median, s |
| ---: | ---: | ---: | ---: | ---: |
| 10,752 | 18.817 | 2.848 | 3.059 | 3.141 |
| 98,304 | 211.831 | 38.398 | 48.864 | 47.343 |

The current two-region ranges were 3.128–3.281 seconds and 40.020–48.410 seconds.
The native 98k range was 37.324–48.135 seconds, so timing variation matters.
The retained baseline ratios are approximately 6.0 and 4.5; they compare
different measurement windows and are not contemporaneous paired medians.

Fresh executions of the **unchanged** old binary help characterize that drift:

| Control | Current measurement | Earlier observation |
| --- | ---: | ---: |
| Native, 98k, median of three | 41.654 s [40.833, 42.074] | 32.073 s |
| Two regions, 10k, one control | 20.466 s | 18.817 s median |
| Two regions, 98k, one control | 235.136 s | 211.831 s median |

All five refreshed controls passed the same numerical checks. The old binary
also became slower, so the historical native comparison does not establish a
native-path regression. Nor does the current native/control difference establish
a reliable native speedup. The much larger stitched-region improvement remains
visible in both the retained and refreshed controls. No new OpenFOAM timings
were taken; the physical comparison uses its retained outputs.

## Profile confirmation

Separate 199 Hz userspace CPU profiles use 65,528-byte DWARF stacks and the same
10k case. Percentages are conditional on a `BoussinesqSolver::step` ancestor,
not wall timers. Inclusive contexts overlap and must not be summed.

| Rank | Canonical cell/face conversion before | After |
| ---: | ---: | ---: |
| 0 | 0.83% | 1.34% |
| 1 | 70.61% | 3.66% |

The after rank-1 `native_face` ancestry alone is 3.47%. The combined measure uses
the same `native_face`/`native_cell` definition as the earlier profile. After
profiles contain 753/751 process samples, with about 69% covering solver steps;
the earlier profiles had 4,502/4,498 samples and 81–82% step coverage. No samples
were reported lost. The shorter runs and unwind coverage limit precision.
Analysis used local symbols with `DEBUGINFOD_URLS` empty to avoid remote symbol
lookup delays. Profiled timings are excluded from the performance medians.

## Scalar indexing control

The unchanged generic scalar diffusion consumer isolates the indexing work from
the new CFD consumers. Its two X-slab regions differ from the CFD Z-slab seam.
All 18 fresh runs passed the existing analytical, residual, checksum and storage
gates, with every result diagnostic—including iterations—exactly matching the
corresponding retained sample.

| Cells | Generic region assembly before → after | Current native | Current region-native |
| ---: | ---: | ---: | ---: |
| 13,824 | 161.84 → 28.50 ms | 13.88 ms | 6.29 ms |
| 97,336 | 1,196.91 → 211.14 ms | 103.83 ms | 33.98 ms |

The generic path improved about 5.7 times against retained measurements, while
provider traversal at 97k stayed approximately 3.53–3.55 ms. The unchanged native
scalar path was about 25% slower than its historical observation, consistent with
the need to distinguish measurement windows. Current reference/region-native
assembly ratios in the same binary were 4.53 and 6.21.

## Validation and evidence

- Indexing: exhaustive independently enumerated retained-face maps, all 64 full
  side-removal combinations, unequal/single-cell dimensions, signed patch
  parameterizations, partial/irregular fallbacks and periodic cases. Ten tests
  passed under LLVM Debug at 1/2/4 ranks and under GCC Release.
- New gradient and transport suites passed GCC Release and LLVM Debug at 1/2/4
  ranks. They cover exact reference parity, stale owned/overlap values, live
  storage aliasing, source callbacks, graph reuse, pressure metrics, six region
  families, ALE rollback/refresh and empty ranks. One overlap-alias fixture is
  deliberately serial-only.
- The affected GCC regression run passed 139 serial tests with three expected
  rank-conditioned skips, plus 12 selected MPI registrations. These include
  turbulence, existing region/lease tests, transport caches and coupled-backend
  parity. New gradient/transport suites are reported separately above. The full
  unrelated suite and hosted CI were not run.

Compact durable summaries are in [qualification/region_cfd_execution](qualification/region_cfd_execution/).
Full manifests, exact commands, hashes, unchanged input/reference checksums,
source patches/new source files, CSVs, logs and captured profiles remain under
`build/verification/region-native-20260911/`. The original protected artifacts
remain under `build/verification/region-operator-20260911/` and
`build/performance-51cbca3/`. The candidate uses the separate
`build/performance-region-native-20260911/` tree: GNU 16.2.1, Release/LTO, Kokkos
Serial, Open MPI 5.0.10 and the existing IF97 headers.

Reproduce with fresh output directories and the retained reference tree:

```sh
cmake --preset GCC-ninja-multi -B build/performance-region-native-20260911 \
  -DSIMPLEFLUID_ENABLE_IF97=ON \
  -DSIMPLEFLUID_IF97_INCLUDE_DIR="$PWD/build/gcc/_deps/simplefluid_if97-src" \
  -DSIMPLEFLUID_BUILD_BENCHMARKS=ON
cmake --build build/performance-region-native-20260911 --config Release \
  --target bottom_heated_bubbly_convection -j 4
source verification/environments.sh
SIMPLEFLUID_BUILD_DIR="$PWD/build/performance-region-native-20260911" \
  SIMPLEFLUID_BUILD_CONFIG=Release export_build_env "$PWD"
python3 verification/run_region_cfd_qualification.py \
  --reference-root build/verification/region-operator-20260911 \
  --build-dir build/performance-region-native-20260911 \
  --output build/verification/region-native-new \
  --cells 10752 98304 --variants piso iso1piso isopiso --repeats 3
```

The runner binds two ranks to physical cores and sets OpenMP/BLAS thread counts
to one. Use `--profile --repeats 1` in a separate output directory for profiles;
never overlap builds, tests, profile analysis or other timing jobs with measured
windows. The scalar runner and its independent reproduction commands are in the
retained `scalar/REPORT.md`.
