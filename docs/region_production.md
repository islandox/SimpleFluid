# Region diffusion production qualification

Scalar diffusion assembly now dispatches through the composite execution view.
The retained ordinal-query kernel remains available to the benchmark as
`reference`; `auto` selects the production dispatch. Both use canonical field
maps, the same coefficient formulas and Tpetra assembly. Compatibility CSR
materialization remains an explicit option and now uses amortized vector growth.

## Recorded Release measurements, 2026-09-11

All 108 current-code benchmark processes passed: 52 in the
[N/R/P matrix](qualification/region_production/after/report.md), 48 in the
[three-sample region study](qualification/region_production/region_repeats/report.md),
and eight in the [large case](qualification/region_production/large/report.md).
The two flattened MPI configurations remain explicitly unsupported. Every
executed true relative residual was below `9.97e-11` and every analytical
solution error was below `2.74e-9`; compact storage and traversal checksum gates also
passed. Builds and correctness runs had finished before these measurements.

The compiler was GNU 16.2.1 Release, using Open MPI 5.0.10 on an Intel i5-12500
under WSL2. MPI used `--bind-to core`; serial processes inherited the host CPU
affinity. All three runs recorded unchanged executable and project-library
hashes. The executable SHA-256 was
`9117845fb9591fd7f5d748b2e76cc79703332011a15c1e4e949f20107b5a409f`; the
shared-library SHA-256 and source snapshot are in each manifest. The source
identity is base `a8eb8bb1584570195b9feef28cd5084772b589dd` plus the retained
working-tree changes.

These measurements predate the final steady-helper correction for zero physical
load. That helper is not called by this benchmark, and the scalar assembly code
was unchanged by that correction. The recorded hashes and source snapshots
identify the measured artifacts; subsequent steady-helper validation is recorded
in [its convergence report](steady_diffusion_convergence.md).

At 32,768 cells and one rank, repeated composite assembly timings were as
follows. Brackets give the three-sample minimum and maximum, in milliseconds.
Native controls also used three samples per region-count configuration.

| R | Reference ms | Production ms | Reference / production | Native median ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 103.142 [101.249, 106.008] | 19.370 [18.917, 19.501] | 5.32 | 31.489 |
| 2 | 384.233 [376.521, 395.368] | 23.567 [22.565, 24.776] | 16.30 | 31.697 |
| 8 | 571.518 [569.199, 574.171] | 25.428 [25.052, 25.759] | 22.48 | 31.765 |
| 32 | 589.533 [581.959, 594.165] | 29.128 [28.750, 29.527] | 20.24 | 31.318 |

The production path removes most of the measured ordinal-query assembly cost.
The remaining region-count dependence is visible, and the generic traversal
diagnostic remains expensive. These timings measure assembled scalar diffusion
on the static conforming Cartesian fixture. Moving geometry, translated
periodicity and coarse/fine equivalence are established by the separate
correctness regressions below.

At 262,144 cells, two regions and one rank, the single-sample results were:

| Representation | Structural mesh KiB | Process peak MiB | Construction s | Production assembly s | Reference assembly s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native compact | 106.422 | 138.039 | 0.00150 | 0.26521 | 0.25359 |
| Compact composite | 109.625 | 156.129 | 0.11860 | 0.18942 | 3.50381 |
| Forced compatibility | 41,749.180 | 439.094 | 0.77377 | 0.16939 | 3.48453 |
| Flattened unstructured | 197,573.945 | 408.539 | 0.89922 | 0.25278 | 0.24446 |

Storage, construction and peak RSS columns use the production process. The
reference assembly column comes from a separate process with the same binary.
Native and flattened select the same kernel in both modes; their differences
show measurement variation. The composite's 3.50381-to-0.18942 second comparison
is an 18.50 ratio and 94.59% assembly-time reduction. The historical 4.264 second
composite assembly in the September 10 report used a different retained binary
and measurement run, so it is not the denominator for this speed ratio.

Forced compatibility construction now takes 0.774 seconds in this large case;
the September 10 record was 218.384 seconds. At 32,768 cells, the freshly repeated
old-binary context was 0.442 seconds and the current single sample was 0.079
seconds. These include indexer construction as well as CSR. The independent
allocation-volume regression directly guards the repaired prefix-copy
pathology. Geometric growth retains spare CSR capacity, which is reflected in
the larger measured compatibility payload; it does not impose a six-face bound
on coarse/fine cells.

The single-sample strong-scaling comparison at 32,768 cells and two regions was:

| P | Reference assembly ms | Production assembly ms | Native assembly ms |
| ---: | ---: | ---: | ---: |
| 1 | 374.853 | 21.926 | 31.960 |
| 2 | 331.962 | 11.819 | 26.387 |
| 4 | 201.567 | 9.392 | 18.412 |

Construction still includes the existing replicated descriptors and canonical
partition/map setup. These small fixed-size runs qualify the assembly change;
they do not establish scalable construction or a new distributed layout.

## Measurement method

The benchmark accepts an optional final `auto|reference` argument. The runner's
`--assemblies auto reference` runs each selection in its own fresh process for
each of the native, composite, materialized and flattened baselines. This keeps
peak RSS independent of another representation or an earlier assembly. MPI
flattened references remain explicitly unsupported.

Each timing in the generated comparison is a median of maximum-rank elapsed
times. The reported comparisons have distinct denominators:

- Reference speed ratio: `T_reference / T_auto`.
- Assembly time reduction: `(T_reference - T_auto) / T_reference`.
- Remaining native ratio: `T_auto / T_native`.
- Fraction of excess over native removed:
  `(T_reference - T_auto) / (T_reference - T_native)`, defined only when
  `T_reference > T_native`. This fraction may exceed 100%.

Normalized nanoseconds per owned cell are
`1e9 * maximum_rank_time / maximum_owned_cell_count`. The two maxima need not
belong to the same rank. They do not represent total CPU time or simultaneous
memory usage. Single samples and small grids are diagnostics; repeated medians
still do not establish whole-solver scaling or a universal performance bound.

## Reproduction and retained artifacts

```sh
export SIMPLEFLUID_BUILD_CONFIG=Release
. verification/environments.sh
export_build_env "$PWD"
cmake --preset "$SIMPLEFLUID_CONFIGURE_PRESET"
simplefluid_build_target region_scaling_benchmark 4
python3 verification/run_region_scaling.py \
  --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-production-new \
  --cells 16,32 --regions 1,2,8,32 --ranks 1,2,4 \
  --fixed-cells 32 --fixed-regions 2 --fixed-ranks 1 \
  --samples 1 --repeats 3 --assemblies auto reference \
  --mpi-args '--bind-to core' --discard-vtu
python3 verification/run_region_scaling.py \
  --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-production-repeat-new \
  --cells 32 --regions 1,2,8,32 --ranks 1 \
  --fixed-cells 32 --fixed-regions 2 --fixed-ranks 1 \
  --samples 3 --repeats 5 --modes native composite \
  --assemblies auto reference --discard-vtu
python3 verification/run_region_scaling.py \
  --binary "$SIMPLEFLUID_BIN_DIR/region_scaling_benchmark" \
  --output /tmp/region-production-large-new \
  --cells 64 --regions 2 --ranks 1 \
  --fixed-cells 64 --fixed-regions 2 --fixed-ranks 1 \
  --samples 1 --repeats 5 --assemblies auto reference --discard-vtu
```

The runner records executable and project shared-library SHA-256 hashes,
tracked source hashes, the source revision and dirty status, source patches and
new source files, CMake cache contents and build configuration hashes, exact
commands, environment, raw JSONL/stderr and CSV summaries. `--discard-vtu`
retains the size and hash of newly generated VTU files while omitting their
contents from the durable report. Generation and writing still occur inside the
measured output phase.

The [before snapshot](qualification/region_production/before/report.md) contains
20 fresh runs of the retained previous-milestone executable on September 11,
2026. Its SHA-256 is
`0566a82b6bd1aa0bec34e3afaf31e2ef7d53affbc00645099b39faff70674550`, matching
the September 10 scaling report. Its embedded source identity is
`a1a5a7a77ab0` plus dirty changes. The working source HEAD was
`a8eb8bb1584570195b9feef28cd5084772b589dd` when these runs began; the retained
binary was not rebuilt from that clean commit. Consequently this snapshot is
context for the prior milestone, while same-binary `reference`/`auto` results
isolate the current dispatch difference. Its VTU contents were omitted from the
repository, with hashes and sizes in the manifest.

## Focused correctness checks

The initial GCC Release regression snapshot passed 75 focused CTest registrations with one intentional serial
rank-gated skip, including 11 MPI registrations. The separately executed
production assembly executable passed four serial tests. These checks exercise
resolved geometry, assembly equality and boundary callback ordering, motion and
rollback, periodic and coarse/fine interfaces, existing leases and interface
indexes, map/halo behavior, and the actual steady diffusion helper. See the
[validation manifest](qualification/region_production/validation/gcc-focused.json)
and [raw CTest log](qualification/region_production/validation/gcc-focused-LastTest.log).

The [LLVM Debug regression snapshot](qualification/region_production/llvm_debug/validation.json)
passed 86 other selected serial tests with three rank-conditioned skips, plus
10 selected MPI registrations. This includes the production/reference assembly,
motion, periodic and coarse/fine cases, native/mapped nonorthogonal operators,
and 12 composite coupled-backend cases.

After the final physical-load normalization correction, the actual steady-helper
suite was rebuilt and passed all nine tests at one, two and four ranks in both
GNU Release and LLVM Debug. Its separate [final reports](steady_diffusion_convergence.md)
record those binaries and the homogeneous/tiny-load checks. This distinguishes
the changed-helper validation from the earlier general-regression snapshots.

The materialization executable passed all three tests in serial and at two and
four ranks. It checks canonical CSR order, unchanged maps, repeated
materialization, coarse/fine logical-face expansion, and requested allocation
volume bounded by final connectivity storage. The allocation check also verifies
that the counter observes the final CSR allocation; it uses no timing threshold.
Its [manifest and logs](qualification/region_production/validation/materialization.json)
retain the executable hash and exact commands.

The existing [GCC Release accuracy study](qualification/region_production/accuracy_gcc_release/accuracy.json)
and [LLVM Debug accuracy study](qualification/region_production/llvm_accuracy/accuracy.json)
passed all three tests at one, two and four ranks. The largest parsed serial/MPI
diagnostic difference was `1.0000004296850662e-12`, within the established
runner tolerance `1e-9`. These studies are numerical regressions of the specified
fixtures; they do not establish general physical validation or hosted CI status.
