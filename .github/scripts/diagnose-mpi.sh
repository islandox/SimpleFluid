#!/usr/bin/env bash
# Diagnose a failed job without replacing its test result with a retry result.
set -uo pipefail

build_dir=$1
configuration=$2
probe="$build_dir/bin/$configuration/testMeshPartitioner"
diagnostics="$build_dir/mpi-diagnostics"
mkdir -p "$diagnostics"
if [[ ! -x "$probe" ]]; then
    printf 'MPI probe was not built: %s\n' "$probe" | tee "$diagnostics/environment.log"
    exit 0
fi

{
    mpiexec --version
    lscpu
    df -h /dev/shm
    ldd "$probe"
    ompi_info --parsable --param btl sm --level 9
} >"$diagnostics/environment.log" 2>&1

filter=MeshPartitionerTest.PartitionsRootOnlySourceGeometry
timeout 30s mpiexec -n 2 "$probe" --gtest_filter="$filter" --gtest_color=no \
    2>&1 | tee "$diagnostics/default-transport.log"
printf 'Default transport exit status: %s\n' "$?"

# Keep ASLR as in the original run. A rank's signal stops gdb before MPI can
# suppress the application backtrace; bound the run if another rank blocks.
timeout 60s mpiexec -n 2 gdb --quiet --batch --return-child-result \
    -ex 'set pagination off' -ex 'set disable-randomization off' \
    -ex run -ex 'x/8i $pc' -ex 'thread apply all bt 30' \
    --args "$probe" --gtest_filter="$filter" --gtest_color=no \
    2>&1 | tee "$diagnostics/backtrace.log"
printf 'Debugger exit status: %s\n' "$?"

# The test environment respects explicit transport overrides. This control
# distinguishes the shared-memory path from the same mesh code over TCP.
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp \
    timeout 30s mpiexec -n 2 "$probe" --gtest_filter="$filter" --gtest_color=no \
    2>&1 | tee "$diagnostics/tcp-transport.log"
printf 'TCP transport exit status: %s\n' "$?"
exit 0
