# Debug-symbol policy fixture

This standalone CMake project uses the production debug-symbol helper without
SimpleFluid's numerical dependencies. The Python runner needs CMake, Ninja,
GCC and/or Clang with LLD. It uses only the Python standard library. GDB checks
run when GDB is available; absence is recorded as a skip.

```sh
python3 cmake/testing/debug_symbols/run.py \
    --build-root /tmp/simplefluid-debug-symbols-check \
    --toolchain gcc --toolchain clang-lld --parallel 2
```

Use `--gcc-cxx`, `--clang-cxx`, or `--gdb` to select tool paths. The build root
must be new or empty, so artifacts cannot be mistaken for fresh evidence.
`summary.json` and individual command logs remain in that directory.

The default policy is checked in Debug, RelWithDebInfo and Release. Separate
split-disabled and compression-disabled builds check both debug configurations.
Target fixtures cover general and configuration-specific IPO, executable,
shared, module, static, object, and imported targets. Checks inspect generated
options, actual compile/link commands, ELF compression headers and decompressed
payloads, executable behavior, retained `.dwo` files, and GDB source breakpoints.
Configure-only negative cases cover unsupported platform defaults/opt-in and a
compiler wrapper that rejects the real compression linker flag.

This is focused build-policy validation, not a full SimpleFluid build, numerical
qualification, MPI validation, or a size/performance benchmark.
