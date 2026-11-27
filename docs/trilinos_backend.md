On Linux, `SIMPLEFLUID_WRAP_STATIC_TRILINOS=ON` places the selected PIC static
Trilinos archives in `libSimpleFluidTrilinosBackend.so`. SimpleFluid libraries
and test executables link to this shared target. The backend is the only target
whose link command receives those archives; MPI and other external TPL
dependencies remain normal dependencies.

The backend exports the archive implementations with their original,
unversioned symbols; only its ownership marker uses
`SIMPLEFLUID_TRILINOS_BACKEND_1.0`. This permits dependency RTTI to coalesce with
consumer definitions as it does with shared Trilinos. This redistributes the selected
Trilinos ABI, not an opaque SimpleFluid-only API: existing templates and tests
can still call Teuchos, Tpetra, Kokkos, and other dependency functions directly.
Rebuild consumers when changing the Trilinos installation or C++ toolchain.

Compile requirements are copied from the selected imported-target closure
without forwarding its static link interface. Whole-archive inclusion makes
the selected implementations available even when SimpleFluid itself does not
reference them. This increases the size of the single backend library while
avoiding a separate archive copy in each executable. The existing per-compiler,
per-configuration build layout is preserved.

The backend is linked without LTO and rejects unresolved references. Consumer
libraries keep their SimpleFluid export boundary and the required dependency
RTTI visibility. The backend boundary test checks the ownership marker and,
with Ninja, verifies that only one link rule receives Trilinos archives.

On Linux with libc++, custom FVM implementation providers compiled with hidden
symbols still need default visibility for vendor declarations. The test
providers compile the same header prelude as the shared PCH when they bypass
PCH reuse. This lets Tpetra type information coalesce with the backend during
field imports while the custom implementation symbols stay hidden.

For libc++ builds using this backend, the FVM export audit checks the core
Teuchos/Tpetra/string RTTI in the backend and requires FVM's direct dependency
on that provider. FVM may omit unused RTTI, while any core RTTI it defines must
remain visible through its versioned bridge. The per-consumer RTTI export
allowlist and limit still apply; the backend's vendor ABI is checked separately.

Set `SIMPLEFLUID_WRAP_STATIC_TRILINOS=OFF` to use direct static linking. Shared
Trilinos installations continue to use their imported shared libraries. The
wrapper defaults off on platforms other than Linux.
