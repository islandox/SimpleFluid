# AMR requested-heap attribution

Stage 6 measures successful C++ allocation requests, not malloc traffic, allocator overhead, RSS, or MPI device memory.
The allocation event establishes its domain for its entire lifetime. A publication phase or a generation retaining a
pointer is not an allocation domain. Do not tag a whole project constructor: plain nested allocations must remain
visible as escapes. Only an actual upstream allocator call, or a narrow third-party constructor, sets the thread-local
domain. Nested scopes restore the previous value on return or exception.

`SIMPLEFLUID_ENABLE_ALLOCATION_AUDIT` is a CMake cache string. The default, `AUTO`, enables attribution only for
`Debug`; `Release` and `RelWithDebInfo` build without it. Set `-DSIMPLEFLUID_ENABLE_ALLOCATION_AUDIT=ON` to investigate
escapes in an optimized build, or `OFF` to disable it even in Debug. The selection is per build configuration;
all translation units using these internal types must agree on the setting.

When disabled, both domain allocator types are aliases of `std::allocator`. There is no account pointer, default
account, mutex, TLS lookup, compiler fence, or test bridge. An explicit `AllocationAccount(limit)` is an audit-only
test utility, not a production budget; it is unavailable rather than silently ignoring a supplied limit. Native
admission, contact arena budgets, paid receipts, and their rejection and lifetime tests remain active. The test-only
raw heap interposer can still check physical extents and phases without recording allocation domains.
The native lifetime test looks up addresses in a separate translation unit so an optimizing caller cannot fold the
lookup to false on the assumption that a freshly allocated pointer cannot already appear in the interposer table.

| Storage | Attribution and admission |
| --- | --- |
| Leaf pages, saved families and their ordered pages | Rebind a domain allocator for private buffers, tree nodes and shared controls. Preserve the public record and range types. |
| Contact halo | Tag private index storage and shared controls. Keep `ContactGroupAllocation` and its `Budget` as the existing arena admission authority; do not debit it twice. |
| Geometry snapshots and owner unions | Distinguish immutable snapshot storage from owner index administration. Preserve owner equality, release semantics and native paid receipts. |
| Native publication | Tag bootstrap, construction and administration allocator requests below existing admission. Preserve strong versus borrowed state lifetimes, mirror accounting, and failure rollback. |
| External libraries | Tag only calls into Teuchos/Tpetra/Kokkos, not project-side input preparation. |

Private allocator state survives rebind and must outlive its allocations, including the last weak shared control.
Default domain accounts have static lifetime; an explicitly supplied account is borrowed. An accepted reservation is
pending until upstream allocation succeeds, so an upstream exception changes neither successful counts nor peak. No
account lock is held while calling the heap. For fixed-layout stores already covered by admission, an empty allocator
adapter tags requests without increasing the size of a vector or a funded control. Its exact bytes/counts are in the
interposer, not in a second admission account. Requested bytes, including the implementation's
actual control-block/node type, are counted on successful allocation and retired by the allocator on deallocation.
Audit counters are observational and distinct from arena credits, profile bounds and cumulative admission limits.
Keep allocator implementation and its types hidden; no new shared-library API or public allocator parameters.
An ELF test executable supplies an optional bridge for image-local allocator code to reach the interposer thread's
domain; an ordinary production executable does not depend on a test library. Compiler fences around the scoped call
prevent optimized builds from eliminating the tag on the assumption that replaceable allocation cannot observe it.

The interposer stores the allocation-time domain in its fixed table. Each live extent belongs to exactly one project,
external, or unattributed bin regardless of aliases, current phase or thread of deletion. An unscoped measured C++
request is an escape from this instrumentation, not proof of a memory leak or even of project origin if an external
caller or test fixture was not isolated. Generation coexistence and release use identity/weak probes independently;
domain totals do not establish which generations own an extent.
