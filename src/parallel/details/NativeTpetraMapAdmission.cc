#include "parallel/details/NativeTpetraMapAdmission.hh"
#include "geometry/details/NativeAllocationGraph.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Tpetra_Details_Behavior.hpp>
#include <Tpetra_Details_Hash.hpp>
#include <Tpetra_Directory.hpp>
#include <cstdlib>
#include <cstring>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <map>
#include <mutex>
#include <optional>
#include <string>

extern char** environ;
namespace SimpleFluid::geometry_detail
{
namespace
{
using Admission = NativeConstructionAdmission;
using Profile = NativeTpetraMapAdmission;
using Pack = DefaultTpetraTypes;
using Map = Pack::map_type;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;
using Device = Pack::device_type;
using Hash = Tpetra::Details::Hash<GO, Device>;
using Offset = Hash::offset_type;
using Pair = Kokkos::pair<GO, LO>;
using Directory = Tpetra::Directory<LO, GO, Pack::node_type>;
using Inner = std::map<std::string, bool>;
using InnerNode = std::_Rb_tree_node<Inner::value_type>;
using Outer = std::map<std::string, Inner>;
using OuterNode = std::_Rb_tree_node<Outer::value_type>;
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > SIZE_MAX - a)
        throw std::invalid_argument("Native map profile sum overflows.");
    return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b)
{
    if (b && a > SIZE_MAX / b)
        throw std::invalid_argument("Native map profile extent overflows.");
    return a * b;
}
template<class... T> std::size_t sum(T... values)
{
    std::size_t result = 0;
    ((result = add(result, values)), ...);
    return result;
}
constexpr std::size_t align8(std::size_t n)
{
    return (n + 7) & ~std::size_t(7);
}
template<class T> constexpr std::size_t record_bytes()
{
    using Functor = Kokkos::Impl::ViewValueFunctor<Device, T>;
    return sizeof(Kokkos::Impl::SharedAllocationRecord<Kokkos::HostSpace, Functor>);
}
constexpr std::size_t header_bytes = sizeof(Kokkos::Impl::SharedAllocationHeader);
constexpr std::size_t map_label = sizeof("Tpetra::FixedHashTable::pairs");
// The longest retained label is copied into SharedAllocationRecord::m_label.
// Initialized counts/ptr/pairs also carry the destruction functor's name;
// two full labels are a finite upper bound for each such record.
constexpr std::size_t retained_labels_per_record = 2 * map_label;
struct LibraryContext
{
    std::shared_ptr<const NativeRetainedExtent> extent;
    std::shared_ptr<const NativeAllocationEnvelope> allocation, extent_envelope;
    std::byte profile_marker{};
    const void* serial_instance = nullptr;
    const void* serial_scratch = nullptr;
};
std::mutex library_mutex;
std::shared_ptr<const LibraryContext> environment_context, serial_context;
struct SerialScratch
{
    const void* instance;
    const void* scratch;
    std::size_t live_bytes;
    std::size_t map_upper;
};
SerialScratch observe_serial_scratch()
{
    constexpr std::size_t minimum_reduction = 512; // Kokkos 5.1.1 SerialInternal.
    const Kokkos::Serial space;
    auto* instance = space.impl_internal_space_instance();
    const std::lock_guard lock(instance->m_instance_mutex);
    const auto& data = instance->m_thread_team_data;
    const auto bytes = Kokkos::Impl::HostThreadTeamData::scratch_size(
        std::max(minimum_reduction, data.pool_reduce_bytes()), std::max(minimum_reduction, data.team_reduce_bytes()),
        data.team_shared_bytes(), data.thread_local_bytes());
    return {instance, data.scratch_buffer(), data.scratch_bytes(), bytes};
}
bool matches_serial_context(const LibraryContext& context, const SerialScratch& actual) noexcept
{
    return context.serial_instance == actual.instance && context.serial_scratch == actual.scratch &&
           context.extent->bytes() == actual.live_bytes;
}
std::shared_ptr<LibraryContext> prepare_context(const Admission& admission, std::size_t bytes)
{
    auto tracked = NativeAllocationEnvelopeAccess::make<LibraryContext>(admission);
    auto candidate = std::move(tracked.owner);
    candidate->allocation = std::move(tracked.envelope);
    auto phase = admission.reserve_profile({bytes});
    auto extent =
        NativeAllocationEnvelopeAccess::make<NativeRetainedExtent>(admission, phase.retain_profile(phase.bytes()));
    candidate->extent = std::move(extent.owner);
    candidate->extent_envelope = std::move(extent.envelope);
    return candidate;
}
constexpr std::size_t environment_context_bytes() noexcept
{
    return 2 * (sizeof(OuterNode) + sizeof(InnerNode));
}
std::shared_ptr<const LibraryContext> initialize_context(const Admission& admission)
{
    const std::lock_guard lock(library_mutex);
    if (environment_context)
        return environment_context;
    auto candidate = prepare_context(admission, environment_context_bytes());
    // These exact functions create two outer and two DEFAULT inner tree nodes
    // in Teuchos_EnvVariables.cpp. Retain the upper context until process exit.
    (void) Tpetra::Details::Behavior::debug("Map");
    (void) Tpetra::Details::Behavior::verbose("Map");
    environment_context = candidate;
    return candidate;
}
std::shared_ptr<LibraryContext> prepare_serial_context(const Admission& admission)
{
    const std::lock_guard lock(library_mutex);
    const auto observed = observe_serial_scratch();
    if (serial_context)
    {
        if (!matches_serial_context(*serial_context, observed))
            throw std::invalid_argument("Native map Serial runtime changed its captured scratch allocation.");
        return {};
    }
    auto candidate = prepare_context(admission, observed.map_upper);
    candidate->serial_instance = observed.instance;
    return candidate;
}
std::shared_ptr<const LibraryContext> capture_serial_context(const std::shared_ptr<LibraryContext>& pending)
{
    const std::lock_guard lock(library_mutex);
    const auto observed = observe_serial_scratch();
    if (serial_context)
    {
        if (!matches_serial_context(*serial_context, observed))
            throw std::invalid_argument("Native map Serial runtime changed its captured scratch allocation.");
    }
    else if (pending && observed.scratch)
    {
        if (pending->serial_instance != observed.instance || observed.live_bytes > pending->extent->bytes())
            throw std::invalid_argument("Native map Serial runtime exceeds its exact prepared scratch profile.");
        if (observed.live_bytes == pending->extent->bytes())
        {
            pending->serial_scratch = observed.scratch;
            serial_context = pending;
        }
    }
    return serial_context;
}
std::size_t environment_scratch()
{
    std::size_t longest = 0;
    for (auto** entry = environ; entry && *entry; ++entry)
        longest = std::max(longest, std::strlen(*entry));
    // Behavior::reject_unrecognized_env_vars splits one entry at a time.
    // token + name/value assign growth (at most 2*length for this STL).
    return add(multiply(5, longest), 5);
}
} // namespace
struct NativeMapAllocation
{
    explicit NativeMapAllocation(const Admission& a) : allocator(a.allocator<Map>()) {}
    ~NativeMapAllocation()
    {
        if (raw)
        {
            if (constructed)
                std::destroy_at(raw);
            allocator.deallocate(raw, 1);
        }
    }
    NativeConstructionAllocator<Map> allocator;
    Map* raw = nullptr;
    bool constructed = false;
    std::byte body_marker{}, control_marker{};
    std::shared_ptr<const NativeRetainedExtent> body;
    std::shared_ptr<const NativeAllocationEnvelope> body_envelope;
};
class NativeMapDelete
{
public:
    using ptr_t = Map;
    std::shared_ptr<NativeMapAllocation> allocation;
    std::shared_ptr<const NativeAllocationEnvelope> envelope;
    std::shared_ptr<const NativeRetainedExtent> control;
    std::shared_ptr<const NativeAllocationEnvelope> control_envelope;
    std::shared_ptr<const LibraryContext> environment, serial;
    void free(Map* map) const noexcept
    {
        if (allocation->raw != map)
            std::terminate();
        std::destroy_at(map);
        allocation->allocator.deallocate(map, 1);
        allocation->raw = nullptr;
        allocation->constructed = false;
        allocation->body.reset(); // actual dynamic map storage died first.
        allocation->body_envelope.reset();
    }
};
class NativeTpetraMapAdmission::Prepared
{
public:
    Prepared(Plan value, Admission a, const std::shared_ptr<Prepared>& same_batch)
        : plan(value), admission(std::move(a)),
          phase(admission.reserve_profile({sum(plan.retained_upper(), plan.scratch_upper())})),
          environment(initialize_context(admission))
    {
        if (same_batch && same_batch->pending_serial)
            pending_serial = same_batch->pending_serial;
        else if (plan.kind == Kind::Noncontiguous && plan.n)
            pending_serial = prepare_serial_context(admission);
        auto tracked = NativeAllocationEnvelopeAccess::make<NativeMapAllocation>(admission, admission);
        allocation = std::move(tracked.owner);
        envelope = std::move(tracked.envelope);
        auto body = NativeAllocationEnvelopeAccess::make<NativeRetainedExtent>(
            admission, phase.retain_profile(plan.retained_upper() - plan.rcp_node));
        allocation->body = std::move(body.owner);
        allocation->body_envelope = std::move(body.envelope);
        auto node =
            NativeAllocationEnvelopeAccess::make<NativeRetainedExtent>(admission, phase.retain_profile(plan.rcp_node));
        control = std::move(node.owner);
        control_envelope = std::move(node.envelope);
        allocation->raw = admission.allocator<Map>().allocate(1); // last throwing native preparation.
        plan.native_receipts = sum(envelope->fixed_allocation_bytes(),
            allocation->body_envelope->fixed_allocation_bytes(), control_envelope->fixed_allocation_bytes());
    }
    Plan plan;
    Admission admission;
    NativePhaseReservation phase;
    std::shared_ptr<const LibraryContext> environment;
    std::shared_ptr<LibraryContext> pending_serial;
    std::shared_ptr<NativeMapAllocation> allocation;
    std::shared_ptr<const NativeAllocationEnvelope> envelope;
    std::shared_ptr<const NativeRetainedExtent> control;
    std::shared_ptr<const NativeAllocationEnvelope> control_envelope;
};
std::size_t NativeTpetraMapAdmission::Plan::retained_upper() const
{
    return sum(directory_object, rcp_node, gid_data, hash_offsets, hash_pairs, retained_kokkos_headers,
        retained_kokkos_records, retained_labels, retained_teuchos_bookkeeping);
}
std::size_t NativeTpetraMapAdmission::Plan::scratch_upper() const
{
    return sum(temporary_gid_copy, temporary_noncontiguous_gids, temporary_hash_counts, temporary_minmax,
        temporary_kokkos_headers_records_labels, temporary_comm_helpers, temporary_profiling_helpers);
}
std::size_t NativeTpetraMapAdmission::library_context_bytes() noexcept
{
    const std::lock_guard lock(library_mutex);
    return environment_context_bytes() + (serial_context ? serial_context->extent->bytes() : 0);
}
std::size_t NativeTpetraMapAdmission::retained_upper(const Teuchos::RCP<const Map>& map)
{
    const auto& receipt = Teuchos::get_dealloc<NativeMapDelete>(map);
    return add(receipt.allocation->body->bytes(), receipt.control->bytes());
}
void NativeTpetraMapAdmission::capture_context_prefix(
    PLIC::details::SealedSourcePrefix& prefix, const std::shared_ptr<const void>& arena)
{
    const std::lock_guard lock(library_mutex);
    if (serial_context && (!Kokkos::is_initialized() ||
            !matches_serial_context(*serial_context, observe_serial_scratch())))
        throw std::invalid_argument("Native map prefix observes a changed Serial scratch allocation.");
    for (const auto& context : {environment_context, serial_context})
    {
        if (!context) continue;
        NativeConstructorOwnerSink::require_envelope(context->allocation, context);
        NativeConstructorOwnerSink::require_envelope(context->extent_envelope, context->extent);
        const auto& state = context->allocation->state();
        if (state != context->extent_envelope->state() || state != context->extent->state())
            throw std::invalid_argument("Native map runtime changed its paired profile allocation domains.");
        const auto counter_bytes = NativeAllocationGraphAccess::validate_state(state);
        const auto& source = state->mirror.owner;
        const bool same_arena = arena && arena.get() == source.get() && !source.owner_before(arena) &&
                                !arena.owner_before(source);
        const bool mirrored_by_source = same_arena && state->mirror.owner_proof.publication_requests();
        prefix.include_library_context_owner(context, 0, false);
        prefix.include_library_context_owner(context->extent, 0, false);
        prefix.include_library_context_owner(context->allocation, context->allocation->fixed_allocation_bytes(),
            mirrored_by_source);
        prefix.include_library_context_owner(context->extent_envelope,
            context->extent_envelope->fixed_allocation_bytes(), mirrored_by_source);
        const std::shared_ptr<const void> profile_alias(context->extent, &context->profile_marker);
        prefix.include_library_context_owner(profile_alias, context->extent->bytes(), mirrored_by_source);
        prefix.include_library_context_owner(state, state->allocation_admission_bytes, mirrored_by_source);
        prefix.include_library_context_owner(state->counter_owner, counter_bytes, mirrored_by_source);
        if (source)
            prefix.include_library_context_owner(source, state->mirror.owner_proof.bytes(), same_arena);
    }
}
bool NativeTpetraMapAdmission::profile_supported() noexcept
{
    if (!native_constructor_capacity_profile_supported()) return false;
#if KOKKOS_VERSION != 50101 || !defined(KOKKOS_ENABLE_SERIAL) || defined(KOKKOS_ENABLE_DEBUG) ||                       \
    defined(HAVE_TPETRA_DEBUG) ||                                                                                  \
    defined(HAVE_TEUCHOS_DEBUG) || defined(HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING) ||                                     \
    defined(HAVE_TEUCHOS_COMM_TIMERS) || defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS) ||                                   \
    defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE) || defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) ||                        \
    defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS) || defined(KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE) ||                        \
    defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
    return false;
#endif
    if constexpr (!std::is_same_v<Pack::memory_space, Kokkos::HostSpace> ||
                  !std::is_same_v<Pack::execution_space, Kokkos::Serial>)
        return false;
    if (!Kokkos::is_initialized() || Kokkos::Profiling::profileLibraryLoaded()) return false;
    for (auto** entry = environ; entry && *entry; ++entry)
        if (std::strncmp(*entry, "TPETRA_", 7) == 0 || std::strncmp(*entry, "KOKKOS_TOOLS_", 13) == 0 ||
            std::strncmp(*entry, "KOKKOS_PROFILE_", 15) == 0)
            return false;
    const std::lock_guard lock(library_mutex);
    if (serial_context && !matches_serial_context(*serial_context, observe_serial_scratch()))
        return false;
    return true;
}
void NativeTpetraMapAdmission::require_profile()
{
    if (!profile_supported())
        throw std::invalid_argument("Unsupported native map compiled/runtime allocation profile.");
}
NativeTpetraMapAdmission::Plan NativeTpetraMapAdmission::plan(Kind kind, std::span<const GO> gids)
{
    require_profile();
    Plan p;
    p.kind = kind;
    p.n = gids.size();
    p.map_object = sizeof(Map); // actual native allocator, excluded from upper.
    p.directory_object = sizeof(Directory);
    p.rcp_node = sizeof(Teuchos::RCPNodeTmpl<Map, NativeMapDelete>);
    p.fingerprint = 0x146f710b6d64b4feULL;
    // The empty Directory owns its actual Teuchos RCP node.
    // No Directory initialization / Distributor / remote lookup is performed.
    p.retained_teuchos_bookkeeping = sizeof(Teuchos::RCPNodeTmpl<Directory, Teuchos::DeallocDelete<Directory>>);
    p.temporary_comm_helpers =
        std::max(sizeof(Teuchos::SumValueReductionOp<int, GO>), sizeof(Teuchos::MaxValueReductionOp<int, GO>)) +
        sizeof(Teuchos::RCPNodeTmpl<Teuchos::ValueTypeReductionOp<int, GO>,
            Teuchos::DeallocDelete<Teuchos::ValueTypeReductionOp<int, GO>>>);
    p.temporary_profiling_helpers =
        sum(environment_scratch(), sizeof(InnerNode), 2 * retained_labels_per_record, std::size_t(64));
#if defined(HAVE_TEUCHOS_KOKKOS_PROFILING) && defined(HAVE_TEUCHOSCORE_KOKKOS)
    // Three nested CommHelpers/MpiComm scope strings. GCC16 stringbuf starts
    // at 512 bytes; every fixed DefaultPack scope label is below 128 chars.
    // Each scope can hold both .str() and ScopedRegion's string conversion.
    p.temporary_profiling_helpers = add(p.temporary_profiling_helpers, 3 * (513 + 2 * 129));
#endif
    if (kind == Kind::ContiguousOwned)
        return p;
    if (p.n)
    {
        std::size_t i = 1;
        while (i < p.n && gids[i - 1] != std::numeric_limits<GO>::max() && gids[i] == gids[i - 1] + 1)
            ++i;
        p.m = p.n - i;
        // FixedHashTable finds a second initial contiguous run in nonContigGids.
        std::size_t run = 0;
        if (p.m)
        {
            run = 1;
            while (run < p.m && gids[i + run - 1] != std::numeric_limits<GO>::max() &&
                   gids[i + run] == gids[i + run - 1] + 1)
                ++run;
        }
        const auto start_index = run ? run - 1 : 0; // exact inclusive endpoint rule.
        p.k = p.m - start_index;
        if (p.k > std::size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Native map hash key count exceeds installed int hash range.");
        p.b = Hash::getRecommendedSize(static_cast<Offset>(p.k));
        p.gid_data = align8(multiply(p.n, sizeof(GO)));
        p.hash_offsets = align8(multiply(add(p.b, 1), sizeof(Offset)));
        p.hash_pairs = align8(multiply(p.k, sizeof(Pair)));
        p.temporary_noncontiguous_gids = align8(multiply(p.m, sizeof(GO)));
        p.temporary_hash_counts = align8(multiply(p.b, sizeof(Offset)));
        // lgMap and ptr exist even for an empty hash; pairs/count/nonContig
        // have no allocation when their requested span is zero.
        const std::size_t retained_records = 2 + bool(p.k);
        p.retained_kokkos_headers = multiply(retained_records, header_bytes);
        p.retained_kokkos_records = sum(record_bytes<GO>(), record_bytes<Offset>(), p.k ? record_bytes<Pair>() : 0);
        p.retained_labels = multiply(retained_records, retained_labels_per_record);
        const std::size_t scratch_records = bool(p.m) + bool(p.b);
        p.temporary_kokkos_headers_records_labels = multiply(scratch_records,
            sum(header_bytes, std::max(record_bytes<GO>(), record_bytes<Offset>()), retained_labels_per_record));
    }
    // Noncontiguous constructors always compute global constants, including
    // empty ranks. The two min/max HostSpace arrays never alias each other.
    p.temporary_minmax = 2 * align8(3 * sizeof(GO));
    p.temporary_kokkos_headers_records_labels = add(p.temporary_kokkos_headers_records_labels,
        2 * sum(header_bytes, record_bytes<GO>(), retained_labels_per_record));
    return p;
}
std::shared_ptr<NativeTpetraMapAdmission::Prepared> NativeTpetraMapAdmission::prepare(
    const Plan& plan, const Admission& admission, const std::shared_ptr<Prepared>& same_batch)
{
    require_profile();
    if (same_batch && (same_batch->admission.state().get() != admission.state().get() ||
            same_batch->admission.state().owner_before(admission.state()) ||
            admission.state().owner_before(same_batch->admission.state())))
        throw std::invalid_argument("Native map preparation batch changes its actual admission domain.");
    return admission.shared<Prepared>(plan, admission, same_batch);
}
NativeTpetraMapAdmission::Owner NativeTpetraMapAdmission::construct(Prepared& prepared, std::span<const GO> gids,
    Tpetra::global_size_t global, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    if (comm.is_null() || !prepared.allocation->raw || gids.size() != prepared.plan.n ||
        global == Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid())
        throw std::invalid_argument("Native map construction differs from its prepared profile.");
    auto* raw = prepared.allocation->raw;
    try
    {
        if (prepared.plan.kind == Kind::ContiguousOwned)
            std::construct_at(raw, global, gids.size(), GO{}, comm);
        else
            std::construct_at(raw, global, Teuchos::ArrayView<const GO>(gids.data(), gids.size()), GO{}, comm);
        prepared.allocation->constructed = true;
    }
    catch (...)
    {
        (void) capture_serial_context(prepared.pending_serial);
        throw;
    }
    Teuchos::RCP<const Map> map;
    Meshes::AMR::adaptive_detail::collective(*comm, [&]
    {
        const auto serial = capture_serial_context(prepared.pending_serial);
        NativeMapDelete deleter{prepared.allocation, prepared.envelope, std::move(prepared.control),
            std::move(prepared.control_envelope), prepared.environment, serial};
        map = Teuchos::rcpWithDealloc(raw, std::move(deleter), true);
    });
    prepared.phase.finish_scratch();
    return {std::move(map), prepared.plan};
}
void NativeTpetraMapAdmission::visit_control(const NativeMapDelete& receipt,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    using Graph = Meshes::AMR::adaptive_detail::NativePublicationOwnerGraph;
    Graph::Context context(graph, retain, bool(accounting), accounting);
    const auto& allocation = receipt.allocation;
    if (!allocation || !receipt.envelope || !receipt.envelope->matches(allocation) || !receipt.control ||
        !receipt.control_envelope || !receipt.control_envelope->matches(receipt.control))
        throw std::invalid_argument("Actual native map has a foreign weak deallocator/control receipt.");
    if (!context.sink.semantic(allocation, NativeConstructorOwnerKind::NativeMesh, receipt.envelope))
        return;
    context.sink.fixed(receipt.envelope);
    graph.native_state(allocation->allocator.state(), retain, bool(accounting));
    context.sink.fixed(receipt.control_envelope);
    if (context.sink.semantic(receipt.control, NativeConstructorOwnerKind::Storage, receipt.control_envelope))
        graph.native_state(receipt.control->state(), retain, bool(accounting));
    const auto control_alias = std::shared_ptr<const void>(receipt.control, &allocation->control_marker);
    if (accounting)
        accounting->authorize_profile_owner(0, control_alias, receipt.control, receipt.control_envelope);
    graph.external(control_alias, receipt.control->bytes(), retain, bool(accounting));
    for (const auto& observed : {receipt.environment, receipt.serial})
    {
        if (!observed || !context.sink.semantic(observed, NativeConstructorOwnerKind::Storage, observed->allocation))
            continue;
        context.sink.fixed(observed->allocation);
        context.sink.fixed(observed->extent_envelope);
        if (context.sink.semantic(observed->extent, NativeConstructorOwnerKind::Storage, observed->extent_envelope))
            graph.native_state(observed->extent->state(), retain, bool(accounting));
        const std::shared_ptr<const void> alias(observed->extent, &observed->profile_marker);
        if (accounting)
            accounting->authorize_profile_owner(0, alias, observed->extent, observed->extent_envelope);
        graph.external(alias, observed->extent->bytes(), retain, bool(accounting));
    }
}
bool NativeTpetraMapAdmission::admits(const Teuchos::RCP<const Map>& map)
{
    if (map.is_null() || !map.is_valid_ptr()) return false;
    const auto receipt = Teuchos::get_optional_dealloc<NativeMapDelete>(map);
    if (receipt.is_null()) return false;
    const auto& value = receipt->allocation;
    return value && value->raw == map.getRawPtr() && value->constructed && receipt->envelope &&
           receipt->envelope->authentic_owner(receipt->envelope) && receipt->envelope->matches(value) &&
           value->body && value->body_envelope && value->body_envelope->matches(value->body) &&
           receipt->control && receipt->control_envelope && receipt->control_envelope->matches(receipt->control);
}
void NativeTpetraMapAdmission::visit(const Teuchos::RCP<const Map>& map, PLIC::details::GeometryOwnerGraphEditor& graph,
    bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (map.is_null())
        throw std::invalid_argument("Native map visitor received an empty resource.");
    const auto& receipt = Teuchos::get_dealloc<NativeMapDelete>(map);
    const auto& allocation = receipt.allocation;
    if (!allocation || allocation->raw != map.getRawPtr() || !allocation->body || !allocation->body_envelope)
        throw std::invalid_argument("Actual native map has a foreign live body receipt.");
    visit_control(receipt, graph, retain, accounting);
    // The actual map object is a separate, conditional lifetime from its RCP
    // node/deallocator header. A weak receipt encountered first cannot suppress
    // this later strong body traversal, nor keep that body alive after free().
    if (accounting)
        accounting->authorize_buffer(PLIC::details::NativeActualBufferProof(
            allocation, receipt.envelope, allocation->allocator.state(), {allocation->raw, sizeof(Map)}));
    if (!graph.external(
            std::shared_ptr<const void>(allocation, allocation->raw), sizeof(Map), retain, bool(accounting)))
        return;
    using Graph = Meshes::AMR::adaptive_detail::NativePublicationOwnerGraph;
    Graph::Context context(graph, retain, bool(accounting), accounting);
    context.sink.fixed(allocation->body_envelope);
    if (context.sink.semantic(allocation->body, NativeConstructorOwnerKind::Storage, allocation->body_envelope))
        graph.native_state(allocation->body->state(), retain, bool(accounting));
    const auto body_alias = std::shared_ptr<const void>(allocation->body, &allocation->body_marker);
    if (accounting)
        accounting->authorize_profile_owner(0, body_alias, allocation->body, allocation->body_envelope);
    graph.external(body_alias, allocation->body->bytes(), retain, bool(accounting));
}
void NativeTpetraMapAdmission::visit_weak(const Teuchos::RCP<const Map>& map,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (map.is_null())
        throw std::invalid_argument("Native weak map receipt is empty.");
    // get_dealloc uses the RCP node, including after the target has expired;
    // it does not dereference or promote that target (installed Teuchos_RCP.hpp).
    visit_control(Teuchos::get_dealloc<NativeMapDelete>(map), graph, retain, accounting);
}
} // namespace SimpleFluid::geometry_detail
