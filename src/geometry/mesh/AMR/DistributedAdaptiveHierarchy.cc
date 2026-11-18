/** @file DistributedAdaptiveHierarchy.cc
 * @brief Distributed complete-family transactions with immutable local state.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/details/NativeAllocationGraph.hh"
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/mesh/AMR/AdaptiveContactSnapshot.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotInternal.hh"
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/AdaptiveLeafOverlay.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DirectoryRegistrationAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNativeRetention.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveOwnerOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativePublicationOperation.hh"
#include "geometry/mesh/AMR/NativeDirectoryAccess.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/ProductFamilyElectionState.hh"
#include "parallel/details/NativePartitionAccess.hh"
#include <Tpetra_Map.hpp>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <set>
#include <variant>
namespace SimpleFluid::Meshes::AMR
{
using namespace adaptive_detail;
namespace
{
constexpr auto family_layout_header_bytes = sizeof(AdaptiveFamilyLayout) + 4 * sizeof(void*);
struct RejectGeometryAdmission
{
    PLIC::GeometryAllocationLedger* ordinary;
    PLIC::GeometryAllocationLedger* bounded;
    int exceptions = std::uncaught_exceptions();
    ~RejectGeometryAdmission()
    {
        if (std::uncaught_exceptions() > exceptions)
        {
            if (ordinary)
                PLIC::details::GeometryAllocationOwnerAccess::poison(*ordinary);
            if (bounded)
                PLIC::details::GeometryAllocationOwnerAccess::poison(*bounded);
        }
    }
};
struct ScopedSourceAdmission
{
    const std::size_t prefix;
    geometry_detail::NativeConstructionAdmission admission;

    ScopedSourceAdmission(std::size_t limit, std::size_t already_admitted, std::shared_ptr<Budget> arena = {})
        : prefix(already_admitted),
          admission(geometry_detail::NativeConstructionCounterAccess::create<Budget>(limit, prefix,
              arena ? NativeSourceOwnerStore::mirror(arena) : geometry_detail::NativeAllocationMirror{}))
    {
    }
    ~ScopedSourceAdmission() { admission.release_prefix(prefix); }
};
} // namespace
// Keeps hierarchy-local admission alive exactly as long as its immutable Data.
struct AdaptivePublishedAdmission
{
    std::shared_ptr<DistributedAdaptiveContactHalo> halo;
    std::size_t bytes = 0;

    ~AdaptivePublishedAdmission()
    {
        if (halo)
            halo->allocation().used -= bytes;
    }
};
struct AdaptiveGeometrySnapshotCache
{
    mutable std::mutex mutex;
    std::array<std::shared_ptr<const AdaptiveGeometrySnapshot>, 6> roots;
    std::array<std::weak_ptr<const AdaptiveGeometrySnapshot>, 6> predecessors;
    std::optional<ContactExternalResidencyLease> lease;
    std::array<std::weak_ptr<const AdaptiveGeometrySnapshot>, 6> sealed_roots;
    std::size_t resident_bytes = 0, lease_resident_bytes = 0, union_censuses = 0, census_root_visits = 0;
    mutable std::size_t admission_queries = 0;
    /** Exact immutable owner set of each cached root, built once when the root is
     * published. Warm cache hits install it into the caller ledger instead of
     * re-walking the snapshot dependency graph. */
    std::array<std::array<geometry_snapshot_detail::PreparedOwnerManifest, 2>, 6> manifests;
    std::size_t manifest_bytes = 0;
    std::array<PLIC::details::GeometryOwnerRootBinding, 6> bindings;
    PLIC::details::GeometryOwnerUnion owner_union;
    PLIC::details::GeometryOwnerUnionWork union_work;
    std::size_t union_updates = 0;
    NativeSourceOwnerStore source_store;
    NativeSourceCoverageBroker coverage_broker;
    NativeGeometryRootStore native_store;
    SourceAccountingPartition source_partition;
    PLIC::details::GeometryOwnerRootBinding sealed_source, sealed_native;
    std::uint64_t revision = 0;
    const NativePublicationOperation* commit_token = nullptr;
    PLIC::details::GeometryOwnerRootBinding binding(const std::shared_ptr<const AdaptiveGeometrySnapshot>& root) const
    {
        if (!root)
            return {};
        return {root, sizeof(*root), GeometrySnapshotAccess::owner_pair(*root)};
    }
    void record_union_work(const PLIC::details::GeometryOwnerUnionWork& work)
    {
        ++union_updates;
        union_work.imported_rows += work.imported_rows;
        for (unsigned channel = 0; channel < 2; ++channel)
        {
            auto& to = union_work.channels[channel];
            const auto& from = work.channels[channel];
            to.edge_visits += from.edge_visits;
            to.dependency_entries += from.dependency_entries;
            to.first_transitions += from.first_transitions;
            to.last_transitions += from.last_transitions;
            to.index.copied_pages += from.index.copied_pages;
            to.index.copied_bytes += from.index.copied_bytes;
            to.index.visited_pages += from.index.visited_pages;
            to.lookup.visited_pages += from.lookup.visited_pages;
        }
        union_work.controls.copied_pages += work.controls.copied_pages;
        union_work.controls.copied_bytes += work.controls.copied_bytes;
        union_work.controls.visited_pages += work.controls.visited_pages;
    }

    /** Cumulative persistent-catalog construction work since the last reset. */
    PLIC::details::CellCatalogDemandWork catalog_work;

    // Every query holds mutex. The seal authenticates the exact immutable
    // root controls, including an empty root, whose union was admitted.
    std::size_t root_bytes() const
    {
        for (std::size_t slot = 0; slot < roots.size(); ++slot)
            if (sealed_roots[slot].lock() != roots[slot] || sealed_roots[slot].owner_before(roots[slot]) ||
                roots[slot].owner_before(sealed_roots[slot]))
                throw std::logic_error("Geometry cache root set differs from its exact residency seal.");
        source_store.require_current();
        const auto same_binding = [](const auto& a, const auto& b)
        {
            return a.header().get() == b.header().get() && !a.header().owner_before(b.header()) &&
                   !b.header().owner_before(a.header());
        };
        if (!same_binding(sealed_source, source_store.binding()) ||
            !same_binding(sealed_native, native_store.binding()))
            throw std::logic_error("Geometry cache source/native root differs from residency seal.");
        if (owner_union.resident_bytes() != resident_bytes)
            throw std::logic_error("Geometry cache owner union differs from residency seal.");
        if (lease && lease->bytes() != lease_resident_bytes)
            throw std::logic_error("Geometry cache uncovered residency differs from its admitted lease.");
        if (lease && lease_resident_bytes != uncovered_bytes(owner_union, lease->arena_identity()))
            throw std::logic_error("Geometry cache exact paid origin differs from its uncovered residency seal.");
        return resident_bytes;
    }
    static std::size_t uncovered_bytes(const PLIC::details::GeometryOwnerUnion& owners,
        const std::shared_ptr<const void>& arena)
    {
        const auto paid = owners.native_paid_bytes(arena);
        if (paid > owners.resident_bytes())
            throw std::logic_error("Native exact same-arena credit exceeds its physical owner union.");
        return owners.resident_bytes() - paid;
    }
    static std::size_t contact_residual(
        const DistributedAdaptiveContactHalo& halo, const PLIC::details::GeometryOwnerUnion& current)
    {
        const auto storage = halo.cache_storage_bytes();
        const auto direct = current.direct_h_paid_bytes(halo.allocation_arena());
        if (direct > storage)
            throw std::logic_error("Current owner union direct-H credit exceeds its actual current halo financial category.");
        return storage - direct;
    }
    std::size_t contact_residual(const DistributedAdaptiveContactHalo& halo) const
    {
        return contact_residual(halo, owner_union);
    }
    void admit_residency(Budget& budget) const
    {
        ++admission_queries;
        const auto bytes = root_bytes();
        budget.add(sizeof(*this));
        budget.add(bytes);
    }
    void require_editable() const
    {
        if (commit_token || revision == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Geometry/source cache is locked or its revision is exhausted.");
    }
    void seal_residency(std::size_t bytes) noexcept
    {
        resident_bytes = bytes;
        lease_resident_bytes = lease ? lease->bytes() : 0;
        sealed_source = source_store.binding();
        sealed_native = native_store.binding();
        for (std::size_t slot = 0; slot < roots.size(); ++slot)
            sealed_roots[slot] = roots[slot];
    }
    void copy_roots_from(const AdaptiveGeometrySnapshotCache& source)
    {
        const auto bytes = source.root_bytes();
        roots = source.roots;
        predecessors = source.predecessors;
        manifests = source.manifests;
        manifest_bytes = source.manifest_bytes;
        bindings = source.bindings;
        owner_union = source.owner_union;
        source_store = source.source_store;
        native_store = source.native_store;
        source_partition = source.source_partition;
        revision = source.revision;

        seal_residency(bytes);
    }
};
struct DistributedAdaptiveHierarchy::Data
{
    using LeafMap = std::map<AdaptiveCellKey, AdaptiveLeaf>;

    std::shared_ptr<AdaptiveGeometrySnapshotCache> geometry_cache = std::make_shared<AdaptiveGeometrySnapshotCache>();
    // Bootstrap stages a bounded import. Adaptation and migration stage only
    // edits over an accepted immutable root.
    std::variant<LeafMap, std::shared_ptr<AdaptiveLeafOverlay>, std::shared_ptr<const AdaptiveLeafLayout>> leaves;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> layout_allocation;
    std::size_t owned_layout_bytes = 0, owned_builder_index_bytes = 0;
    std::size_t staged_body_bound = 0, staged_family_bound = 0;
    AdaptiveLeafBodyUsage body_usage;
    PLIC::details::OwnerManifestPair initial_kernel_owners;
    std::shared_ptr<const ContactExternalResidencyLease> bootstrap_family_admission;
    std::shared_ptr<const ContactExternalResidencyLease> bound_layout_admission;
    // Bootstrap imports and received migration capsules use temporary rows only.
    std::map<AdaptiveBytes, AdaptiveFamily> staged_families;
    std::shared_ptr<const AdaptiveFamilyLayout> family_layout;
    std::shared_ptr<const ProductFamilyElectionState> product_election;
    bool active_product = false;
    bool native_source_enabled = false;
    std::shared_ptr<NativePublicationCache> native_publication_cache;
    std::shared_ptr<const AdaptiveProviderKernel> kernel;
    std::shared_ptr<const DistributedAdaptiveContactHalo> contact_halo;
    std::shared_ptr<const AdaptiveContactSnapshot> contacts;
    std::shared_ptr<const AdaptivePublishedAdmission> admission;
    std::shared_ptr<const DistributedAncestryDirectory> cells, family_directory;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    DistributedAdaptiveOptions options;
    std::shared_ptr<const void> token, topology_token, partition_token;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> token_allocation, topology_allocation;
    std::shared_ptr<const CompositePartition<>> native;
    Teuchos::RCP<const DefaultTpetraTypes::map_type> owned_map;
    std::shared_ptr<const NativeSourceBoundInput::Map> map_token;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> map_token_allocation;
    std::uint64_t native_epoch = 0;
    std::uint64_t topology = 0, partition = 0, cell_begin = 0, global_cells = 0;
    bool bootstrap_family_registration_validated = false;
    [[nodiscard]] DirectorySourceCredit directory_credit(bool family, Budget& budget, std::size_t& held) const
    {
        return DirectorySourceCredit(family ? *family_directory : *cells, budget, held);
    }
    [[nodiscard]] FamilyDirectoryProjection directory_family_projection() const
    {
        if (family_layout)
            return FamilyDirectoryProjection(family_layout, topology);
        if (!bootstrap_family_registration_validated)
            throw std::logic_error("Bootstrap directory projection lacks validated family proofs.");
        return FamilyDirectoryProjection(staged_families, topology);
    }
    void finish_directory_overlap() const noexcept
    {
        DirectoryRegistrationAccess::finish_overlap(*cells);
        DirectoryRegistrationAccess::finish_overlap(*family_directory);
    }
    [[nodiscard]] const LeafMap& owned() const { return std::get<LeafMap>(leaves); }
    [[nodiscard]] LeafMap& edit_owned() { return std::get<LeafMap>(leaves); }
    [[nodiscard]] AdaptiveLeafOverlay& overlay() { return *std::get<std::shared_ptr<AdaptiveLeafOverlay>>(leaves); }
    [[nodiscard]] const std::shared_ptr<const AdaptiveLeafLayout>& layout() const
    {
        return std::get<std::shared_ptr<const AdaptiveLeafLayout>>(leaves);
    }
    [[nodiscard]] const AdaptiveLeafPages& pages() const { return layout()->pages_; }
    [[nodiscard]] AdaptiveLeafTransition transition_from(const Data& source, std::shared_ptr<Budget> arena) const
    {
        return AdaptiveLeafTransition(source.layout(), layout(), std::move(arena));
    }
    [[nodiscard]] std::size_t family_layout_residency() const noexcept
    {
        const auto rows =
            family_layout ? family_layout_header_bytes + AdaptiveFamilyStorage::storage_bytes(*family_layout) : 0;
        return rows +
               (product_election ? ProductFamilyElectionState::shell_bytes() + product_election->storage_bytes() : 0);
    }
    [[nodiscard]] std::size_t family_body_residency() const
    {
        return body_usage.storage_bytes() + (body_usage.payload_bytes() - body_usage.leaf_payload_bytes());
    }
    [[nodiscard]] std::size_t active_size() const
    {
        if (const auto* map = std::get_if<LeafMap>(&leaves))
            return map->size();
        return layout()->size();
    }
    [[nodiscard]] bool contains_active(const AdaptiveCellKey& key) const
    {
        if (const auto* map = std::get_if<LeafMap>(&leaves))
            return map->contains(key);
        return layout()->contains(key);
    }
    template<class Visitor> void visit_active(Visitor&& visit) const
    {
        if (const auto* map = std::get_if<LeafMap>(&leaves))
        {
            for (const auto& [key, row] : *map)
            {
                record_pipeline_work(PipelineWork::ActiveLeafVisit);
                visit(row);
            }
        }
        else
            for (const auto& row : *layout())
            {
                record_pipeline_work(PipelineWork::ActiveLeafVisit);
                visit(row);
            }
    }
    void set_identity(const geometry_detail::NativeConstructionAdmission* native_storage = nullptr)
    {
        if (native_storage)
        {
            auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<char>(*native_storage, 0);
            token = std::move(tracked.owner);
            token_allocation = std::move(tracked.envelope);
        }
        else
            token = std::make_shared<const char>(0);
        partition_token = token;
        if (!topology_token)
        {
            topology_token = token;
            topology_allocation = token_allocation;
        }
    }
    void set_layout(AdaptiveLeafPages pages, std::shared_ptr<const ContactExternalResidencyLease> admission,
        const geometry_detail::NativeConstructionAdmission* native_storage = nullptr)
    {
        if (native_storage)
        {
            auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<AdaptiveLeafLayout>(
                *native_storage, AdaptiveLeafLayout::NativePreparedKey{}, std::move(pages), std::move(admission));
            leaves = std::shared_ptr<const AdaptiveLeafLayout>(std::move(tracked.owner));
            layout_allocation = std::move(tracked.envelope);
        }
        else
            leaves = std::shared_ptr<const AdaptiveLeafLayout>(
                new const AdaptiveLeafLayout(std::move(pages), std::move(admission)));
    }
    [[nodiscard]] NativePublicationSourceObservations source_observations() const
    {
        using Observation = PLIC::details::NativeWeakOwnerObservation;
        NativePublicationSourceObservations result;
        if (layout_allocation) result.layout = Observation::native(layout(), layout_allocation);
        if (token_allocation) result.identity = Observation::native(token, token_allocation);
        if (topology_allocation) result.topology = Observation::native(topology_token, topology_allocation);
        if (map_token_allocation)
            result.ownership = Observation::native(map_token, map_token_allocation);
        else if (!native && token_allocation)
            result.ownership = Observation::native(partition_token, token_allocation);
        if (native_source_enabled)
        {
            result.authority = Observation::native(NativeKernelAuthority::observation(kernel).lock(),
                NativeKernelAuthority::allocation_observation(kernel));
            result.subfaces = AdaptiveSubfaceSnapshot::weak_observation(contact_halo->subface_snapshot());
        }
        return result;
    }
    [[nodiscard]] PLIC::details::SealedSourcePrefix source_prefix_for_refresh() const;
    [[nodiscard]] std::shared_ptr<const ContactExternalResidencyLease> leaf_body_admission() const
    {
        return bound_layout_admission ? bound_layout_admission : layout()->admission_;
    }
    [[nodiscard]] std::size_t staged_buffer_bytes() const
    {
        Budget total{std::numeric_limits<std::size_t>::max()};
        for (const auto& [key, row] : owned())
            total.add(AdaptiveLeafPages::buffer_bytes(row));
        return total.used;
    }
    void publish_bootstrap(const std::shared_ptr<Budget>& arena, std::size_t buffer_bytes)
    {
        auto row = edit_owned().begin();
        auto pages = AdaptiveLeafPages::build_moving(
            owned().size(), [&]() -> AdaptiveLeaf&& { return std::move((row++)->second); }, arena, buffer_bytes);
        {
            AdaptiveLeafBodyDelta delta(arena);
            pages.visit_leaf_pages([&](auto rows) { delta.record(rows, true); });
            body_usage = delta.apply({}).usage;
        }
        {
            AdaptiveLeafBodyDelta delta(arena);
            for (const auto& [key, family] : staged_families)
                delta.record(std::span<const AdaptiveLeaf>(&family.parent, 1), true);
            body_usage = delta.apply(body_usage, AdaptiveLeafBodyRole::Family).usage;
        }
        const auto external = std::make_shared<std::size_t>(0);
        auto family_admission = std::make_shared<ContactExternalResidencyLease>(arena, external);
        family_admission->resize(family_layout_header_bytes);
        auto family_row = staged_families.begin();
        family_layout = std::make_shared<const AdaptiveFamilyLayout>(AdaptiveFamilyStorage::build_from(
            staged_families.size(), [&]() -> const AdaptiveFamily& { return (family_row++)->second; }, arena,
            topology));
        staged_families.clear();
        const auto leaf_bytes = body_usage.leaf_payload_bytes();
        const auto family_bytes = body_usage.payload_bytes() - leaf_bytes;
        if (leaf_bytes > staged_body_bound || family_bytes > staged_family_bound)
            throw std::logic_error("Initial adaptive bodies exceed staged residency.");
        // The joint decoder already admitted these owners. Separate their
        // credits so an escaped leaf layout cannot pin any saved-parent body.
        arena->limit += staged_body_bound + staged_family_bound;
        auto leaf_admission = std::make_shared<ContactExternalResidencyLease>(arena, external);
        leaf_admission->resize(leaf_bytes);
        family_admission->resize(family_bytes + family_layout_header_bytes);
        bootstrap_family_admission = std::move(family_admission);
        const auto layout_bytes = pages.retained_bytes() + leaf_bytes;
        if (native_source_enabled)
        {
            ScopedSourceAdmission native_storage(arena->limit, arena->used);
            set_layout(std::move(pages), std::move(leaf_admission), &native_storage.admission);
        }
        else
            set_layout(std::move(pages), std::move(leaf_admission));
        owned_layout_bytes -= owned_builder_index_bytes + buffer_bytes + staged_body_bound;
        owned_layout_bytes += layout_bytes;
        owned_builder_index_bytes = staged_body_bound = staged_family_bound = 0;
    }

    ~Data()
    {
        contacts.reset();
        // Release owner storage before the accounting token can release its lease.
        leaves.emplace<std::shared_ptr<const AdaptiveLeafLayout>>();
        staged_families.clear();
        family_layout.reset();
        product_election.reset();
        body_usage = {};
        bootstrap_family_admission.reset();
        cells.reset();
        family_directory.reset();
        admission.reset();
        contact_halo.reset();
    }
};
namespace
{
constexpr auto leaf_layout_header_bytes = sizeof(AdaptiveLeafLayout) + sizeof(Budget) + 8 * sizeof(void*);
constexpr auto candidate_layout_header_bytes =
    leaf_layout_header_bytes + sizeof(ContactExternalResidencyLease) + 4 * sizeof(void*);
constexpr auto bootstrap_layout_header_bytes =
    candidate_layout_header_bytes + sizeof(ContactExternalResidencyLease) + sizeof(std::size_t) + 8 * sizeof(void*);
constexpr auto legacy_layout_allocation = sizeof(AdaptiveLeafLayout) + 4 * sizeof(void*);
constexpr auto overlay_header_bytes = sizeof(AdaptiveLeafOverlay) + 4 * sizeof(void*);

// Session scratch is charged against the same ledger as the contact halo.
// Charges remain conservative through the scope and are released together.
struct ClosureScratch
{
    Budget& budget;
    std::size_t held = 0;
    ~ClosureScratch() { budget.used -= held; }
    void add(std::size_t n)
    {
        budget.add(n);
        held += n;
    }
    void drop(std::size_t n)
    {
        held -= n;
        budget.used -= n;
    }
    template<class T> void append(std::vector<T>& v)
    {
        const auto before = budget.used;
        try
        {
            budget.append_capacity(v);
        }
        catch (...)
        {
            held += budget.used - before;
            throw;
        }
        held += budget.used - before;
    }
};
std::uint64_t transaction(const Teuchos::Comm<int>& comm)
{
    static std::atomic<std::uint64_t> counter{0};
    std::uint64_t id = 0;
    if (comm.getRank() == 0)
    {
        auto previous = counter.load();
        while (previous != std::numeric_limits<std::uint64_t>::max())
            if (counter.compare_exchange_weak(previous, previous + 1))
            {
                id = previous + 1;
                break;
            }
    }
    ::SimpleFluid::owner_packet_broadcast_words(comm, 0, std::span(&id, 1));
    if (!id)
        throw std::overflow_error("Distributed transaction identity exhausted.");
    return id;
}
void agree(const Teuchos::Comm<int>& comm, std::span<const std::uint64_t> descriptor)
{
    std::array<std::uint64_t, 8> root{};
    collective(comm, [&]
    {
        if (descriptor.size() > root.size())
            throw std::invalid_argument("Distributed semantic descriptor exceeds its fixed scalar inventory.");
        std::copy(descriptor.begin(), descriptor.end(), root.begin());
    });
    ::SimpleFluid::owner_packet_broadcast_words(comm, 0, std::span(root).first(descriptor.size()));
    collective(comm, [&]
    {
        if (!std::equal(descriptor.begin(), descriptor.end(), root.begin()))
            throw std::invalid_argument("Distributed semantic descriptors differ.");
    });
}

std::size_t fetch_key_bytes(const AdaptiveCellKey& key);
void fetch_copy_charge(Budget& budget, const AdaptiveLeaf& leaf);
void fetch_copy_charge(Budget& budget, const AdaptiveFamily& family);
void fetch_copy_charge(Budget& budget, const AdaptiveFamilyView& family);
std::size_t fetch_retention(std::span<const AdaptiveLeaf>, std::span<const AdaptiveFamily>, std::size_t, bool,
    std::size_t* leaf_bytes = nullptr, bool dictionary_prepaid = false);
void intern_state_legacy(DistributedAdaptiveHierarchy::Data& state)
{
    std::vector<AdaptiveLeaf> leaves;
    std::vector<AdaptiveFamily> families;
    for (const auto& [key, leaf] : state.owned())
        leaves.push_back(leaf);
    for (const auto& [key, family] : state.staged_families)
        families.push_back(family);
    auto packet = encode_adaptive_batch(leaves, families, state.options.byte_budget);
    auto shared = decode_adaptive_batch(packet, state.options.byte_budget);
    std::size_t leaf_retained = 0;
    const auto retained =
        fetch_retention(shared.leaves, shared.families, state.options.byte_budget, false, &leaf_retained);
    state.staged_family_bound = retained - leaf_retained;
    Budget row_metadata{state.options.byte_budget};
    for (const auto& leaf : shared.leaves)
        fetch_copy_charge(row_metadata, leaf);
    if (leaf_retained < row_metadata.used)
        throw std::logic_error("Initial leaf residency lacks its row metadata.");
    state.staged_body_bound = leaf_retained - row_metadata.used;
    Budget layout{std::numeric_limits<std::size_t>::max(), bootstrap_layout_header_bytes -
        (state.native_source_enabled ? legacy_layout_allocation : 0)};
    layout.add(leaf_retained);
    state.owned_builder_index_bytes = shared.leaves.size() * sizeof(AdaptiveLeaf);
    state.edit_owned().clear();
    state.staged_families.clear();
    for (auto& leaf : shared.leaves)
    {
        const auto entry = sizeof(DistributedAdaptiveHierarchy::Data::LeafMap::value_type) + 4 * sizeof(void*) +
                           fetch_key_bytes(leaf.key);
        state.owned_builder_index_bytes += entry;
        layout.add(entry);
        state.edit_owned().emplace(leaf.key, std::move(leaf));
    }
    state.owned_layout_bytes = layout.used;
    for (auto& family : shared.families)
        state.staged_families.emplace(family.key, std::move(family));
}
bool descendant(const AdaptiveCellKey& parent, const AdaptiveCellKey& child)
{
    if (parent.region != child.region || parent.ancestry.index() != child.ancestry.index())
        return false;
    auto prefix = [](const auto& a, const auto& b)
    { return a.path.size() <= b.path.size() && std::equal(a.path.begin(), a.path.end(), b.path.begin()); };
    return std::visit(
        [&](const auto& p)
        {
            using T = std::decay_t<decltype(p)>;
            const auto& c = std::get<T>(child.ancestry);
            if constexpr (std::is_same_v<T, CartesianCellAncestry>)
            {
                if (c.level < p.level || c.level - p.level >= 64)
                    return false;
                for (unsigned a = 0; a < 3; ++a)
                    if ((c.index[a] >> (c.level - p.level)) != p.index[a])
                        return false;
                return true;
            }
            else if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                return p.root == c.root && prefix(p, c);
            else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                return p.base.root == c.base.root && p.base.root_template == c.base.root_template &&
                       p.axial.root == c.axial.root && prefix(p.base, c.base) && prefix(p.axial, c.axial);
            else
                return p == c;
        },
        parent.ancestry);
}
void validate_family(const AdaptiveProviderKernel& kernel, const AdaptiveFamily& family)
{
    kernel.validate(family.parent);
    if (family.key != adaptive_family_key(family.parent.key, family.split) || family.anchor.empty() ||
        !descendant(family.parent.key, decode_adaptive_key(family.anchor)))
        throw std::invalid_argument("Family key/anchor is not actual typed ancestry.");
    auto expected = kernel.split(family.parent, family.split).family.children, actual = family.children;
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    if (expected != actual || std::adjacent_find(actual.begin(), actual.end()) != actual.end())
        throw std::invalid_argument("Family capsule does not name complete immediate provider children.");
}
// Small-fixture reference for the historical full Product election. The
// production path below uses sharded changed-proposal receipts.
OwnerExchange closure_exchange(
    std::span<const OwnerPacket>, const Teuchos::RCP<const Teuchos::Comm<int>>&, ClosureScratch&);
void full_product_family_reference(DistributedAdaptiveHierarchy::Data& state, ClosureScratch& retained)
{
    ClosureScratch work{retained.budget};
    std::vector<OwnerPacket> packets;
    auto remaining = [&]
    {
        const auto cap = work.budget.limit - work.budget.used;
        if (!cap)
            throw std::invalid_argument("Product family workspace lease exhausted.");
        work.add(cap);
        return cap;
    };
    auto packet_for = [&](int rank, std::span<const AdaptiveFamily> families)
    {
        work.append(packets);
        const auto cap = remaining();
        auto bytes = encode_adaptive_batch({}, families, cap);
        if (bytes.capacity() > cap)
            throw std::invalid_argument("Product family packet exceeds lease.");
        work.drop(cap - bytes.capacity());
        packets.push_back({rank, std::move(bytes)});
    };
    auto admit_decoded = [&](const AdaptiveBatch& batch, std::size_t cap)
    {
        Budget measured{cap};
        measured.add(fetch_retention(batch.leaves, batch.families, cap, false));
        if (batch.leaves.capacity() - batch.leaves.size() > cap / sizeof(AdaptiveLeaf) ||
            batch.families.capacity() - batch.families.size() > cap / sizeof(AdaptiveFamily))
            throw std::invalid_argument("Product decoded vector capacity exceeds lease.");
        measured.add((batch.leaves.capacity() - batch.leaves.size()) * sizeof(AdaptiveLeaf));
        measured.add((batch.families.capacity() - batch.families.size()) * sizeof(AdaptiveFamily));
        work.drop(cap - measured.used);
        return measured.used;
    };
    collective(*state.comm,
        [&]
        {
            std::erase_if(state.staged_families,
                [](const auto& entry) { return entry.second.parent.provider == AdaptiveProviderKind::Product; });
            std::map<AdaptiveBytes, AdaptiveFamily> local;
            state.visit_active(
                [&](const AdaptiveLeaf& leaf)
                {
                    const auto& key = leaf.key;
                    if (leaf.provider == AdaptiveProviderKind::Product)
                    {
                        const auto cap = remaining();
                        auto ancestry = state.kernel->ancestry_families(leaf);
                        const auto size = fetch_retention({}, ancestry, cap, false);
                        work.drop(cap - size);
                        for (auto& family : ancestry)
                        {
                            family.revision = state.topology;
                            const auto dynamic = fetch_key_bytes(key);
                            if (work.budget.limit < 256 || dynamic > work.budget.limit / 4 - 64)
                                throw std::invalid_argument("Product family anchor exceeds lease.");
                            work.add(4 * (dynamic + 64));
                            family.anchor = encode_adaptive_key(key);
                            auto found = local.find(family.key);
                            if (found == local.end())
                            {
                                work.add(sizeof(decltype(local)::value_type) + 4 * sizeof(void*) + family.key.size());
                                local.emplace(family.key, std::move(family));
                            }
                            else
                            {
                                const auto revision = std::max(found->second.parent.revision, family.parent.revision);
                                if (decode_adaptive_key(family.anchor) < decode_adaptive_key(found->second.anchor))
                                    found->second = std::move(family);
                                found->second.parent.revision = revision;
                            }
                        }
                    }
                });
            std::map<int, std::vector<AdaptiveFamily>> groups;
            for (auto& [key, family] : local)
            {
                const int rank = authority(key, state.comm->getSize());
                auto found = groups.find(rank);
                if (found == groups.end())
                {
                    work.add(sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                    found = groups.emplace(rank, std::vector<AdaptiveFamily>{}).first;
                }
                work.append(found->second);
                found->second.push_back(std::move(family));
            }
            for (const auto& [rank, families] : groups)
                packet_for(rank, families);
        });
    auto incoming = closure_exchange(packets, state.comm, work);
    struct Candidate
    {
        AdaptiveFamily family;
        int owner;
    };
    std::vector<Candidate> candidates;
    std::vector<AdaptiveBytes> anchors;
    collective(*state.comm,
        [&]
        {
            for (const auto& packet : incoming.packets)
            {
                const auto cap = remaining();
                auto batch = decode_adaptive_batch(packet.bytes, cap);
                (void) admit_decoded(batch, cap);
                if (!batch.leaves.empty())
                    throw std::invalid_argument("Unexpected leaf in factor authority packet.");
                for (auto& family : batch.families)
                {
                    validate_family(*state.kernel, family);
                    if (authority(family.key, state.comm->getSize()) != state.comm->getRank())
                        throw std::invalid_argument("Misrouted product factor family.");
                    work.append(anchors);
                    work.add(family.anchor.size());
                    anchors.push_back(family.anchor);
                    work.append(candidates);
                    candidates.push_back({std::move(family), packet.peer});
                }
            }
        });
    std::size_t lookup_cap = 0;
    collective(*state.comm, [&] { lookup_cap = remaining(); });
    auto owners = state.cells->lookup(anchors, true, lookup_cap);
    collective(*state.comm,
        [&]
        {
            Budget result{lookup_cap};
            if (owners.capacity() > lookup_cap / sizeof(AdaptiveOwner))
                throw std::invalid_argument("Product family owner result exceeds lease.");
            result.add(owners.capacity() * sizeof(AdaptiveOwner));
            for (const auto& owner : owners)
                result.add(owner.key.capacity());
            work.drop(lookup_cap - result.used);
        });
    packets.clear();
    collective(*state.comm,
        [&]
        {
            std::map<AdaptiveBytes, Candidate> elected;
            for (std::size_t i = 0; i < candidates.size(); ++i)
            {
                auto candidate = std::move(candidates[i]);
                if (owners[i].owner != candidate.owner)
                    throw std::invalid_argument("Product family candidate lacks actual anchor ownership.");
                auto old = elected.find(candidate.family.key);
                if (old == elected.end())
                {
                    work.add(sizeof(decltype(elected)::value_type) + 4 * sizeof(void*) + candidate.family.key.size());
                    elected.emplace(candidate.family.key, std::move(candidate));
                }
                else
                {
                    auto& previous = old->second;
                    const auto& a = previous.family.parent;
                    const auto& b = candidate.family.parent;
                    if (previous.family.children.size() > work.budget.limit / sizeof(AdaptiveCellKey) ||
                        candidate.family.children.size() > work.budget.limit / sizeof(AdaptiveCellKey))
                        throw std::invalid_argument("Product child comparison exceeds lease.");
                    work.add(previous.family.children.size() * sizeof(AdaptiveCellKey));
                    work.add(candidate.family.children.size() * sizeof(AdaptiveCellKey));
                    for (const auto& key : previous.family.children)
                        work.add(fetch_key_bytes(key));
                    for (const auto& key : candidate.family.children)
                        work.add(fetch_key_bytes(key));
                    auto ac = previous.family.children, bc = candidate.family.children;
                    std::sort(ac.begin(), ac.end());
                    std::sort(bc.begin(), bc.end());
                    if (ac != bc || a.domain != b.domain || a.material != b.material || a.provider != b.provider ||
                        !state.kernel->geometry(a)->equivalent(*state.kernel->geometry(b)))
                        throw std::invalid_argument("Product factor authorities disagree on saved exact parent.");
                    const auto revision = std::max(a.revision, b.revision);
                    if (decode_adaptive_key(candidate.family.anchor) < decode_adaptive_key(previous.family.anchor))
                        previous = std::move(candidate);
                    previous.family.parent.revision = revision;
                }
            }
            std::map<int, std::vector<AdaptiveFamily>> groups;
            for (auto& [key, candidate] : elected)
            {
                auto found = groups.find(candidate.owner);
                if (found == groups.end())
                {
                    work.add(sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                    found = groups.emplace(candidate.owner, std::vector<AdaptiveFamily>{}).first;
                }
                work.append(found->second);
                found->second.push_back(std::move(candidate.family));
            }
            for (const auto& [rank, families] : groups)
                packet_for(rank, families);
        });
    auto elected = closure_exchange(packets, state.comm, work);
    collective(*state.comm,
        [&]
        {
            for (const auto& packet : elected.packets)
            {
                const auto cap = remaining();
                auto batch = decode_adaptive_batch(packet.bytes, cap);
                const auto decoded = admit_decoded(batch, cap);
                work.held -= decoded;
                retained.held += decoded;
                if (!batch.leaves.empty())
                    throw std::invalid_argument("Unexpected elected leaf packet.");
                for (auto& family : batch.families)
                {
                    retained.add(
                        sizeof(decltype(state.staged_families)::value_type) + 4 * sizeof(void*) + family.key.size());
                    if (packet.peer != authority(family.key, state.comm->getSize()) ||
                        !state.contains_active(decode_adaptive_key(family.anchor)) ||
                        !state.staged_families.emplace(family.key, std::move(family)).second)
                        throw std::invalid_argument("Invalid elected product factor authority.");
                }
            }
        });
}
std::size_t fetch_key_bytes(const AdaptiveCellKey& key);
std::shared_ptr<const DistributedAncestryDirectory> register_directory(DistributedAdaptiveHierarchy::Data& state,
    std::span<const AdaptiveOwner> entries, ClosureScratch* retained, const DirectorySourceCredit* source = nullptr,
    const FamilyDirectoryProjection* projection = nullptr)
{
    if (!retained)
    {
        auto directory = std::make_shared<const DistributedAncestryDirectory>(DirectoryRegistrationAccess::derive(
            entries, state.comm, state.options.byte_budget, state.options.byte_budget, nullptr, projection));
        DirectoryRegistrationAccess::finish_overlap(*directory);
        return directory;
    }
    std::shared_ptr<std::optional<DistributedAncestryDirectory>> holder;
    collective(*state.comm,
        [&]
        {
            retained->add(sizeof(std::optional<DistributedAncestryDirectory>) - sizeof(DistributedAncestryDirectory) +
                          4 * sizeof(void*));
            holder = std::make_shared<std::optional<DistributedAncestryDirectory>>();
        });
    std::size_t cap = 0;
    collective(*state.comm,
        [&]
        {
            cap = retained->budget.limit - retained->budget.used;
            if (!cap)
                throw std::invalid_argument("Directory registration shared lease exhausted.");
            retained->add(cap);
        });
    holder->emplace(
        DirectoryRegistrationAccess::derive(entries, state.comm, state.options.byte_budget, cap, source, projection));
    std::shared_ptr<const DistributedAncestryDirectory> directory(holder, &holder->value());
    collective(*state.comm,
        [&]
        {
            const auto resident = directory->storage_bytes();
            const auto shared = DirectoryRegistrationAccess::statistics(*directory).shared_fact_bytes;
            if (shared > resident || resident - shared > cap)
                throw std::invalid_argument("Directory registration exceeded shared lease.");
            retained->drop(cap - (resident - shared));
            if (source)
                DirectoryRegistrationAccess::transfer_shared(*source, *directory, retained->budget, retained->held);
            else if (shared)
                throw std::logic_error("Directory shared facts have no exact predecessor lease.");
        });
    return directory;
}

void register_cells(DistributedAdaptiveHierarchy::Data& state, ClosureScratch* retained = nullptr,
    const DistributedAdaptiveHierarchy::Data* source = nullptr, ClosureScratch* source_overlap = nullptr)
{
    const PipelinePhaseScope audit(PipelinePhase::Directory);
    Budget fallback{state.options.byte_budget};
    ClosureScratch scratch{retained ? retained->budget : fallback};
    const std::uint64_t local = state.active_size();
    std::uint64_t prefix = 0;
    collective(*state.comm,
        [&]
        {
            if (local > std::numeric_limits<std::uint64_t>::max() / std::uint64_t(state.comm->getSize()))
                throw std::overflow_error("Global adaptive cell count exceeds safe prefix bound.");
        });
    Teuchos::scan(*state.comm, Teuchos::REDUCE_SUM, 1, &local, &prefix);
    Teuchos::reduceAll(*state.comm, Teuchos::REDUCE_SUM, 1, &local, &state.global_cells);
    state.cell_begin = prefix - local;
    std::vector<AdaptiveOwner> cells;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            std::uint64_t ordinal = 0;
            state.visit_active(
                [&](const AdaptiveLeaf& leaf)
                {
                    const auto& key = leaf.key;
                    state.active_product |= leaf.provider == AdaptiveProviderKind::Product;
                    // Admit the owning vector capacity and the exact encoded key copy
                    // BEFORE either allocation is materialized.
                    const auto key_bytes = encoded_adaptive_key_size(key);
                    if (retained)
                    {
                        scratch.append(cells);
                        // Preserve the established small-budget threshold: it is a
                        // policy guard on key dynamic storage, not on wire size.
                        const auto dynamic = fetch_key_bytes(key);
                        if (scratch.budget.limit < 256 || dynamic > scratch.budget.limit / 4 - 64)
                            throw std::invalid_argument("Directory cell key exceeds shared lease.");
                        scratch.add(key_bytes);
                    }
                    else
                    {
                        budget.append_capacity(cells);
                        budget.add(key_bytes);
                    }
                    auto bytes = encode_adaptive_key(key);
                    if (bytes.size() != key_bytes)
                        throw std::logic_error("Directory cell key encoding differs from its admitted size.");
                    cells.push_back(
                        {std::move(bytes), state.comm->getRank(), leaf.revision, state.cell_begin + ordinal++});
                });
        });
    std::optional<DirectorySourceCredit> credit;
    if (source)
        collective(*state.comm,
            [&]
            {
                if (!source_overlap || !retained || &source_overlap->budget != &retained->budget)
                    throw std::logic_error("Directory source overlap does not belong to the current operation.");
                credit.emplace(source->directory_credit(false, retained->budget, source_overlap->held));
            });
    state.cells = register_directory(state, cells, retained, credit ? &*credit : nullptr);
    std::vector<AdaptiveBytes> partner_keys;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            state.visit_active(
                [&](const AdaptiveLeaf& leaf)
                {
                    for (const auto& partner : state.kernel->matching_partners(leaf))
                    {
                        const auto key_bytes = encoded_adaptive_key_size(partner);
                        if (retained)
                        {
                            scratch.append(partner_keys);
                            // Same policy guard as register_cells: key dynamic storage,
                            // not the encoded wire size.
                            const auto dynamic = fetch_key_bytes(partner);
                            if (scratch.budget.limit < 256 || dynamic > scratch.budget.limit / 4 - 64)
                                throw std::invalid_argument("Directory partner key exceeds shared lease.");
                            scratch.add(key_bytes);
                        }
                        else
                        {
                            budget.append_capacity(partner_keys);
                            budget.add(key_bytes);
                        }
                        auto bytes = encode_adaptive_key(partner);
                        if (bytes.size() != key_bytes)
                            throw std::logic_error("Directory partner key encoding differs from its admitted size.");
                        partner_keys.push_back(std::move(bytes));
                    }
                });
        });
    if (retained)
    {
        std::size_t cap = 0;
        collective(*state.comm,
            [&]
            {
                cap = scratch.budget.limit - scratch.budget.used;
                if (!cap)
                    throw std::invalid_argument("Directory partner lookup lease exhausted.");
                scratch.add(cap);
            });
        (void) state.cells->lookup(partner_keys, true, cap);
        scratch.drop(cap);
    }
    else
        (void) state.cells->lookup(partner_keys);
}

void register_families(DistributedAdaptiveHierarchy::Data& state, ClosureScratch* retained = nullptr,
    const DistributedAdaptiveHierarchy::Data* source = nullptr, ClosureScratch* source_overlap = nullptr)
{
    const PipelinePhaseScope audit(PipelinePhase::Directory);
    Budget fallback{state.options.byte_budget};
    ClosureScratch scratch{retained ? retained->budget : fallback};
    std::vector<AdaptiveOwner> families;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            const auto emit = [&](const auto& family)
            {
                // Admit vector growth and the exact source key copy before the
                // AdaptiveOwner element (and its deep key buffer) is constructed.
                if (retained)
                {
                    scratch.append(families);
                    scratch.add(family.key.size());
                }
                else
                {
                    budget.append_capacity(families);
                    budget.add(family.key.size());
                }
                families.push_back({family.key, state.comm->getRank(), family.revision});
            };
            if (state.family_layout)
                for (const auto family : *state.family_layout)
                    emit(family);
            else
                for (const auto& [key, family] : state.staged_families)
                {
                    validate_family(*state.kernel, family);
                    if (!state.contains_active(decode_adaptive_key(family.anchor)))
                        throw std::invalid_argument("Saved parent authority lost actual descendant owner.");
                    emit(family);
                }
            state.bootstrap_family_registration_validated = true;
        });
    std::optional<DirectorySourceCredit> credit;
    std::optional<FamilyDirectoryProjection> projection;
    collective(*state.comm,
        [&]
        {
            projection.emplace(state.directory_family_projection());
            if (source)
            {
                if (!source_overlap || !retained || &source_overlap->budget != &retained->budget)
                    throw std::logic_error("Family directory source lease belongs to a different operation.");
                credit.emplace(source->directory_credit(true, retained->budget, source_overlap->held));
            }
        });
    state.family_directory = register_directory(state, families, retained, credit ? &*credit : nullptr, &*projection);
}
std::optional<AdaptiveCellKey> original_cover(
    const DistributedAdaptiveHierarchy::Data& source, const AdaptiveCellKey& key)
{
    if (source.layout()->contains(key))
        return key;
    AdaptiveCellKey ancestor = key;
    if (auto* p = std::get_if<CartesianCellAncestry>(&ancestor.ancestry))
    {
        while (p->level)
        {
            *p = p->parent();
            if (source.layout()->contains(ancestor))
                return ancestor;
        }
    }
    else if (auto* p = std::get_if<ConvexCellAncestry>(&ancestor.ancestry))
    {
        while (!p->path.empty())
        {
            p->path.pop_back();
            if (source.layout()->contains(ancestor))
                return ancestor;
        }
    }
    else if (auto* p = std::get_if<ProductCellAncestry>(&ancestor.ancestry))
    {
        for (;;)
        {
            auto probe = ancestor;
            auto& axial = std::get<ProductCellAncestry>(probe.ancestry).axial;
            for (;;)
            {
                if (source.layout()->contains(probe))
                    return probe;
                if (axial.path.empty())
                    break;
                axial.path.pop_back();
            }
            if (p->base.path.empty())
                break;
            p->base.path.pop_back();
        }
    }
    return {};
}
bool repeated_factor(const AdaptiveCellKey& origin, const AdaptiveCellKey& current, AdaptiveSplit split)
{
    return std::visit(
        [&](const auto& old)
        {
            using T = std::decay_t<decltype(old)>;
            const auto& now = std::get<T>(current.ancestry);
            if constexpr (std::is_same_v<T, CartesianCellAncestry>)
                return now.level > old.level;
            else if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                return now.path.size() > old.path.size();
            else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                return (split != AdaptiveSplit::Axial && now.base.path.size() > old.base.path.size()) ||
                       (split != AdaptiveSplit::Base && now.axial.path.size() > old.axial.path.size());
            else
                return true;
        },
        origin.ancestry);
}
std::size_t closure_packet_storage(const std::vector<OwnerPacket>& packets)
{
    if (packets.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(OwnerPacket))
        throw std::invalid_argument("Closure packet capacity overflows.");
    std::size_t bytes = packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
    {
        if (packet.bytes.capacity() > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::invalid_argument("Closure packet capacity overflows.");
        bytes += packet.bytes.capacity();
    }
    return bytes;
}
OwnerExchange closure_exchange(
    std::span<const OwnerPacket> packets, const Teuchos::RCP<const Teuchos::Comm<int>>& comm, ClosureScratch& memory)
{
    std::size_t cap = 0;
    collective(*comm,
        [&]
        {
            cap = memory.budget.limit - memory.budget.used;
            if (!cap)
                throw std::invalid_argument("Closure request transport budget exhausted.");
            memory.add(cap);
        });
    auto result = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(packets, comm, cap);
    collective(*comm,
        [&]
        {
            const auto retained = closure_packet_storage(result.packets);
            if (retained > cap)
                throw std::invalid_argument("Closure request transport exceeded lease.");
            memory.drop(cap - retained);
        });
    return result;
}
using AnchorIndex = std::map<AdaptiveBytes, std::set<AdaptiveBytes>>;
void anchor_add(AnchorIndex& index, const AdaptiveBytes& anchor, const AdaptiveBytes& family, ClosureScratch& memory)
{
    auto it = index.find(anchor);
    if (it == index.end())
    {
        memory.add(sizeof(AnchorIndex::value_type) + 4 * sizeof(void*));
        memory.add(anchor.size());
        it = index.emplace(anchor, std::set<AdaptiveBytes>{}).first;
    }
    if (!it->second.contains(family))
    {
        memory.add(sizeof(AdaptiveBytes) + 4 * sizeof(void*));
        memory.add(family.size());
        it->second.insert(family);
    }
}
void remove_anchor(AnchorIndex& index, const AdaptiveFamily& family)
{
    auto it = index.find(family.anchor);
    if (it == index.end())
        return;
    it->second.erase(family.key);
    if (it->second.empty())
        index.erase(it);
}
std::size_t fetch_key_bytes(const AdaptiveCellKey& key);
void charge_encoded_key(ClosureScratch& memory, const AdaptiveCellKey& key)
{
    const auto size = fetch_key_bytes(key);
    if (memory.budget.limit < 256 || size > memory.budget.limit / 4 - 64)
        throw std::invalid_argument("Adaptive key encoding exceeds closure budget.");
    memory.add(4 * (size + 64));
}
void fetch_copy_charge(Budget& budget, const AdaptiveLeaf& leaf);
void fetch_copy_charge(Budget& budget, const AdaptiveFamily& family);
void fetch_copy_charge(Budget& budget, const AdaptiveFamilyView& family);
template<class Value> void charge_candidate_copy(ClosureScratch& memory, const Value& value)
{
    const auto before = memory.budget.used;
    try
    {
        fetch_copy_charge(memory.budget, value);
    }
    catch (...)
    {
        memory.held += memory.budget.used - before;
        throw;
    }
    memory.held += memory.budget.used - before;
}
// Non-Product rows enter this journal only when changed. Its reverse index
// covers pending non-Product rows; the accepted index supplies untouched anchors.
class FamilyJournal
{
public:
    struct Change
    {
        std::optional<AdaptiveFamily> family;
        bool accepted_parent = false;
    };

    explicit FamilyJournal(const AdaptiveFamilyLayout& source) : source_(source) {}
    [[nodiscard]] bool contains(const AdaptiveBytes& key) const
    {
        const auto changed = edits_.find(key);
        return changed == edits_.end() ? source_.contains(key) : changed->second.family.has_value();
    }
    void insert(AdaptiveFamily&& family, ClosureScratch& memory)
    {
        if (contains(family.key))
            throw std::invalid_argument("Duplicate retained family capsule.");
        set(std::move(family), false, memory);
    }
    void set_product(AdaptiveFamily&& family, const AdaptiveProviderKernel& kernel, ClosureScratch& memory)
    {
        if (const auto old = source_.find(family.key); family.parent.provider != AdaptiveProviderKind::Product ||
                                                       (old && old->parent.provider != AdaptiveProviderKind::Product))
            throw std::invalid_argument("Product election conflicts with a retained family.");
        // The shard validated at the anchor revision. Only the actual
        // kernel may authorize reuse at the projected maximum revision.
        const bool accepted_proof = kernel.product_family_proof_revision_invariant(&family.parent);
        set(std::move(family), accepted_proof, memory);
    }
    void erase(const AdaptiveBytes& key, ClosureScratch& memory)
    {
        if (!contains(key))
            throw std::invalid_argument("Retired family capsule is absent.");
        auto& change = slot(key, memory);
        if (change.family && change.family->parent.provider != AdaptiveProviderKind::Product)
            remove_anchor(anchors_, *change.family);
        change = {};
        if (!source_.contains(key))
            edits_.erase(key);
    }
    void move_anchor(const AdaptiveBytes& from, const AdaptiveBytes& to, ClosureScratch& memory)
    {
        if (from == to)
            return;
        for (;;)
        {
            const auto pending = anchors_.find(from);
            if (pending == anchors_.end())
                break;
            auto& family = *edits_.at(*pending->second.begin()).family;
            remove_anchor(anchors_, family);
            memory.add(to.size());
            family.anchor = to;
            anchor_add(anchors_, to, family.key, memory);
        }
        source_.visit_anchored(from,
            [&](AdaptiveFamilyView view)
            {
                if (view.parent.provider == AdaptiveProviderKind::Product || edits_.contains(view.key))
                    return;
                charge_candidate_copy(memory, view);
                auto family = view.materialize();
                memory.add(to.size());
                family.anchor = to;
                set(std::move(family), true, memory);
            });
    }
    [[nodiscard]] const auto& edits() const noexcept { return edits_; }
    void clear()
    {
        anchors_.clear();
        edits_.clear();
    }

private:
    using Edits = std::map<AdaptiveBytes, Change>;
    Change& slot(const AdaptiveBytes& key, ClosureScratch& memory)
    {
        auto found = edits_.find(key);
        if (found == edits_.end())
        {
            memory.add(sizeof(Edits::value_type) + 4 * sizeof(void*) + key.size());
            found = edits_.try_emplace(key).first;
        }
        return found->second;
    }
    void set(AdaptiveFamily&& family, bool accepted_parent, ClosureScratch& memory)
    {
        auto& change = slot(family.key, memory);
        if (change.family && change.family->parent.provider != AdaptiveProviderKind::Product)
            remove_anchor(anchors_, *change.family);
        if (family.parent.provider != AdaptiveProviderKind::Product)
            anchor_add(anchors_, family.anchor, family.key, memory);
        change = {std::move(family), accepted_parent};
    }

    const AdaptiveFamilyLayout& source_;
    Edits edits_;
    AnchorIndex anchors_;
};

#include "ProductFamilyElection.ipp"

std::shared_ptr<ProductFamilyElectionState> bootstrap_product_families(DistributedAdaptiveHierarchy::Data& state)
{
    const int local = state.active_product;
    int any = 0;
    Teuchos::reduceAll(*state.comm, Teuchos::REDUCE_MAX, 1, &local, &any);
    std::shared_ptr<ProductFamilyElectionState> edit;
    collective(*state.comm,
        [&]
        {
            std::erase_if(state.staged_families,
                [](const auto& row) { return row.second.parent.provider == AdaptiveProviderKind::Product; });
            if (any)
                edit = ProductFamilyElectionState::create(std::make_shared<Budget>(Budget{state.options.byte_budget}));
        });
    if (!any)
        return {};
    ClosureScratch retained{*edit->keeper()};
    std::vector<ProductFamilyResult> rows;
    collect_product_families(state, nullptr, {}, 0, *edit, retained, rows);
    collective(*state.comm,
        [&]
        {
            for (auto& row : rows)
            {
                retained.add(sizeof(decltype(state.staged_families)::value_type) + 4 * sizeof(void*) + row.key.size());
                if (!row.family || !state.staged_families.emplace(std::move(row.key), std::move(*row.family)).second)
                    throw std::invalid_argument("Bootstrap Product family conflicts with a saved capsule.");
            }
        });
    return edit;
}

void edit_product_families(DistributedAdaptiveHierarchy::Data& candidate,
    const DistributedAdaptiveHierarchy::Data& source, std::span<const AdaptiveCellKey> dirty, std::uint64_t transaction,
    FamilyJournal& journal, ClosureScratch& source_overlap, ClosureScratch& candidate_memory,
    ClosureScratch& family_memory)
{
    using persistent_ordered_detail::checked_sum;
    const int local = candidate.active_product || source.product_election || !candidate.staged_families.empty();
    int any = 0;
    Teuchos::reduceAll(*candidate.comm, Teuchos::REDUCE_MAX, 1, &local, &any);
    if (!any)
        return;
    // Wire-decoded capsules have already passed their original migration
    // proofs. The changed authority receipts determine actual residency.
    candidate.staged_families.clear();
    ClosureScratch nested{candidate_memory.budget};
    std::size_t cap = 0;
    std::shared_ptr<ProductFamilyElectionState> edit;
    collective(*candidate.comm,
        [&]
        {
            cap = nested.budget.limit - nested.budget.used;
            nested.add(cap);
            const auto inherited = source.product_election ? source.product_election->storage_bytes() : 0;
            edit = ProductFamilyElectionState::create(
                std::make_shared<Budget>(Budget{checked_sum(cap, inherited)}), source.product_election.get());
        });
    ClosureScratch retained{*edit->keeper()};
    std::vector<ProductFamilyResult> rows;
    collect_product_families(candidate, &source, dirty, transaction, *edit, retained, rows);
    collective(*candidate.comm,
        [&]
        {
            const auto shared = edit->inherited_bytes();
            const auto new_bytes =
                checked_sum(ProductFamilyElectionState::shell_bytes(), edit->storage_bytes() - shared);
            const auto kept = checked_sum(new_bytes, retained.held);
            if (kept > cap || shared > source_overlap.held)
                throw std::logic_error("Product election residency differs from its admitted union.");
            nested.drop(cap - kept);
            nested.held -= kept;
            source_overlap.held -= shared;
            candidate_memory.held += checked_sum(new_bytes, shared);
            family_memory.held += retained.held;
            retained.drop(retained.held);
            edit->finish_overlap();
            candidate.product_election = std::move(edit);
            for (auto& row : rows)
            {
                if (row.family)
                    journal.set_product(std::move(*row.family), *candidate.kernel, family_memory);
                else if (!source.family_layout->contains(row.key))
                    throw std::invalid_argument("Product winner tombstone lacks an accepted owner.");
                else if (journal.contains(row.key))
                    journal.erase(row.key, family_memory);
            }
        });
}

std::size_t foreign_source_fixed_bytes(const DistributedAdaptiveHierarchy::Data& state)
{
    const auto cache = state.native_publication_cache ? state.native_publication_cache->allocation : nullptr;
    const std::array allocations{cache, state.layout_allocation, state.token_allocation, state.topology_allocation};
    struct FixedOwners
    {
        // Each of four envelopes can hold one State, Counter and mirror.
        std::array<std::shared_ptr<const void>, 16> owners;
        std::size_t size = 0, bytes = 0;
    } fixed;
    const geometry_detail::NativeAllocationGraphSink sink{&fixed,
        [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes, bool)
        {
            auto& value = *static_cast<FixedOwners*>(raw);
            const auto end = value.owners.begin() + value.size;
            if (std::any_of(value.owners.begin(), end, [&](const auto& other)
                    { return other.get() == owner.get() && !other.owner_before(owner) && !owner.owner_before(other); }))
                return false;
            if (end == value.owners.end())
                throw std::logic_error("Native source fixed owner inventory exceeds its typed bound.");
            value.owners[value.size++] = owner;
            value.bytes = persistent_ordered_detail::checked_sum(value.bytes, bytes);
            return true;
        }};
    const auto arena = state.contact_halo ? state.contact_halo->allocation_arena() : nullptr;
    for (const auto& allocation : allocations)
    {
        if (!allocation) continue;
        const auto& origin = allocation->state()->mirror.owner;
        if (arena && origin.get() == arena.get() && !origin.owner_before(arena) && !arena.owner_before(origin))
            continue;
        geometry_detail::NativeAllocationGraphAccess::fixed(allocation, sink, false);
    }
    return fixed.bytes;
}
void charge_source_base_residency(Budget& budget, const DistributedAdaptiveHierarchy::Data& state)
{
    // A source fixed allocation can be paid in a detached input domain or
    // mirrored into this halo. Only the first case is an input charge.
    budget.add(foreign_source_fixed_bytes(state));
    budget.add(state.initial_kernel_owners.payload_bytes());
    budget.add(state.initial_kernel_owners.index_admission_bytes());
    budget.add(state.owned_layout_bytes);
    if (state.admission)
    {
        budget.add(state.admission->bytes);
        if (state.bootstrap_family_admission)
            budget.add(state.body_usage.payload_bytes() - state.body_usage.leaf_payload_bytes());
        return;
    }
    budget.add(sizeof(DistributedAdaptiveHierarchy::Data));
    budget.add(state.family_body_residency());
    budget.add(state.family_layout_residency());
    budget.add(state.staged_family_bound);
    budget.add(state.cells->storage_bytes());
    budget.add(state.family_directory->storage_bytes());
    for (const auto& [key, family] : state.staged_families)
    {
        budget.add(sizeof(decltype(state.staged_families)::value_type) + 4 * sizeof(void*) + key.capacity());
        fetch_copy_charge(budget, family);
    }
}
void charge_source_residency(Budget& budget, const DistributedAdaptiveHierarchy::Data& state,
    std::size_t& outside_publications, std::uint64_t* revision = nullptr)
{
    if (state.geometry_cache)
    {
        std::lock_guard lock(state.geometry_cache->mutex);
        state.geometry_cache->admit_residency(budget);
        if (revision) *revision = state.geometry_cache->revision;
        if (!state.geometry_cache->source_store.empty())
        {
            const auto contacts = state.geometry_cache->source_store.contact_credit(*state.contact_halo);
            if (contacts > state.geometry_cache->owner_union.resident_bytes())
                throw std::logic_error("Source/contact fork partition differs from its exact sealed component.");
            // fork_candidate imports H itself. Its caller owns U minus the
            // exact already represented contact component, plus B's residual.
            budget.used -= contacts;
            budget.add(state.geometry_cache->source_partition.exclusive_source_admission);
            const auto& arena = state.contact_halo->allocation_arena();
            const auto live = NativeSourceOwnerStore::publication_mirror_bytes(arena);
            const auto in_union = state.geometry_cache->owner_union.publication_h_paid_bytes(arena);
            if (in_union > live)
                throw std::logic_error("Publication union credit exceeds the actual outstanding mirror requests.");
            outside_publications = live - in_union;
            budget.add(outside_publications);
            return;
        }
    }
    charge_source_base_residency(budget, state);
}

/** Close a newly built hierarchy's current source before it becomes visible.
 * All call sites own private Data and have completed the physical edit. The
 * accepted hierarchy is never mutated by this bootstrap/replacement step. */
void refresh_native_source(DistributedAdaptiveHierarchy::Data& state, DistributedAdaptiveContactHalo& halo,
    ClosureScratch* source_overlap = nullptr, ClosureScratch* candidate_memory = nullptr,
    bool copy_publication_cache = false, std::initializer_list<ClosureScratch*> transient_memory = {})
{
    if (!state.native_source_enabled) return;
    if (!state.contact_halo || state.contact_halo.get() != &halo || !state.layout() ||
        !state.bootstrap_family_admission)
        throw std::invalid_argument("Native source publication lacks its final detached source floors.");
    const auto caller_live = [&]
    {
        std::size_t bytes = 0;
        const auto add = [&](const ClosureScratch* memory)
        {
            if (!memory) return;
            if (&memory->budget != &halo.allocation())
                throw std::invalid_argument("Source publication has a foreign caller scratch arena.");
            bytes = persistent_ordered_detail::checked_sum(bytes, memory->held);
        };
        add(source_overlap);
        add(candidate_memory);
        for (auto* memory : transient_memory) add(memory);
        return bytes;
    };
    // This is a private final halo. Retire its old metadata overlap, and mark
    // every caller-owned scratch category before reading direct contact H.
    // No accepted halo or source floor is changed by this step.
    halo.bind_external_retained_bytes(caller_live());
    auto& cache = *state.geometry_cache;
    const std::lock_guard geometry_lock(cache.mutex);
    cache.require_editable();
    if (cache.lease)
        throw std::logic_error("Source refresh requires a private new hierarchy cache.");
    Budget source_base{state.options.byte_budget};
    charge_source_base_residency(source_base, state);
    const auto prefix = state.source_prefix_for_refresh();
    const geometry_detail::NativeConstructionAdmission admission(
        geometry_detail::NativeConstructionCounterAccess::create<Budget>(state.options.byte_budget, prefix.bytes(),
            NativeSourceOwnerStore::mirror(state.contact_halo->allocation_arena())));
    struct PrefixRelease
    {
        const geometry_detail::NativeConstructionAdmission& admission;
        std::size_t bytes;
        ~PrefixRelease() { admission.release_prefix(bytes); }
    } release{admission, prefix.bytes()};
    NativeSourceBoundInput bound_input;
    if (state.native && !state.map_token)
    {
        if (!geometry_detail::NativePartitionAccess::allocation(state.native) ||
            !geometry_detail::NativeTpetraMapAdmission::admits(state.owned_map))
            throw std::invalid_argument("Qualified source binding lacks its actual admitted partition or owned map.");
        auto token = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceBoundInput::Map>(
            admission, state.owned_map);
        state.map_token = std::move(token.owner);
        state.map_token_allocation = std::move(token.envelope);
        bound_input = {state.native, state.map_token, state.map_token_allocation};
    }

    const auto body_group = [&](AdaptiveLeafBodyRole role,
                                std::shared_ptr<const ContactExternalResidencyLease> floor)
    {
        const auto bytes = role == AdaptiveLeafBodyRole::Leaf ? state.body_usage.leaf_payload_bytes() :
            state.body_usage.payload_bytes() - state.body_usage.leaf_payload_bytes();
        const auto& arena = state.contact_halo->allocation_arena();
        const bool same_arena = floor && floor->arena_identity().get() == arena.get() &&
            !floor->arena_identity().owner_before(arena) && !arena.owner_before(floor->arena_identity());
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
        if (!same_arena)
        {
            // Bootstrap's leaf arena is foreign to the halo. Mirror only its
            // exact physical role; its original floor grants no halo credit.
            auto mirrored = geometry_detail::NativeAllocationEnvelopeAccess::make<ContactExternalResidencyLease>(
                admission, state.contact_halo->external_residency_lease());
            if (source_overlap && state.initial_kernel_owners.owners[0].owner_count())
            {
                if (bytes > source_overlap->held)
                    throw std::logic_error("Initial source owner floor exceeds its pre-admitted source input.");
                mirrored.owner->adopt_precharged(bytes);
                source_overlap->held -= bytes;
            }
            else
                mirrored.owner->resize(bytes);
            floor = std::move(mirrored.owner);
            allocation = std::move(mirrored.envelope);
        }
        return NativeSourceOwnerStore::body_coverage(state.body_usage, role, std::move(floor), admission,
            state.topology, std::move(allocation));
    };
    std::array<std::shared_ptr<const NativeSourceCoverage>, 4> groups{
        body_group(AdaptiveLeafBodyRole::Leaf, state.leaf_body_admission()),
        body_group(AdaptiveLeafBodyRole::Family, state.bootstrap_family_admission), {}, {}};
    halo.bind_external_retained_bytes(caller_live());
    auto contact = halo.source_coverage(admission);
    for (std::size_t slot = 0; slot < contact.coverage.size(); ++slot)
        groups[slot + 2] = std::move(contact.coverage[slot]);
    auto coverage = NativeSourceOwnerStore::composite_coverage(cache.source_store.coverage(),
        groups, admission, state.topology);
    auto components = cache.source_store.components();
    components[0] = state.body_usage.owners();
    components[1] = std::move(contact.owners);
    if (!components[2].owners[0].owner_count())
        components[2] = state.initial_kernel_owners;
    if (!components[2].owners[0].owner_count())
        throw std::invalid_argument("Native source lacks its pre-admitted factory kernel owners.");
    if (copy_publication_cache)
        state.native_publication_cache = NativePublicationSession::copy_cache(state.native_publication_cache, admission);
    std::shared_ptr<const NativePublicationIndexes> indexes;
    NativeSourcePublicationValues publication_values;
    {
        const std::lock_guard publication_lock(state.native_publication_cache->mutex);
        const auto& allocation = state.native_publication_cache->allocation;
        if (!allocation || !allocation->authentic_owner(allocation) || !allocation->matches(state.native_publication_cache))
            throw std::invalid_argument("Source publication controller has a foreign actual allocation/control.");
        indexes = state.native_publication_cache->indexes;
        publication_values = {allocation, state.native_publication_cache->receipt, std::move(bound_input),
            state.source_observations(), {}};
    }
    auto component_paid = cache.source_store.component_paid();
    component_paid[0] = state.body_usage.paid_receipts();
    component_paid[1] = std::move(contact.paid);
    auto source = NativeSourceOwnerStore::assemble(cache.source_store, std::move(components), admission, nullptr,
        std::move(indexes), {state.body_usage.administration(), state.contact_halo->source_owner_administration()},
        std::move(coverage), state.contact_halo->subface_snapshot(), std::move(component_paid),
        contact.projected_admission_bytes, state.contact_halo->allocation_arena(), std::move(publication_values),
        std::move(contact.side_indexes));
    // A bind may have replaced its copied foreign controller with a new
    // same-halo allocation. Recompute B before sealing its disjoint component.
    source_base.used = 0;
    charge_source_base_residency(source_base, state);
    const auto projected_source = persistent_ordered_detail::checked_sum(state.body_usage.projected_bytes(),
        persistent_ordered_detail::checked_sum(foreign_source_fixed_bytes(state),
            persistent_ordered_detail::checked_sum(state.initial_kernel_owners.payload_bytes(),
                state.initial_kernel_owners.index_admission_bytes())));
    auto partition = SourceAccountingPartition::split(source_base.used, projected_source, 0, 0);
    // First source issuance is a full physical capture. Native bookkeeping
    // admits its actual allocations, while no source-paid authority is assumed
    // for owners not yet present in the accepted source root.
    std::size_t cursor = persistent_ordered_detail::checked_sum(partition.residual_bytes(),
        sizeof(AdaptiveGeometrySnapshotCache));
    PLIC::GeometryAllocationLedger ordinary, bounded;
    PLIC::details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(
        ordinary, admission, cursor, state.options.byte_budget);
    PLIC::details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(
        bounded, admission, cursor, state.options.byte_budget);
    auto owners = source.replace_in(cache.source_store, cache.owner_union, ordinary, bounded, cursor,
        state.options.byte_budget);
    partition.exclusive_contact_admission = AdaptiveGeometrySnapshotCache::contact_residual(halo, owners);
    Budget combined{state.options.byte_budget, partition.residual_bytes()};
    if (combined.used > combined.limit)
        throw std::invalid_argument("Native source exclusive input and contact residency exceed budget.");
    combined.add(sizeof(AdaptiveGeometrySnapshotCache));
    combined.add(owners.resident_bytes());
    auto lease = state.contact_halo->external_residency_lease();
    const auto uncovered = AdaptiveGeometrySnapshotCache::uncovered_bytes(owners, lease.arena_identity());
    if (source_overlap || candidate_memory)
    {
        const auto available = persistent_ordered_detail::checked_sum(
            source_overlap ? source_overlap->held : 0, candidate_memory ? candidate_memory->held : 0);
        if (uncovered > available)
            throw std::logic_error("New source union exceeds its exact same-arena pre-admitted source overlap.");
        lease.adopt_precharged(uncovered);
        const auto inherited = source_overlap ? std::min(uncovered, source_overlap->held) : 0;
        if (source_overlap) source_overlap->held -= inherited;
        if (candidate_memory) candidate_memory->held -= uncovered - inherited;
    }
    else lease.resize(uncovered);
    cache.source_store.swap(source);
    cache.owner_union = std::move(owners);
    cache.source_partition = partition;
    cache.lease.emplace(std::move(lease));
    state.initial_kernel_owners = {};
    cache.seal_residency(cache.owner_union.resident_bytes());
    ++cache.revision;
}

void inherit_geometry_predecessors(
    AdaptiveGeometrySnapshotCache& candidate, const AdaptiveGeometrySnapshotCache& source)
{
    std::lock_guard lock(source.mutex);
    source.require_editable();
    candidate.copy_roots_from(source);
    for (std::size_t slot = 0; slot < source.roots.size(); ++slot)
        candidate.predecessors[slot] = source.roots[slot];
}

void normalize_candidate(DistributedAdaptiveHierarchy::Data& candidate, ClosureScratch& leaf_memory)
{
    ClosureScratch scratch{leaf_memory.budget};
    std::vector<AdaptiveLeaf> selected;
    const auto append = [&](const AdaptiveLeaf& row)
    {
        scratch.append(selected);
        charge_candidate_copy(scratch, row);
        selected.push_back(row);
    };
    if (candidate.kernel->supports_leaf_local_repack())
        candidate.overlay().visit_edits(
            [&](const AdaptiveLeafPages::EditView& edit)
            {
                if (edit.leaf)
                    append(*edit.leaf);
            });
    else
        candidate.overlay().visit_owned(append);

    if (selected.empty())
        return;
    const auto cap = scratch.budget.limit - scratch.budget.used;
    if (!cap)
        throw std::invalid_argument("Adaptive leaf repack lease exhausted.");
    scratch.add(cap);
    auto packed = candidate.kernel->repack(selected);
    const auto retained = fetch_retention(packed, {}, cap, false);
    const auto spare = packed.capacity() - packed.size();
    if (spare > (cap - retained) / sizeof(AdaptiveLeaf))
        throw std::invalid_argument("Adaptive repack vector capacity exceeds lease.");
    scratch.drop(cap - retained - spare * sizeof(AdaptiveLeaf));
    if (packed.size() != selected.size())
        throw std::invalid_argument("Adaptive repack changed its active selection.");
    std::sort(packed.begin(), packed.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    for (std::size_t i = 0; i < packed.size(); ++i)
    {
        auto& row = packed[i];
        const auto& original = selected[i];
        if (row.key != original.key || row.revision != original.revision || row.material != original.material ||
            row.provider != original.provider || row.domain != original.domain ||
            row.parent_family != original.parent_family)
            throw std::invalid_argument("Adaptive repack changed a typed leaf binding.");
        candidate.kernel->validate(row);
        row.work = original.work;
        candidate.overlay().replace_active(std::move(row));
    }
    scratch.held -= retained;
    leaf_memory.held += retained;
}

struct SealedLeafStorage
{
    std::shared_ptr<Budget> arena;
    std::size_t new_pages = 0, new_index = 0;
    std::vector<AdaptiveCellKey> product_edits;
};

SealedLeafStorage seal_candidate_layout(DistributedAdaptiveHierarchy::Data& candidate,
    const DistributedAdaptiveHierarchy::Data& source, ClosureScratch& source_overlap, ClosureScratch& candidate_memory,
    ClosureScratch& leaf_memory, ClosureScratch& family_memory, DistributedAdaptiveContactHalo& halo)
{
    const bool native_layout = candidate.native_source_enabled;
    const auto layout_header = candidate_layout_header_bytes - (native_layout ? legacy_layout_allocation : 0);
    candidate_memory.add(layout_header);
    ClosureScratch page_memory{candidate_memory.budget};
    std::vector<AdaptiveLeafPages::EditView> edits;
    const auto count = candidate.overlay().edit_count();
    if (count > page_memory.budget.limit / sizeof(AdaptiveLeafPages::EditView))
        throw std::invalid_argument("Adaptive page edit view exceeds budget.");
    page_memory.add(count * sizeof(AdaptiveLeafPages::EditView));
    edits.reserve(count);
    if (edits.capacity() != count)
        throw std::invalid_argument("Adaptive page edit view capacity differs from admission.");
    std::vector<AdaptiveCellKey> product_edits;
    candidate.overlay().visit_edits(
        [&](const auto& edit)
        {
            edits.push_back(edit);
            const auto* previous = source.layout()->find(edit.key);
            if ((previous && previous->provider == AdaptiveProviderKind::Product) ||
                (edit.leaf && edit.leaf->provider == AdaptiveProviderKind::Product))
            {
                family_memory.append(product_edits);
                family_memory.add(fetch_key_bytes(edit.key));
                product_edits.push_back(edit.key);
            }
        });

    const auto cap = page_memory.budget.limit - page_memory.budget.used;
    page_memory.add(cap);
    auto arena = std::make_shared<Budget>(Budget{cap});
    AdaptiveLeafPages pages;
    AdaptiveLeafBodyDelta::Result bodies;
    std::size_t new_pages = 0;
    {
        AdaptiveLeafBodyDelta delta(arena);
        const AdaptiveLeafPages::PageChanges callbacks{
            [&](auto rows) { delta.record(rows, false); }, [&](auto rows) { delta.record(rows, true); }};
        pages = AdaptiveLeafPages::edit_borrowed(candidate.overlay().source(), edits, arena, nullptr, &callbacks);
        new_pages = arena->used - delta.metadata_bytes();
        bodies = delta.apply(source.body_usage);
    }
    if (pages.size() != candidate.overlay().size())
        throw std::logic_error("Sealed adaptive selection differs from its edit view.");
    std::optional<std::size_t> previous;
    for (const auto& edit : edits)
    {
        const auto ordinal = pages.ordinal_of(edit.key);
        if (!edit.leaf)
        {
            if (ordinal)
                throw std::logic_error("Sealed adaptive selection retains a retired key.");
            continue;
        }
        if (!ordinal || (previous && *ordinal <= *previous))
            throw std::logic_error("Sealed adaptive leaf order differs from its edit view.");
        const auto& actual = pages.at_ordinal(*ordinal);
        if (actual.revision != edit.leaf->revision || actual.native_cell != edit.leaf->native_cell ||
            actual.native_recipe != edit.leaf->native_recipe || actual.geometry != edit.leaf->geometry)
            throw std::logic_error("Sealed adaptive leaf binding differs from its edit view.");
        previous = ordinal;
    }

    const auto new_storage = arena->used;
    const auto new_index = new_storage - new_pages;
    page_memory.drop(cap - new_storage);
    Budget resident{candidate_memory.budget.limit};
    const auto page_bytes = pages.retained_bytes();
    const auto leaf_bytes = bodies.usage.leaf_payload_bytes();
    if (new_pages > page_bytes || bodies.new_payload_bytes > leaf_bytes)
        throw std::logic_error("Adaptive leaf residency differs from new allocations.");
    const auto shared_pages = page_bytes - new_pages;
    const auto shared_leaf = leaf_bytes - bodies.new_payload_bytes;
    resident.add(page_bytes);
    resident.add(leaf_bytes);
    resident.add(layout_header);
    if (shared_pages > source.pages().retained_bytes() || shared_pages > source_overlap.held ||
        shared_leaf > source_overlap.held - shared_pages || shared_leaf > source.body_usage.payload_bytes() ||
        bodies.new_payload_bytes > leaf_memory.held)
        throw std::logic_error("Adaptive layout residency exceeds its source or delta admission.");

    auto lease = std::make_shared<ContactExternalResidencyLease>(halo.external_residency_lease());
    ScopedSourceAdmission native_storage(candidate.options.byte_budget, halo.allocation().used,
        halo.allocation_arena());
    candidate.set_identity(native_layout ? &native_storage.admission : nullptr);
    // The edit views still borrow keys and leaves from the candidate overlay.
    auto publication_cache =
        NativePublicationSession::edited(source.native_publication_cache, edits, native_storage.admission);
    candidate.set_layout(std::move(pages), lease, native_layout ? &native_storage.admission : nullptr);
    candidate.body_usage = std::move(bodies.usage);
    candidate_memory.drop(overlay_header_bytes);
    lease->adopt_precharged(resident.used);
    candidate.owned_layout_bytes = resident.used;
    source_overlap.held -= shared_pages + shared_leaf;
    candidate_memory.held -= layout_header;
    candidate_memory.held += new_index;
    page_memory.held -= new_storage;
    leaf_memory.held -= bodies.new_payload_bytes;
    candidate.contacts = halo.seal_contacts(candidate.transition_from(source, arena));
    candidate.native_publication_cache = std::move(publication_cache);
    // A decoder or split may have charged an owner first through a leaf that
    // repacking replaced. Saved parents can still pin it until family seal.
    return {std::move(arena), new_pages, new_index, std::move(product_edits)};
}

bool same_family_bodies(const AdaptiveLeaf& left, const AdaptiveLeaf& right)
{
    if (left.geometry != right.geometry || left.native_recipe != right.native_recipe ||
        left.join_recipe != right.join_recipe || left.factor_recipe != right.factor_recipe ||
        left.surfaces.size() != right.surfaces.size())
        return false;
    for (std::size_t face = 0; face < left.surfaces.size(); ++face)
        if (left.surfaces[face].geometry != right.surfaces[face].geometry ||
            left.surfaces[face].native_whole_geometry != right.surfaces[face].native_whole_geometry)
            return false;
    return true;
}

void seal_candidate_families(DistributedAdaptiveHierarchy::Data& candidate,
    const DistributedAdaptiveHierarchy::Data& source, const SealedLeafStorage& leaf_storage, FamilyJournal& journal,
    ClosureScratch& source_overlap, ClosureScratch& candidate_memory, ClosureScratch& leaf_memory,
    ClosureScratch& family_memory)
{
    using persistent_ordered_detail::checked_product;
    using persistent_ordered_detail::checked_sum;
    candidate_memory.add(family_layout_header_bytes);
    ClosureScratch edit_memory{candidate_memory.budget};
    std::vector<AdaptiveFamilyStorage::Edit> edits;
    edit_memory.add(checked_product(journal.edits().size(), sizeof(AdaptiveFamilyStorage::Edit)));
    edits.reserve(journal.edits().size());
    if (edits.capacity() != journal.edits().size())
        throw std::invalid_argument("Saved family edit capacity differs from admission.");
    for (const auto& [key, change] : journal.edits())
    {
        const auto previous = source.family_layout->find(key);
        if (change.family)
        {
            const auto& family = *change.family;
            const auto anchor = decode_adaptive_key(family.anchor);
            if (!candidate.contains_active(anchor) || !descendant(family.parent.key, anchor))
                throw std::invalid_argument("Saved family change lacks an active typed descendant anchor.");
            if (!change.accepted_parent)
                validate_family(*candidate.kernel, family);
        }
        edits.push_back({key, previous ? std::optional(previous->revision) : std::nullopt,
            change.family ? &*change.family : nullptr});
    }

    ClosureScratch index_memory{candidate_memory.budget};
    const auto cap = index_memory.budget.limit - index_memory.budget.used;
    index_memory.add(cap);
    const auto& arena = leaf_storage.arena;
    const auto source_family_bytes = AdaptiveFamilyStorage::storage_bytes(*source.family_layout);
    // The independent page arena imports the accepted family union. The
    // enclosing operation already holds those same bytes as source overlap.
    arena->limit = checked_sum(checked_sum(arena->used, cap), source_family_bytes);
    std::size_t new_payload = 0;
    auto layout = [&]
    {
        AdaptiveLeafBodyDelta delta(arena);
        AdaptiveFamilyStorage::Work work;
        auto result = AdaptiveFamilyStorage::edited(*source.family_layout, edits, arena, candidate.topology, work,
            [&](const AdaptiveLeaf* before, const AdaptiveLeaf* after)
            {
                if (before && after && same_family_bodies(*before, *after))
                    return;
                if (before)
                    delta.record(std::span(before, 1), false);
                if (after)
                    delta.record(std::span(after, 1), true);
            });
        auto bodies = delta.apply(candidate.body_usage, AdaptiveLeafBodyRole::Family, &source.body_usage);
        new_payload = bodies.new_payload_bytes;
        candidate.body_usage = std::move(bodies.usage);
        return result;
    }();
    const auto shared_family = AdaptiveFamilyStorage::inherited_bytes(layout);
    const auto new_family = AdaptiveFamilyStorage::storage_bytes(layout) - shared_family;
    const auto reserved = checked_sum(checked_sum(leaf_storage.new_pages, new_family), source_family_bytes);
    if (reserved > arena->used)
        throw std::logic_error("Saved family page residency exceeds the working arena.");
    const auto new_index = arena->used - reserved;
    if (new_index >= leaf_storage.new_index)
    {
        const auto added = checked_sum(new_family, new_index - leaf_storage.new_index);
        if (added > cap)
            throw std::logic_error("Saved family allocation exceeds its new-storage lease.");
        index_memory.drop(cap - added);
        index_memory.held -= added;
        candidate_memory.held += added;
    }
    else
    {
        if (new_family > cap)
            throw std::logic_error("Saved family allocation exceeds its new-storage lease.");
        index_memory.drop(cap - new_family);
        index_memory.held -= new_family;
        candidate_memory.held += new_family;
        candidate_memory.drop(leaf_storage.new_index - new_index);
    }

    const auto index_bytes = candidate.body_usage.storage_bytes();
    const auto family_bytes = candidate.body_usage.payload_bytes() - candidate.body_usage.leaf_payload_bytes();
    if (new_index > index_bytes || new_payload > family_bytes || shared_family > source_family_bytes)
        throw std::logic_error("Adaptive family residency differs from new allocations.");
    const auto shared_index = index_bytes - new_index;
    const auto shared_payload = family_bytes - new_payload;
    const auto shared = checked_sum(checked_sum(shared_index, shared_payload), shared_family);
    if (shared_index > source.body_usage.storage_bytes() || shared > source_overlap.held)
        throw std::logic_error("Saved family residency exceeds its source admission.");

    const auto family_charge = std::min(new_payload, family_memory.held);
    const auto leaf_charge = new_payload - family_charge;
    if (leaf_charge > leaf_memory.held)
        throw std::logic_error("Saved family residency exceeds its candidate admission.");
    source_overlap.held -= shared;
    family_memory.held -= family_charge;
    leaf_memory.held -= leaf_charge;
    candidate_memory.held += shared + new_payload;
    // The outer source lease retains replaced allocations through acceptance.
    AdaptiveFamilyStorage::finish_overlap(layout);
    candidate.family_layout = std::make_shared<const AdaptiveFamilyLayout>(std::move(layout));
    edits.clear();
    journal.clear();
    family_memory.drop(family_memory.held);
    leaf_memory.drop(leaf_memory.held);
}

void close_candidate(DistributedAdaptiveHierarchy::Data& candidate, const DistributedAdaptiveHierarchy::Data& source,
    std::span<const AdaptiveOwner> initially_retired, std::span<const AdaptiveOwner> initially_inserted,
    DistributedAdaptiveContactHalo& frontier, ContactHaloContext context, FamilyJournal& journal,
    ClosureScratch& memory, ClosureScratch& family_memory, ClosureScratch& leaf_memory)
{
    DistributedAdaptiveOwnerOverlay overlay(*source.cells, candidate.comm, frontier.allocation());
    overlay.apply(initially_retired, initially_inserted);
    ContactOwnedLeafLookup leaf_lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
    { return candidate.overlay().at(key); };
    std::vector<ContactRetiredLeaf> initial_retire_keys;
    std::vector<AdaptiveCellKey> initial_insert_keys;
    collective(*candidate.comm,
        [&]
        {
            for (const auto& item : initially_retired)
            {
                memory.append(initial_retire_keys);
                if (item.key.size() > memory.budget.limit / 2)
                    throw std::invalid_argument("Initial contact key exceeds closure budget.");
                memory.add(2 * item.key.size());
                initial_retire_keys.push_back({decode_adaptive_key(item.key), item.revision});
            }
            for (const auto& item : initially_inserted)
            {
                memory.append(initial_insert_keys);
                if (item.key.size() > memory.budget.limit / 2)
                    throw std::invalid_argument("Initial contact key exceeds closure budget.");
                memory.add(2 * item.key.size());
                initial_insert_keys.push_back(decode_adaptive_key(item.key));
            }
        });
    context.epoch = candidate.native_epoch;
    context.topology = candidate.topology;
    context.partition = candidate.partition;
    ++context.round;
    frontier.apply_delta(initial_retire_keys, initial_insert_keys, context, leaf_lookup);
    std::vector<ContactRetiredLeaf>().swap(initial_retire_keys);
    std::vector<AdaptiveCellKey>().swap(initial_insert_keys);
    for (unsigned round = 0; round < candidate.options.maximum_closure_rounds; ++round)
    {
        ClosureScratch scratch{frontier.allocation()};
        auto needed_lease = frontier.lease_required_refinements();
        const auto needed = needed_lease.marks();
        std::uint64_t local = needed.size(), global = 0;
        Teuchos::reduceAll(*candidate.comm, Teuchos::REDUCE_SUM, 1, &local, &global);
        if (!global)
            return;
        std::vector<AdaptiveBytes> keys;
        std::vector<AdaptiveSplit> kinds;
        collective(*candidate.comm,
            [&]
            {
                for (const auto& mark : needed)
                {
                    const auto cap = scratch.budget.limit - scratch.budget.used;
                    if (!cap)
                        throw std::invalid_argument("Closure refinement group lease exhausted.");
                    scratch.add(cap);
                    auto members = candidate.kernel->refinement_group(mark.key, mark.split);
                    Budget member_budget{cap};
                    if (members.capacity() > cap / sizeof(AdaptiveCellKey))
                        throw std::invalid_argument("Closure refinement group exceeds lease.");
                    member_budget.add(members.capacity() * sizeof(AdaptiveCellKey));
                    for (const auto& member : members)
                        member_budget.add(fetch_key_bytes(member));
                    scratch.drop(cap - member_budget.used);
                    for (const auto& member : members)
                    {
                        scratch.append(keys);
                        scratch.append(kinds);
                        const auto dynamic = fetch_key_bytes(member);
                        if (scratch.budget.limit < 256 || dynamic > (scratch.budget.limit - 256) / 4)
                            throw std::invalid_argument("Closure key encoding exceeds budget.");
                        scratch.add(4 * (dynamic + 64));
                        auto key = encode_adaptive_key(member);
                        keys.push_back(std::move(key));
                        kinds.push_back(mark.split);
                    }
                }
            });
        std::vector<OwnerPacket> packets;
        {
            auto owners_lease = overlay.lookup(keys);
            const auto& owners = owners_lease.values();
            collective(*candidate.comm,
                [&]
                {
                    for (std::size_t i = 0; i < owners.size(); ++i)
                    {
                        scratch.append(packets);
                        if (scratch.budget.limit < 128 || owners[i].key.size() > (scratch.budget.limit - 128) / 4)
                            throw std::invalid_argument("Closure request exceeds budget.");
                        scratch.add(4 * (owners[i].key.size() + 32));
                        Writer w;
                        w.bytes(owners[i].key);
                        w.word(std::uint64_t(kinds[i]));
                        w.word(owners[i].revision);
                        packets.push_back({owners[i].owner, std::move(w.data)});
                    }
                });
        }
        auto requests = closure_exchange(packets, candidate.comm, scratch);
        std::vector<AdaptiveOwner> retired, inserted;
        std::vector<ContactRetiredLeaf> retired_keys;
        std::vector<AdaptiveCellKey> inserted_keys;
        collective(*candidate.comm,
            [&]
            {
                std::map<AdaptiveCellKey, AdaptiveSplit> marks;
                for (const auto& packet : requests.packets)
                {
                    if (packet.bytes.size() > scratch.budget.limit / 4)
                        throw std::invalid_argument("Closure request decode exceeds budget.");
                    scratch.add(4 * packet.bytes.size());
                    scratch.add(sizeof(std::pair<const AdaptiveCellKey, AdaptiveSplit>) + 4 * sizeof(void*));
                    Reader r{packet.bytes};
                    auto key = decode_adaptive_key(r.bytes());
                    auto split = r.word(), revision = r.word();
                    r.done();
                    const auto* row = candidate.overlay().find(key);
                    if (split > 2 || !row || row->revision != revision)
                        throw std::invalid_argument("Stale distributed closure group.");
                    auto [mark, inserted] = marks.emplace(key, AdaptiveSplit(split));
                    if (!inserted && mark->second != AdaptiveSplit(split))
                    {
                        if (row->provider != AdaptiveProviderKind::Product)
                            throw std::invalid_argument("Incompatible native closure factors.");
                        mark->second = AdaptiveSplit::Isotropic;
                    }
                }
                for (const auto& [key, split] : marks)
                {
                    const auto origin = original_cover(source, key);
                    if (!origin || repeated_factor(*origin, key, split))
                        throw std::invalid_argument(
                            "Closure would require multiple source generations or undo coarsening.");
                    const auto split_cap = memory.budget.limit - memory.budget.used;
                    if (!split_cap)
                        throw std::invalid_argument("Closure split lease exhausted.");
                    memory.add(split_cap);
                    auto result = candidate.kernel->split(candidate.overlay().at(key), split);
                    std::size_t leaf_bytes = 0;
                    const auto split_retained = fetch_retention(result.children,
                        std::span<const AdaptiveFamily>(&result.family, 1), split_cap, false, &leaf_bytes);
                    memory.drop(split_cap - split_retained);
                    memory.held -= split_retained;
                    leaf_memory.held += leaf_bytes;
                    family_memory.held += split_retained - leaf_bytes;
                    auto old_key = encode_adaptive_key(key);
                    scratch.append(retired);
                    scratch.append(retired_keys);
                    scratch.add(old_key.size() + fetch_key_bytes(key));
                    retired.push_back({old_key, candidate.comm->getRank(), candidate.overlay().at(key).revision});
                    retired_keys.push_back({key, candidate.overlay().at(key).revision});
                    (void) candidate.overlay().retire(key);
                    journal.move_anchor(old_key, result.family.anchor, family_memory);
                    for (auto& child : result.children)
                    {
                        scratch.append(inserted);
                        scratch.append(inserted_keys);
                        const auto dynamic = fetch_key_bytes(child.key);
                        if (scratch.budget.limit < 256 || dynamic > (scratch.budget.limit - 256) / 4)
                            throw std::invalid_argument("Closure child key exceeds budget.");
                        scratch.add(4 * (dynamic + 64));
                        inserted.push_back({encode_adaptive_key(child.key), candidate.comm->getRank(), child.revision});
                        inserted_keys.push_back(child.key);
                        if (!candidate.overlay().insert(std::move(child)))
                            throw std::invalid_argument("Closure child already active.");
                    }
                    // Product factor capsules are regenerated by final register_state; a derived
                    // cross-factor capsule can already exist for this exact family.
                    if (result.family.parent.provider != AdaptiveProviderKind::Product)
                        journal.insert(std::move(result.family), family_memory);
                }
                if (candidate.overlay().size() > candidate.options.maximum_owned)
                    throw std::invalid_argument("Closure local leaf budget exhausted.");
            });
        overlay.apply(retired, inserted);
        ++context.round;
        frontier.apply_delta(retired_keys, inserted_keys, context, leaf_lookup);
    }
    throw std::invalid_argument("Distributed closure round budget exhausted.");
}
} // namespace
DistributedAdaptiveHierarchy::DistributedAdaptiveHierarchy(std::vector<AdaptiveLeaf> owned,
    std::vector<AdaptiveFamily> families, std::shared_ptr<const AdaptiveProviderKernel> kernel,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, DistributedAdaptiveOptions options)
{
    if (comm.is_null())
        throw std::invalid_argument("Distributed hierarchy needs communicator.");
    const bool local_native_source = NativeKernelAuthority::supports_source_validation(kernel) &&
        std::all_of(owned.begin(), owned.end(), [&](const auto& leaf)
            { return NativeKernelAuthority::supports_source_validation(kernel, leaf); }) &&
        std::all_of(families.begin(), families.end(), [&](const auto& family)
            { return NativeKernelAuthority::supports_source_validation(kernel, family.parent); });
    const bool native_source = !::SimpleFluid::owner_packet_failure_agreement(*comm, !local_native_source);
    std::shared_ptr<Data> state;
    collective(*comm,
        [&]
        {
            if (!kernel || !options.byte_budget || !options.maximum_owned || !options.maximum_tiles ||
                !options.maximum_closure_rounds || !std::isfinite(options.maximum_span_ratio) ||
                options.maximum_span_ratio < 1 || owned.size() > options.maximum_owned)
                throw std::invalid_argument("Invalid distributed hierarchy limits.");
            Budget initial{options.byte_budget};
            initial.add(sizeof(Data) + sizeof(AdaptiveGeometrySnapshotCache) + 4 * sizeof(void*));
            PLIC::details::OwnerManifestPair kernel_owners;
            if (native_source)
            {
                PLIC::details::GeometryOwnerGraphEditor graph({}, 0, initial.used, initial.limit);
                NativePublicationOwnerGraph::Context context(graph, true, false);
                NativeKernelAuthority::visit(kernel, context.sink,
                    NativePublicationOwnerGraph::Context::identity, NativePublicationOwnerGraph::Context::geometry);
                kernel_owners = NativeSourceOwnerStore::single(graph.finish());
            }
            ScopedSourceAdmission native_storage(options.byte_budget, initial.used);
            state = std::make_shared<Data>();
            state->comm = comm;
            state->options = options;
            state->kernel = std::move(kernel);
            state->native_source_enabled = native_source;
            state->initial_kernel_owners = std::move(kernel_owners);
            state->native_publication_cache = NativePublicationSession::make_cache(native_storage.admission);
            state->set_identity(native_source ? &native_storage.admission : nullptr);
        });
    collective(*comm,
        [&]
        {
            Budget budget{options.byte_budget};
            budget.add(sizeof(Data) + sizeof(AdaptiveGeometrySnapshotCache) + 4 * sizeof(void*));
            budget.add(state->initial_kernel_owners.payload_bytes());
            budget.add(state->initial_kernel_owners.index_admission_bytes());
            budget.add(foreign_source_fixed_bytes(*state));
            budget.add(bootstrap_layout_header_bytes - (state->token_allocation ? legacy_layout_allocation : 0));
            for (auto& leaf : owned)
            {
                state->kernel->validate(leaf);
                budget.add(encoded_adaptive_key_size(leaf.key) + leaf.domain.size() + leaf.parent_family.size());
                budget.add(sizeof(AdaptiveLeaf));
                if (!state->edit_owned().emplace(leaf.key, std::move(leaf)).second)
                    throw std::invalid_argument("Duplicate owned adaptive leaf.");
            }
            for (auto& family : families)
            {
                if (!state->owned().contains(decode_adaptive_key(family.anchor)))
                    throw std::invalid_argument("Family capsule must reside with actual anchor descendant.");
                state->kernel->validate(family.parent);
                budget.add(family.key.size() + family.anchor.size() + family.children.size() * sizeof(AdaptiveCellKey));
                budget.add(sizeof(AdaptiveFamily));
                if (!state->staged_families.emplace(family.key, std::move(family)).second)
                    throw std::invalid_argument("Duplicate retained family capsule.");
            }
        });
    std::vector<AdaptiveLeaf>().swap(owned);
    std::vector<AdaptiveFamily>().swap(families);
    const std::array<std::uint64_t, 6> description{options.byte_budget, options.maximum_owned, options.maximum_tiles,
        options.maximum_closure_rounds, std::bit_cast<std::uint64_t>(options.maximum_span_ratio), 1};
    agree(*comm, description);
    AdaptiveBytes kernel_descriptor;
    collective(*comm,
        [&]
        {
            kernel_descriptor = state->kernel->descriptor();
            if (kernel_descriptor.size() > options.byte_budget)
                throw std::invalid_argument("Kernel semantic descriptor exceeds budget.");
        });
    std::uint64_t descriptor_size = kernel_descriptor.size(), root_size = descriptor_size;
    Teuchos::broadcast(*comm, 0, 1, &root_size);
    collective(*comm,
        [&]
        {
            if (root_size != descriptor_size || root_size > std::size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Kernel semantic descriptors have different lengths.");
        });
    AdaptiveBytes root_descriptor;
    collective(*comm, [&] { root_descriptor = kernel_descriptor; });
    Teuchos::broadcast(*comm, 0, int(root_size), root_descriptor.data());
    collective(*comm,
        [&]
        {
            if (root_descriptor != kernel_descriptor)
                throw std::invalid_argument("Kernel semantic descriptors differ.");
        });
    register_cells(*state);
    auto product_election = bootstrap_product_families(*state);
    collective(*comm,
        [&]
        {
            intern_state_legacy(*state);
            if (product_election)
            {
                // The intern operation creates the final shared byte owners.
                // Seed the elected index from those actual published controls.
                ProductElectedPartDelta parts(product_election->keeper());
                for (const auto& [key, family] : state->staged_families)
                    if (family.parent.provider == AdaptiveProviderKind::Product)
                        parts.record(family.parent, true);
                parts.apply(*product_election);
                state->product_election = std::move(product_election);
            }
        });
    register_families(*state);
    collective(*comm,
        [&]
        {
            Budget overlap{options.byte_budget};
            charge_source_base_residency(overlap, *state);
            const auto buffer_bytes = state->staged_buffer_bytes();
            if (buffer_bytes > state->owned_layout_bytes)
                throw std::logic_error("Initial adaptive page buffers exceed staged residency.");
            // The staged map keeps its nodes until publication. Moved row
            // buffers are already included in the source residency charge.
            const auto cap = overlap.limit - overlap.used + buffer_bytes;
            state->publish_bootstrap(std::make_shared<Budget>(Budget{cap}), buffer_bytes);
        });
    const auto& layout = *state->layout();
    Budget source_input{options.byte_budget};
    collective(*comm, [&]
    {
        if (state->initial_kernel_owners.owners[0].owner_count())
        {
            charge_source_base_residency(source_input, *state);
            source_input.add(sizeof(AdaptiveGeometrySnapshotCache));
        }
    });
    AdaptiveClosureOptions closure_options;
    closure_options.byte_budget = options.byte_budget;
    closure_options.contact_budget = options.maximum_tiles;
    closure_options.tile_budget = options.maximum_tiles;
    closure_options.maximum_span_ratio = options.maximum_span_ratio;
    std::unique_ptr<void, void (*)(void*)> shell(nullptr, [](void* pointer) { ::operator delete(pointer); });
    collective(*comm, [&]
    {
        if (sizeof(DistributedAdaptiveContactHalo) > source_input.limit - source_input.used)
            throw std::invalid_argument("Initial contact halo shell exceeds the source input budget.");
        shell.reset(::operator new(sizeof(DistributedAdaptiveContactHalo)));
    });
    auto halo = std::unique_ptr<DistributedAdaptiveContactHalo>(new (shell.get()) DistributedAdaptiveContactHalo(
        *state->kernel, comm, closure_options,
        ContactHaloContext{state->native_epoch, state->topology, state->partition, 0, 0}, ContactHaloBeforeSend{},
        source_input.used));
    (void) shell.release();
    std::shared_ptr<DistributedAdaptiveContactHalo> source_halo;
    ClosureScratch source_memory{halo->allocation(), source_input.used};
    {
        ClosureScratch memory{halo->allocation()};
        std::vector<AdaptiveLeaf> records;
        collective(*comm,
            [&]
            {
                if (layout.size() > memory.budget.limit / sizeof(AdaptiveLeaf))
                    throw std::invalid_argument("Initial contact records exceed budget.");
                memory.add(layout.size() * sizeof(AdaptiveLeaf));
                records.reserve(layout.size());
                if (records.capacity() != layout.size())
                    throw std::invalid_argument("Initial contact record capacity differs from admission.");
                for (const auto& leaf : layout)
                {
                    const auto before = memory.budget.used;
                    try
                    {
                        fetch_copy_charge(memory.budget, leaf);
                    }
                    catch (...)
                    {
                        memory.held += memory.budget.used - before;
                        throw;
                    }
                    memory.held += memory.budget.used - before;
                    records.push_back(leaf);
                }
            });
        ContactOwnedLeafLookup lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
        { return layout.at(key); };
        halo->bootstrap_canonical(records, lookup, state->layout(), state->native_source_enabled);
    }
    const auto initial_requirements = halo->lease_required_refinements();
    collective(*comm,
        [&]
        {
            if (!initial_requirements.marks().empty())
                throw std::invalid_argument("Initial distributed hierarchy contact grading is inadmissible.");
        });
    collective(*comm, [&] { state->contacts = halo->seal_contacts(state->layout()); });
    collective(*comm, [&] { source_halo = std::shared_ptr<DistributedAdaptiveContactHalo>(std::move(halo)); });
    state->contact_halo = source_halo;
    collective(*comm, [&] { refresh_native_source(*state, *source_halo, &source_memory); });
    source_memory.drop(source_memory.held);
    source_halo->bind_external_retained_bytes(0);
    data_ = std::move(state);
}
const std::shared_ptr<const AdaptiveLeafLayout>& DistributedAdaptiveHierarchy::leaf_layout() const noexcept
{
    return data_->layout();
}
std::vector<AdaptiveFamily> adaptive_detail::product_family_reference(const DistributedAdaptiveHierarchy& hierarchy)
{
    auto state = *hierarchy.data_;
    Budget budget{state.options.byte_budget};
    ClosureScratch retained{budget};
    full_product_family_reference(state, retained);
    std::vector<AdaptiveFamily> result;
    result.reserve(state.staged_families.size());
    for (auto& [key, family] : state.staged_families)
        result.push_back(std::move(family));
    return result;
}
const DistributedAncestryDirectory& adaptive_detail::DirectoryRegistrationAccess::cells(
    const DistributedAdaptiveHierarchy& state)
{
    return *state.data_->cells;
}
const std::shared_ptr<const DistributedAncestryDirectory>& adaptive_detail::DirectoryRegistrationAccess::cell_owner(
    const DistributedAdaptiveHierarchy& state)
{
    return state.data_->cells;
}
const DistributedAncestryDirectory& adaptive_detail::DirectoryRegistrationAccess::families(
    const DistributedAdaptiveHierarchy& state)
{
    return *state.data_->family_directory;
}
std::shared_ptr<const adaptive_detail::AdaptiveContactSnapshot> adaptive_detail::contact_snapshot(
    const DistributedAdaptiveHierarchy& hierarchy)
{
    if (!hierarchy.geometry_unchanged() || !hierarchy.data_->contacts)
        throw std::invalid_argument("Closed-contact snapshot source is stale or incomplete.");
    return hierarchy.data_->contacts;
}
std::shared_ptr<const adaptive_detail::AdaptiveSubfaceSnapshot> adaptive_detail::subface_snapshot(
    const DistributedAdaptiveHierarchy& hierarchy)
{
    if (!hierarchy.geometry_unchanged() || !hierarchy.data_->contact_halo)
        throw std::invalid_argument("Canonical subface snapshot source is stale or incomplete.");
    return hierarchy.data_->contact_halo->subface_snapshot();
}
const adaptive_detail::DistributedAdaptiveContactHalo& adaptive_detail::GeometrySnapshotAccess::contact_cache(
    const DistributedAdaptiveHierarchy& hierarchy)
{
    if (!hierarchy.geometry_unchanged() || !hierarchy.data_->contact_halo)
        throw std::invalid_argument("Canonical contact cache source is stale or incomplete.");
    return *hierarchy.data_->contact_halo;
}
const adaptive_detail::DistributedAdaptiveContactHalo& adaptive_detail::contact_cache(
    const DistributedAdaptiveHierarchy& hierarchy)
{
    return GeometrySnapshotAccess::contact_cache(hierarchy);
}
const AdaptiveLeafLayout& DistributedAdaptiveHierarchy::owned_leaves() const noexcept
{
    return *leaf_layout();
}
const AdaptiveFamilyLayout& DistributedAdaptiveHierarchy::owned_families() const noexcept
{
    return *data_->family_layout;
}
std::uint64_t DistributedAdaptiveHierarchy::topology_generation() const noexcept
{
    return data_->topology;
}
std::uint64_t DistributedAdaptiveHierarchy::partition_generation() const noexcept
{
    return data_->partition;
}
const std::shared_ptr<const void>& DistributedAdaptiveHierarchy::identity() const noexcept
{
    return data_->token;
}
const Teuchos::RCP<const Teuchos::Comm<int>>& DistributedAdaptiveHierarchy::communicator() const noexcept
{
    return data_->comm;
}
const DistributedAdaptiveOptions& DistributedAdaptiveHierarchy::options() const noexcept
{
    return data_->options;
}
std::uint64_t DistributedAdaptiveHierarchy::global_cells() const noexcept
{
    return data_->global_cells;
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> DistributedAdaptiveHierarchy::geometry(const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return data_->kernel->geometry(leaf);
}
bool DistributedAdaptiveHierarchy::has_native_cartesian_geometry() const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return AMR::has_native_cartesian_geometry(*data_->kernel);
}
std::optional<AdaptiveCartesianGeometryBounds> DistributedAdaptiveHierarchy::native_cartesian_geometry_bounds(
    const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return AMR::native_cartesian_geometry_bounds(*data_->kernel, leaf);
}
AdaptiveCellMetrics DistributedAdaptiveHierarchy::metrics(const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return data_->kernel->metrics(leaf);
}
std::vector<AdaptiveSurface> DistributedAdaptiveHierarchy::surfaces(const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return data_->kernel->surfaces(leaf);
}
bool DistributedAdaptiveHierarchy::geometry_unchanged() const noexcept
{
    if (!data_->native)
        return true;
    try
    {
        return data_->native->geometry_unchanged() &&
               data_->native->geometry()->geometry_epoch() == data_->native_epoch;
    }
    catch (...)
    {
        return false;
    }
}
HierarchyObservation DistributedAdaptiveHierarchy::observation() const noexcept
{
    std::uint64_t epoch = data_->native_epoch;
    try
    {
        if (data_->native)
            epoch = data_->native->geometry()->geometry_epoch();
    }
    catch (...)
    {
        epoch = std::numeric_limits<std::uint64_t>::max();
    }
    std::shared_ptr<const void> ownership = data_->map_token
                                            ? std::shared_ptr<const void>(data_->map_token, data_->owned_map.getRawPtr())
                                            : data_->native  ? std::shared_ptr<const void>(data_->native)
                                                             : data_->partition_token;
    return {data_->token, data_->topology_token, ownership, data_->topology, data_->partition, epoch,
        data_->native ? data_->native->geometry().get() : nullptr, data_->comm.getRawPtr()};
}
const std::shared_ptr<const CompositePartition<>>& DistributedAdaptiveHierarchy::native_partition() const noexcept
{
    return data_->native;
}
std::shared_ptr<const DistributedAdaptiveHierarchy> DistributedAdaptiveHierarchy::with_partition(
    std::shared_ptr<const CompositePartition<>> partition, Teuchos::RCP<const DefaultTpetraTypes::map_type> map) const
{
    std::shared_ptr<Data> bound;
    Budget binding_overlap{data_->options.byte_budget};
    std::size_t source_charge = 0, outside_publications = 0;
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || !partition || !partition->geometry_unchanged() ||
                partition->plan().communicator().getRawPtr() != data_->comm.getRawPtr() ||
                !partition->geometry()->sparse_native() || partition->geometry()->num_cells() != data_->global_cells ||
                partition->plan().owned_cells().size() != owned_leaves().size())
                throw std::invalid_argument("Foreign, incomplete or moved distributed native binding.");
            if (map.is_null() || map->getComm().getRawPtr() != data_->comm.getRawPtr() ||
                map->getGlobalNumElements() != data_->global_cells ||
                map->getLocalNumElements() != owned_leaves().size() || map->getIndexBase() != 0)
                throw std::invalid_argument("Foreign or incomplete actual distributed owned map.");
            const bool exact_receipt =
                NativePublicationSession::receipt_matches(NativePublicationSession::capture(*this), partition, map);
            std::uint64_t id = data_->cell_begin;
            std::size_t ordinal = 0;
            for (const auto& leaf : owned_leaves())
            {
                if (partition->plan().owned_cells()[ordinal] != id ||
                    map->getGlobalElement(ordinal) != DefaultTpetraTypes::global_ordinal_type(id) ||
                    partition->geometry()->sparse_cell_owner(id) != data_->comm->getRank())
                    throw std::invalid_argument("Native binding does not own actual typed cell GID.");
                const auto [region, local_cell] = partition->geometry()->native_cell(id);
                (void) local_cell;
                const auto family = partition->geometry()->region_layout(region).family;
                if ((leaf.provider == AdaptiveProviderKind::Cartesian && family != RegionLayout::Family::Rectilinear) ||
                    (leaf.provider == AdaptiveProviderKind::Product && family != RegionLayout::Family::Extruded))
                    throw std::invalid_argument("Procedural adaptive family was flattened in native binding.");
                if (!exact_receipt)
                    record_pipeline_work(PipelineWork::CellGeometryRequest);
                if (!exact_receipt && !partition->geometry()->certified_sparse_provider_cell_geometry(id)->equivalent(
                                          *data_->kernel->geometry(leaf)))
                    throw std::invalid_argument("Native binding differs from actual typed cell geometry.");
                ++id;
                ++ordinal;
            }
            std::uint64_t revision = 0;
            charge_source_residency(binding_overlap, *data_, outside_publications, &revision);
            source_charge = binding_overlap.used;
            binding_overlap.add(sizeof(Data) + sizeof(AdaptiveGeometrySnapshotCache) + 4 * sizeof(void*));
            binding_overlap.add(sizeof(AdaptivePublishedAdmission) + 4 * sizeof(void*));
            bound = std::make_shared<Data>(*data_);
            bound->admission.reset();
            bound->geometry_cache = std::make_shared<AdaptiveGeometrySnapshotCache>();
            {
                std::lock_guard lock(data_->geometry_cache->mutex);
                if (data_->geometry_cache->revision != revision)
                    throw std::invalid_argument("Distributed source residency changed during native binding.");
                if (exact_receipt)
                {
                    const auto& root = data_->geometry_cache->native_store.root();
                    if (!root || !root->payload() || root->payload()->partition.get() != partition.get() ||
                        root->payload()->partition.owner_before(partition) ||
                        partition.owner_before(root->payload()->partition))
                        throw std::invalid_argument("Native binding receipt is no longer the current output.");
                }
                bound->geometry_cache->copy_roots_from(*data_->geometry_cache);
            }
            bound->native = std::move(partition);
            bound->owned_map = map;
            bound->map_token.reset();
            bound->map_token_allocation.reset();
            if (!exact_receipt)
            {
                binding_overlap.add(sizeof(NativeSourceBoundInput::Map) + 4 * sizeof(void*));
                bound->map_token = std::make_shared<const NativeSourceBoundInput::Map>(map);
            }
            bound->native_epoch = bound->native->geometry()->geometry_epoch();
        });
    const auto& plan = bound->native->plan();
    agree(*data_->comm, std::array<std::uint64_t, 4>{
                            plan.collective_identity(), plan.global_cells(), plan.global_faces(), plan.global_nodes()});
    // Binding certifies each native cell against the unchanged canonical leaf;
    // support identities and periodic images remain owned by that same provider.
    collective(*data_->comm,
        [&]
        {
            if (!data_->contact_halo)
                throw std::invalid_argument("Native binding has no contact cache.");
        });
    const auto id = transaction(*data_->comm);
    auto rebound = data_->contact_halo->fork_candidate(
        ContactHaloContext{bound->native_epoch, bound->topology, bound->partition, 0, id}, binding_overlap.used);
    std::shared_ptr<DistributedAdaptiveContactHalo> keepalive;
    ClosureScratch old_output_overlap{rebound->allocation(), outside_publications};
    ClosureScratch source_memory{rebound->allocation(), source_charge - outside_publications};
    ClosureScratch bound_memory{rebound->allocation(), binding_overlap.used - source_charge};
    std::shared_ptr<AdaptivePublishedAdmission> admission;
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || !bound->native->geometry_unchanged() ||
                bound->native->geometry()->geometry_epoch() != bound->native_epoch)
                throw std::invalid_argument("Native binding geometry changed while rebinding contact cache.");
            keepalive = std::shared_ptr<DistributedAdaptiveContactHalo>(std::move(rebound));
            bound->contacts = keepalive->seal_contacts(bound->layout());
            admission = std::make_shared<AdaptivePublishedAdmission>();
            // Move the shared layout and family reservations into the bound
            // arena before sealing its new source. The qualified fork carries
            // the exact contact component itself. Refresh checks the remaining
            // uncovered union after the layout and family floors are in place.
            const auto geometry_bytes = bound->geometry_cache->root_bytes();
            const bool carry_legacy_geometry =
                bound->geometry_cache->source_store.empty() && !bound->native_source_enabled;
            const auto geometry_carry = carry_legacy_geometry ? geometry_bytes : 0;
            const auto layout_bytes = bound->owned_layout_bytes;
            using persistent_ordered_detail::checked_sum;
            const auto directory_bytes =
                checked_sum(bound->cells->storage_bytes(), bound->family_directory->storage_bytes());
            const auto family_bytes = checked_sum(
                checked_sum(bound->family_body_residency(), bound->family_layout_residency()), directory_bytes);
            if (layout_bytes > source_memory.held || geometry_carry > source_memory.held - layout_bytes ||
                family_bytes > source_memory.held - layout_bytes - geometry_carry)
                throw std::logic_error("Shared root residency exceeds admitted source overlap.");
            source_memory.held -= family_bytes;
            bound_memory.held += family_bytes;
            bound->bootstrap_family_admission.reset();
            if (carry_legacy_geometry)
            {
                // This existing compatibility cache contains only the six
                // geometry roots. Preserve its complete legacy lease when an
                // unsupported optimized profile leaves SourceStrong absent.
                bound->geometry_cache->lease.emplace(keepalive->external_residency_lease());
                bound->geometry_cache->lease->adopt_precharged(geometry_carry);
                source_memory.held -= geometry_carry;
                bound->geometry_cache->seal_residency(geometry_carry);
            }
            bound_memory.add(sizeof(ContactExternalResidencyLease) + 4 * sizeof(void*));
            auto layout_admission = std::make_shared<ContactExternalResidencyLease>(keepalive->external_residency_lease());
            layout_admission->adopt_precharged(layout_bytes);
            bound->bound_layout_admission = std::move(layout_admission);
            source_memory.held -= layout_bytes;
        });
    collective(*data_->comm, [&]
    {
        const auto family_body = bound->body_usage.payload_bytes() - bound->body_usage.leaf_payload_bytes();
        if (family_body > bound_memory.held)
            throw std::logic_error("Bound family body coverage exceeds its pre-admitted source amount.");
        bound_memory.add(sizeof(ContactExternalResidencyLease) + 4 * sizeof(void*));
        auto family_admission = std::make_shared<ContactExternalResidencyLease>(keepalive->external_residency_lease());
        family_admission->adopt_precharged(family_body);
        bound_memory.held -= family_body;
        bound->bootstrap_family_admission = std::move(family_admission);
    });
    bound->contact_halo = keepalive;
    collective(*data_->comm, [&]
    {
        refresh_native_source(*bound, *keepalive, &source_memory, &bound_memory, true, {&old_output_overlap});
    });
    admission->halo = keepalive;
    admission->bytes = bound_memory.held;
    keepalive->bind_external_retained_bytes(admission->bytes);
    bound_memory.held = 0;
    bound->contact_halo = keepalive;
    bound->admission = std::move(admission);
    return std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(std::move(bound)));
}
namespace
{
std::size_t fetch_key_bytes(const AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& p) -> std::size_t
        {
            if constexpr (requires { p.path; })
                return p.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (requires { p.base.path; })
                return (p.base.path.capacity() + p.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
void fetch_copy_charge(Budget& budget, const AdaptiveLeaf& leaf)
{
    budget.add(sizeof(AdaptiveLeaf));
    budget.add(fetch_key_bytes(leaf.key));
    budget.add(leaf.domain.capacity());
    budget.add(leaf.parent_family.capacity());
    if (leaf.surfaces.capacity() > budget.limit / sizeof(AdaptiveSurface))
        throw std::invalid_argument("Needed leaf surface count exceeds lease.");
    budget.add(leaf.surfaces.capacity() * sizeof(AdaptiveSurface));
    for (const auto& face : leaf.surfaces)
    {
        budget.add(face.key.capacity());
        budget.add(face.boundary.capacity());
        budget.add(face.authority_image.payload_bytes());
    }
}
template<class Family> void fetch_family_copy_charge(Budget& budget, const Family& family)
{
    budget.add(sizeof(AdaptiveFamily));
    budget.add(family.key.capacity());
    budget.add(family.anchor.capacity());
    if (family.children.capacity() > budget.limit / sizeof(AdaptiveCellKey))
        throw std::invalid_argument("Needed family child count exceeds lease.");
    budget.add(family.children.capacity() * sizeof(AdaptiveCellKey));
    for (const auto& key : family.children)
        budget.add(fetch_key_bytes(key));
    fetch_copy_charge(budget, family.parent);
}
void fetch_copy_charge(Budget& budget, const AdaptiveFamily& family)
{
    fetch_family_copy_charge(budget, family);
}
void fetch_copy_charge(Budget& budget, const AdaptiveFamilyView& family)
{
    fetch_family_copy_charge(budget, family);
}
// Per-message decoded retention admission. Shared procedural dictionaries are
// counted once, while separately decoded cell/face certificates are counted per
// occurrence. The sender reserves growth slack; the receiver checks capacities.
std::size_t fetch_retention(std::span<const AdaptiveLeaf> leaves, std::span<const AdaptiveFamily> families,
    std::size_t limit, bool reserve_growth, std::size_t* leaf_bytes, bool dictionary_prepaid)
{
    Budget bytes{limit};
    adaptive_detail::IdentityBudget identities(reserve_growth);
    std::set<const void*> objects;
    auto charge = [&](std::size_t n)
    {
        if (reserve_growth)
        {
            if (n > limit / 4)
                throw std::invalid_argument("Needed decoded record exceeds retention lease.");
            n *= 4;
        }
        bytes.add(n);
    };
    auto unique = [&](const void* p)
    {
        if (!p || objects.contains(p))
            return false;
        bytes.add(8 * sizeof(void*));
        objects.insert(p);
        return true;
    };
    auto part = [&](const std::shared_ptr<const AdaptiveBytes>& p)
    {
        if (p && !dictionary_prepaid && unique(p.get()))
            charge(sizeof(AdaptiveBytes) + p->capacity());
    };
    auto leaf = [&](const AdaptiveLeaf& value)
    {
        Budget metadata{limit};
        fetch_copy_charge(metadata, value);
        charge(metadata.used);
        if (value.native_recipe && unique(value.native_recipe.get()))
        {
            charge(sizeof(NativeFragmentDescriptor) + value.native_recipe->view.capacity() +
                   value.native_recipe->parts.capacity() * sizeof(std::shared_ptr<const AdaptiveBytes>));
            for (const auto& p : value.native_recipe->parts)
                part(p);
        }
        part(value.join_recipe);
        if (value.factor_recipe && unique(value.factor_recipe.get()))
        {
            charge(sizeof(AdaptiveFactorRecipe));
            part(value.factor_recipe->root);
            part(value.factor_recipe->base);
            part(value.factor_recipe->axis);
        }
        if (value.geometry)
            identities.retain(bytes, value.geometry);
        const auto face_body = [&](const auto& face)
        {
            if (face)
            {
                charge(sizeof(PLIC::PlanarFace<ArithmeticMode::Ordinary>));
                identities.face(bytes, *face);
            }
        };
        for (const auto& face : value.surfaces)
        {
            face_body(face.geometry);
            face_body(face.native_whole_geometry);
        }
    };
    for (const auto& value : leaves)
        leaf(value);
    if (leaf_bytes)
        *leaf_bytes = bytes.used;
    for (const auto& value : families)
    {
        Budget metadata{limit};
        fetch_copy_charge(metadata, value);
        charge(metadata.used);
        leaf(value.parent);
    }
    return bytes.used;
}
AdaptiveBatch fetch_records(const DistributedAdaptiveHierarchy::Data& state, std::span<const AdaptiveBytes> keys,
    bool family, std::size_t cap, bool leased, bool preserve_source = false)
{
    const auto& directory = family ? state.family_directory : state.cells;
    const auto packet_cap = leased ? cap / 8 : cap;
    const auto exchange_cap = leased ? cap / 2 : cap;
    const auto batch_cap = leased ? cap / 4 : cap;
    const auto owner_cap = leased ? std::optional<std::size_t>(cap / 4) : std::nullopt;
    collective(*state.comm,
        [&]
        {
            if (leased && packet_cap < 256)
                throw std::invalid_argument("Needed-record lease is too small.");
        });
    auto owners = directory->lookup(keys, true, owner_cap);
    const auto check_packets = [&](const auto& packets)
    {
        if (!leased)
            return;
        Budget live{cap / 4};
        if (packets.capacity() > (cap / 4) / sizeof(OwnerPacket))
            throw std::invalid_argument("Needed packet metadata exceeds transient lease.");
        live.add(packets.capacity() * sizeof(OwnerPacket));
        for (const auto& packet : packets)
            live.add(packet.bytes.capacity());
    };
    std::vector<OwnerPacket> packets;
    collective(*state.comm,
        [&]
        {
            Budget budget{packet_cap};
            std::map<int, Writer> groups;
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                budget.add(keys[i].size() + 24);
                auto& w = groups[owners[i].owner];
                w.word(i);
                w.bytes(keys[i]);
                w.word(owners[i].revision);
            }
            for (auto& [rank, w] : groups)
            {
                budget.packet(w.data.size());
                packets.push_back({rank, std::move(w.data)});
            }
        });
    collective(*state.comm, [&] { check_packets(packets); });
    auto incoming =
        ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(packets, state.comm, exchange_cap);
    packets.clear();
    collective(*state.comm,
        [&]
        {
            Budget budget{leased ? cap / 4 : cap};
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                Writer order;
                std::vector<AdaptiveLeaf> leaves;
                std::vector<AdaptiveFamily> families;
                if (leased)
                {
                    // Count framed requests without allocating key copies. Reserve
                    // the outer record array before any immutable source copy.
                    Reader scan{packet.bytes};
                    std::size_t count = 0;
                    while (scan.offset < scan.data.size())
                    {
                        (void) scan.word();
                        const auto length = scan.word();
                        if (length > scan.data.size() - scan.offset)
                            throw std::invalid_argument("Invalid needed-record request framing.");
                        scan.offset += length;
                        (void) scan.word();
                        ++count;
                    }
                    const auto record_bytes = family ? sizeof(AdaptiveFamily) : sizeof(AdaptiveLeaf);
                    if (count > budget.limit / (2 * record_bytes))
                        throw std::invalid_argument("Needed source-record vector exceeds lease.");
                    budget.add(2 * count * record_bytes);
                    if (family)
                    {
                        families.reserve(count);
                        if (families.capacity() > 2 * count)
                            throw std::invalid_argument("Needed family vector capacity exceeds admission.");
                    }
                    else
                    {
                        leaves.reserve(count);
                        if (leaves.capacity() > 2 * count)
                            throw std::invalid_argument("Needed leaf vector capacity exceeds admission.");
                    }
                }
                while (r.offset < r.data.size())
                {
                    auto index = r.word();
                    auto key = r.bytes();
                    auto revision = r.word();
                    budget.add(key.size() + 16);
                    order.word(index);
                    if (family)
                    {
                        const auto row = state.family_layout->find(key);
                        if (!row || row->revision != revision)
                            throw std::invalid_argument("Missing or stale needed family capsule.");
                        if (leased)
                            fetch_copy_charge(budget, *row);
                        families.push_back(row->materialize());
                    }
                    else
                    {
                        auto typed = decode_adaptive_key(key);
                        const auto* leaf = state.layout()->find(typed);
                        if (!leaf || leaf->revision != revision)
                            throw std::invalid_argument("Missing or stale needed owned leaf.");
                        if (leased)
                            fetch_copy_charge(budget, *leaf);
                        leaves.push_back(*leaf);
                    }
                }
                // Publication-only compact drafts must inspect the original
                // certificate presence and keep its native ordinal/recipe pair.
                // Repacking can erase supplied geometry or remap ordinals.
                if (!family && !leased && !preserve_source)
                {
                    std::map<AdaptiveCellKey, AdaptiveLeaf> unique;
                    for (const auto& leaf : leaves)
                        unique.emplace(leaf.key, leaf);
                    std::vector<AdaptiveLeaf> selected;
                    for (const auto& [key, leaf] : unique)
                        selected.push_back(leaf);
                    auto packed = state.kernel->repack(selected);
                    unique.clear();
                    for (auto& leaf : packed)
                        unique.emplace(leaf.key, std::move(leaf));
                    for (auto& leaf : leaves)
                        leaf = unique.at(leaf.key);
                }
                const auto retention = leased ? fetch_retention(leaves, families, batch_cap, true) : 0;
                if (leased)
                {
                    // The owner table, received requests, copies/order, prior replies
                    // and codec arena coexist until this reply has been constructed.
                    Budget live{cap};
                    live.add(owners.capacity() * sizeof(AdaptiveOwner));
                    for (const auto& owner : owners)
                        live.add(owner.key.capacity());
                    live.add(incoming.packets.capacity() * sizeof(OwnerPacket));
                    for (const auto& input : incoming.packets)
                        live.add(input.bytes.capacity());
                    live.add(budget.used);
                    live.add(order.data.capacity());
                    live.add(packets.capacity() * sizeof(OwnerPacket));
                    for (const auto& output : packets)
                        live.add(output.bytes.capacity());
                    live.add(2 * packet_cap); // encoder vector growth, not only logical wire size
                    live.add(cap / 4);        // pending reply allocation
                }
                const auto encoded_batch = encode_adaptive_batch(leaves, families, packet_cap);
                Writer reply;
                if (leased)
                {
                    if (encoded_batch.capacity() > 2 * packet_cap)
                        throw std::invalid_argument("Needed-record encoded capacity exceeds its growth lease.");
                    Budget live{cap / 4};
                    if (packets.capacity() > live.limit / sizeof(OwnerPacket))
                        throw std::invalid_argument("Needed reply metadata exceeds lease.");
                    live.add(packets.capacity() * sizeof(OwnerPacket));
                    for (const auto& packet : packets)
                        live.add(packet.bytes.capacity());
                    live.add(24);
                    live.add(order.data.size());
                    live.add(encoded_batch.size());
                    reply.data.reserve(24 + order.data.size() + encoded_batch.size());
                }
                reply.bytes(order.data);
                if (leased)
                    reply.word(retention);
                reply.bytes(encoded_batch);
                budget.packet(reply.data.size());
                packets.push_back({packet.peer, std::move(reply.data)});
            }
        });
    if (leased)
        incoming = {};
    collective(*state.comm, [&] { check_packets(packets); });
    auto replies =
        ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(packets, state.comm, exchange_cap);
    if (leased)
        std::vector<OwnerPacket>().swap(packets);
    AdaptiveBatch result;
    collective(*state.comm,
        [&]
        {
            Budget budget{leased ? cap / 4 : cap};
            if (leased)
            {
                if (keys.size() > budget.limit / (sizeof(AdaptiveLeaf) + sizeof(AdaptiveFamily) + 1))
                    throw std::invalid_argument("Needed result metadata exceeds retention lease.");
                budget.add(keys.size()); // packed seen flags, conservatively one byte per key
            }
            std::vector<bool> seen(keys.size());
            if (family)
            {
                budget.add(keys.size() * sizeof(AdaptiveFamily));
                result.families.resize(keys.size());
            }
            else
            {
                budget.add(keys.size() * sizeof(AdaptiveLeaf));
                result.leaves.resize(keys.size());
            }
            std::size_t remaining_retention = batch_cap;
            for (const auto& packet : replies.packets)
            {
                budget.add(packet.bytes.size());
                Reader r{packet.bytes};
                auto order = r.bytes();
                const auto retention = leased ? r.word() : batch_cap;
                if (leased && retention > remaining_retention)
                    throw std::invalid_argument("Cumulative needed-record retention exceeds lease.");
                // One half of this reservation is the bounded codec arena;
                // the other half is decoded record retention. Thus previous
                // replies and this decode share one cumulative pool.
                auto batch = decode_adaptive_batch(r.bytes(), leased ? std::min(batch_cap, retention / 2) : batch_cap);
                if (leased)
                {
                    if (fetch_retention(batch.leaves, batch.families, retention / 2, false) > retention / 2)
                        throw std::invalid_argument("Needed-record decoded capacities exceed admission.");
                    remaining_retention -= retention;
                }
                r.done();
                Reader indices{order};
                if ((family && !batch.leaves.empty()) || (!family && !batch.families.empty()))
                    throw std::invalid_argument("Needed-record owner returned wrong payload type.");
                const auto count = family ? batch.families.size() : batch.leaves.size();
                if (count != order.size() / 8 || order.size() % 8)
                    throw std::invalid_argument("Needed-record owner count differs from ordered request.");
                for (std::size_t i = 0; i < count; ++i)
                {
                    auto index = indices.word();
                    if (index >= keys.size() || seen[index] || packet.peer != owners[index].owner)
                        throw std::invalid_argument("Foreign, repeated or misordered needed-record reply.");
                    seen[index] = true;
                    if (family)
                    {
                        auto& value = batch.families[i];
                        if (value.key != keys[index] || value.revision != owners[index].revision)
                            throw std::invalid_argument("Needed family changed exact identity or revision.");
                        validate_family(*state.kernel, value);
                        result.families[index] = std::move(value);
                    }
                    else
                    {
                        auto& value = batch.leaves[i];
                        if (encode_adaptive_key(value.key) != keys[index] || value.revision != owners[index].revision)
                            throw std::invalid_argument("Needed leaf changed exact identity or revision.");
                        state.kernel->validate(value);
                        result.leaves[index] = std::move(value);
                    }
                }
            }
            for (bool present : seen)
                if (!present)
                    throw std::invalid_argument("Missing needed-record owner reply.");
        });
    return result;
}
} // namespace
std::vector<AdaptiveOwner> DistributedAdaptiveHierarchy::lookup_cells(
    std::span<const AdaptiveCellKey> keys, bool require_all, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    std::vector<AdaptiveBytes> encoded;
    collective(*data_->comm,
        [&]
        {
            Budget budget{transient_cap ? cap / 8 : cap};
            if (transient_cap)
            {
                if (keys.size() > (cap / 8) / sizeof(AdaptiveBytes))
                    throw std::invalid_argument("Needed key metadata exceeds transient lease.");
                budget.add(keys.size() * sizeof(AdaptiveBytes));
                encoded.reserve(keys.size());
            }
            for (const auto& key : keys)
            {
                if (transient_cap)
                {
                    const auto dynamic = fetch_key_bytes(key);
                    if (budget.limit < 256 || dynamic > budget.limit / 2 - 128)
                        throw std::invalid_argument("Needed key encoding exceeds transient lease.");
                    budget.add(2 * (128 + dynamic)); // framing plus vector growth, BEFORE encoding
                }
                auto bytes = encode_adaptive_key(key);
                if (!transient_cap)
                    budget.add(bytes.size());
                encoded.push_back(std::move(bytes));
            }
        });
    return data_->cells->lookup(
        encoded, require_all, transient_cap ? std::optional<std::size_t>(cap - cap / 8) : std::nullopt);
}
std::vector<AdaptiveOwner> DistributedAdaptiveHierarchy::lookup_families(
    std::span<const AdaptiveBytes> keys, bool require_all, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    return data_->family_directory->lookup(keys, require_all, transient_cap);
}
AdaptiveBytes DistributedAdaptiveHierarchy::configuration_descriptor() const
{
    return data_->kernel->descriptor();
}
const std::shared_ptr<const AdaptiveProviderKernel>&
DistributedAdaptiveHierarchy::native_publication_kernel() const noexcept
{
    return data_->kernel;
}
std::vector<AdaptiveLeaf> DistributedAdaptiveHierarchy::fetch_leaves(
    std::span<const AdaptiveCellKey> keys, std::optional<std::size_t> transient_cap) const
{
    return fetch_leaves_impl(keys, transient_cap, false);
}
std::vector<AdaptiveLeaf> adaptive_detail::fetch_native_source_leaves(
    const DistributedAdaptiveHierarchy& hierarchy, std::span<const AdaptiveCellKey> keys)
{
    return hierarchy.fetch_leaves_impl(keys, {}, true);
}
std::vector<AdaptiveLeaf> DistributedAdaptiveHierarchy::fetch_leaves_impl(
    std::span<const AdaptiveCellKey> keys, std::optional<std::size_t> transient_cap, bool preserve_source) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    std::vector<AdaptiveBytes> encoded;
    collective(*data_->comm,
        [&]
        {
            Budget budget{transient_cap ? cap / 8 : cap};
            if (transient_cap)
            {
                if (keys.size() > (cap / 8) / sizeof(AdaptiveBytes))
                    throw std::invalid_argument("Needed key metadata exceeds transient lease.");
                budget.add(keys.size() * sizeof(AdaptiveBytes));
                encoded.reserve(keys.size());
            }
            for (const auto& key : keys)
            {
                if (transient_cap)
                {
                    const auto dynamic = fetch_key_bytes(key);
                    if (budget.limit < 256 || dynamic > budget.limit / 2 - 128)
                        throw std::invalid_argument("Needed key encoding exceeds transient lease.");
                    budget.add(2 * (128 + dynamic)); // framing plus vector growth, BEFORE encoding
                }
                auto bytes = encode_adaptive_key(key);
                if (!transient_cap)
                    budget.add(bytes.size());
                encoded.push_back(std::move(bytes));
            }
        });
    return fetch_records(
        *data_, encoded, false, transient_cap ? cap - cap / 8 : cap, transient_cap.has_value(), preserve_source)
        .leaves;
}
std::vector<AdaptiveFamily> DistributedAdaptiveHierarchy::fetch_families(
    std::span<const AdaptiveBytes> keys, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    return fetch_records(*data_, keys, true, cap, transient_cap.has_value()).families;
}
std::shared_ptr<const AdaptiveGeometrySnapshot> DistributedAdaptiveHierarchy::geometry_snapshot(
    std::span<const AdaptiveLeaf* const> leaves, ArithmeticMode mode, GeometrySnapshotPurpose purpose,
    std::size_t& operation_used, std::size_t operation_limit, PLIC::GeometryAllocationLedger* supplied_ordinary,
    PLIC::GeometryAllocationLedger* supplied_bounded) const
{
    const RejectGeometryAdmission reject{supplied_ordinary, supplied_bounded};
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed geometry snapshot source moved.");
    for (const auto* leaf : leaves)
    {
        if (!leaf)
            throw std::invalid_argument("Geometry snapshot lacks an authenticated source leaf.");
        if (const auto* owned_leaf = leaf_layout()->find(leaf->key))
        {
            const auto& source = *owned_leaf;
            if (source.key != leaf->key || source.revision != leaf->revision || source.material != leaf->material ||
                source.domain != leaf->domain || source.provider != leaf->provider ||
                source.native_cell != leaf->native_cell ||
                !GeometrySnapshotBinding::same_auxiliary_recipes(source, *leaf) ||
                (purpose == GeometrySnapshotPurpose::State && source.geometry != leaf->geometry))
                throw std::invalid_argument("Geometry snapshot source differs from current owned binding.");
            if (source.native_recipe != leaf->native_recipe)
            {
                if (!source.native_recipe || !leaf->native_recipe)
                    throw std::invalid_argument("Geometry snapshot source recipe owner differs.");
                const auto& first = *source.native_recipe;
                const auto& second = *leaf->native_recipe;
                if (first.kind != second.kind || first.view != second.view || first.parts.size() != second.parts.size())
                    throw std::invalid_argument("Geometry snapshot source recipe differs.");
                for (std::size_t part = 0; part < first.parts.size(); ++part)
                    if (!first.parts[part] || !second.parts[part] || *first.parts[part] != *second.parts[part])
                        throw std::invalid_argument("Geometry snapshot source recipe payload differs.");
            }
        }
    }
    const auto mode_slot = static_cast<unsigned>(mode);
    if (mode_slot > 2 ||
        (purpose != GeometrySnapshotPurpose::State && purpose != GeometrySnapshotPurpose::OriginalSources))
        throw std::invalid_argument("Unknown geometry snapshot policy.");
    const auto slot = 2 * mode_slot + (purpose == GeometrySnapshotPurpose::OriginalSources ? 1 : 0);
    PLIC::GeometryAllocationLedger local_ordinary, local_bounded;
    auto& operation_ordinary = supplied_ordinary ? *supplied_ordinary : local_ordinary;
    auto& operation_bounded = supplied_bounded ? *supplied_bounded : local_bounded;
    std::lock_guard lock(data_->geometry_cache->mutex);
    data_->geometry_cache->require_editable();
    auto& current = data_->geometry_cache->roots[slot];
    const auto controls = geometry_construction_controls();
    const bool compatible = current && current->matches_layout(leaf_layout()) && current->stable_controls() &&
                            controls.supported && current->construction_controls() == controls;
    if (compatible && purpose == GeometrySnapshotPurpose::State &&
        std::all_of(leaves.begin(), leaves.end(), [&](const auto* leaf) { return leaf && current->find(*leaf); }))
    {
        data_->geometry_cache->owner_union.import_subset(data_->geometry_cache->bindings[slot], operation_ordinary,
            operation_bounded, operation_used, operation_limit);
        return current;
    }
    // Failed construction preserves the spent admission prefix while leaving
    // the published cache unchanged. Both budgets admit old/new root overlap.
    auto pending_used = operation_used;
    struct SpentCounter
    {
        std::size_t& caller;
        std::size_t& pending;
        ~SpentCounter() { caller = std::max(caller, pending); }
    } spent{operation_used, pending_used};
    // A retained immutable cache value may belong to the previous layout.
    // Never return it directly on a mismatch; build still authenticates every
    // selected leaf's exact source/control/revision before reusing a child.
    auto predecessor = current && current->stable_controls() && controls.supported &&
        current->construction_controls() == controls ? current : nullptr;
    bool historical = false;
    if (!current && controls.supported && purpose == GeometrySnapshotPurpose::State)
    {
        predecessor = data_->geometry_cache->predecessors[slot].lock();
        if (predecessor && (!predecessor->stable_controls() || predecessor->construction_controls() != controls))
            predecessor.reset();
        historical = bool(predecessor);
    }
    if (!predecessor && purpose == GeometrySnapshotPurpose::OriginalSources)
        predecessor = data_->geometry_cache->roots[2 * mode_slot];
    auto candidate = AdaptiveGeometrySnapshot::build(*this, leaves, mode, purpose, pending_used, operation_limit,
        operation_ordinary, operation_bounded, predecessor, {}, &data_->geometry_cache->catalog_work);
    Budget hierarchy_budget{data_->options.byte_budget};
    hierarchy_budget.add(sizeof(AdaptiveGeometrySnapshotCache));
    hierarchy_budget.add(data_->geometry_cache->manifest_bytes);
    if (!data_->geometry_cache->source_store.empty())
    {
        hierarchy_budget.add(data_->geometry_cache->source_partition.exclusive_source_admission);
        hierarchy_budget.add(data_->geometry_cache->contact_residual(*data_->contact_halo));
    }
    else
    {
        charge_source_base_residency(hierarchy_budget, *data_);
        if (data_->contact_halo)
            hierarchy_budget.add(data_->contact_halo->cache_storage_bytes());
    }
    PLIC::GeometryAllocationLedger ordinary_owners, bounded_owners;
    const auto base_bytes = hierarchy_budget.used;
    auto candidate_binding = data_->geometry_cache->binding(candidate);
    PLIC::details::GeometryOwnerRootBinding historical_binding;
    auto overlap = data_->geometry_cache->owner_union;
    if (historical)
    {
        historical_binding = data_->geometry_cache->binding(predecessor);
        PLIC::details::GeometryOwnerUnionWork work;
        overlap = overlap.staged(nullptr, &historical_binding, ordinary_owners, bounded_owners, hierarchy_budget.used,
            hierarchy_budget.limit, &work);
        data_->geometry_cache->record_union_work(work);
    }
    PLIC::details::GeometryOwnerUnionWork owner_work;
    overlap = overlap.staged(nullptr, &candidate_binding, ordinary_owners, bounded_owners, hierarchy_budget.used,
        hierarchy_budget.limit, &owner_work);
    data_->geometry_cache->record_union_work(owner_work);
    auto overlap_bytes = hierarchy_budget.used - base_bytes;
    const auto refresh_candidate = [&]
    {
        auto updated = data_->geometry_cache->binding(candidate);
        PLIC::details::GeometryOwnerUnionWork work;
        overlap = overlap.staged(&candidate_binding, &updated, ordinary_owners, bounded_owners, hierarchy_budget.used,
            hierarchy_budget.limit, &work);
        candidate_binding = std::move(updated);
        data_->geometry_cache->record_union_work(work);
        overlap_bytes = hierarchy_budget.used - base_bytes;
    };
    if (!data_->geometry_cache->lease)
        data_->geometry_cache->lease.emplace(data_->contact_halo->external_residency_lease());
    auto& lease = *data_->geometry_cache->lease;
    auto optional_reservation = [&](std::size_t new_owners, std::size_t index_frontier)
    {
        namespace G = PLIC::details::geometry_demand;
        const auto incoming = geometry_snapshot_detail::checked_sum(new_owners, index_frontier);
        const auto rows = geometry_snapshot_detail::checked_sum(overlap.manifests().owners[0].owner_count(), incoming);
        const auto refresh_edges = geometry_snapshot_detail::checked_sum(
            4, geometry_snapshot_detail::checked_product(index_frontier, G::leaf_capacity));
        // Rewritten snapshot metadata uses at most one additional admitted
        // State. Its typed keys are State and Counter, with a flat Budget pin.
        const std::size_t typed_growth =
            PLIC::details::GeometryAllocationOwnerAccess::native_state(operation_ordinary) ? 2 : 0;
        const auto refresh_typed = typed_growth || candidate_binding.payload().owners[0].has_native_administration() ?
            std::optional{typed_growth} : std::nullopt;
        auto reserve = overlap.administration_byte_bound(
            refresh_edges, std::min(refresh_edges, rows), index_frontier, incoming, 0, refresh_typed);
        const auto retire = [&](const PLIC::details::GeometryOwnerRootBinding& binding)
        {
            if (!binding)
                return std::size_t(0);
            // Optional preparation is Ordinary-only. A future producer that
            // adds another channel must supply its separate mutation envelope.
            if (binding.payload().owners[1].owner_count())
                return std::numeric_limits<std::size_t>::max();
            const auto& payload = binding.payload().owners[0];
            const auto metadata = payload.metadata_owner_count_bound();
            const auto edges = geometry_snapshot_detail::checked_sum(
                2, geometry_snapshot_detail::checked_product(metadata, G::leaf_capacity));
            const auto retire_typed = payload.has_native_administration() ? std::optional{typed_growth} : std::nullopt;
            return overlap.administration_byte_bound(edges, std::min(edges, rows), metadata, 0, incoming, retire_typed);
        };
        for (const auto* binding : {&data_->geometry_cache->bindings[slot], &historical_binding})
        {
            const auto bytes = retire(*binding);
            if (bytes == std::numeric_limits<std::size_t>::max())
                return bytes;
            reserve = geometry_snapshot_detail::checked_sum(reserve, bytes);
        }
        return reserve;
    };
    const geometry_snapshot_detail::OptionalOwnerReservation publication{&optional_reservation,
        [](void* raw, std::size_t owners, std::size_t frontier)
        { return (*static_cast<const decltype(optional_reservation)*>(raw))(owners, frontier); }};
    constexpr std::size_t maximum_stream_batch = 4;
    if (compatible && mode == ArithmeticMode::Ordinary && purpose == GeometrySnapshotPurpose::State &&
        leaves.size() <= maximum_stream_batch)
        for (const auto* leaf : leaves)
        {
            if (current->find(*leaf))
                continue;
            const auto prepare = [&](std::uint64_t first)
            {
                const auto demand_growth = overlap_bytes - std::min(overlap_bytes, lease.bytes());
                const auto arena_available = lease.available();
                const auto optional_headroom = std::min(hierarchy_budget.limit - hierarchy_budget.used,
                    arena_available - std::min(demand_growth, arena_available));
                if (candidate->prepare_cartesian_page(*this, *leaf, pending_used, operation_limit, optional_headroom,
                        first, operation_ordinary, publication))
                {
                    refresh_candidate();
                }
            };
            const auto block = leaf->native_cell / geometry_snapshot_detail::prepared_cartesian_page_size *
                               geometry_snapshot_detail::prepared_cartesian_page_size;
            prepare(block);
            // Look one block ahead once the request reaches the end of its own
            // block, so the next block's page exists before its first cell is
            // selected and can therefore cover every cell of that block.
            if (leaf->native_cell + maximum_stream_batch >=
                block + geometry_snapshot_detail::prepared_cartesian_page_size)
                prepare(block + geometry_snapshot_detail::prepared_cartesian_page_size);
        }
    if (compatible && mode == ArithmeticMode::Ordinary && purpose == GeometrySnapshotPurpose::State &&
        leaves.size() <= maximum_stream_batch)
    {
        // Cells whose page was prepared after they were already selected keep
        // their per-build source until they adopt the page here. Only genuinely
        // different sources are rebound, so already adopted cells cost nothing.
        std::array<std::size_t, maximum_stream_batch> ordinals;
        std::array<AdaptiveGeometrySnapshot::PreparedSourceSlot, maximum_stream_batch> prepared;
        std::size_t count = 0;
        for (const auto* leaf : leaves)
        {
            const auto ordinal = candidate->find(*leaf);
            if (!ordinal)
                continue;
            const auto slot = candidate->prepared_slot(*leaf);
            if (!slot.valid)
                continue;
            const PLIC::CellGeometryView<ArithmeticMode::Ordinary> view(candidate->ordinary_catalog(), *ordinal);
            if (!view.is_procedural())
                continue;
            const auto current = PLIC::details::NativeCellGeometrySourceAccess::source_cell(view);
            if (current.source.get() == slot.source.get())
                continue;
            ordinals[count] = *ordinal;
            prepared[count++] = slot;
        }
        if (count)
        {
            const auto growth = overlap_bytes - std::min(overlap_bytes, lease.bytes());
            const auto arena_available = lease.available();
            const auto remaining = std::min({operation_limit - std::min(pending_used, operation_limit),
                hierarchy_budget.limit - hierarchy_budget.used, arena_available - std::min(growth, arena_available)});
            if (candidate->can_rebind_prepared_slots(count, remaining, publication))
            {
                candidate->rebind_prepared_slots(std::span(ordinals).first(count), std::span(prepared).first(count),
                    operation_ordinary, pending_used, pending_used + remaining, &data_->geometry_cache->catalog_work);
                refresh_candidate();
            }
        }
    }
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed geometry snapshot source moved before publication.");
    PLIC::details::GeometryOwnerUnion final_union = overlap;
    if (data_->geometry_cache->bindings[slot])
    {
        PLIC::details::GeometryOwnerUnionWork work;
        final_union = overlap.staged(&data_->geometry_cache->bindings[slot], nullptr, ordinary_owners, bounded_owners,
            hierarchy_budget.used, hierarchy_budget.limit, &work);
        data_->geometry_cache->record_union_work(work);
    }
    if (historical_binding)
    {
        PLIC::details::GeometryOwnerUnionWork work;
        final_union = final_union.staged(&historical_binding, nullptr, ordinary_owners, bounded_owners,
            hierarchy_budget.used, hierarchy_budget.limit, &work);
        data_->geometry_cache->record_union_work(work);
    }
    const auto overlap_residency = AdaptiveGeometrySnapshotCache::uncovered_bytes(overlap, lease.arena_identity());
    const auto final_residency = AdaptiveGeometrySnapshotCache::uncovered_bytes(final_union, lease.arena_identity());
    const auto previous_lease_bytes = lease.bytes();
    try
    {
        lease.resize(std::max(overlap_residency, final_residency));
        lease.resize(final_residency);
    }
    catch (...)
    {
        lease.resize(previous_lease_bytes);
        throw;
    }
    current = candidate;
    data_->geometry_cache->bindings[slot] = std::move(candidate_binding);
    data_->geometry_cache->manifests[slot] = data_->geometry_cache->bindings[slot].payload().owners;
    data_->geometry_cache->owner_union = std::move(final_union);
    data_->geometry_cache->manifest_bytes = 0;
    data_->geometry_cache->seal_residency(data_->geometry_cache->owner_union.resident_bytes());
    data_->geometry_cache->predecessors[slot].reset();
    operation_used = pending_used;
    return candidate;
}
std::shared_ptr<const AdaptiveGeometrySnapshot> adaptive_detail::GeometrySnapshotAccess::native_draft(
    const DistributedAdaptiveHierarchy& hierarchy, const AdaptiveNativeDraft& draft,
    std::span<const AdaptiveLeaf* const> selected, const HierarchyObservation& source_observation, ArithmeticMode mode,
    GeometrySnapshotPurpose purpose, std::size_t& used, std::size_t limit,
    PLIC::GeometryAllocationLedger& ordinary_owners, PLIC::GeometryAllocationLedger& bounded_owners)
{
    const RejectGeometryAdmission reject{&ordinary_owners, &bounded_owners};
    const auto& state = *hierarchy.data_;
    if (!hierarchy.geometry_unchanged() || hierarchy.observation() != source_observation)
        throw std::invalid_argument("Native geometry draft has a stale source observation.");
    const auto& layout = hierarchy.leaf_layout();
    if (draft.source_layout != layout || draft.source_layout.owner_before(layout) ||
        layout.owner_before(draft.source_layout) || draft.global_cells != state.global_cells ||
        draft.owned.size() != layout->size())
        throw std::invalid_argument("Native geometry draft lacks complete current ownership.");
    const auto mode_slot = static_cast<unsigned>(mode);
    if (mode_slot > 2 ||
        (purpose != GeometrySnapshotPurpose::State && purpose != GeometrySnapshotPurpose::OriginalSources))
        throw std::invalid_argument("Unknown native geometry draft snapshot policy.");
    Budget budget{limit, used};
    struct SpentCounter
    {
        std::size_t& caller;
        Budget& budget;
        ~SpentCounter() { caller = std::max(caller, budget.used); }
    } spent{used, budget};
    struct Row
    {
        const AdaptiveLeaf* leaf = nullptr;
        bool owned = false;
        const AdaptiveNativeCellBinding* binding = nullptr;
    };
    std::map<AdaptiveCellKey, Row> rows;
    bool provider_compatible = true;
    const auto same_native_recipe = [](const auto& first, const auto& second)
    {
        if (first == second)
            return true;
        if (!first || !second || first->kind != second->kind || first->view != second->view ||
            first->parts.size() != second->parts.size())
            return false;
        for (std::size_t part = 0; part < first->parts.size(); ++part)
            if (!first->parts[part] || !second->parts[part] || *first->parts[part] != *second->parts[part])
                return false;
        return true;
    };
    for (const auto* values : {&draft.owned, &draft.ghosts})
        for (const auto& leaf : *values)
        {
            const bool owned = values == &draft.owned;
            budget.add(sizeof(decltype(rows)::value_type) + 4 * sizeof(void*) + fetch_key_bytes(leaf.key));
            if (!rows.emplace(leaf.key, Row{&leaf, owned, nullptr}).second)
                throw std::invalid_argument("Native geometry draft repeats a full source key.");
            if (owned)
            {
                const auto* found = draft.source_layout->find(leaf.key);
                if (!found)
                    throw std::invalid_argument("Native geometry draft contains foreign ownership.");
                const auto& source = *found;
                if (source.revision != leaf.revision || source.provider != leaf.provider ||
                    source.material != leaf.material || source.domain != leaf.domain ||
                    source.native_cell != leaf.native_cell ||
                    !same_native_recipe(source.native_recipe, leaf.native_recipe) ||
                    !GeometrySnapshotBinding::same_auxiliary_capsules(source, leaf))
                    throw std::invalid_argument("Native geometry draft changed its provider source binding.");
                provider_compatible = provider_compatible && source.geometry == leaf.geometry &&
                                      GeometrySnapshotBinding::same_auxiliary_recipes(source, leaf);
            }
            else if (leaf.join_recipe && leaf.geometry && leaf.provider == AdaptiveProviderKind::Polyhedral)
                provider_compatible = false;
        }
    for (const auto& binding : draft.cell_bindings)
    {
        if (!binding.full)
            continue;
        const auto found = rows.find(binding.key);
        if (found == rows.end() || found->second.binding)
            throw std::invalid_argument("Native geometry draft full binding lacks its unique source row.");
        const auto& row = found->second;
        const auto& leaf = *row.leaf;
        if (binding.owner < 0 || binding.owner >= state.comm->getSize() ||
            row.owned != (binding.owner == state.comm->getRank()) || binding.provider != leaf.provider ||
            binding.material != leaf.material || binding.native_cell != leaf.native_cell ||
            binding.native_recipe != leaf.native_recipe)
            throw std::invalid_argument("Native geometry draft full binding changed its source authority.");
        found->second.binding = &binding;
    }
    if (std::any_of(rows.begin(), rows.end(), [](const auto& row) { return !row.second.binding; }))
        throw std::invalid_argument("Native geometry draft lacks a full binding for its source row.");
    std::vector<geometry_snapshot_detail::PublicationSource> publication;
    geometry_snapshot_detail::charge_items(
        budget, selected.size(), sizeof(geometry_snapshot_detail::PublicationSource));
    publication.reserve(selected.size());
    if (publication.capacity() > selected.size())
        geometry_snapshot_detail::charge_items(
            budget, publication.capacity() - selected.size(), sizeof(geometry_snapshot_detail::PublicationSource));
    bool has_published_geometry = false;
    for (const auto* leaf : selected)
    {
        if (!leaf)
            throw std::invalid_argument("Native geometry draft selection lacks a source row.");
        const auto found = rows.find(leaf->key);
        if (found == rows.end() || found->second.leaf != leaf)
            throw std::invalid_argument("Native geometry selection is not a row of its authenticated draft.");
        publication.push_back({leaf, found->second.binding});
        has_published_geometry = has_published_geometry || bool(leaf->geometry);
    }
    if (provider_compatible && !has_published_geometry)
        return hierarchy.geometry_snapshot(
            selected, mode, purpose, budget.used, limit, &ordinary_owners, &bounded_owners);
    // A publication-derived representation is not a provider-input cache key.
    // Exact matching source entries can still be reused, while differing rows
    // remain operation-owned and cannot poison the hierarchy's provider cache.
    std::lock_guard lock(state.geometry_cache->mutex);
    const auto slot = 2 * mode_slot + (purpose == GeometrySnapshotPurpose::OriginalSources ? 1 : 0);
    const auto& previous = state.geometry_cache->roots[slot] ? state.geometry_cache->roots[slot]
                                                             : state.geometry_cache->roots[2 * mode_slot];
    auto result = AdaptiveGeometrySnapshot::build(hierarchy, selected, mode, purpose, budget.used, limit,
        ordinary_owners, bounded_owners, previous, publication, &state.geometry_cache->catalog_work);
    Budget hierarchy_budget{state.options.byte_budget};
    hierarchy_budget.add(sizeof(AdaptiveGeometrySnapshotCache));
    hierarchy_budget.add(state.geometry_cache->manifest_bytes);
    if (!state.geometry_cache->source_store.empty())
    {
        hierarchy_budget.add(state.geometry_cache->source_partition.exclusive_source_admission);
        hierarchy_budget.add(state.geometry_cache->contact_residual(*state.contact_halo));
    }
    else
    {
        charge_source_base_residency(hierarchy_budget, state);
        hierarchy_budget.add(state.contact_halo->cache_storage_bytes());
    }
    PLIC::GeometryAllocationLedger hierarchy_ordinary, hierarchy_bounded;
    const auto result_binding = state.geometry_cache->binding(result);
    const auto overlap = state.geometry_cache->owner_union.staged(
        nullptr, &result_binding, hierarchy_ordinary, hierarchy_bounded, hierarchy_budget.used, hierarchy_budget.limit);
    (void) overlap;
    if (!hierarchy.geometry_unchanged() || hierarchy.observation() != source_observation)
        throw std::invalid_argument("Native geometry draft source changed during snapshot acquisition.");
    return result;
}
adaptive_detail::GeometryCacheAdmissionStatistics adaptive_detail::GeometrySnapshotAccess::cache_admission_statistics(
    const DistributedAdaptiveHierarchy& hierarchy)
{
    const auto& cache = *hierarchy.data_->geometry_cache;
    std::lock_guard lock(cache.mutex);
    return {sizeof(cache), cache.root_bytes(), cache.lease ? cache.lease->bytes() : 0, cache.admission_queries,
        cache.union_censuses, cache.census_root_visits, cache.catalog_work.cumulative_metadata_bytes,
        cache.catalog_work.pages.copied_pages, cache.catalog_work.pages.visited_pages,
        cache.catalog_work.pages.copied_source_rows, cache.catalog_work.pages.visited_source_rows, cache.union_updates,
        cache.union_work.channels[0].edge_visits + cache.union_work.channels[1].edge_visits,
        cache.union_work.channels[0].dependency_entries + cache.union_work.channels[1].dependency_entries,
        cache.union_work.channels[0].first_transitions + cache.union_work.channels[1].first_transitions,
        cache.union_work.channels[0].last_transitions + cache.union_work.channels[1].last_transitions,
        cache.union_work.channels[0].index.copied_pages + cache.union_work.channels[1].index.copied_pages +
            cache.union_work.controls.copied_pages,
        cache.union_work.channels[0].index.copied_bytes + cache.union_work.channels[1].index.copied_bytes +
            cache.union_work.controls.copied_bytes,
        cache.union_work.channels[0].index.visited_pages + cache.union_work.channels[1].index.visited_pages +
            cache.union_work.controls.visited_pages,
        cache.union_work.channels[0].lookup.visited_pages + cache.union_work.channels[1].lookup.visited_pages,
        cache.union_work.imported_rows};
}
void adaptive_detail::GeometrySnapshotAccess::visit_cache_owner_union(const DistributedAdaptiveHierarchy& hierarchy,
    const std::function<void(const PLIC::details::geometry_demand::OwnerRecord&)>& rows,
    PLIC::details::geometry_demand::MetadataSink metadata)
{
    const auto& cache = *hierarchy.data_->geometry_cache;
    std::lock_guard lock(cache.mutex);
    for (const auto& root : cache.owner_union.manifests().owners)
        root.visit(rows);
    cache.owner_union.visit_administration(metadata);
}
void adaptive_detail::GeometrySnapshotAccess::visit_native_cache_bindings(const DistributedAdaptiveHierarchy& hierarchy,
    const std::function<void(const PLIC::details::GeometryOwnerRootBinding&)>& binding,
    const std::function<void(const PLIC::details::NativePaidExtentCharge&)>& charge)
{
    const auto& cache = *hierarchy.data_->geometry_cache;
    std::lock_guard lock(cache.mutex);
    binding(cache.source_store.binding());
    binding(cache.native_store.binding());
    struct ChargeVisitor
    {
        const std::function<void(const PLIC::details::NativePaidExtentCharge&)>& visit;
    } visitor{charge};
    cache.owner_union.native_administration_.domain_paid.visit(
        {&visitor, [](void* raw, const auto& value, std::size_t) { static_cast<ChargeVisitor*>(raw)->visit(value); }});
}
void adaptive_detail::GeometrySnapshotAccess::admit_cache_residency(
    const DistributedAdaptiveHierarchy& hierarchy, Budget& budget)
{
    const auto& cache = *hierarchy.data_->geometry_cache;
    std::lock_guard lock(cache.mutex);
    cache.admit_residency(budget);
}
void DistributedAdaptiveHierarchy::visit_geometry_snapshots(
    const std::function<void(const AdaptiveGeometrySnapshot&)>& visitor) const
{
    std::lock_guard lock(data_->geometry_cache->mutex);
    for (const auto& root : data_->geometry_cache->roots)
        if (root)
            visitor(*root);
}
DistributedAdaptiveHierarchy::Storage DistributedAdaptiveHierarchy::storage() const
{
    Storage result;
    result.owned_leaves = owned_leaves().size();
    result.family_capsules = owned_families().size();
    result.directory_entries = data_->cells->local_entries() + data_->family_directory->local_entries();
    result.directory_bytes = data_->cells->storage_bytes() + data_->family_directory->storage_bytes();
    result.owned_bytes = sizeof(Data) + sizeof(AdaptiveLeafLayout);
    result.owned_bytes += sizeof(AdaptiveGeometrySnapshotCache);
    if (data_->contact_halo)
        result.owned_bytes += data_->contact_halo->cache_storage_bytes();
    std::set<const void*> objects, geometry;
    auto key_dynamic = [](const AdaptiveCellKey& key)
    {
        return std::visit(
            [](const auto& p) -> std::size_t
            {
                using T = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                    return p.path.capacity() * sizeof(std::uint64_t);
                else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                    return (p.base.path.capacity() + p.axial.path.capacity()) * sizeof(std::uint64_t);
                else
                    return 0;
            },
            key.ancestry);
    };
    auto part = [&](const std::shared_ptr<const AdaptiveBytes>& value)
    {
        if (value && objects.insert(value.get()).second)
            result.procedural_bytes += sizeof(AdaptiveBytes) + value->capacity();
    };
    PLIC::GeometryAllocationLedger identity_allocations;
    std::size_t identity_counter = 0;
    {
        std::lock_guard lock(data_->geometry_cache->mutex);
        PLIC::GeometryAllocationLedger bounded_owners;
        const auto before = identity_allocations.payload_bytes();
        for (const auto& channels : data_->geometry_cache->manifests)
            for (const auto& manifest : channels)
                result.owned_bytes += manifest.metadata_payload_bytes();
        for (const auto& root : data_->geometry_cache->roots)
            if (root)
            {
                root->visit_external_owners([&](const void* owner, std::size_t) { objects.insert(owner); });
                root->retain(
                    identity_allocations, bounded_owners, identity_counter, std::numeric_limits<std::size_t>::max());
            }
        result.owned_bytes += identity_allocations.payload_bytes() - before + bounded_owners.payload_bytes();
    }

    const auto payload_delta = [&](const auto& value)
    {
        const auto before = identity_allocations.payload_bytes();
        identity_allocations.retain(value, identity_counter, std::numeric_limits<std::size_t>::max());
        return identity_allocations.payload_bytes() - before;
    };
    const auto face_bytes = [&](const auto& face) -> std::size_t
    {
        if (!face)
            return 0;
        const auto before = identity_allocations.payload_bytes();
        identity_allocations.retain_planar_face(face, identity_counter, std::numeric_limits<std::size_t>::max());
        return identity_allocations.payload_bytes() - before;
    };
    auto leaf = [&](const AdaptiveLeaf& value, std::size_t& bytes)
    {
        bytes += key_dynamic(value.key) + value.domain.capacity() + value.parent_family.capacity() +
                 value.surfaces.capacity() * sizeof(AdaptiveSurface);
        if (value.native_recipe && objects.insert(value.native_recipe.get()).second)
        {
            result.procedural_bytes +=
                sizeof(NativeFragmentDescriptor) + value.native_recipe->view.capacity() +
                value.native_recipe->parts.capacity() * sizeof(std::shared_ptr<const AdaptiveBytes>);
            for (const auto& p : value.native_recipe->parts)
                part(p);
        }
        part(value.join_recipe);
        if (value.factor_recipe && objects.insert(value.factor_recipe.get()).second)
        {
            result.procedural_bytes += sizeof(AdaptiveFactorRecipe);
            part(value.factor_recipe->root);
            part(value.factor_recipe->base);
            part(value.factor_recipe->axis);
        }
        if (value.geometry && geometry.insert(value.geometry.get()).second)
            bytes += payload_delta(value.geometry);
        for (const auto& face : value.surfaces)
        {
            // authority_image is an inline Ordinary displacement, already in
            // the AdaptiveSurface capacity counted above, with no owner edge.
            bytes += face.key.capacity() + face.boundary.capacity();
            bytes += face_bytes(face.geometry);
            bytes += face_bytes(face.native_whole_geometry);
        }
    };
    result.owned_bytes += data_->pages().retained_bytes() + data_->body_usage.storage_bytes();
    for (const auto& value : owned_leaves())
    {
        leaf(value, result.owned_bytes);
        // The page already counts the row's exclusively owned buffers.
        result.owned_bytes -= AdaptiveLeafPages::buffer_bytes(value);
    }
    result.ancestor_bytes += data_->family_layout_residency();
    for (const auto family : owned_families())
    {
        leaf(family.parent, result.ancestor_bytes);
        result.ancestor_bytes -= AdaptiveLeafPages::buffer_bytes(family.parent);
    }
    if (data_->native)
    {
        result.native_bytes = data_->native->geometry()->storage_report().measured_bytes();
        const auto& plan = data_->native->plan();
        result.native_map_entries = plan.cell_ids().size() + plan.face_ids().size() + plan.node_ids().size();
        result.native_bytes +=
            plan.cell_ids().capacity() * sizeof(std::uint64_t) + plan.cell_owner_ranks().capacity() * sizeof(int) +
            plan.face_ids().capacity() * sizeof(std::uint64_t) + plan.node_ids().capacity() * sizeof(std::uint64_t);
    }
    return result;
}
DistributedAdaptiveHierarchy::Plan DistributedAdaptiveHierarchy::plan_adaptation(
    std::span<const AdaptiveMark> refine, std::span<const AdaptiveBytes> coarsen) const
{
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    std::map<AdaptiveCellKey, AdaptiveSplit> marks;
    std::vector<AdaptiveBytes> family_keys;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            if (data_->topology == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Distributed topology generation exhausted.");
            for (const auto& mark : refine)
            {
                budget.add(sizeof(mark) + encoded_adaptive_key_size(mark.key));
                if (!owned_leaves().contains(mark.key) || !marks.emplace(mark.key, mark.split).second)
                    throw std::invalid_argument("Refinement mark is repeated or not locally owned.");
            }
            std::set<AdaptiveBytes> unique;
            for (const auto& key : coarsen)
            {
                budget.add(key.size());
                if (unique.insert(key).second)
                    family_keys.push_back(key);
            }
        });
    std::vector<AdaptiveBytes> group_keys;
    std::vector<AdaptiveSplit> group_splits;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& [key, split] : marks)
                for (const auto& member : data_->kernel->refinement_group(key, split))
                {
                    auto bytes = encode_adaptive_key(member);
                    budget.add(bytes.size() + sizeof(AdaptiveSplit));
                    group_keys.push_back(std::move(bytes));
                    group_splits.push_back(split);
                }
        });
    auto group_owners = data_->cells->lookup(group_keys);
    std::vector<OwnerPacket> group_requests;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (std::size_t i = 0; i < group_owners.size(); ++i)
            {
                Writer w;
                w.bytes(group_owners[i].key);
                w.word(std::uint64_t(group_splits[i]));
                w.word(group_owners[i].revision);
                budget.packet(w.data.size());
                group_requests.push_back({group_owners[i].owner, std::move(w.data)});
            }
        });
    auto group_incoming = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(
        group_requests, data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            marks.clear();
            for (const auto& packet : group_incoming.packets)
            {
                Reader r{packet.bytes};
                auto key = decode_adaptive_key(r.bytes());
                auto split = r.word(), revision = r.word();
                r.done();
                const auto* leaf = owned_leaves().find(key);
                if (split > 2 || !leaf || leaf->revision != revision)
                    throw std::invalid_argument("Invalid distributed native refinement group.");
                auto [it, inserted] = marks.emplace(key, AdaptiveSplit(split));
                if (!inserted && it->second != AdaptiveSplit(split))
                    throw std::invalid_argument("Conflicting distributed factor marks.");
            }
        });
    const auto plan_id = transaction(*data_->comm);
    std::uint64_t local_changes = refine.size() + coarsen.size(), global_changes = 0;
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &local_changes, &global_changes);
    if (!global_changes)
    {
        Plan plan;
        plan.source_ = data_->token;
        plan.observation_ = observation();
        plan.topology_ = data_->topology;
        plan.partition_ = data_->partition;
        plan.transaction_ = plan_id;
        plan.candidate_ = std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(data_));
        return plan;
    }
    auto owners = data_->family_directory->lookup(family_keys);
    std::vector<OwnerPacket> requests;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& owner : owners)
            {
                Writer w;
                w.bytes(owner.key);
                w.word(owner.revision);
                budget.packet(w.data.size());
                requests.push_back({owner.owner, std::move(w.data)});
            }
        });
    auto incoming = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(
        requests, data_->comm, data_->options.byte_budget);
    std::set<AdaptiveBytes> requested;
    collective(*data_->comm,
        [&]
        {
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto key = r.bytes();
                auto revision = r.word();
                r.done();
                const auto family = owned_families().find(key);
                if (!family || family->revision != revision)
                    throw std::invalid_argument("Stale family authority request.");
                requested.insert(std::move(key));
            }
        });
    std::vector<AdaptiveBytes> grouped_families;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            std::set<AdaptiveBytes> unique;
            for (const auto& key : requested)
            {
                const auto family = owned_families().at(key);
                const auto before = budget.used;
                fetch_copy_charge(budget, family);
                auto group = data_->kernel->coarsening_group(family.materialize());
                budget.used = before;
                for (auto& member : group)
                {
                    budget.add(member.size());
                    if (unique.insert(member).second)
                        grouped_families.push_back(std::move(member));
                }
            }
        });
    auto grouped_owners = data_->family_directory->lookup(grouped_families);
    requests.clear();
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& owner : grouped_owners)
            {
                Writer w;
                w.bytes(owner.key);
                w.word(owner.revision);
                budget.packet(w.data.size());
                requests.push_back({owner.owner, std::move(w.data)});
            }
        });
    incoming = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(
        requests, data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            requested.clear();
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto key = r.bytes();
                auto revision = r.word();
                r.done();
                const auto family = owned_families().find(key);
                if (!family || family->revision != revision)
                    throw std::invalid_argument("Invalid native coarsening group authority.");
                requested.insert(std::move(key));
            }
        });
    std::vector<AdaptiveBytes> child_keys;
    std::vector<AdaptiveBytes> child_families;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& key : requested)
                for (const auto& child : owned_families().at(key).children)
                {
                    auto encoded = encode_adaptive_key(child);
                    budget.add(encoded.size() + key.size());
                    child_keys.push_back(std::move(encoded));
                    child_families.push_back(key);
                }
        });
    owners = data_->cells->lookup(child_keys);
    requests.clear();
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (std::size_t i = 0; i < owners.size(); ++i)
            {
                Writer w;
                w.bytes(child_families[i]);
                w.bytes(owners[i].key);
                w.word(owners[i].revision);
                budget.packet(w.data.size());
                requests.push_back({owners[i].owner, std::move(w.data)});
            }
        });
    incoming = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(
        requests, data_->comm, data_->options.byte_budget);
    std::vector<OwnerPacket> replies;
    std::set<AdaptiveCellKey> retiring;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto family = r.bytes();
                auto key = decode_adaptive_key(r.bytes());
                auto revision = r.word();
                r.done();
                const auto* leaf = owned_leaves().find(key);
                if (!leaf || leaf->revision != revision || marks.contains(key) || !retiring.insert(key).second)
                    throw std::invalid_argument("Incomplete, conflicting or stale source sibling family.");
                Writer w;
                w.bytes(family);
                w.bytes(encode_adaptive_leaf(*leaf));
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto children = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(
        replies, data_->comm, data_->options.byte_budget);
    std::vector<AdaptiveBytes>().swap(family_keys);
    std::vector<AdaptiveBytes>().swap(group_keys);
    std::vector<AdaptiveSplit>().swap(group_splits);
    std::vector<AdaptiveOwner>().swap(group_owners);
    std::vector<OwnerPacket>().swap(group_requests);
    group_incoming = {};
    std::vector<AdaptiveBytes>().swap(grouped_families);
    std::vector<AdaptiveOwner>().swap(grouped_owners);
    std::vector<AdaptiveBytes>().swap(child_keys);
    std::vector<AdaptiveBytes>().swap(child_families);
    std::vector<AdaptiveOwner>().swap(owners);
    std::vector<OwnerPacket>().swap(requests);
    incoming = {};
    std::vector<OwnerPacket>().swap(replies);
    const auto source_observation = observation();
    ContactHaloContext closure_context{
        source_observation.geometry, source_observation.topology, source_observation.partition, 0, plan_id};
    std::unique_ptr<DistributedAdaptiveContactHalo> frontier;
    std::shared_ptr<DistributedAdaptiveContactHalo> candidate_keepalive;
    collective(*data_->comm,
        [&]
        {
            if (!data_->contact_halo)
                throw std::invalid_argument("Distributed contact cache needs ownership reinitialization.");
        });
    Budget overlap{data_->options.byte_budget};
    std::size_t outside_publications = 0;
    collective(*data_->comm,
        [&]
        {
            charge_source_residency(overlap, *data_, outside_publications);
            overlap.add(closure_packet_storage(children.packets));
            for (const auto& [key, split] : marks)
                overlap.add(sizeof(decltype(marks)::value_type) + 4 * sizeof(void*) + fetch_key_bytes(key));
            for (const auto& key : retiring)
                overlap.add(sizeof(AdaptiveCellKey) + 4 * sizeof(void*) + fetch_key_bytes(key));
            for (const auto& key : requested)
                overlap.add(sizeof(AdaptiveBytes) + 4 * sizeof(void*) + key.capacity());
        });
    frontier = data_->contact_halo->fork_candidate(closure_context, overlap.used);
    ClosureScratch closure_memory{frontier->allocation()};
    ClosureScratch old_output_overlap{frontier->allocation(), outside_publications};
    ClosureScratch initial_memory{frontier->allocation(), overlap.used - outside_publications};
    ClosureScratch candidate_memory{frontier->allocation()};
    ClosureScratch leaf_memory{frontier->allocation()};
    ClosureScratch family_memory{frontier->allocation()};
    FamilyJournal family_journal(owned_families());
    std::map<AdaptiveBytes, std::vector<AdaptiveLeaf>> families;
    std::vector<AdaptiveOwner> initially_retired, initially_inserted;
    std::shared_ptr<Data> next;
    collective(*data_->comm,
        [&]
        {
            for (const auto& packet : children.packets)
            {
                if (packet.bytes.size() > initial_memory.budget.limit / 4)
                    throw std::invalid_argument("Initial family decode exceeds budget.");
                initial_memory.add(4 * packet.bytes.size());
                Reader r{packet.bytes};
                auto family = r.bytes();
                if (!requested.contains(family))
                    throw std::invalid_argument("Foreign family child response.");
                auto leaf = decode_adaptive_leaf(r.bytes());
                r.done();
                families[family].push_back(std::move(leaf));
            }
            candidate_memory.add(sizeof(Data) + 4 * sizeof(void*) + overlay_header_bytes);
            next = std::make_shared<Data>();
            next->kernel = data_->kernel;
            next->native_source_enabled = data_->native_source_enabled;
            next->comm = data_->comm;
            next->options = data_->options;
            inherit_geometry_predecessors(*next->geometry_cache, *data_->geometry_cache);
            next->leaves.emplace<std::shared_ptr<AdaptiveLeafOverlay>>(
                std::make_shared<AdaptiveLeafOverlay>(data_->pages(), candidate_memory.budget));
            if (data_->topology == std::numeric_limits<std::uint64_t>::max() ||
                data_->partition == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Distributed adaptive generation exhausted.");
            next->topology = data_->topology + 1;
            next->partition = data_->partition + 1;
            for (const auto& key : retiring)
            {
                initial_memory.append(initially_retired);
                charge_encoded_key(initial_memory, key);
                initially_retired.push_back(
                    {encode_adaptive_key(key), data_->comm->getRank(), owned_leaves().at(key).revision});
                (void) next->overlay().retire(key);
            }
            for (const auto& key : requested)
            {
                const auto family = owned_families().at(key);
                ClosureScratch parent_memory{initial_memory.budget};
                charge_candidate_copy(parent_memory, family);
                auto capsule = family.materialize();
                const auto cap = initial_memory.budget.limit - initial_memory.budget.used;
                if (!cap)
                    throw std::invalid_argument("Initial coarsening lease exhausted.");
                initial_memory.add(cap);
                auto parent = data_->kernel->coarsen(capsule, families[key]);
                const auto retained = fetch_retention(std::span<const AdaptiveLeaf>(&parent, 1), {}, cap, false);
                initial_memory.drop(cap - retained);
                initial_memory.held -= retained;
                leaf_memory.held += retained;
                const auto parent_anchor = encode_adaptive_key(parent.key);
                family_journal.erase(key, family_memory);
                for (const auto& child : family.children)
                    family_journal.move_anchor(encode_adaptive_key(child), parent_anchor, family_memory);
                initial_memory.append(initially_inserted);
                charge_encoded_key(initial_memory, parent.key);
                initially_inserted.push_back(
                    {encode_adaptive_key(parent.key), data_->comm->getRank(), parent.revision});
                if (!next->overlay().insert(std::move(parent)))
                    throw std::invalid_argument("Coarsened parent already active.");
            }
            for (const auto& [key, kind] : marks)
            {
                const auto cap = initial_memory.budget.limit - initial_memory.budget.used;
                if (!cap)
                    throw std::invalid_argument("Initial refinement lease exhausted.");
                initial_memory.add(cap);
                auto result = data_->kernel->split(owned_leaves().at(key), kind);
                std::size_t leaf_bytes = 0;
                const auto retained = fetch_retention(
                    result.children, std::span<const AdaptiveFamily>(&result.family, 1), cap, false, &leaf_bytes);
                initial_memory.drop(cap - retained);
                initial_memory.held -= retained;
                leaf_memory.held += leaf_bytes;
                family_memory.held += retained - leaf_bytes;
                initial_memory.append(initially_retired);
                charge_encoded_key(initial_memory, key);
                initially_retired.push_back(
                    {encode_adaptive_key(key), data_->comm->getRank(), owned_leaves().at(key).revision});
                (void) next->overlay().retire(key);
                family_journal.move_anchor(encode_adaptive_key(key), result.family.anchor, family_memory);
                for (auto& leaf : result.children)
                {
                    initial_memory.add(sizeof(AdaptiveLeaf));
                    charge_encoded_key(initial_memory, leaf.key);
                    data_->kernel->validate(leaf);
                    initial_memory.append(initially_inserted);
                    charge_encoded_key(initial_memory, leaf.key);
                    initially_inserted.push_back(
                        {encode_adaptive_key(leaf.key), data_->comm->getRank(), leaf.revision});
                    if (!next->overlay().insert(std::move(leaf)))
                        throw std::invalid_argument("Refined child already active.");
                }
                if (result.family.parent.provider != AdaptiveProviderKind::Product)
                    family_journal.insert(std::move(result.family), family_memory);
            }
            if (next->overlay().size() > data_->options.maximum_owned)
                throw std::invalid_argument("Distributed local leaf budget exhausted.");
        });
    close_candidate(*next, *data_, initially_retired, initially_inserted, *frontier, closure_context, family_journal,
        closure_memory, family_memory, leaf_memory);
    marks.clear();
    retiring.clear();
    requested.clear();
    children = {};
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || source_observation != observation())
                throw std::invalid_argument("Distributed source geometry changed during closure.");
        });
    SealedLeafStorage leaf_storage;
    collective(*data_->comm,
        [&]
        {
            normalize_candidate(*next, leaf_memory);
            leaf_storage = seal_candidate_layout(
                *next, *data_, initial_memory, candidate_memory, leaf_memory, family_memory, *frontier);
        });
    register_cells(*next, &candidate_memory, data_.get(), &initial_memory);
    edit_product_families(*next, *data_, leaf_storage.product_edits, plan_id, family_journal, initial_memory,
        candidate_memory, family_memory);
    std::vector<AdaptiveCellKey>().swap(leaf_storage.product_edits);
    collective(*data_->comm,
        [&]
        {
            seal_candidate_families(*next, *data_, leaf_storage, family_journal, initial_memory, candidate_memory,
                leaf_memory, family_memory);
        });
    register_families(*next, &candidate_memory, data_.get(), &initial_memory);
    std::shared_ptr<AdaptivePublishedAdmission> admission;
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || source_observation != observation())
                throw std::invalid_argument("Distributed source changed during candidate publication.");
            candidate_memory.add(sizeof(AdaptivePublishedAdmission) + 4 * sizeof(void*));
            candidate_keepalive = std::shared_ptr<DistributedAdaptiveContactHalo>(std::move(frontier));
            admission = std::make_shared<AdaptivePublishedAdmission>();
        });
    collective(*data_->comm, [&]
    {
        const auto family_body = next->body_usage.payload_bytes() - next->body_usage.leaf_payload_bytes();
        if (family_body > candidate_memory.held)
            throw std::logic_error("Detached family body coverage exceeds its exact candidate admission.");
        candidate_memory.add(sizeof(ContactExternalResidencyLease) + 4 * sizeof(void*));
        auto family_admission = std::make_shared<ContactExternalResidencyLease>(candidate_keepalive->external_residency_lease());
        family_admission->adopt_precharged(family_body);
        candidate_memory.held -= family_body;
        next->bootstrap_family_admission = std::move(family_admission);
    });
    next->finish_directory_overlap();
    next->contact_halo = candidate_keepalive;
    collective(*data_->comm, [&] { refresh_native_source(*next, *candidate_keepalive, &initial_memory, &candidate_memory, false,
            {&closure_memory, &leaf_memory, &family_memory, &old_output_overlap}); });
    admission->halo = candidate_keepalive;
    admission->bytes = candidate_memory.held;
    candidate_keepalive->bind_external_retained_bytes(admission->bytes);
    candidate_memory.held = 0;
    next->contact_halo = candidate_keepalive;
    next->admission = std::move(admission);
    Plan plan;
    plan.source_ = data_->token;
    plan.observation_ = source_observation;
    plan.topology_ = data_->topology;
    plan.partition_ = data_->partition;
    plan.transaction_ = plan_id;
    plan.candidate_ =
        std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(std::move(next)));
    return plan;
}
DistributedAdaptiveHierarchy::Plan DistributedAdaptiveHierarchy::plan_migration(std::span<const int> destinations) const
{
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    const auto source_observation = observation();
    NativeMigrationDiagnostics diagnostics;
    std::vector<ContactRetiredLeaf> migrated_retired;
    std::vector<AdaptiveCellKey> migrated_inserted;
    const auto id = transaction(*data_->comm);
    std::vector<OwnerPacket> packets;
    std::uint64_t local_changes = 0, changes = 0;
    collective(*data_->comm,
        [&]
        {
            if (destinations.size() != owned_leaves().size() ||
                data_->partition == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Invalid ownership migration layout or exhausted generation.");
            for (const auto destination : destinations)
            {
                if (destination < 0 || destination >= data_->comm->getSize())
                    throw std::invalid_argument("Invalid migration destination.");
                local_changes += destination != data_->comm->getRank();
            }
        });
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &local_changes, &changes);
    if (!changes)
    {
        Plan plan;
        plan.source_ = data_->token;
        plan.observation_ = observation();
        plan.topology_ = data_->topology;
        plan.partition_ = data_->partition;
        plan.transaction_ = id;
        plan.candidate_ = std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(data_));
        return plan;
    }
    const auto local_rank = data_->comm->getRank();
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            struct Group
            {
                std::vector<AdaptiveLeaf> leaves;
                std::vector<AdaptiveFamily> families;
            };
            std::map<int, Group> groups;
            std::size_t ordinal = 0;
            for (const auto& leaf : owned_leaves())
            {
                const auto destination = destinations[ordinal++];
                if (destination == local_rank)
                    continue;
                budget.add(sizeof(ContactRetiredLeaf) + fetch_key_bytes(leaf.key));
                migrated_retired.push_back({leaf.key, leaf.revision});
                fetch_copy_charge(budget, leaf);
                auto& group = groups[destination];
                group.leaves.push_back(leaf);
                ClosureScratch key_memory{budget};
                charge_encoded_key(key_memory, leaf.key);
                owned_families().visit_anchored(encode_adaptive_key(leaf.key),
                    [&](AdaptiveFamilyView family)
                    {
                        fetch_copy_charge(budget, family);
                        group.families.push_back(family.materialize());
                    });
            }
            for (auto& [rank, group] : groups)
            {
                std::ranges::sort(group.families, {}, &AdaptiveFamily::key);
                auto bytes = encode_adaptive_batch(group.leaves, group.families, data_->options.byte_budget);
                budget.packet(bytes.size());
                packets.push_back({rank, std::move(bytes)});
            }
        });
    collective(*data_->comm,
        [&]
        {
            std::set<int> peers;
            for (const auto& packet : packets)
            {
                diagnostics.peak_native_packet_buffers += packet.bytes.capacity();
                diagnostics.remote_sent_payload_bytes += packet.bytes.size();
                ++diagnostics.remote_sent_packets;
                peers.insert(packet.peer);
            }
            diagnostics.peak_native_packet_buffers += packets.capacity() * sizeof(OwnerPacket);
            diagnostics.remote_sent_peers = peers.size();
        });
    auto incoming = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(
        packets, data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            std::set<int> peers;
            for (const auto& packet : incoming.packets)
            {
                if (packet.peer == local_rank)
                    throw std::logic_error("Remote migration returned a self packet.");
                diagnostics.remote_received_payload_bytes += packet.bytes.size();
                ++diagnostics.remote_received_packets;
                peers.insert(packet.peer);
            }
            diagnostics.remote_received_peers = peers.size();
            if (incoming.stats.peak_scratch_bytes >
                std::numeric_limits<std::size_t>::max() - diagnostics.peak_native_packet_buffers)
                throw std::overflow_error("Native packet peak counter overflow.");
            diagnostics.peak_native_packet_buffers += incoming.stats.peak_scratch_bytes;
        });
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || source_observation != observation() || !data_->contact_halo)
                throw std::invalid_argument("Migration source contact cache is stale.");
        });
    ContactHaloContext context{
        source_observation.geometry, source_observation.topology, source_observation.partition, 0, id};
    Budget overlap{data_->options.byte_budget};
    std::size_t outside_publications = 0;
    collective(*data_->comm,
        [&]
        {
            charge_source_residency(overlap, *data_, outside_publications);
            overlap.add(closure_packet_storage(packets));
            overlap.add(closure_packet_storage(incoming.packets));
            if (migrated_retired.capacity() > overlap.limit / sizeof(ContactRetiredLeaf))
                throw std::invalid_argument("Migration contact delta exceeds budget.");
            overlap.add(migrated_retired.capacity() * sizeof(ContactRetiredLeaf));
            for (const auto& item : migrated_retired)
                overlap.add(fetch_key_bytes(item.key));
        });
    auto frontier = data_->contact_halo->fork_candidate(context, overlap.used);
    std::shared_ptr<DistributedAdaptiveContactHalo> candidate_keepalive;
    ClosureScratch old_output_overlap{frontier->allocation(), outside_publications};
    ClosureScratch migration_memory{frontier->allocation(), overlap.used - outside_publications};
    ClosureScratch candidate_memory{frontier->allocation()};
    ClosureScratch leaf_memory{frontier->allocation()};
    ClosureScratch family_memory{frontier->allocation()};
    FamilyJournal family_journal(owned_families());
    std::shared_ptr<Data> next;
    collective(*data_->comm,
        [&]
        {
            candidate_memory.add(sizeof(Data) + 4 * sizeof(void*) + overlay_header_bytes);
            next = std::make_shared<Data>();
            next->kernel = data_->kernel;
            next->native_source_enabled = data_->native_source_enabled;
            next->comm = data_->comm;
            next->options = data_->options;
            inherit_geometry_predecessors(*next->geometry_cache, *data_->geometry_cache);
            next->topology = data_->topology;
            next->partition = data_->partition + 1;
            next->topology_token = data_->topology_token;
            next->topology_allocation = data_->topology_allocation;
            next->leaves.emplace<std::shared_ptr<AdaptiveLeafOverlay>>(
                std::make_shared<AdaptiveLeafOverlay>(data_->pages(), candidate_memory.budget));
            for (const auto& retired : migrated_retired)
                if (!next->overlay().retire(retired.key))
                    throw std::invalid_argument("Migrated retirement lost its source leaf.");
            for (const auto& retired : migrated_retired)
            {
                ClosureScratch key_memory{migration_memory.budget};
                charge_encoded_key(key_memory, retired.key);
                owned_families().visit_anchored(encode_adaptive_key(retired.key),
                    [&](AdaptiveFamilyView family)
                    {
                        if (family.parent.provider != AdaptiveProviderKind::Product)
                            family_journal.erase(family.key, family_memory);
                    });
            }
            for (const auto& packet : incoming.packets)
            {
                const auto cap = migration_memory.budget.limit - migration_memory.budget.used;
                if (!cap)
                    throw std::invalid_argument("Migration decode lease exhausted.");
                migration_memory.add(cap);
                auto batch = decode_adaptive_batch(packet.bytes, cap);
                std::size_t leaf_bytes = 0;
                const auto retained = fetch_retention(batch.leaves, batch.families, cap, false, &leaf_bytes);
                migration_memory.drop(cap - retained);
                migration_memory.held -= retained;
                leaf_memory.held += leaf_bytes;
                family_memory.held += retained - leaf_bytes;
                for (auto& leaf : batch.leaves)
                {
                    data_->kernel->validate(leaf);
                    migration_memory.append(migrated_inserted);
                    migration_memory.add(fetch_key_bytes(leaf.key));
                    migrated_inserted.push_back(leaf.key);
                    if (!next->overlay().insert(std::move(leaf)))
                        throw std::invalid_argument("Duplicate migrated leaf owner.");
                }
                for (auto& family : batch.families)
                {
                    if (family.parent.provider != AdaptiveProviderKind::Product)
                        family_journal.insert(std::move(family), family_memory);
                    else
                    {
                        // A moved Product capsule still has its original wire proof.
                        // The complete compatibility election replaces this staging row.
                        validate_family(*data_->kernel, family);
                        family_memory.add(sizeof(decltype(next->staged_families)::value_type) + 4 * sizeof(void*) +
                                          family.key.size());
                        if (!next->staged_families.emplace(family.key, std::move(family)).second)
                            throw std::invalid_argument("Duplicate migrated Product family authority.");
                    }
                }
            }
            if (next->overlay().size() > data_->options.maximum_owned)
                throw std::invalid_argument("Migrated local leaf budget exhausted.");
            normalize_candidate(*next, leaf_memory);
            for (const auto& [key, family] : next->staged_families)
                if (!next->overlay().contains(decode_adaptive_key(family.anchor)))
                    throw std::invalid_argument("Migrated capsule lost actual anchor.");
        });
    ContactOwnedLeafLookup lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
    { return next->overlay().at(key); };
    context.epoch = next->native_epoch;
    context.partition = next->partition;
    ++context.round;
    frontier->apply_delta(migrated_retired, migrated_inserted, context, lookup);
    std::vector<ContactRetiredLeaf>().swap(migrated_retired);
    std::vector<AdaptiveCellKey>().swap(migrated_inserted);
    std::vector<OwnerPacket>().swap(packets);
    incoming = {};
    {
        const auto requirements = frontier->lease_required_refinements();
        collective(*data_->comm,
            [&]
            {
                if (!requirements.marks().empty())
                    throw std::invalid_argument("Ownership migration changed contact grading.");
            });
    }
    SealedLeafStorage leaf_storage;
    collective(*data_->comm,
        [&]
        {
            leaf_storage = seal_candidate_layout(
                *next, *data_, migration_memory, candidate_memory, leaf_memory, family_memory, *frontier);
        });
    register_cells(*next, &candidate_memory, data_.get(), &migration_memory);
    edit_product_families(*next, *data_, leaf_storage.product_edits, id, family_journal, migration_memory,
        candidate_memory, family_memory);
    std::vector<AdaptiveCellKey>().swap(leaf_storage.product_edits);
    collective(*data_->comm,
        [&]
        {
            seal_candidate_families(*next, *data_, leaf_storage, family_journal, migration_memory, candidate_memory,
                leaf_memory, family_memory);
        });
    register_families(*next, &candidate_memory, data_.get(), &migration_memory);
    std::shared_ptr<AdaptivePublishedAdmission> admission;
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || source_observation != observation())
                throw std::invalid_argument("Migration source changed during contact rebind.");
            candidate_memory.add(sizeof(AdaptivePublishedAdmission) + 4 * sizeof(void*));
            candidate_keepalive = std::shared_ptr<DistributedAdaptiveContactHalo>(std::move(frontier));
            admission = std::make_shared<AdaptivePublishedAdmission>();
        });
    collective(*data_->comm, [&]
    {
        const auto family_body = next->body_usage.payload_bytes() - next->body_usage.leaf_payload_bytes();
        if (family_body > candidate_memory.held)
            throw std::logic_error("Detached family body coverage exceeds its exact candidate admission.");
        candidate_memory.add(sizeof(ContactExternalResidencyLease) + 4 * sizeof(void*));
        auto family_admission = std::make_shared<ContactExternalResidencyLease>(candidate_keepalive->external_residency_lease());
        family_admission->adopt_precharged(family_body);
        candidate_memory.held -= family_body;
        next->bootstrap_family_admission = std::move(family_admission);
    });
    next->finish_directory_overlap();
    next->contact_halo = candidate_keepalive;
    collective(*data_->comm, [&] { refresh_native_source(*next, *candidate_keepalive, &migration_memory, &candidate_memory, false,
            {&leaf_memory, &family_memory, &old_output_overlap}); });
    admission->halo = candidate_keepalive;
    admission->bytes = candidate_memory.held;
    candidate_keepalive->bind_external_retained_bytes(admission->bytes);
    candidate_memory.held = 0;
    next->contact_halo = candidate_keepalive;
    next->admission = std::move(admission);
    Plan plan;
    plan.native_migration_ = diagnostics;
    plan.source_ = data_->token;
    plan.observation_ = source_observation;
    plan.topology_ = data_->topology;
    plan.partition_ = data_->partition;
    plan.transaction_ = id;
    plan.candidate_ =
        std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(std::move(next)));
    return plan;
}
DistributedAdaptiveHierarchy::Plan DistributedAdaptiveHierarchy::plan_migration(
    const AdaptiveBalancePlan& proposal) const
{
    collective(*data_->comm,
        [&]
        {
            if (proposal.source.identity != data_->token || proposal.source.topology != data_->topology ||
                proposal.source.partition != data_->partition || proposal.keys.size() != owned_leaves().size() ||
                proposal.revisions.size() != owned_leaves().size() ||
                proposal.localdestinations.size() != owned_leaves().size())
                throw std::invalid_argument("Foreign, stale or incomplete measured ownership proposal.");
            if (proposal.accepted && !proposal.proposed.wire_bytes_exact)
                throw std::invalid_argument("Measured ownership proposal lacks actual packed stream qualification.");
            std::size_t i = 0;
            for (const auto& leaf : owned_leaves())
            {
                if (proposal.keys[i] != leaf.key || proposal.revisions[i] != leaf.revision)
                    throw std::invalid_argument("Measured ownership proposal changed local key/revision order.");
                ++i;
            }
        });
    return plan_migration(proposal.localdestinations);
}
std::shared_ptr<const DistributedAdaptiveHierarchy> DistributedAdaptiveHierarchy::apply(const Plan& plan) const
{
    agree(*data_->comm, std::array<std::uint64_t, 1>{plan.transaction_});
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || plan.observation_ != observation() || plan.source_ != data_->token ||
                plan.topology_ != data_->topology || plan.partition_ != data_->partition || !plan.candidate_)
                throw std::invalid_argument("Foreign or stale distributed adaptation plan.");
        });
    return plan.candidate_;
}
std::shared_ptr<const DistributedAdaptiveHierarchy> DistributedAdaptiveHierarchy::apply(
    const Plan& plan, std::shared_ptr<const DistributedAdaptiveHierarchy> candidate) const
{
    (void) apply(plan);
    collective(*data_->comm,
        [&]
        {
            if (!candidate || candidate->identity() != plan.candidate_->identity() ||
                candidate->topology_generation() != plan.candidate_->topology_generation() ||
                candidate->partition_generation() != plan.candidate_->partition_generation() ||
                candidate->leaf_layout() != plan.candidate_->leaf_layout() || !candidate->native_partition() ||
                !candidate->geometry_unchanged())
                throw std::invalid_argument("Native-bound candidate does not belong to this source-pinned plan.");
        });
    return candidate;
}
AdaptiveNativeDraft DistributedAdaptiveHierarchy::native_draft() const
{
    return native_draft_impl(false);
}
AdaptiveNativeDraft adaptive_detail::compact_adaptive_native_draft(const DistributedAdaptiveHierarchy& hierarchy)
{
    return hierarchy.native_draft_impl(true);
}
AdaptiveNativeDraft DistributedAdaptiveHierarchy::native_draft_impl(bool compact_procedural) const
{
    const PipelinePhaseScope audit(PipelinePhase::NativePublication);
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    AdaptiveNativeDraft result;
    result.source_layout = leaf_layout();
    collective(*data_->comm,
        [&]
        {
            // A provider enum is not an allocation guarantee. Only the actual
            // kernel's explicit bounded materialization capability can select
            // the capacity contract. Any unsupported/supplied record preserves
            // the complete call's original cumulative expanded-size guard.
            const bool bounded = compact_procedural && !owned_leaves().empty() &&
                                 std::all_of(owned_leaves().begin(), owned_leaves().end(), [&](const auto& leaf)
                                     { return data_->kernel->compact_native_materialization_bound(leaf) != 0; });
            if (bounded)
            {
                adaptive_detail::CompactNativeOwnedBudget budget(data_->options.byte_budget);
                auto report = [&](const char* stage)
                {
                    if (!std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
                        return;
                    const auto stats = budget.statistics();
                    std::fprintf(stderr,
                        "ADAPTIVE_NATIVE_OWNED rank=%d mode=bounded_cartesian stage=%s leaves=%zu "
                        "retained_bytes=%zu live_peak_bytes=%zu cumulative_bytes=%zu serialized_bytes=%zu "
                        "transient_geometry_peak_bytes=%zu encoding_scratch_peak_bytes=%zu "
                        "materialization_peak_bytes=%zu limit=%zu scope=owned_record_capacity_and_scratch\n",
                        data_->comm->getRank(), stage, stats.leaves, stats.retained_bytes, stats.live_peak_bytes,
                        stats.cumulative_bytes, stats.serialized_bytes, stats.transient_geometry_peak_bytes,
                        stats.encoding_scratch_peak_bytes, stats.materialization_peak_bytes,
                        data_->options.byte_budget);
                };
                try
                {
                    for (const auto& leaf : owned_leaves())
                        if (!budget.append_bounded(result.owned, leaf, *data_->kernel))
                            throw std::invalid_argument("Native provider changed bounded materialization capability.");
                }
                catch (...)
                {
                    report("rejected");
                    throw;
                }
                report("complete");
            }
            else
            {
                Budget budget{data_->options.byte_budget};
                for (const auto& leaf : owned_leaves())
                {
                    auto transient = leaf;
                    record_pipeline_work(PipelineWork::CellGeometryRequest);
                    transient.geometry = data_->kernel->geometry(leaf);
                    record_pipeline_work(PipelineWork::SurfaceGeometryRequest);
                    transient.surfaces = data_->kernel->surfaces(leaf);
                    budget.add(encode_adaptive_leaf(transient).size());
                    if (adaptive_detail::compact_native_geometry(leaf, compact_procedural))
                        transient.geometry.reset();
                    result.owned.push_back(std::move(transient));
                }
            }
        });
    result.global_cells = data_->global_cells;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            std::uint64_t gid = data_->cell_begin;
            for (const auto& leaf : owned_leaves())
            {
                const auto& key = leaf.key;
                budget.add(sizeof(AdaptiveNativeCellBinding) + encoded_adaptive_key_size(key));
                record_pipeline_work(PipelineWork::NumericOutput);
                const auto values = data_->kernel->metrics(leaf);
                AdaptiveNativeCellBinding binding;
                binding.key = key;
                binding.gid = gid++;
                binding.material = leaf.material;
                binding.native_cell = leaf.native_cell;
                binding.owner = data_->comm->getRank();
                binding.provider = leaf.provider;
                binding.full = true;
                binding.volume = values.volume;
                binding.centroid = values.centroid;
                binding.native_recipe = leaf.native_recipe;
                result.cell_bindings.push_back(std::move(binding));
            }
        });
    AdaptiveAuthorityStats authority_stats;
    const auto& subfaces = data_->contact_halo->subface_snapshot();
    collective(*data_->comm,
        [&]
        {
            if (!subfaces)
                throw std::invalid_argument("Native publication lacks its completed subface source.");
        });
    const auto source_leaf = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
    {
        const auto ordinal = owned_leaves().ordinal_of(key);
        if (!ordinal || *ordinal >= result.owned.size() || result.owned[*ordinal].key != key)
            throw std::invalid_argument("Native subface source is not in the current owned selection.");
        return result.owned[*ordinal];
    };
    result.tiles = publish_adaptive_subfaces(*subfaces, source_leaf, data_->comm, data_->options.byte_budget,
        data_->options.maximum_tiles, &authority_stats, &result.nodes);
    result.global_faces = authority_stats.global_faces;
    result.global_nodes = authority_stats.global_nodes;
    collective(*data_->comm,
        [&]
        {
            struct Boundary
            {
                std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> pieces;
                bool periodic_image = false;
            };
            std::map<AdaptiveCellKey, Boundary> incidence;
            for (const auto& tile : result.tiles)
            {
                const bool image = !tile.second_image_identity.is_zero();
                auto& first = incidence[tile.first];
                first.pieces.push_back(tile.geometry);
                first.periodic_image = first.periodic_image || image;
                if (tile.second)
                {
                    auto& second = incidence[*tile.second];
                    second.pieces.push_back(!image ? tile.geometry->reversed()
                                                   : tile.geometry->reversed()->translated(
                                                         PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                                             {}, tile.second_image_identity)));
                    second.periodic_image = second.periodic_image || image;
                }
            }
            for (const auto& leaf : result.owned)
            {
                // Regenerate from the original provider record, never from an
                // expanded draft that could remap a procedural cell ordinal.
                if (!leaf.geometry)
                    record_pipeline_work(PipelineWork::CellGeometryRequest);
                const auto geometry =
                    leaf.geometry ? leaf.geometry : data_->kernel->geometry(owned_leaves().at(leaf.key));
                const auto& boundary = incidence.at(leaf.key);
                if (boundary.periodic_image || leaf.surfaces.size() != geometry->num_faces())
                    certify_adaptive_physical_boundary(geometry, boundary.pieces, data_->options.byte_budget);
                else if ((record_pipeline_work(PipelineWork::PhysicalShellReconstruction),
                             !PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(boundary.pieces)
                                 ->equivalent(*geometry)))
                    throw std::invalid_argument("Authority tiles do not form complete exact local shell.");
            }
        });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
thread_local PipelineObserver pipeline_observer;
thread_local PipelinePhase pipeline_phase = PipelinePhase::Candidate;
} // namespace
PipelineObserver set_pipeline_observer(PipelineObserver next) noexcept
{
    const auto previous = pipeline_observer;
    pipeline_observer = next;
    if (next.enter)
        next.enter(next.context, pipeline_phase);
    return previous;
}
void record_pipeline_work(PipelineWork work, std::size_t count) noexcept
{
    if (pipeline_observer.work)
        pipeline_observer.work(pipeline_observer.context, work, count);
}
PipelinePhase switch_pipeline_phase(PipelinePhase next) noexcept
{
    const auto previous = pipeline_phase;
    pipeline_phase = next;
    if (pipeline_observer.enter)
        pipeline_observer.enter(pipeline_observer.context, next);
    return previous;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
PLIC::GeometryStatistics GeometrySnapshotAccess::pipeline_geometry_statistics()
{
    return PLIC::geometry_statistics();
}
const void* GeometrySnapshotAccess::pipeline_geometry_statistics_domain()
{
    return PLIC::geometry_statistics_domain();
}
void GeometrySnapshotAccess::reset_pipeline_geometry_statistics()
{
    PLIC::reset_geometry_statistics();
}
void GeometrySnapshotAccess::visit_pipeline_graph(
    const DistributedAdaptiveHierarchy& hierarchy, const PipelineGraphVisit& visitor)
{
    const auto& state = *hierarchy.data_;
    using Kind = PipelineGraphOwner::Kind;
    const auto object = [&](const void* owner, std::size_t bytes) { visitor({owner, bytes, Kind::Object}); };
    const auto buffer = [&](const auto& values)
    {
        if (values.capacity())
            visitor({values.data(),
                values.capacity() * sizeof(typename std::remove_cvref_t<decltype(values)>::value_type), Kind::Buffer});
    };
    const auto key = [&](const AdaptiveCellKey& value)
    {
        std::visit(
            [&](const auto& ancestry)
            {
                if constexpr (requires { ancestry.path; })
                    buffer(ancestry.path);
                else if constexpr (requires { ancestry.base.path; })
                {
                    buffer(ancestry.base.path);
                    buffer(ancestry.axial.path);
                }
            },
            value.ancestry);
    };
    object(&state, sizeof(state));
    object(state.token.get(), sizeof(char));
    object(state.topology_token.get(), sizeof(char));
    object(state.partition_token.get(), sizeof(char));
    object(state.geometry_cache.get(), sizeof(AdaptiveGeometrySnapshotCache));
    object(state.layout().get(), sizeof(AdaptiveLeafLayout));
    state.pages().visit_payload(object);
    state.body_usage.visit_storage(object);
    if (state.family_layout)
    {
        object(state.family_layout.get(), sizeof(AdaptiveFamilyLayout));
        AdaptiveFamilyStorage::visit_payload(*state.family_layout, object);
    }
    const auto directory = [&](const auto& value)
    {
        DirectoryRegistrationAccess::visit_storage(*value, [&](const void* address, std::size_t bytes, bool numeric)
            { visitor({address, bytes, numeric ? Kind::NumericOutput : Kind::Object}); });
    };
    directory(state.cells);
    directory(state.family_directory);
    if (state.product_election)
        state.product_election->visit_payload(object);
    for (const auto& channels : state.geometry_cache->manifests)
        for (const auto& manifest : channels)
            manifest.visit_payload(object);
    if (state.admission)
        object(state.admission.get(), sizeof(AdaptivePublishedAdmission));
    PLIC::GeometryAllocationLedger ordinary, bounded;
    std::size_t used = 0;
    const auto shared_bytes = [&](const auto& owner)
    {
        if (owner)
            ordinary.retain_external_owner(
                owner, sizeof(AdaptiveBytes) + owner->capacity(), used, std::numeric_limits<std::size_t>::max());
    };
    const auto leaf = [&](const AdaptiveLeaf& value)
    {
        key(value.key);
        buffer(value.domain);
        buffer(value.parent_family);
        buffer(value.surfaces);
        if (value.native_recipe)
        {
            ordinary.retain_external_owner(value.native_recipe,
                sizeof(NativeFragmentDescriptor) + value.native_recipe->view.capacity() +
                    value.native_recipe->parts.capacity() * sizeof(std::shared_ptr<const AdaptiveBytes>),
                used, std::numeric_limits<std::size_t>::max());
            buffer(value.native_recipe->view);
            buffer(value.native_recipe->parts);
            for (const auto& part : value.native_recipe->parts)
            {
                shared_bytes(part);
                if (part)
                    buffer(*part);
            }
        }
        shared_bytes(value.join_recipe);
        if (value.join_recipe)
            buffer(*value.join_recipe);
        if (value.factor_recipe)
        {
            ordinary.retain_external_owner(
                value.factor_recipe, sizeof(AdaptiveFactorRecipe), used, std::numeric_limits<std::size_t>::max());
            shared_bytes(value.factor_recipe->root);
            shared_bytes(value.factor_recipe->base);
            shared_bytes(value.factor_recipe->axis);
        }
        if (value.geometry)
            ordinary.retain(value.geometry, used, std::numeric_limits<std::size_t>::max());
        for (const auto& surface : value.surfaces)
        {
            buffer(surface.key);
            if (surface.boundary.capacity())
                visitor({surface.boundary.data(), surface.boundary.capacity(), Kind::Buffer});
            if (surface.geometry)
                ordinary.retain_planar_face(surface.geometry, used, std::numeric_limits<std::size_t>::max());
            if (surface.native_whole_geometry)
                ordinary.retain_planar_face(
                    surface.native_whole_geometry, used, std::numeric_limits<std::size_t>::max());
        }
    };
    for (const auto& value : hierarchy.owned_leaves())
        leaf(value);
    for (const auto family : hierarchy.owned_families())
    {
        leaf(family.parent);
        buffer(family.key);
        buffer(family.anchor);
        buffer(family.children);
        for (const auto& child : family.children)
            key(child);
    }
    hierarchy.visit_geometry_snapshots(
        [&](const AdaptiveGeometrySnapshot& snapshot)
        {
            snapshot.visit_external_owners(
                [&](const void* identity, std::size_t bytes) { visitor({identity, bytes, Kind::Geometry}); });
            snapshot.retain(ordinary, bounded, used, std::numeric_limits<std::size_t>::max());
        });
    const auto geometries = [&](const PLIC::GeometryAllocationLedger& ledger)
    {
        PLIC::details::GeometryAllocationOwnerAccess::visit(ledger,
            [&](std::uint8_t, const void* identity, std::size_t bytes) { visitor({identity, bytes, Kind::Geometry}); });
    };
    geometries(ordinary);
    geometries(bounded);
    if (state.contact_halo)
    {
        object(state.contact_halo.get(), sizeof(DistributedAdaptiveContactHalo));
        state.contact_halo->for_each_borrowed_contact_payload([&](const ContactPayloadAllocation& allocation)
            { visitor({allocation.identity.address, allocation.payload_bytes, Kind::Object}); });
        state.contact_halo->for_each_borrowed_geometry_ledger(geometries);
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
NativeDirectoryLookup NativePublicationSession::lookup_cells(const DistributedAdaptiveHierarchy& hierarchy,
    std::span<const AdaptiveCellKey* const> keys, const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto& data = *hierarchy.data_;
    std::optional<geometry_detail::NativeConstructorStorageOwner> storage;
    std::vector<AdaptiveBytes> encoded;
    collective(*data.comm, [&]
    {
        if (!hierarchy.geometry_unchanged() || !data.cells)
            throw std::invalid_argument("Native borrowed-key lookup has a stale source directory.");
        storage.emplace(geometry_detail::native_constructor_storage(admission));
        storage->owner->reserve(encoded, keys.size());
        for (const auto* key : keys)
        {
            if (!key) throw std::invalid_argument("Native borrowed-key lookup contains a null key.");
            encoded.push_back(NativeAdaptiveKeyCodec::encode(*key, *storage));
        }
    });
    // Every rank has admitted encoding and wrapper storage before the native
    // directory enters its own collective transport stages.
    return NativeDirectoryAccess::lookup(*data.cells, encoded, admission, true);
}
NativePublicationInputs NativePublicationSession::capture(const DistributedAdaptiveHierarchy& hierarchy)
{
    const auto& data = *hierarchy.data_;
    if (!hierarchy.geometry_unchanged() || !data.contact_halo)
        throw std::invalid_argument("Native publication has moved or incomplete source state.");
    const std::lock_guard geometry_lock(data.geometry_cache->mutex);
    const auto& source = data.geometry_cache->source_store;
    const auto& bound = source.bound_input();
    if (data.map_token_allocation && (!bound.partition || !bound.map_token || !data.map_token ||
            bound.partition.get() != data.native.get() ||
            bound.partition.owner_before(data.native) || data.native.owner_before(bound.partition) ||
            bound.map_token.get() != data.map_token.get() || bound.map_token.owner_before(data.map_token) ||
            data.map_token.owner_before(bound.map_token) || bound.map_token_allocation != data.map_token_allocation ||
            bound.map_token->getRawPtr() != data.owned_map.getRawPtr() ||
            !bound.map_token->shares_resource(data.owned_map)))
        throw std::invalid_argument("Native source detached input differs from the actual bound partition/map.");
    return {hierarchy.leaf_layout(), data.kernel, data.contact_halo->subface_snapshot(), data.native_publication_cache,
        hierarchy.observation(), data.comm, data.cell_begin, data.global_cells, data.options.byte_budget,
        !data.native || data.map_token_allocation ? source.coverage() : nullptr, source.observations(),
        data.map_token ? std::shared_ptr<const void>(data.map_token) : data.partition_token};
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
NativePublicationOperation::NativePublicationOperation(const DistributedAdaptiveHierarchy& hierarchy)
    : hierarchy_(&hierarchy)
{
    if (hierarchy.communicator().is_null())
        throw std::invalid_argument("Native operation lacks communicator.");
    collective(*hierarchy.communicator(), [&]
    {
        if (!hierarchy.geometry_unchanged() || !hierarchy.data_->contact_halo)
            throw std::invalid_argument("Native operation source is stale or incomplete.");
        inputs_ = NativePublicationSession::capture(hierarchy);
        source_data_ = hierarchy.data_;
        cache_ = hierarchy.data_->geometry_cache;
        const std::lock_guard geometry_lock(cache_->mutex);
        const std::lock_guard publication_lock(inputs_.cache->mutex);
        if (cache_->commit_token || inputs_.cache->commit_token || !inputs_.source_coverage ||
            cache_->source_store.coverage() != inputs_.source_coverage)
            throw std::invalid_argument("Native source accounting is absent or publication is in progress.");
        (void) cache_->root_bytes();
        accepted_union_ = cache_->owner_union;
        staged_union_ = accepted_union_;
        accepted_source_ = cache_->source_store;
        accepted_native_ = cache_->native_store;
        staged_source_ = accepted_source_;
        staged_native_ = accepted_native_;
        partition_ = cache_->source_partition;
        // H changes when native outputs outside this current union retire.
        // Only exact canonical direct-H extents inside U may leave that scalar
        // category; external group/subset leases are already excluded by H.
        partition_.exclusive_contact_admission = cache_->contact_residual(*hierarchy.data_->contact_halo);
        cache_revision_ = cache_->revision;
        publication_revision_ = inputs_.cache->revision;
        prefix_.emplace(PLIC::details::SealedSourcePrefix(accepted_union_, persistent_ordered_detail::checked_sum(
            partition_.residual_bytes(), sizeof(AdaptiveGeometrySnapshotCache))));
        const auto& arena = hierarchy.data_->contact_halo->allocation_arena();
        geometry_detail::NativeTpetraMapAdmission::capture_context_prefix(*prefix_, arena);
        admission_.emplace(geometry_detail::NativeConstructionCounterAccess::create<Budget>(
            hierarchy.options().byte_budget, prefix_->bytes(), NativeSourceOwnerStore::make_mirror(arena, true)));
        accounting_.emplace(*admission_, *prefix_);
        accounting_->import_current(accepted_union_);
        coverage_tx_.emplace(cache_->coverage_broker, *admission_);
    });
}
const geometry_detail::NativeConstructionAdmission& NativePublicationOperation::admission() const noexcept
{
    return *admission_;
}
PLIC::details::NativeGeometryAccounting& NativePublicationOperation::geometry_accounting() noexcept
{
    return *accounting_;
}
void NativePublicationOperation::prepare_source_coverage()
{
    if (token_ || !coverage_tx_)
        throw std::logic_error("Source coverage transaction is absent or already locked.");
    coverage_tx_->prepare();
}
void NativePublicationOperation::stage_source(NativeSourceOwnerStore source)
{
    if (token_ || source_staged_)
        throw std::logic_error("Native source root was already staged or locked.");
    auto next = source.replace_in(accepted_source_, staged_union_, *accounting_);
    staged_source_.swap(source);
    staged_union_ = std::move(next);
    source_staged_ = true;
}
void NativePublicationOperation::stage_native_result(std::shared_ptr<const NativePublicationRoot> root)
{
    if (token_ || native_staged_ || !root)
        throw std::logic_error("Native output root was already staged, locked or absent.");
    staged_union_ = accepted_native_.stage(std::move(root), staged_union_, accounting_->ordinary(),
        accounting_->bounded(), accounting_->cursor(), accounting_->limit(), staged_native_);
    native_staged_ = true;
}
void NativePublicationOperation::prepare_commit(NativePublicationSession& session)
{
    if (!native_staged_ || token_ || session.admission_.state() != admission_->state() ||
        session.inputs_.cache != inputs_.cache || !session.result_ || !session.receipt_)
        throw std::logic_error("Native operation/session result is incomplete or foreign.");
    session.recheck(*hierarchy_);
    if (!source_staged_)
    {
        accepted_union_.import_subset(accepted_source_.binding(), accounting_->ordinary(), accounting_->bounded(),
            accounting_->cursor(), accounting_->limit());
        auto source = NativeSourceOwnerStore::assemble(accepted_source_, accepted_source_.components(),
            *admission_, &*accounting_, session.indexes_, accepted_source_.input_administration(), {}, {}, {}, {}, {},
            NativeSourcePublicationValues{inputs_.cache->allocation, session.receipt_, accepted_source_.bound_input(),
                accepted_source_.observations(), coverage_tx_->staged_broker()});
        stage_source(std::move(source));
    }
    const std::lock_guard geometry_lock(cache_->mutex);
    const std::lock_guard publication_lock(inputs_.cache->mutex);
    if (cache_->commit_token || inputs_.cache->commit_token || cache_->revision != cache_revision_ ||
        inputs_.cache->revision != publication_revision_ || session.cache_revision_ != publication_revision_ ||
        cache_revision_ == std::numeric_limits<std::uint64_t>::max() ||
        publication_revision_ == std::numeric_limits<std::uint64_t>::max())
        throw std::invalid_argument("Native publication cache changed during staging.");
    (void) cache_->root_bytes();
    const auto old = cache_->lease ? cache_->lease->bytes() : 0;
    const auto& arena = hierarchy_->data_->contact_halo->allocation_arena();
    const auto peak = std::max(AdaptiveGeometrySnapshotCache::uncovered_bytes(accepted_union_, arena),
        AdaptiveGeometrySnapshotCache::uncovered_bytes(staged_union_, arena));
    if (cache_->lease)
        growth_.emplace(cache_->lease->prepare_mirror(peak > old ? peak - old : 0));
    else
    {
        growth_.emplace(hierarchy_->data_->contact_halo->external_residency_lease());
        growth_->resize(peak);
    }
    if (cache_->lease)
        transfer_.emplace(cache_->lease->prepare_transfer(*growth_, growth_->bytes()));
    cache_->commit_token = inputs_.cache->commit_token = this;
    token_ = true;
}
void NativePublicationOperation::commit(NativePublicationSession& session) noexcept
{
    if (!token_ || committed_)
        std::terminate();
    const std::lock_guard geometry_lock(cache_->mutex);
    const std::lock_guard publication_lock(inputs_.cache->mutex);
    if (cache_->commit_token != this || inputs_.cache->commit_token != this)
        std::terminate();
    coverage_tx_->commit();
    if (transfer_)
        transfer_->commit();
    else
        cache_->lease.swap(growth_);
    cache_->source_store.swap(staged_source_);
    cache_->native_store.swap(staged_native_);
    using std::swap;
    swap(cache_->owner_union, staged_union_);
    cache_->seal_residency(cache_->owner_union.resident_bytes());
    ++cache_->revision;
    session.commit_locked();
    committed_ = true;
    // Keep the short publication token until old operation pins retire and
    // the mirrored overlap lease shrinks outside this no-throw swap phase.
}
NativePublicationOperation::~NativePublicationOperation()
{
    if (transfer_ || coverage_tx_)
    {
        const std::lock_guard geometry_lock(cache_->mutex);
        transfer_.reset();
        coverage_tx_.reset();
    }
    // The prefix outlives every ledger/root pin it authenticates. Output
    // allocator charges retire at their actual allocation lifetimes.
    accounting_.reset();
    accepted_source_ = {};
    staged_source_ = {};
    accepted_native_ = {};
    staged_native_ = {};
    accepted_union_ = {};
    staged_union_ = {};
    inputs_.layout.reset();
    inputs_.kernel.reset();
    inputs_.subfaces.reset();
    source_data_.reset();
    if (token_)
    {
        const std::lock_guard geometry_lock(cache_->mutex);
        const std::lock_guard publication_lock(inputs_.cache->mutex);
        if (committed_ && cache_->lease)
            cache_->lease->resize(AdaptiveGeometrySnapshotCache::uncovered_bytes(
                cache_->owner_union, cache_->lease->arena_identity()));
        if (committed_) cache_->lease_resident_bytes = cache_->lease ? cache_->lease->bytes() : 0;
        cache_->commit_token = inputs_.cache->commit_token = nullptr;
    }
    inputs_ = {};
    growth_.reset();
    cache_.reset();
    if (admission_ && prefix_)
        admission_->release_prefix(prefix_->bytes());
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace SimpleFluid::Meshes::AMR
{
PLIC::details::SealedSourcePrefix DistributedAdaptiveHierarchy::Data::source_prefix_for_refresh() const
{
    return adaptive_detail::NativeSourceOwnerStore::prefix_for_refresh(*this);
}
} // namespace SimpleFluid::Meshes::AMR
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
PLIC::details::SealedSourcePrefix NativeSourceOwnerStore::prefix_for_refresh(
    const DistributedAdaptiveHierarchy::Data& state)
{
    if (!state.geometry_cache || !state.contact_halo || !state.layout())
        throw std::invalid_argument("Native source prefix requires the actual complete private hierarchy state.");
    const auto& cache = *state.geometry_cache;
    // The refresh caller holds this exact cache's mutex. This verifies all
    // current source/native/snapshot root controls and the union generation;
    // neither a caller-supplied union nor an arbitrary scalar is accepted.
    (void) cache.root_bytes();
    cache.require_editable();
    if (cache.lease)
        throw std::invalid_argument("Native source refresh prefix requires a private replacement cache.");
    std::size_t residual = 0;
    if (cache.source_store.empty())
    {
        Budget source{state.options.byte_budget};
        charge_source_base_residency(source, state);
        residual = persistent_ordered_detail::checked_sum(source.used, state.contact_halo->cache_storage_bytes());
    }
    else
        residual = persistent_ordered_detail::checked_sum(cache.source_partition.exclusive_source_admission,
            cache.contact_residual(*state.contact_halo));
    return PLIC::details::SealedSourcePrefix(cache.owner_union,
        persistent_ordered_detail::checked_sum(residual, sizeof(AdaptiveGeometrySnapshotCache)));
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
