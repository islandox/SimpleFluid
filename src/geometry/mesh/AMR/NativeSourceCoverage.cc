#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include <algorithm>
#include <exception>
#include <limits>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Source subset coverage overflows.");
    return a + b;
}
template<class A, class B> bool same(const std::shared_ptr<A>& a, const std::shared_ptr<B>& b) noexcept
{ return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); }
}
struct NativeSourceCoveragePool
{
    std::weak_ptr<ContactExternalResidencyLease> floor;
    std::weak_ptr<const NativeSourceCoverage> membership;
    std::size_t original = 0, active = 0, covered = 0;
    AdaptiveLeafBodyRole role = AdaptiveLeafBodyRole::Leaf;
    bool manifest = false;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation, membership_allocation;
};
void NativeSourceOwnerStore::seal_coverage_floor(const std::shared_ptr<NativeSourceCoverage>& group,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto& floor = group->floor_;
    if (floor->return_channel_)
    {
        auto pool = std::static_pointer_cast<NativeSourceCoveragePool>(floor->return_channel_);
        const auto same_membership = [&]
        {
            if (pool->manifest != group->manifest_membership_) return false;
            if (!pool->manifest)
                return pool->role == group->role_ &&
                    same(pool->membership_allocation, group->body_.membership_control());
            const auto accepted = pool->membership.lock();
            return accepted && pool->membership_allocation &&
                pool->membership_allocation->authentic_owner(pool->membership_allocation) &&
                pool->membership_allocation->matches(accepted) && accepted->same_membership(*group);
        };
        if (!same_membership())
            throw std::invalid_argument("Source coverage floor cannot change its exact issued member controls.");
        group->pool_ = std::move(pool);
        group->require_current();
        return;
    }
    auto pool = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceCoveragePool>(admission);
    pool.owner->allocation = pool.envelope;
    pool.owner->floor = floor;
    pool.owner->role = group->role_;
    pool.owner->manifest = group->manifest_membership_;
    if (pool.owner->manifest)
    {
        pool.owner->membership = group;
        pool.owner->membership_allocation = group->allocation_;
    }
    else
        pool.owner->membership_allocation = group->body_.membership_control();
    pool.owner->original = floor->bytes();
    pool.owner->covered = group->covered_;
    group->pool_ = std::move(pool.owner);
    group->require_current();
    floor->return_channel_ = group->pool_;
}
std::shared_ptr<const NativeSourceCoverage> NativeSourceOwnerStore::body_coverage(const AdaptiveLeafBodyUsage& body,
    AdaptiveLeafBodyRole role, std::shared_ptr<const ContactExternalResidencyLease> floor,
    const geometry_detail::NativeConstructionAdmission& admission, std::uint64_t revision,
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> floor_allocation)
{
    if (!floor || !floor->arena_identity())
        throw std::invalid_argument("Source body group lacks its actual detached admission floor.");
    auto result = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceCoverage>(admission);
    result.owner->body_ = body;
    result.owner->role_ = role;
    result.owner->floor_ = std::const_pointer_cast<ContactExternalResidencyLease>(std::move(floor));
    if (floor_allocation)
        geometry_detail::NativeConstructorOwnerSink::require_envelope(floor_allocation, result.owner->floor_);
    result.owner->floor_allocation_ = std::move(floor_allocation);
    result.owner->revision_ = revision;
    result.owner->allocation_ = result.envelope;
    result.owner->covered_ = role == AdaptiveLeafBodyRole::Leaf ? body.leaf_payload_bytes() :
        body.payload_bytes() - body.leaf_payload_bytes();
    seal_coverage_floor(result.owner, admission);
    return std::move(result.owner);
}
std::shared_ptr<const NativeSourceCoverage> NativeSourceOwnerStore::manifest_coverage(
    PLIC::details::OwnerManifestPair members, std::shared_ptr<const ContactExternalResidencyLease> floor,
    const geometry_detail::NativeConstructionAdmission& admission, std::uint64_t revision,
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> floor_allocation)
{
    if (!floor || !floor->arena_identity() || members.owners[1].owner_count())
        throw std::invalid_argument("Source manifest coverage requires its detached floor and Ordinary physical owners.");
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceCoverage>(admission);
    tracked.owner->allocation_ = tracked.envelope;
    tracked.owner->floor_ = std::const_pointer_cast<ContactExternalResidencyLease>(std::move(floor));
    if (floor_allocation)
        geometry_detail::NativeConstructorOwnerSink::require_envelope(floor_allocation, tracked.owner->floor_);
    tracked.owner->floor_allocation_ = std::move(floor_allocation);
    tracked.owner->manifest_membership_ = true;
    tracked.owner->members_ = std::move(members);
    tracked.owner->covered_ = tracked.owner->members_.payload_bytes();
    tracked.owner->revision_ = revision;
    seal_coverage_floor(tracked.owner, admission);
    std::shared_ptr<const NativeSourceCoverage> group = std::move(tracked.owner);
    return group;
}
std::shared_ptr<const NativeSourceCoverage> NativeSourceOwnerStore::composite_coverage(
    const std::shared_ptr<const NativeSourceCoverage>& previous,
    std::span<const std::shared_ptr<const NativeSourceCoverage>> parts,
    const geometry_detail::NativeConstructionAdmission& admission, std::uint64_t revision)
{
    if (parts.empty() || parts.size() > 8 || !parts.front())
        throw std::invalid_argument("Source coverage component inventory is invalid.");
    auto group = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceCoverage>(admission);
    group.owner->allocation_ = group.envelope;
    group.owner->part_count_ = parts.size();
    group.owner->manifest_membership_ = true;
    group.owner->revision_ = revision;
    const auto& origin = parts.front()->floor()->arena_identity();
    for (std::size_t slot = 0; slot < parts.size(); ++slot)
    {
        if (!parts[slot]) continue;
        if (parts[slot]->part_count_ || !same(parts[slot]->floor()->arena_identity(), origin))
            throw std::invalid_argument("Composite source coverage mixes foreign arenas or nested inventories.");
        parts[slot]->require_current();
        group.owner->parts_[slot] = parts[slot];
    }
    std::size_t cursor = 0, overlap = previous ? previous->members_.overlap_bytes() : 0;
    PLIC::details::GeometryOwnerGraphEditor ordinary(previous ? previous->members_.owners[0] :
        PLIC::details::geometry_demand::GeometryOwnerManifest{}, 0, cursor, admission.counters().limit,
        nullptr, nullptr, admission.state());
    PLIC::details::GeometryOwnerGraphEditor bounded(previous ? previous->members_.owners[1] :
        PLIC::details::geometry_demand::GeometryOwnerManifest{}, 1, cursor, admission.counters().limit,
        nullptr, nullptr, admission.state());
    ordinary.pair_with(bounded, overlap);
    for (std::size_t slot = 0; slot < std::max(parts.size(), previous ? previous->part_count_ : 0); ++slot)
    {
        const auto* old = previous && slot < previous->part_count_ ? previous->parts_[slot].get() : nullptr;
        const auto* next = slot < parts.size() ? parts[slot].get() : nullptr;
        struct Context { PLIC::details::GeometryOwnerGraphEditor& graph; } context{ordinary};
        const PLIC::details::NativePaidExtentGroup::MemberSink sink{&context,
            [](void* raw, auto kind, const auto& owner, auto bytes, bool retain)
            {
                auto& value = *static_cast<Context*>(raw);
                value.graph.owner({{0, kind, owner.get()}, owner, bytes, 1}, retain);
            }};
        if (next) next->visit_delta(old, sink);
        else if (old)
        {
            if (old->manifest_membership_)
            {
                struct Retire { PLIC::details::NativePaidExtentGroup::MemberSink sink; } retire{sink};
                old->members_.owners[0].visit_added({}, {&retire, [](void* raw, const auto& row)
                { auto& value = *static_cast<Retire*>(raw); value.sink.visit(value.sink.context, row.key.kind, row.owner, row.bytes, false); }});
            }
            else AdaptiveLeafBodyUsage{}.visit_role_delta(old->body_, old->role_, sink);
        }
    }
    group.owner->members_ = seal_default(ordinary, bounded, overlap);
    group.owner->covered_ = group.owner->members_.payload_bytes();
    group.owner->require_current();
    return std::move(group.owner);
}
PLIC::details::NativePaidExtentGroup NativeSourceOwnerStore::coverage_group(
    const std::shared_ptr<const NativeSourceCoverage>& group)
{
    if (!group || !group->covered_bytes()) return {};
    group->require_current();
    const auto& origin = group->part_count_ ? group->parts_[0]->floor()->arena_identity() : group->floor_->arena_identity();
    return PLIC::details::NativePaidExtentGroup(group, group->part_count_ ? std::static_pointer_cast<const void>(group) :
        std::static_pointer_cast<const void>(group->pool_), origin,
        group->covered_, group->revision_,
        [](const NativeSourceCoverage* members, auto kind, const auto& owner, auto bytes)
        { return members->contains(kind, owner, bytes); },
        [](const NativeSourceCoverage* a, const NativeSourceCoverage* b) noexcept
        { return a->same_membership(*b); },
        [](const NativeSourceCoverage* previous, const NativeSourceCoverage* next, auto sink)
        {
            if (next) next->visit_delta(previous, sink);
            else if (previous)
            {
                const auto* old = previous;
                if (old->manifest_membership_)
                {
                    struct Retire { PLIC::details::NativePaidExtentGroup::MemberSink sink; } retire{sink};
                    old->members_.owners[0].visit_added({}, {&retire, [](void* raw, const auto& row)
                    { auto& value = *static_cast<Retire*>(raw); value.sink.visit(value.sink.context, row.key.kind, row.owner, row.bytes, false); }});
                }
                else AdaptiveLeafBodyUsage{}.visit_role_delta(old->body_, old->role_, sink);
            }
        });
}
void NativeSourceOwnerStore::project_coverage(const std::shared_ptr<const NativeSourceCoverage>& group,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!group) return;
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(group, geometry_detail::NativeConstructorOwnerKind::NativeMesh, group->allocation_)) return;
    context.sink.fixed(group->allocation_);
    const auto metadata = [&](const PLIC::details::OwnerManifestPair& pair,
                              const PLIC::details::NativeOwnerReceipts* paid = nullptr)
    {
        struct Context { PLIC::details::GeometryOwnerGraphEditor& graph; bool retain, prepaid; } c{graph, retain, bool(accounting)};
        const PLIC::details::geometry_demand::MetadataSink meta{&c,
            [](void* raw, const auto& owner, std::size_t, std::size_t admitted)
            { auto& value = *static_cast<Context*>(raw); return value.graph.external(owner, admitted, value.retain, value.prepaid); }, nullptr,
            [](void* raw, const auto& charge)
            { static_cast<Context*>(raw)->graph.record_paid_extent(charge); }};
        if (paid)
            paid->visit_administration(meta.with_financial(graph, retain));
        for (const auto& root : pair.owners)
            root.visit_dependencies(meta.with_financial(graph, retain), {&c, [](void* raw, const auto& row)
                { auto& value = *static_cast<Context*>(raw); value.graph.owner(row, value.retain, value.prepaid); }});
    };
    if (group->manifest_membership_) metadata(group->members_);
    else
    {
        metadata(group->body_.owners(), &group->body_.paid_receipts());
        group->body_.project_administration(graph, retain);
        struct Context { PLIC::details::GeometryOwnerGraphEditor& graph; bool retain; } c{graph, retain};
        const PLIC::details::geometry_demand::MetadataSink sink{&c,
            [](void* raw, const auto& owner, std::size_t, std::size_t admitted)
            { auto& value = *static_cast<Context*>(raw); return value.graph.external(owner, admitted, value.retain); }, nullptr,
            [](void* raw, const auto& charge) { static_cast<Context*>(raw)->graph.record_paid_extent(charge); }};
        group->body_.administration().visit(sink.with_financial(graph, retain));
    }
    if (group->part_count_)
    {
        for (std::size_t slot = 0; slot < group->part_count_; ++slot)
            project_coverage(group->parts_[slot], graph, retain, accounting);
        return;
    }
    if (group->floor_allocation_) context.sink.fixed(group->floor_allocation_);
    const bool floor_transition = group->floor_allocation_
        ? context.sink.semantic(group->floor_, geometry_detail::NativeConstructorOwnerKind::Storage,
              group->floor_allocation_)
        : graph.external(group->floor_, sizeof(ContactExternalResidencyLease) + 4 * sizeof(void*), retain, bool(accounting));
    if (floor_transition)
    {
        graph.external(group->floor_->arena_, sizeof(Budget) + 4 * sizeof(void*), retain, bool(accounting));
        graph.external(group->floor_->total_, sizeof(std::size_t) + 4 * sizeof(void*), retain, bool(accounting));
        project_coverage_pool(group->pool_, context.sink);
    }
    // Direct group.pool field is distinct from floor.return_channel.
    project_coverage_pool(group->pool_, context.sink);
}
void NativeSourceOwnerStore::project_coverage_pool(const std::shared_ptr<NativeSourceCoveragePool>& pool,
    const geometry_detail::NativeConstructorOwnerSink& sink)
{
    if (sink.semantic(pool, geometry_detail::NativeConstructorOwnerKind::Storage, pool->allocation))
    {
        sink.fixed(pool->allocation);
        if (pool->membership_allocation) sink.fixed(pool->membership_allocation);
    }
}
void NativeSourceOwnerStore::project_coverage_descriptor(const std::shared_ptr<const NativeSourceCoverage>& group,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!group || !group->covered_) return;
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    project_coverage(group, graph, retain, accounting); // descriptor membership root
    if (group->part_count_)
        project_coverage(group, graph, retain, accounting); // descriptor detached coverage
    else project_coverage_pool(group->pool_, context.sink);
    const auto& floor = group->part_count_ ? group->parts_[0]->floor_ : group->floor_;
    graph.external(floor->arena_, sizeof(Budget) + 4 * sizeof(void*), retain, bool(accounting));
}
void NativeSourceOwnerStore::project_subset(const std::shared_ptr<const NativeSourceSubsetCoverage>& subset,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!subset) return;
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(subset, geometry_detail::NativeConstructorOwnerKind::NativeMesh, subset->allocation_)) return;
    context.sink.fixed(subset->allocation_);
    context.sink.storage(subset->metadata_);
    for (const auto& owner : subset->owners_)
        if (context.sink.semantic(owner, geometry_detail::NativeConstructorOwnerKind::Storage, owner->allocation_))
        {
            context.sink.fixed(owner->allocation_);
            // Weak issuer observation keeps only its measured allocation/control
            // tail, never its member tree, source floor or controller.
            context.sink.fixed(owner->issuer_allocation_);
            if (owner->owner_fixed_) graph.weak_observation(owner->owner_fixed_, retain);
            graph.external(owner->lease_.arena_, sizeof(Budget) + 4 * sizeof(void*), retain, bool(accounting));
            graph.external(owner->lease_.total_, sizeof(std::size_t) + 4 * sizeof(void*), retain, bool(accounting));
        }
}
bool NativeSourceOwnerStore::validate_owner_coverage(const std::shared_ptr<const void>& erased,
    const std::shared_ptr<const void>& physical, const std::shared_ptr<const void>& arena,
    std::uint8_t kind, std::size_t bytes, std::size_t snapshot, std::uint64_t revision)
{
    const auto owner = std::static_pointer_cast<const NativeSourceOwnerCoverage>(erased);
    return owner && revision == 0 && snapshot >= bytes && owner->matches(kind, physical, bytes) &&
        owner->allocation_ && owner->allocation_->authentic_owner(owner->allocation_) &&
        owner->allocation_->matches(owner) && same(owner->lease_.arena_identity(), arena) &&
        ((owner->published_ && !owner->lease_.transfer_pending_ && owner->lease_.bytes() == bytes) ||
            (!owner->published_ && owner->prepared_ && owner->lease_.transfer_pending_ && owner->lease_.bytes() == 0));
}
void NativeSourceOwnerStore::project_owner_coverage(const std::shared_ptr<const void>& erased,
    const geometry_detail::NativeConstructorOwnerSink& sink)
{
    const auto owner = std::static_pointer_cast<const NativeSourceOwnerCoverage>(erased);
    if (!owner || !owner->allocation_ || !owner->allocation_->matches(owner) ||
        (!owner->published_ && !owner->prepared_) ||
        (owner->published_ && owner->lease_.bytes() != owner->bytes_))
        throw std::invalid_argument("Source financial pin has foreign controls or changed live coverage.");
    if (!sink.semantic(owner, geometry_detail::NativeConstructorOwnerKind::Storage, owner->allocation_)) return;
    sink.fixed(owner->allocation_);
    sink.fixed(owner->issuer_allocation_);
    if (owner->owner_fixed_)
    {
        if (!sink.financial_graph)
            throw std::invalid_argument("Source financial pin lacks physical weak-control projection.");
        sink.financial_graph->weak_observation(owner->owner_fixed_, sink.financial_retain);
    }
    sink.edge(owner->lease_.arena_, geometry_detail::NativeConstructorOwnerKind::Storage,
        sizeof(Budget) + 4 * sizeof(void*));
    sink.edge(owner->lease_.total_, geometry_detail::NativeConstructorOwnerKind::Storage,
        sizeof(std::size_t) + 4 * sizeof(void*));
}
void NativeSourceOwnerStore::inspect_owner_coverage(const std::shared_ptr<const void>& erased,
    const PLIC::details::NativeFinancialInspectionSink& sink)
{
    // This diagnostic reads the concrete fields independently. It never calls
    // the production projector or follows the weak issuer into source storage.
    const auto owner = std::static_pointer_cast<const NativeSourceOwnerCoverage>(erased);
    if (!owner || !owner->allocation_ || !owner->allocation_->matches(owner) || !sink.owner || !sink.fixed)
        throw std::invalid_argument("Incomplete source financial pin inspection.");
    const auto kind = PLIC::details::GeometryOwnerGraphEditor::external_kind();
    sink.owner(sink.context, kind, owner, 0);
    sink.fixed(sink.context, owner->allocation_);
    sink.fixed(sink.context, owner->issuer_allocation_);
    sink.weak_observation(owner->owner_fixed_);
    sink.owner(sink.context, kind, owner->lease_.arena_, sizeof(Budget) + 4 * sizeof(void*));
    sink.owner(sink.context, kind, owner->lease_.total_, sizeof(std::size_t) + 4 * sizeof(void*));
}
void NativeSourceOwnerStore::record_subset(const std::shared_ptr<const NativeSourceSubsetCoverage>& subset,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
{
    if (!subset) return;
    for (const auto& owner : subset->owners())
    {
        const auto actual = owner->owner_.lock();
        if (!actual || (!owner->published_ && !owner->prepared_) ||
            (owner->published_ && owner->lease_.bytes() != owner->bytes_) ||
            !owner->matches(owner->kind_, actual, owner->bytes_))
            throw std::invalid_argument("Detached source subset changes its canonical owner/control.");
        graph.record_subset_charge(owner->kind_, actual, owner->bytes_, owner->lease_.arena_identity(),
            owner, owner->bytes_, 0, project_owner_coverage, inspect_owner_coverage, validate_owner_coverage, retain,
            owner->owner_fixed_, observe_coverage(owner));
    }
}
PLIC::details::NativeWeakOwnerObservation NativeSourceOwnerStore::observe_coverage(
    const std::shared_ptr<const NativeSourceOwnerCoverage>& owner)
{
    return PLIC::details::NativeWeakOwnerObservation::native(owner, owner->allocation_);
}
bool NativeSourceOwnerCoverage::matches(std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t bytes) const noexcept
{
    return kind_ == kind && bytes_ == bytes && owner && owner.get() == address_ &&
        !owner_.owner_before(owner) && !owner.owner_before(owner_) &&
        (!owner_fixed_ || owner_fixed_.matches_live(owner));
}
NativeSourceOwnerCoverage::~NativeSourceOwnerCoverage()
{
    if (!published_ || !lease_.bytes()) return;
    if (const auto pool = issuer_.lock())
        if (const auto floor = pool->floor.lock())
        {
            if (lease_.bytes() != bytes_ || pool->active < bytes_ || pool->active > pool->original ||
                floor->bytes() != pool->original - pool->active)
                std::terminate();
            // This pair was checked at issuance; its total is conserved through
            // every split. Returning it cannot allocate or exceed original.
            floor->return_subset_coverage(lease_);
            pool->active -= bytes_;
        }
}
void NativeSourceCoverage::require_current() const
{
    if (part_count_)
    {
        for (std::size_t slot = 0; slot < part_count_; ++slot)
            if (parts_[slot]) parts_[slot]->require_current();
        if (covered_ != members_.payload_bytes())
            throw std::invalid_argument("Composite source coverage changed its canonical member roots.");
        return;
    }
    if (!floor_ || !pool_ || !same(pool_->floor.lock(), floor_) ||
        floor_->bytes() > pool_->original || pool_->active > pool_->original - floor_->bytes() ||
        floor_->bytes() + pool_->active != pool_->original || covered_ > pool_->original ||
        covered_ != pool_->covered || pool_->active > covered_)
        throw std::invalid_argument("Source coverage floor/control changed after group issuance.");
    if (floor_allocation_)
        geometry_detail::NativeConstructorOwnerSink::require_envelope(floor_allocation_, floor_);
}
bool NativeSourceCoverage::contains(std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    require_current();
    if (manifest_membership_)
        return members_.owners[0].admits(0, kind, owner, bytes) || members_.owners[1].admits(1, kind, owner, bytes);
    const auto role = body_.physical_role(kind, owner, bytes);
    return role && *role == role_;
}
bool NativeSourceCoverage::same_membership(const NativeSourceCoverage& other) const noexcept
{
    if (manifest_membership_ != other.manifest_membership_) return false;
    if (manifest_membership_) return members_.owners[0].same_root(other.members_.owners[0]) &&
        members_.owners[1].same_root(other.members_.owners[1]);
    return role_ == other.role_ && body_.same_membership(other.body_);
}
void NativeSourceCoverage::visit_delta(const NativeSourceCoverage* previous,
    PLIC::details::NativePaidExtentGroup::MemberSink sink) const
{
    require_current();
    if (manifest_membership_ && (!previous || previous->manifest_membership_))
    {
        for (unsigned channel = 0; channel < 2; ++channel)
        {
            const auto& old = previous ? previous->members_.owners[channel] :
                PLIC::details::geometry_demand::GeometryOwnerManifest{};
            struct Context { PLIC::details::NativePaidExtentGroup::MemberSink sink; bool retain; } add{sink, true}, retire{sink, false};
            const auto visit = [](void* raw, const PLIC::details::geometry_demand::OwnerRecord& row)
            { auto& value = *static_cast<Context*>(raw); value.sink.visit(value.sink.context, row.key.kind, row.owner, row.bytes, value.retain); };
            members_.owners[channel].visit_added(old, {&add, visit});
            old.visit_added(members_.owners[channel], {&retire, visit});
        }
        return;
    }
    if (previous && previous->role_ == role_)
        body_.visit_role_delta(previous->body_, role_, sink);
    else
    {
        if (previous)
        {
            struct Reverse { PLIC::details::NativePaidExtentGroup::MemberSink sink; } reverse{sink};
            AdaptiveLeafBodyUsage{}.visit_role_delta(previous->body_, previous->role_,
                {&reverse, [](void* raw, auto kind, const auto& owner, auto bytes, bool retain)
                    { auto& value = *static_cast<Reverse*>(raw); value.sink.visit(value.sink.context, kind, owner, bytes, retain); }});
        }
        body_.visit_role_delta({}, role_, sink);
    }
}
struct NativeSourceCoverageTransaction::Batch
{
    std::shared_ptr<const NativeSourceCoverage> source;
    geometry_detail::NativeConstructorVector<std::shared_ptr<NativeSourceOwnerCoverage>> created;
    geometry_detail::NativeConstructorVector<ContactExternalResidencyLease::TransferPart> parts;
    std::optional<ContactExternalResidencyLease::PreparedBatch> transfer;
    std::size_t bytes = 0;
    Batch(std::shared_ptr<const NativeSourceCoverage> group, const geometry_detail::NativeConstructionAdmission& admission)
        : source(std::move(group)), created(admission.allocator<std::shared_ptr<NativeSourceOwnerCoverage>>()),
          parts(admission.allocator<ContactExternalResidencyLease::TransferPart>()) {}
};
NativeSourceCoverageTransaction::NativeSourceCoverageTransaction(NativeSourceCoverageBroker& broker,
    geometry_detail::NativeConstructionAdmission admission)
    : broker_(broker), admission_(std::move(admission)), staged_(broker.root_),
      batches_(admission_.allocator<std::shared_ptr<Batch>>())
{
    if (broker_.pending_) throw std::invalid_argument("Source coverage cache already has a staged transaction.");
    broker_.pending_ = true;
}
NativeSourceCoverageTransaction::~NativeSourceCoverageTransaction()
{
    // Cancel prepared floor reservations before candidate subset leases die.
    for (const auto& batch : batches_) batch->transfer.reset();
    broker_.pending_ = false;
}
void NativeSourceCoverageBroker::project(PLIC::details::GeometryOwnerGraphEditor& graph, bool retain,
    PLIC::details::NativeGeometryAccounting* accounting) const
{
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    const auto project_page = [&](auto&& self, const Pages::Ptr& node) -> void
    {
        if (!node || !Pages::visit_native_node(node, context.sink)) return;
        for (const auto& row : node->rows)
            graph.weak_observation(row.coverage_fixed, retain);
        for (const auto& child : node->children) self(self, child);
    };
    project_page(project_page, root_);
}
NativeSourceCoverageBroker NativeSourceCoverageTransaction::staged_broker() const
{
    if (!prepared_ || committed_)
        throw std::logic_error("Source coverage broker is outside its prepared staging window.");
    NativeSourceCoverageBroker result;
    result.root_ = staged_;
    return result;
}
std::shared_ptr<const NativeSourceSubsetCoverage> NativeSourceCoverageTransaction::collect(
    const std::shared_ptr<const NativeSourceCoverage>& group,
    std::span<const PLIC::details::geometry_demand::OwnerRecord> owners,
    const PLIC::details::NativeOwnerReceipts* physical_receipts)
{
    if (prepared_ || failed_) throw std::logic_error("Source subset transaction cannot resume after preparation/rejection.");
    struct RejectFailedCollection
    {
        bool& failed;
        const int exceptions = std::uncaught_exceptions();
        ~RejectFailedCollection()
        {
            if (std::uncaught_exceptions() > exceptions) failed = true;
        }
    } reject{failed_};
    if (!group) throw std::logic_error("Cannot extend absent source subset coverage.");
    group->require_current();
    for (std::size_t index = 0; index < owners.size(); ++index)
    {
        const auto& owner = owners[index];
        if (!owner.owner || owner.key.address != owner.owner.get() || !owner.references || owner.key.channel != 0)
            throw std::invalid_argument("Source subset requires canonical Ordinary owner records.");
        if (index)
        {
            const auto& previous = owners[index - 1];
            if (previous.key.kind > owner.key.kind || (previous.key.kind == owner.key.kind &&
                !std::less<const void*>{}(previous.key.address, owner.key.address)))
                throw std::invalid_argument("Source subset owner records must be sorted and physically unique.");
        }
    }
    auto result = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceSubsetCoverage>(admission_);
    result.owner->allocation_ = result.envelope;
    result.owner->metadata_ = geometry_detail::native_constructor_storage(admission_);
    result.owner->metadata_.owner->reserve(result.owner->owners_, owners.size());
    using Broker = NativeSourceCoverageBroker;
    auto arena = std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission_.state()->counter_owner));
    Broker::Pages editor(arena, nullptr, {}, admission_.state());
    for (const auto& owner : owners)
    {
        if (!owner.bytes || !group->contains(owner.key.kind, owner.owner, owner.bytes)) continue;
        std::shared_ptr<const NativeSourceCoverage> owner_group = group;
        if (group->part_count_)
        {
            owner_group.reset();
            for (std::size_t slot = 0; slot < group->part_count_; ++slot)
                if (group->parts_[slot] && group->parts_[slot]->contains(owner.key.kind, owner.owner, owner.bytes))
                { owner_group = group->parts_[slot]; break; }
            if (!owner_group) throw std::logic_error("Composite source member lacks its actual coverage floor.");
        }
        const Broker::Row key{owner.key.kind, owner.owner.get(), {}, {}};
        const auto* old = Broker::Pages::lower_bound(staged_, [&](const auto& row) { return Broker::Less{}(row, key); });
        if (old && Broker::Less{}(key, *old)) old = nullptr;
        std::shared_ptr<const NativeSourceOwnerCoverage> coverage = old ? old->coverage.lock() : nullptr;
        if (coverage && !coverage->matches(owner.key.kind, owner.owner, owner.bytes))
            throw std::invalid_argument("Source subset cache changes a live canonical physical owner.");
        if (!coverage)
        {
            PLIC::details::NativeWeakOwnerObservation physical;
            if (physical_receipts)
            {
                struct Physical
                {
                    const PLIC::details::geometry_demand::OwnerRecord& row;
                    const std::shared_ptr<Budget>& arena;
                    PLIC::details::NativeWeakOwnerObservation& observation;
                    bool directly_paid = false;
                } value{owner, owner_group->floor_->arena_identity(), physical};
                physical_receipts->visit_owner(owner.key.kind, owner.owner,
                    {&value, [](void* raw, const auto& charge, std::size_t)
                    {
                        auto& actual = *static_cast<Physical*>(raw);
                        if (charge.bytes() != actual.row.bytes) return;
                        if (charge.direct_h() && same(charge.origin(), actual.arena))
                            actual.directly_paid = true;
                        if (!actual.observation)
                            actual.observation = charge.detached_source_observation();
                    }});
                if (value.directly_paid) continue;
                if (!physical)
                    throw std::invalid_argument("Native source subset owner has no actual weak-control witness.");
            }
            auto found_batch = std::find_if(batches_.begin(), batches_.end(),
                [&](const auto& value) { return value->source.get() == owner_group.get(); });
            std::shared_ptr<Batch> batch;
            if (found_batch == batches_.end())
            {
                batch = admission_.shared<Batch>(owner_group, admission_);
                batches_.push_back(batch);
            }
            else batch = *found_batch;
            auto target = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSourceOwnerCoverage>(admission_);
            target.owner->allocation_ = target.envelope;
            target.owner->lease_ = owner_group->floor_->prepare_mirror(0);
            target.owner->issuer_ = owner_group->pool_;
            target.owner->issuer_allocation_ = owner_group->pool_->allocation;
            target.owner->owner_ = owner.owner;
            target.owner->owner_fixed_ = std::move(physical);
            target.owner->address_ = owner.owner.get();
            target.owner->kind_ = owner.key.kind;
            target.owner->bytes_ = owner.bytes;
            batch->created.push_back(target.owner);
            coverage = std::move(target.owner);
            const Broker::Row row{key.kind, key.address, coverage,
                NativeSourceOwnerStore::observe_coverage(coverage)};
            staged_ = editor.edit(staged_, key, &row, old != nullptr);
        }
        result.owner->bytes_ = sum(result.owner->bytes_, coverage->bytes());
        result.owner->owners_.push_back(std::move(coverage));
    }
    return std::move(result.owner);
}
void NativeSourceCoverageTransaction::prepare()
{
    if (prepared_ || failed_) throw std::logic_error("Source subset transaction cannot resume after preparation/rejection.");
    try
    {
        for (const auto& batch : batches_)
        {
            batch->source->require_current();
            std::size_t outgoing = 0;
            batch->parts.reserve(batch->created.size());
            for (const auto& target : batch->created)
            {
                outgoing = sum(outgoing, target->bytes_);
                batch->parts.push_back({&target->lease_, target->bytes_});
            }
            if (batch->source->pool_->active > batch->source->covered_ ||
                outgoing > batch->source->covered_ - batch->source->pool_->active)
                throw std::invalid_argument("Source subset coverage exceeds its independently held source footprint.");
            std::sort(batch->parts.begin(), batch->parts.end(), [](const auto& a, const auto& b)
                { return std::less<ContactExternalResidencyLease*>{}(a.target, b.target); });
            batch->transfer.emplace(batch->source->floor_->prepare_batch(batch->parts));
            batch->bytes = outgoing;
            for (const auto& target : batch->created) target->prepared_ = true;
        }
        prepared_ = true;
    }
    catch (...)
    {
        for (const auto& batch : batches_)
        {
            batch->transfer.reset();
            for (const auto& target : batch->created) target->prepared_ = false;
        }
        failed_ = true;
        throw;
    }
}
void NativeSourceCoverageTransaction::commit() noexcept
{
    if (!prepared_ || committed_) std::terminate();
    for (const auto& batch : batches_)
    {
        batch->transfer->commit();
        batch->source->pool_->active += batch->bytes;
        for (const auto& target : batch->created)
        {
            target->published_ = true;
            target->prepared_ = false;
        }
    }
    using std::swap;
    swap(broker_.root_, staged_);
    committed_ = true;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
