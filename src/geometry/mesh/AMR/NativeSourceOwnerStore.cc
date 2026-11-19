#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeContactFunding.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativeAdaptivePayloadAccess.hh"
#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "parallel/details/NativePartitionAccess.hh"
#include <algorithm>
#include <limits>
#include <utility>
#include <mutex>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Native source accounting overflows.");
    return a + b;
}
}
std::size_t SourceAccountingPartition::residual_bytes() const
{
    return sum(exclusive_source_admission, exclusive_contact_admission);
}
SourceAccountingPartition SourceAccountingPartition::split(std::size_t source, std::size_t projected_source,
    std::size_t contacts, std::size_t projected_contacts)
{
    if (projected_source > source || projected_contacts > contacts)
        throw std::logic_error("Native source projection exceeds its exact replaced charge component.");
    return {source - projected_source, contacts - projected_contacts, projected_source, projected_contacts};
}
void NativeSourceOwnerStore::require_current() const
{
    if (!root_) return;
    if (!root_->allocation || !root_->allocation->authentic_owner(root_->allocation) ||
        !root_->allocation->matches(root_) || binding_.header().get() != root_.get() ||
        binding_.header().owner_before(root_) || root_.owner_before(binding_.header()))
        throw std::invalid_argument("SourceStrong root/control differs from its actual allocation seal.");
    if (root_->coverage) root_->coverage->require_current();
}
const PLIC::details::OwnerManifestPair& NativeSourceOwnerStore::owners() const noexcept
{
    static const PLIC::details::OwnerManifestPair empty;
    return root_ ? root_->owners : empty;
}
namespace
{
// All detached halo/native scalar updates share this allocation-free lock.
// Mutable hierarchy operations remain serialized by their existing cache token.
std::mutex native_halo_mirror_mutex;
template<bool Publication> bool precharge_halo(void* raw, std::size_t bytes) noexcept
{
    const std::lock_guard lock(native_halo_mirror_mutex);
    auto& budget = *static_cast<Budget*>(raw);
    if (budget.used > budget.limit || bytes > budget.limit - budget.used ||
        bytes > std::numeric_limits<std::size_t>::max() - budget.cumulative)
    {
        record_pipeline_budget_rejection({budget.used, bytes, budget.limit, budget.cumulative});
        return false;
    }
    budget.used += bytes;
    if constexpr (Publication) budget.publication_mirror_live += bytes;
    budget.peak = std::max(budget.peak, budget.used);
    budget.cumulative += bytes;
    return true;
}
template<bool Publication> void retire_halo(void* raw, std::size_t bytes) noexcept
{
    const std::lock_guard lock(native_halo_mirror_mutex);
    auto& budget = *static_cast<Budget*>(raw);
    if (bytes > budget.used) std::terminate();
    if constexpr (Publication)
    {
        if (bytes > budget.publication_mirror_live) std::terminate();
        budget.publication_mirror_live -= bytes;
    }
    budget.used -= bytes;
}
}
geometry_detail::NativeAllocationMirror NativeSourceOwnerStore::mirror(const std::shared_ptr<Budget>& arena)
{
    return make_mirror(arena, false);
}
std::size_t NativeSourceOwnerStore::publication_mirror_bytes(const std::shared_ptr<Budget>& arena)
{
    const std::lock_guard lock(native_halo_mirror_mutex);
    if (!arena || arena->publication_mirror_live > arena->used)
        throw std::invalid_argument("Native publication mirror has an invalid live request total.");
    return arena->publication_mirror_live;
}
geometry_detail::NativeAllocationMirror NativeSourceOwnerStore::make_mirror(
    const std::shared_ptr<Budget>& arena, bool publication)
{
    if (!arena)
        throw std::invalid_argument("Native halo mirror requires its actual owned arena.");
    geometry_detail::NativeAllocationMirror result;
    result.owner = arena;
    result.owner_proof = geometry_detail::NativeMirrorOwnerProof(result.owner, sizeof(Budget) + 4 * sizeof(void*),
        geometry_detail::NativeMirrorExtentKind::ProfiledUpper, publication);
    result.try_precharge = publication ? precharge_halo<true> : precharge_halo<false>;
    result.retire_callback = publication ? retire_halo<true> : retire_halo<false>;
    return result;
}
const std::shared_ptr<const AdaptiveSubfaceSnapshot>& NativeSourceOwnerStore::subfaces() const noexcept
{
    static const std::shared_ptr<const AdaptiveSubfaceSnapshot> empty;
    return root_ ? root_->subfaces : empty;
}
const std::shared_ptr<const NativeSourceCoverage>& NativeSourceOwnerStore::coverage() const noexcept
{
    static const std::shared_ptr<const NativeSourceCoverage> empty;
    return root_ ? root_->coverage : empty;
}
const std::array<PLIC::details::GeometryNativeAdministration, 2>&
NativeSourceOwnerStore::input_administration() const noexcept
{
    static const std::array<PLIC::details::GeometryNativeAdministration, 2> empty;
    return root_ ? root_->input_administration : empty;
}
const std::shared_ptr<const NativePublicationIndexes>& NativeSourceOwnerStore::indexes() const noexcept
{
    static const std::shared_ptr<const NativePublicationIndexes> empty;
    return root_ ? root_->indexes : empty;
}
const NativeSourceBoundInput& NativeSourceOwnerStore::bound_input() const noexcept
{
    static const NativeSourceBoundInput empty;
    return root_ ? root_->publication_values.input : empty;
}
const NativePublicationSourceObservations& NativeSourceOwnerStore::observations() const noexcept
{
    static const NativePublicationSourceObservations empty;
    return root_ ? root_->publication_values.source : empty;
}
const std::array<PLIC::details::OwnerManifestPair, 4>& NativeSourceOwnerStore::components() const noexcept
{
    static const std::array<PLIC::details::OwnerManifestPair, 4> empty;
    return root_ ? root_->components : empty;
}
const std::array<PLIC::details::NativeOwnerReceipts, 4>& NativeSourceOwnerStore::component_paid() const noexcept
{
    static const std::array<PLIC::details::NativeOwnerReceipts, 4> empty;
    return root_ ? root_->component_paid : empty;
}
std::size_t NativeSourceOwnerStore::contact_credit(const DistributedAdaptiveContactHalo& halo) const
{
    require_current();
    if (!root_) return 0;
    const auto owners = halo.source_owner_manifest();
    const auto subfaces = halo.subface_snapshot();
    const auto& arena = halo.allocation_arena();
    if (!root_->components[1].owners[0].same_root(owners) ||
        root_->subfaces.get() != subfaces.get() || root_->subfaces.owner_before(subfaces) ||
        subfaces.owner_before(root_->subfaces) || root_->contact_arena.get() != arena.get() ||
        root_->contact_arena.owner_before(arena) || arena.owner_before(root_->contact_arena))
        throw std::invalid_argument("Source contact credit has foreign physical roots, subfaces or arena controls.");
    if (!root_->contact_projected_admission)
        throw std::invalid_argument("Native source contact fork lacks its exact imported financial scope.");
    return *root_->contact_projected_admission;
}
PLIC::details::OwnerManifestPair NativeSourceOwnerStore::single(
    PLIC::details::geometry_demand::GeometryOwnerManifest ordinary)
{
    return PLIC::details::OwnerManifestPair::sealed({std::move(ordinary), {}}, 0);
}
NativeSourceOwnerStore NativeSourceOwnerStore::assemble(const NativeSourceOwnerStore& previous,
    std::array<PLIC::details::OwnerManifestPair, 4> components,
    const geometry_detail::NativeConstructionAdmission& admission, PLIC::details::NativeGeometryAccounting* accounting,
    std::shared_ptr<const NativePublicationIndexes> indexes,
    std::array<PLIC::details::GeometryNativeAdministration, 2> input_administration,
    std::shared_ptr<const NativeSourceCoverage> coverage,
    std::shared_ptr<const AdaptiveSubfaceSnapshot> subfaces,
    std::optional<std::array<PLIC::details::NativeOwnerReceipts, 4>> component_paid,
    std::optional<std::size_t> contact_projected_admission, std::shared_ptr<const void> contact_arena,
    std::optional<NativeSourcePublicationValues> publication_values,
    std::optional<NativeContactSideIndexes> contact_side_indexes)
{
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<Root>(admission);
    if (accounting)
        accounting->authorize_semantic(0, PLIC::details::GeometryOwnerGraphEditor::external_kind(),
            tracked.owner, tracked.envelope);
    tracked.owner->components = std::move(components);
    tracked.owner->component_paid = component_paid ? std::move(*component_paid) : previous.component_paid();
    tracked.owner->contact_projected_admission = contact_arena ? contact_projected_admission :
        (previous.root_ ? previous.root_->contact_projected_admission : std::nullopt);
    tracked.owner->contact_arena = contact_arena ? std::move(contact_arena) :
        (previous.root_ ? previous.root_->contact_arena : std::shared_ptr<const void>{});
    tracked.owner->indexes = std::move(indexes);
    tracked.owner->publication_values = publication_values ? std::move(*publication_values) :
        (previous.root_ ? previous.root_->publication_values : NativeSourcePublicationValues{});
    tracked.owner->input_administration = std::move(input_administration);
    tracked.owner->coverage = coverage ? std::move(coverage) : previous.coverage();
    tracked.owner->subfaces = subfaces ? std::move(subfaces) : previous.subfaces();
    tracked.owner->contact_side_indexes = contact_side_indexes ? std::move(*contact_side_indexes) :
        (previous.root_ ? previous.root_->contact_side_indexes : NativeContactSideIndexes{});
    tracked.owner->group = coverage_group(tracked.owner->coverage);
    std::size_t local_cursor = 0;
    auto& cursor = accounting ? accounting->cursor() : local_cursor;
    const auto limit = accounting ? accounting->limit() : admission.counters().limit;
    std::size_t overlap = previous.owners().overlap_bytes();
    PLIC::details::GeometryOwnerGraphEditor ordinary(previous.owners().owners[0], 0, cursor, limit,
        accounting ? &accounting->ordinary() : nullptr, nullptr, admission.state(),
        previous.root_ ? &previous.root_->paid : nullptr);
    PLIC::details::GeometryOwnerGraphEditor bounded(previous.owners().owners[1], 1, cursor, limit,
        accounting ? &accounting->bounded() : nullptr, nullptr, admission.state());
    // Component manifests contribute flat allocation pins. A publication
    // proof can also visit the same convex, planar and identity owners
    // directly. Its first and last geometry edges own their own child walks.
    ordinary.distinguish_geometry_references();
    bounded.distinguish_geometry_references();
    ordinary.pair_with(bounded, overlap);
    ordinary.native_fixed(tracked.envelope, true, bool(accounting));
    // Closed typed financial receipts travel with their actual component roots.
    // Shared receipt pages prune unchanged component deltas on warm publication.
    for (std::size_t slot = 0; slot < tracked.owner->component_paid.size(); ++slot)
        tracked.owner->component_paid[slot].visit_added(previous.component_paid()[slot],
            {&ordinary, [](void* raw, const auto& charge, std::size_t)
                { static_cast<PLIC::details::GeometryOwnerGraphEditor*>(raw)->record_paid_extent(charge); }});
    struct RetiredComponentReceipt
    {
        PLIC::details::GeometryOwnerGraphEditor& graph;
        const std::array<PLIC::details::NativeOwnerReceipts, 4>& current;
    } retired_component{ordinary, tracked.owner->component_paid};
    for (std::size_t slot = 0; slot < tracked.owner->component_paid.size(); ++slot)
        previous.component_paid()[slot].visit_added(tracked.owner->component_paid[slot],
            {&retired_component, [](void* raw, const auto& charge, std::size_t)
                {
                    auto& value = *static_cast<RetiredComponentReceipt*>(raw);
                    if (std::none_of(value.current.begin(), value.current.end(),
                            [&](const auto& receipts) { return receipts.contains(charge); }))
                        value.graph.forget_paid_extent(charge);
                }});

    // Each component is an actual held manifest. Project metadata first/last
    // and its physical record slots; shared B+ child controls prune the visit.
    // This is the typed direct ownership protocol, not a flat diagnostic census.
    for (const bool retain : {true, false})
    {
        const auto side_indexes = [&](const auto& roots)
        {
            for (const auto& root : roots) project_contact_side_index(root, ordinary, retain, bool(accounting));
        };
        if (!retain && previous.root_) side_indexes(previous.root_->contact_side_indexes);
        for (std::size_t slot = 0; slot < tracked.owner->components.size(); ++slot)
            for (unsigned channel = 0; channel < 2; ++channel)
            {
                auto& graph = channel ? bounded : ordinary;
                struct Context { PLIC::details::GeometryOwnerGraphEditor& graph; bool retain, prepaid; }
                    context{graph, retain, bool(accounting)};
                const PLIC::details::geometry_demand::MetadataSink metadata{&context,
                    [](void* raw, const auto& owner, std::size_t, std::size_t admitted)
                    {
                        auto& c = *static_cast<Context*>(raw);
                        return c.graph.external(owner, admitted, c.retain, c.prepaid);
                    }, nullptr, [](void* raw, const auto& charge)
                    { static_cast<Context*>(raw)->graph.record_paid_extent(charge); }};
                const PLIC::details::geometry_demand::GeometryOwnerManifest::RecordSink records{&context,
                    [](void* raw, const auto& record)
                    {
                        auto& c = *static_cast<Context*>(raw);
                        c.graph.owner(record, c.retain, c.prepaid);
                    }};
                const auto& part = (retain ? tracked.owner->components : previous.components())[slot];
                part.owners[channel].visit_dependencies(metadata.with_financial(graph, retain), records);
            }
        struct AdministrationContext { PLIC::details::GeometryOwnerGraphEditor& graph; bool retain, prepaid; }
            administration_context{ordinary, retain, bool(accounting)};
        const PLIC::details::geometry_demand::MetadataSink administration_sink{&administration_context,
            [](void* raw, const auto& owner, std::size_t, std::size_t admitted)
            {
                auto& c = *static_cast<AdministrationContext*>(raw);
                return c.graph.external(owner, admitted, c.retain, c.prepaid);
            }, nullptr, [](void* raw, const auto& charge)
            { static_cast<AdministrationContext*>(raw)->graph.record_paid_extent(charge); }};
        for (const auto& administration : retain ? tracked.owner->input_administration : previous.input_administration())
            administration.visit(administration_sink.with_financial(ordinary, retain));
        for (const auto& receipts : retain ? tracked.owner->component_paid : previous.component_paid())
            receipts.visit_administration(administration_sink.with_financial(ordinary, retain));
        if (retain) AdaptiveSubfaceSnapshot::project_owners(tracked.owner->subfaces, ordinary, true);
        const auto& coverage = retain ? tracked.owner->coverage : previous.coverage();
        project_coverage(coverage, ordinary, retain, accounting);
        project_coverage_descriptor(coverage, ordinary, retain, accounting);
        if (!retain) AdaptiveSubfaceSnapshot::project_owners(previous.subfaces(), ordinary, false);
        if (retain) side_indexes(tracked.owner->contact_side_indexes);
        NativePublicationSession::project_indexes(retain ? tracked.owner->indexes : previous.indexes(),
            ordinary, retain, accounting);
        const auto values = retain ? tracked.owner->publication_values :
            (previous.root_ ? previous.root_->publication_values : NativeSourcePublicationValues{});
        if (values.cache_allocation) ordinary.native_fixed(values.cache_allocation, retain, bool(accounting));
        values.source.project(ordinary, retain);
        values.broker.project(ordinary, retain, accounting);
        if (const auto& input = values.input; input.partition)
        {
            NativePublicationOwnerGraph::Context context(ordinary, retain, bool(accounting), accounting);
            geometry_detail::NativePartitionAccess::visit(input.partition, context.sink, ordinary, retain, accounting);
            if (context.sink.semantic(input.map_token, geometry_detail::NativeConstructorOwnerKind::NativeMesh,
                    input.map_token_allocation))
            {
                context.sink.fixed(input.map_token_allocation);
                geometry_detail::NativeTpetraMapAdmission::visit(*input.map_token, ordinary, retain, accounting);
            }
        }
        NativePublicationSession::project_receipt(values.receipt, ordinary, retain, accounting);
    }
    if (previous.root_)
        ordinary.native_fixed(previous.root_->allocation, false, bool(accounting));
    tracked.owner->owners = accounting ? accounting->finish_pair(ordinary, bounded, overlap) :
        seal_default(ordinary, bounded, overlap);
    tracked.owner->paid = ordinary.paid_receipts();
    tracked.owner->allocation = std::move(tracked.envelope);
    NativeSourceOwnerStore result;
    result.root_ = std::move(tracked.owner);
    result.binding_ = PLIC::details::GeometryOwnerRootBinding(result.root_, 0, result.root_->owners,
        &result.root_->paid, result.root_->group.valid() ? &result.root_->group : nullptr);
    return result;
}
void NativeSourceOwnerStore::leaf(const AdaptiveLeaf& row, PLIC::details::GeometryOwnerGraphEditor& graph,
    bool retain, bool prepaid, PLIC::details::NativeGeometryAccounting* accounting)
{
    NativePublicationOwnerGraph::Context context(graph, retain, prepaid, accounting);
    const auto part = [&](const std::shared_ptr<const AdaptiveBytes>& value)
    {
        if (value && !NativeAdaptivePayloadAccess::bytes(value, context.sink))
            graph.external(value, sum(sizeof(*value), value->capacity()), retain, prepaid);
    };
    NativePublicationOwnerGraph::descriptor(row.native_recipe, graph, retain, prepaid, accounting);
    part(row.join_recipe);
    if (row.factor_recipe && !NativeAdaptivePayloadAccess::factor(row.factor_recipe, context.sink) &&
        graph.external_with_children(row.factor_recipe, sizeof(AdaptiveFactorRecipe), retain, prepaid))
    {
        part(row.factor_recipe->root);
        part(row.factor_recipe->base);
        part(row.factor_recipe->axis);
    }
    if (row.geometry)
        graph.geometry(row.geometry, retain);
    for (const auto& face : row.surfaces)
    {
        if (face.geometry) graph.planar(face.geometry, retain);
        if (face.native_whole_geometry) graph.planar(face.native_whole_geometry, retain);
    }
}
PLIC::details::OwnerManifestPair NativeSourceOwnerStore::seal_default(
    PLIC::details::GeometryOwnerGraphEditor& ordinary, PLIC::details::GeometryOwnerGraphEditor& bounded,
    std::size_t overlap)
{
    return PLIC::details::OwnerManifestPair::sealed({ordinary.finish(), bounded.finish()}, overlap);
}
NativeSourceOwnerStore NativeSourceOwnerStore::staged(PLIC::details::OwnerManifestPair owners,
    PLIC::details::NativeGeometryAccounting& accounting)
{
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<Root>(accounting.admission());
    accounting.authorize_semantic(0, PLIC::details::GeometryOwnerGraphEditor::external_kind(),
        tracked.owner, tracked.envelope);
    std::size_t overlap = owners.overlap_bytes();
    PLIC::details::GeometryOwnerGraphEditor ordinary(owners.owners[0], 0, accounting.cursor(), accounting.limit(),
        &accounting.ordinary());
    PLIC::details::GeometryOwnerGraphEditor bounded(owners.owners[1], 1, accounting.cursor(), accounting.limit(),
        &accounting.bounded());
    ordinary.pair_with(bounded, overlap);
    ordinary.native_fixed(tracked.envelope, true, true);
    tracked.owner->owners = accounting.finish_pair(ordinary, bounded, overlap);
    tracked.owner->paid = ordinary.paid_receipts();
    tracked.owner->allocation = std::move(tracked.envelope);
    NativeSourceOwnerStore result;
    result.root_ = std::move(tracked.owner);
    // The root header is a native allocation. Its actual fixed envelope is
    // projected by the operation before sealing owners; no guessed control or
    // second sizeof(Root) is charged through this binding.
    result.binding_ = PLIC::details::GeometryOwnerRootBinding(result.root_, 0, result.root_->owners,
        &result.root_->paid, result.root_->group.valid() ? &result.root_->group : nullptr);
    return result;
}
PLIC::details::GeometryOwnerUnion NativeSourceOwnerStore::replace_in(const NativeSourceOwnerStore& accepted,
    const PLIC::details::GeometryOwnerUnion& current, PLIC::details::NativeGeometryAccounting& accounting,
    PLIC::details::GeometryOwnerUnionWork* work) const
{
    return current.staged(accepted.binding_ ? &accepted.binding_ : nullptr, binding_ ? &binding_ : nullptr,
        accounting.ordinary(), accounting.bounded(), accounting.cursor(), accounting.limit(), work);
}
PLIC::details::GeometryOwnerUnion NativeSourceOwnerStore::replace_in(const NativeSourceOwnerStore& accepted,
    const PLIC::details::GeometryOwnerUnion& current, PLIC::GeometryAllocationLedger& ordinary,
    PLIC::GeometryAllocationLedger& bounded, std::size_t& used, std::size_t limit,
    PLIC::details::GeometryOwnerUnionWork* work) const
{
    return current.staged(accepted.binding_ ? &accepted.binding_ : nullptr, binding_ ? &binding_ : nullptr,
        ordinary, bounded, used, limit, work);
}
void NativeSourceOwnerStore::swap(NativeSourceOwnerStore& other) noexcept
{
    using std::swap;
    swap(root_, other.root_);
    swap(binding_, other.binding_);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
