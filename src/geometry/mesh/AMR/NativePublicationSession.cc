#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/mesh/AMR/NativePublicationOutputGraph.hh"
#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
#include "parallel/details/NativePartitionAccess.hh"
#include "geometry/mesh/AMR/NativeImmutableSequenceAccess.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Admission = geometry_detail::NativeConstructionAdmission;
using Face = PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>;
using Planar = PLIC::details::NativePlanarOwnerAccess;
Admission component_admission(const NativePublicationInputs& inputs)
{
    if (inputs.comm.is_null())
        throw std::invalid_argument("Native publication requires a communicator.");
    std::optional<Admission> admission;
    collective(*inputs.comm, [&]
        { admission.emplace(geometry_detail::NativeConstructionCounterAccess::create<Budget>(inputs.byte_budget)); });
    return std::move(*admission);
}
template<class A, class B> bool same_control(const std::shared_ptr<A>& a, const std::shared_ptr<B>& b) noexcept
{
    return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
bool same_double(double a, double b) noexcept
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
std::uint64_t issuance(const Teuchos::Comm<int>& comm)
{
    static std::atomic<std::uint64_t> next{1};
    std::uint64_t value = 0;
    collective(comm,
        [&]
        {
            if (comm.getRank() == 0)
            {
                value = next.fetch_add(1);
                if (!value || value == std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument("Native ticket issuance overflow.");
            }
        });
    ::SimpleFluid::owner_packet_broadcast_words(comm, 0, std::span(&value, 1));
    return value;
}
Face native_face(const MultiRegionMesh::Region& region, std::uint64_t face,
    const geometry_detail::NativeConstructorStorageOwner& scratch)
{
    const auto& admission = scratch.owner->admission();
    return std::visit(
        [&](const auto& provider) -> Face
        {
            const auto& geometry = provider.geometry();
            if constexpr (std::is_same_v<std::decay_t<decltype(provider)>, DistributedExplicitRegion>)
                if (const auto source = NativeSparseAccess::source_provider_geometry(geometry))
                    return Planar::cell_face(*source, face, admission);
            if constexpr (requires { geometry.certified_face_geometry(face); })
                if (auto exact = geometry.certified_face_geometry(face))
                    return exact;
            if (provider.layout().family == RegionLayout::Family::Cylindrical)
                throw std::invalid_argument("Analytic native source has no planar certificate.");
            std::vector<PLIC::GeometryPoint> points;
            const auto nodes = provider.topology().face_nodes(face);
            scratch.owner->reserve(points, nodes.size());
            for (const auto node : nodes)
            {
                const auto point = geometry.node_coordinates(node);
                points.push_back({point.x, point.y, point.z});
            }
            const auto area = geometry.face_area_vector(face);
            return Planar::oriented_like(Planar::make(points, admission), {area.x, area.y, area.z}, admission);
        },
        region);
}
const NativePublicationFragmentRow* find_fragment(
    const NativePublicationIndexes& indexes, const std::shared_ptr<const NativeFragmentDescriptor>& descriptor)
{
    const NativePublicationFragmentRow probe{descriptor, {}, {}, 0};
    const NativePublicationFragmentLess less;
    const auto row = NativePublicationIndexes::Fragments::lower_bound(
        indexes.fragments, [&](const auto& value) { return less(value, probe); });
    return row && !less(probe, *row) ? row : nullptr;
}
void edit_fragment(
    NativePublicationIndexes& indexes, const NativeCellSourceProof& source, bool retain, const Admission& admission)
{
    if (!source.fragment() || !source.source().native_recipe)
        return;
    const auto& descriptor = source.source().native_recipe;
    const auto previous = find_fragment(indexes, descriptor);
    NativePublicationFragmentRow row{descriptor, source.fragment(), geometry_construction_controls(), 0};
    if (previous)
        row = *previous;
    if (retain)
    {
        if (row.references == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Native fragment reference count overflows.");
        ++row.references;
    }
    else
    {
        if (!previous || !row.references)
            throw std::logic_error("Native fragment retirement has no accepted incoming edge.");
        --row.references;
    }
    NativePublicationIndexes::Fragments editor(
        std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission.state()->counter_owner)), nullptr, {},
        admission.state());
    indexes.fragments = editor.edit(indexes.fragments, row, row.references ? &row : nullptr, previous != nullptr);
}
std::vector<Face> source_faces(
    const NativeCellSourceProof& source, const geometry_detail::NativeConstructorStorageOwner& scratch)
{
    const auto& admission = scratch.owner->admission();
    std::vector<Face> result;
    if (!source.fragment())
    {
        if (!source.source().geometry)
            throw std::invalid_argument("Explicit native source lacks exact geometry.");
        scratch.owner->reserve(result, source.source().geometry->num_faces());
        for (std::size_t face = 0; face < source.source().geometry->num_faces(); ++face)
            result.push_back(Planar::cell_face(*source.source().geometry, face, admission));
        return result;
    }
    scratch.owner->reserve(result, source.faces().size());
    for (std::size_t index = 0; index < source.faces().size(); ++index)
    {
        const auto& route = source.faces()[index];
        if (std::any_of(source.faces().begin(), source.faces().begin() + index,
                [&](const auto& previous) { return previous.native_face == route.native_face; }))
            continue;
        auto face = native_face(source.fragment()->region(), route.native_face, scratch);
        result.push_back(route.outward_sign == 1 ? std::move(face) : Planar::reversed(*face, admission));
    }
    return result;
}
} // namespace
class NativePublicationReceipt final
{
public:
    NativePublicationSourceObservations source;
    PLIC::details::NativeWeakOwnerObservation partition, view;
    std::weak_ptr<const NativePublicationViewData> result;
    Teuchos::RCP<const DefaultTpetraTypes::map_type> map;
    GeometryConstructionControls controls;
    std::uint64_t source_topology = 0, source_partition = 0, source_geometry = 0;
    const void* source_native_geometry = nullptr;
    const void* source_ownership_address = nullptr;
    std::uint64_t output_geometry = 0, transaction = 0, cell_begin = 0;
    std::uint64_t cells = 0, faces = 0, nodes = 0;
    std::size_t local_cells = 0;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
};
void NativePublicationSourceObservations::project(PLIC::details::GeometryOwnerGraphEditor& graph, bool retain) const
{
    for (const auto* value : {&layout, &identity, &topology, &ownership, &authority, &subfaces})
        if (*value) graph.weak_observation(*value, retain);
}
std::shared_ptr<NativePublicationCache> NativePublicationSession::make_cache(const Admission& admission)
{
    auto cache = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationCache>(admission);
    cache.owner->allocation = cache.envelope;
    return std::move(cache.owner);
}
std::shared_ptr<NativePublicationCache> NativePublicationSession::copy_cache(
    const std::shared_ptr<NativePublicationCache>& source, const Admission& admission)
{
    auto cache = make_cache(admission);
    if (source)
    {
        const std::lock_guard lock(source->mutex);
        if (source->commit_token)
            throw std::invalid_argument("Cannot copy a publication cache during its accepted commit.");
        // Immutable values are shared; the new controller has its own mutex,
        // revision/token and current-result receipt. No predecessor is held.
        cache->indexes = source->indexes;
    }
    return cache;
}
void NativePublicationSession::project_receipt(const std::shared_ptr<const NativePublicationReceipt>& receipt,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!receipt) return;
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(receipt, geometry_detail::NativeConstructorOwnerKind::NativeMesh,
            receipt->allocation)) return;
    context.sink.fixed(receipt->allocation);
    receipt->source.project(graph, retain);
    graph.weak_observation(receipt->partition, retain);
    graph.weak_observation(receipt->view, retain);
    geometry_detail::NativeTpetraMapAdmission::visit_weak(receipt->map, graph, retain, accounting);
}
const NativePublicationIndexRow* NativePublicationIndexes::find(const AdaptiveCellKey& key) const
{
    const auto row = Pages::lower_bound(sources, [&](const auto& value) { return value.key() < key; });
    return row && row->key() == key ? row : nullptr;
}
NativePublicationSession::NativePublicationSession(NativePublicationInputs inputs)
    : NativePublicationSession(inputs, component_admission(inputs))
{
}
NativePublicationSession::NativePublicationSession(NativePublicationInputs inputs, Admission admission,
    PLIC::details::NativeGeometryAccounting* accounting, NativeSourceCoverageTransaction* coverage_transaction)
    : inputs_(std::move(inputs)), admission_(std::move(admission)), accounting_(accounting),
      coverage_transaction_(coverage_transaction)
{
    if (inputs_.comm.is_null())
        throw std::invalid_argument("Incomplete private native publication communicator.");
    collective(*inputs_.comm,
        [&]
        {
            if (!inputs_.layout || !inputs_.kernel || !inputs_.cache || !inputs_.subfaces)
                throw std::invalid_argument("Incomplete private native publication inputs.");
            if (accounting_ && accounting_->admission().state() != admission_.state())
                throw std::invalid_argument("Native session has a foreign operation accounting domain.");
            {
                const std::lock_guard lock(inputs_.cache->mutex);
                indexes_ = inputs_.cache->indexes;
                cache_revision_ = inputs_.cache->revision;
            }
            if (!indexes_)
            {
                auto index =
                    geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationIndexes>(admission_);
                index.owner->allocation = index.envelope;
                for (const auto& leaf : *inputs_.layout)
                    index.owner->owned_keys = index.owner->owned_keys.with_key(leaf.key, admission_);
                indexes_ = std::move(index.owner);
            }
            if (indexes_->owned_keys.size() != inputs_.layout->size())
                throw std::invalid_argument("Native source indexes have stale owned key membership.");
        });
    issuance_ = issuance(*inputs_.comm);
}
std::shared_ptr<const NativeSourceSubsetCoverage> NativePublicationSession::retain_source_subset(
    const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& geometry)
{
    if (!geometry || !coverage_transaction_ || !inputs_.source_coverage) return {};
    if (!accounting_) throw std::logic_error("Qualified source subset has no operation accounting bridge.");
    PLIC::details::GeometryOwnerGraphEditor graph({}, 0, accounting_->cursor(), accounting_->limit(),
        &accounting_->ordinary(), nullptr, admission_.state());
    graph.geometry(geometry, true);
    auto owners = NativeSourceOwnerStore::single(graph.finish());
    return retain_source_subset(owners, graph.paid_receipts());
}
std::shared_ptr<const NativeSourceSubsetCoverage> NativePublicationSession::retain_source_subset(
    const PLIC::details::OwnerManifestPair& members, const PLIC::details::NativeOwnerReceipts& receipts)
{
    if (!coverage_transaction_ || !inputs_.source_coverage) return {};
    if (members.owners[1].owner_count())
        throw std::invalid_argument("Native output subset may retain only its actual Ordinary source owners.");
    auto storage = geometry_detail::native_constructor_storage(admission_);
    std::vector<PLIC::details::geometry_demand::OwnerRecord> owners;
    storage.owner->reserve(owners, members.owners[0].owner_count());
    members.owners[0].visit([&](const auto& owner) { owners.push_back(owner); });
    return coverage_transaction_->collect(inputs_.source_coverage, owners, &receipts);
}
void NativePublicationSession::project_indexes(const std::shared_ptr<const NativePublicationIndexes>& indexes,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!indexes) return;
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(indexes, geometry_detail::NativeConstructorOwnerKind::NativeMesh, indexes->allocation))
        return;
    context.sink.fixed(indexes->allocation);
    const auto source_pages = [&](const auto& self, const NativePublicationIndexes::Pages::Ptr& page) -> void
    {
        if (!page || !NativePublicationIndexes::Pages::visit_native_node(page, context.sink)) return;
        for (const auto& child : page->children) self(self, child);
        for (const auto& row : page->rows)
        {
            project_source(row.source, graph, retain, bool(accounting), accounting);
            const auto& incidence = row.incidence;
            if (!incidence || !context.sink.semantic(incidence,
                    geometry_detail::NativeConstructorOwnerKind::NativeMesh, incidence->allocation_)) continue;
            context.sink.fixed(incidence->allocation_);
            context.sink.storage(incidence->metadata_);
            project_source(incidence->source_, graph, retain, bool(accounting), accounting);
            project_remote_physical(incidence->remote_, graph, retain, accounting);
            for (const auto& piece : incidence->pieces_) piece.project_owners(graph, retain);
        }
    };
    source_pages(source_pages, indexes->sources);
    const auto fragments = [&](const auto& self, const NativePublicationIndexes::Fragments::Ptr& page) -> void
    {
        if (!page || !NativePublicationIndexes::Fragments::visit_native_node(page, context.sink)) return;
        for (const auto& child : page->children) self(self, child);
        for (const auto& row : page->rows)
        {
            NativePublicationOwnerGraph::descriptor(row.descriptor, graph, retain, bool(accounting), accounting);
            NativePublicationOwnerGraph::fragment(row.fragment, graph, retain, bool(accounting), accounting);
        }
    };
    fragments(fragments, indexes->fragments);
    NativePublicationOutputGraph::catalog(indexes->owned_keys, graph, retain, accounting);
    NativePublicationOutputGraph::catalog(indexes->remote, graph, retain, accounting);
    NativePublicationOutputGraph::numbering(indexes->numbering, graph, retain, accounting);
}
void NativePublicationSession::project_remote_physical(const std::shared_ptr<const NativeRemoteIncidenceRecord>& record,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!record) return;
    NativePublicationOwnerGraph::Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(record, geometry_detail::NativeConstructorOwnerKind::NativeMesh, record->allocation_))
        return;
    context.sink.fixed(record->allocation_);
    context.sink.storage(record->metadata_);
    const auto sequence = [&](const auto& value)
    {
        NativeImmutableSequenceAccess::visit(value, context.sink);
    };
    for (const auto& tile : record->tiles_)
    {
        sequence(tile.domain);
        sequence(tile.surface);
        sequence(tile.node_gids);
        sequence(tile.second_node_gids);
        if (tile.geometry) graph.planar(tile.geometry, retain);
        graph.displacement_identity(tile.second_image_identity, retain);
    }
    for (const auto& facts : record->facts_)
        for (const auto& point : facts.points) graph.point_identity(point, retain);
}
void NativePublicationSession::project_source(const std::shared_ptr<const NativeCellSourceProof>& source,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid,
    PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!source) return;
    NativePublicationOwnerGraph::Context context(graph, retain, prepaid, accounting);
    if (!context.sink.semantic(source, geometry_detail::NativeConstructorOwnerKind::NativeMesh, source->allocation_))
        return;
    context.sink.fixed(source->allocation_);
    context.sink.fixed(source->authority_allocation_);
    NativeSourceBinding::graph(source->binding_, graph, retain, prepaid, accounting);
    NativePublicationOwnerGraph::fragment(source->fragment_, graph, retain, prepaid, accounting);
    context.sink.storage(source->metadata_);
}
NativeSubfaceOccurrence NativePublicationSession::occurrence(const CanonicalSubfaceView& view)
{
    return {view.page_, view.slot_, view.side_};
}
bool NativePublicationSession::qualified() const noexcept
{
    return accounting_ ? NativeKernelAuthority::supports_source_validation(inputs_.kernel)
                       : NativeKernelAuthority::has_numerical_verifier(inputs_.kernel);
}
void NativePublicationSession::stage_remote_membership(
    std::span<const std::shared_ptr<const NativeCellSourceProof>> current)
{
    for (std::size_t index = 0; index < current.size(); ++index)
        if (!current[index] || inputs_.layout->contains(current[index]->source().key) ||
            (index && !(current[index - 1]->source().key < current[index]->source().key)))
            throw std::invalid_argument("Remote membership must be an exact sorted current source selection.");
    auto next = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationIndexes>(admission_);
    *next.owner = *indexes_;
    next.owner->allocation = next.envelope;
    const auto arena =
        std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission_.state()->counter_owner));
    NativePublicationIndexes::Pages editor(arena, nullptr, {}, admission_.state());
    for (std::size_t ordinal = 0; ordinal < indexes_->remote.size(); ++ordinal)
    {
        const auto& key = indexes_->remote.at_ordinal(ordinal);
        const auto found = std::lower_bound(current.begin(), current.end(), key,
            [](const auto& value, const auto& key) { return value->source().key < key; });
        if (found != current.end() && (*found)->source().key == key)
            continue;
        next.owner->remote = next.owner->remote.edited(key, {}, arena, nullptr, admission_.state());
        if (!inputs_.layout->contains(key))
            if (const auto row = next.owner->find(key))
            {
                const auto retired = *row;
                edit_fragment(*next.owner, *row->source, false, admission_);
                next.owner->sources = editor.edit(next.owner->sources, retired, nullptr, true);
            }
    }
    for (const auto& proof : current)
    {
        const auto row = next.owner->find(proof->source().key);
        if (!row || !same_control(row->source, proof))
            throw std::invalid_argument("Remote membership uses a foreign session source.");
        if (!next.owner->remote.ordinal_of(proof->source().key))
        {
            auto key = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeRemoteKey>(admission_);
            key.owner->metadata_ = geometry_detail::native_constructor_storage(admission_);
            key.owner->key_.region = proof->source().key.region;
            std::visit(
                [&](const auto& ancestry)
                {
                    using T = std::decay_t<decltype(ancestry)>;
                    if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                    {
                        ConvexCellAncestry copy;
                        copy.root = ancestry.root;
                        key.owner->metadata_.owner->copy(copy.path, std::span<const std::uint64_t>(ancestry.path));
                        key.owner->key_.ancestry = std::move(copy);
                    }
                    else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                    {
                        ProductCellAncestry copy;
                        copy.base.root = ancestry.base.root;
                        copy.base.root_template = ancestry.base.root_template;
                        copy.axial.root = ancestry.axial.root;
                        key.owner->metadata_.owner->copy(
                            copy.base.path, std::span<const std::uint64_t>(ancestry.base.path));
                        key.owner->metadata_.owner->copy(
                            copy.axial.path, std::span<const std::uint64_t>(ancestry.axial.path));
                        key.owner->key_.ancestry = std::move(copy);
                    }
                    else
                        key.owner->key_.ancestry = ancestry;
                },
                proof->source().key.ancestry);
            key.owner->allocation_ = key.envelope;
            next.owner->remote = next.owner->remote.edited(
                proof->source().key, std::move(key.owner), arena, nullptr, admission_.state());
        }
    }
    indexes_ = std::move(next.owner);
}
void NativePublicationSession::stage_numbering(std::shared_ptr<const NativePublicationNumbering> numbering)
{
    auto next = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationIndexes>(admission_);
    *next.owner = *indexes_;
    next.owner->allocation = next.envelope;
    next.owner->numbering = std::move(numbering);
    indexes_ = std::move(next.owner);
}
NativeSparseSeal NativePublicationSession::seal(const std::shared_ptr<const NativeCellSourceProof>& source,
    const std::shared_ptr<const NativeCellIncidenceProof>& incidence) const
{
    const auto row = source ? indexes_->find(source->source().key) : nullptr;
    if (!row || !same_control(row->source, source) || !same_control(row->incidence, incidence) || !incidence ||
        !same_control(incidence->source_, source))
        throw std::invalid_argument("Sparse seal lacks current session source/incidence.");
    return NativeSparseSeal(source, incidence);
}
void NativePublicationSession::recheck(const DistributedAdaptiveHierarchy& hierarchy) const
{
    const auto current = capture(hierarchy);
    if (!same_control(current.layout, inputs_.layout) || !same_control(current.kernel, inputs_.kernel) ||
        !same_control(current.subfaces, inputs_.subfaces) || current.observation != inputs_.observation ||
        !same_control(current.observation.identity, inputs_.observation.identity) ||
        !same_control(current.observation.topology_identity, inputs_.observation.topology_identity) ||
        !same_control(current.observation.partition_identity, inputs_.observation.partition_identity) ||
        !same_control(current.ownership_owner, inputs_.ownership_owner) ||
        !current.comm.shares_resource(inputs_.comm) || current.comm.getRawPtr() != inputs_.comm.getRawPtr() ||
        geometry_construction_controls() != inputs_.subfaces->controls())
        throw std::invalid_argument("Native source changed during publication.");
}
void NativePublicationSession::replace(NativePublicationIndexRow row)
{
    auto next = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationIndexes>(admission_);
    *next.owner = *indexes_;
    next.owner->allocation = next.envelope;
    const auto previous = indexes_->find(row.key());
    if (!previous || !same_control(previous->source, row.source))
    {
        edit_fragment(*next.owner, *row.source, true, admission_);
        if (previous)
            edit_fragment(*next.owner, *previous->source, false, admission_);
    }
    NativePublicationIndexes::Pages editor(
        std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission_.state()->counter_owner)), nullptr, {},
        admission_.state());
    next.owner->sources = editor.edit(indexes_->sources, row, &row, previous != nullptr);
    indexes_ = std::move(next.owner);
}
std::shared_ptr<const NativeCellSourceProof> NativePublicationSession::source(const AdaptiveLeaf& leaf)
{
    if (inputs_.layout->find(leaf.key) != &leaf)
        throw std::invalid_argument("Native source is not the exact current owner-local row.");
    return source_impl(leaf, {});
}
std::shared_ptr<const NativeCellSourceProof> NativePublicationSession::source_impl(
    const AdaptiveLeaf& leaf, std::optional<NativeSourceTicket> remote_ticket)
{
    if (!qualified())
        throw std::invalid_argument("Custom native providers require full compatibility publication.");
    const auto controls = geometry_construction_controls();
    if (!controls.supported)
        throw std::invalid_argument("Unsupported native source construction controls.");
    const auto authority = NativeKernelAuthority::observation(inputs_.kernel).lock();
    if (const auto previous = indexes_->find(leaf.key))
    {
        const auto& proof = previous->source;
        if (proof && proof->allocation_ && proof->allocation_->authentic_owner(proof->allocation_) &&
            proof->allocation_->matches(proof) && proof->binding_->matches(leaf) && proof->controls_ == controls &&
            same_control(proof->authority_.lock(), authority) &&
            ((remote_ticket && proof->ticket_ == *remote_ticket) ||
                (!remote_ticket && proof->ticket_.issuer == inputs_.comm->getRank())))
            return proof;
    }
    if (accounting_ && !NativeKernelAuthority::supports_source_validation(inputs_.kernel, leaf))
        throw std::invalid_argument("Native operation requires an admitted source verifier for this exact leaf.");
    const bool numerical_verifier = NativeKernelAuthority::has_numerical_verifier(inputs_.kernel, leaf);
    if (!numerical_verifier) inputs_.kernel->validate(leaf);
    auto scratch = geometry_detail::native_constructor_storage(admission_);
    auto controlled = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeCellSourceProof>(
        admission_, NativeCellSourceProof::ConstructionKey{});
    auto& proof = *controlled.owner;
    proof.metadata_ = geometry_detail::native_constructor_storage(admission_);
    proof.binding_ = NativeSourceBinding::capture(leaf, admission_);
    proof.controls_ = controls;
    proof.authority_ = authority;
    proof.authority_allocation_ = NativeKernelAuthority::allocation_observation(inputs_.kernel);
    if (!numerical_verifier) proof.metrics_ = inputs_.kernel->metrics(leaf);
    proof.native_cell_ = leaf.provider == AdaptiveProviderKind::Polyhedral ? 0 : leaf.native_cell;
    {
        if (leaf.provider == AdaptiveProviderKind::Polyhedral)
        {
            if (numerical_verifier)
                proof.metrics_ = NativeKernelAuthority::prepare_source(inputs_.kernel, leaf, admission_);
            auto fragment = NativeSparseAccess::source_cell_fragment(
                leaf, proof.metrics_, admission_, retain_source_subset(leaf.geometry));
            proof.fragment_ =
                NativeFragmentAccess::freeze(std::move(fragment.region), std::move(fragment.metadata), admission_);
        }
        else
        {
            const auto prior_fragment = find_fragment(*indexes_, leaf.native_recipe);
            proof.fragment_ = prior_fragment && prior_fragment->controls == controls
                                  ? prior_fragment->fragment
                                  : NativeFragmentAccess::materialize(leaf.native_recipe, admission_);
        }
        if (numerical_verifier)
        {
            const auto& allocation = proof.fragment_->allocation();
            if (!allocation || !allocation->authentic_owner(allocation) || !allocation->matches(proof.fragment_))
                throw std::invalid_argument("Native source validator has a foreign fragment control.");
            const auto count = NativeKernelAuthority::source_face_count(inputs_.kernel, leaf);
            proof.metadata_.owner->reserve(proof.faces_, count);
            proof.faces_.resize(count);
            const auto metrics =
                NativeKernelAuthority::validate_source(inputs_.kernel, leaf, *proof.fragment_, admission_, proof.faces_);
            if (leaf.provider == AdaptiveProviderKind::Polyhedral &&
                (!same_double(metrics.volume, proof.metrics_.volume) ||
                    !std::equal(metrics.centroid.begin(), metrics.centroid.end(), proof.metrics_.centroid.begin(), same_double)))
                throw std::invalid_argument("Native cell provider metrics differ from its verified original source.");
            proof.metrics_ = metrics;
            for (std::size_t surface = 0; surface < count; ++surface)
            {
                const auto& route = proof.faces_[surface];
                if (route.surface != surface || (route.outward_sign != 1 && route.outward_sign != -1))
                    throw std::invalid_argument("Native provider did not certify each original source face route.");
            }
        }
        else
        {
            const auto& region = proof.fragment_->region();
            std::vector<Face> outward;
            std::visit(
                [&](const auto& provider)
                {
                    if (proof.native_cell_ >= provider.layout().cells)
                        throw std::invalid_argument("Native source ordinal exceeds exact retained fragment.");
                    const auto faces = provider.topology().cell_faces(proof.native_cell_);
                    scratch.owner->reserve(outward, faces.size());
                    for (const auto face : faces)
                    {
                        auto physical = native_face(region, face, scratch);
                        outward.push_back(provider.topology().owner_cell(face) == proof.native_cell_
                                              ? std::move(physical)
                                              : Planar::reversed(*physical, admission_));
                    }
                    const auto volume = provider.geometry().cell_volume(proof.native_cell_);
                    const auto center = provider.geometry().cell_centroid(proof.native_cell_);
                    if (!same_double(volume, proof.metrics_.volume) || !same_double(center.x, proof.metrics_.centroid[0]) ||
                        !same_double(center.y, proof.metrics_.centroid[1]) ||
                        !same_double(center.z, proof.metrics_.centroid[2]))
                        throw std::invalid_argument("Native source metrics differ from exact provider source.");
                },
                region);
            record_pipeline_work(PipelineWork::CellGeometryRequest);
            const auto canonical = inputs_.kernel->geometry(leaf);
            record_pipeline_work(PipelineWork::PhysicalShellReconstruction);
            if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(outward)->equivalent(*canonical))
                throw std::invalid_argument("Native retained fragment differs from source convex geometry.");
            const auto supports = inputs_.kernel->surfaces(leaf);
            proof.metadata_.owner->reserve(proof.faces_, supports.size());
            for (std::size_t surface = 0; surface < supports.size(); ++surface)
            {
                std::optional<NativeSourceFaceRoute> match;
                std::visit(
                    [&](const auto& provider)
                    {
                        std::size_t index = 0;
                        for (const auto face : provider.topology().cell_faces(proof.native_cell_))
                        {
                            if (outward[index]->contains(*supports[surface].geometry) &&
                                !outward[index]->oppositely_oriented(*supports[surface].geometry))
                            {
                                if (match)
                                    throw std::invalid_argument("Ambiguous native source support mapping.");
                                match = NativeSourceFaceRoute{
                                    surface, face, provider.topology().owner_cell(face) == proof.native_cell_ ? 1 : -1};
                            }
                            ++index;
                        }
                    },
                    region);
                if (!match)
                    throw std::invalid_argument("Native source support has no exact provider face.");
                proof.faces_.push_back(*match);
            }
        }
    }
    proof.ticket_ = remote_ticket.value_or(NativeSourceTicket{inputs_.comm->getRank(), issuance_, next_ordinal_++});
    proof.allocation_ = controlled.envelope;
    auto result = std::shared_ptr<const NativeCellSourceProof>(std::move(controlled.owner));
    replace({result, {}});
    return result;
}
void NativePublicationSession::compact_remote(
    const NativeCellSourceProof& source, NativeRemoteIncidenceRecord& record) const
{
    auto scratch = geometry_detail::native_constructor_storage(admission_);
    const auto originals = source_faces(source, scratch);
    std::vector<std::uint64_t> native_faces;
    scratch.owner->reserve(native_faces, source.faces().size());
    for (const auto& face : source.faces())
        if (std::find(native_faces.begin(), native_faces.end(), face.native_face) == native_faces.end())
            native_faces.push_back(face.native_face);
    record.metadata_.owner->reserve(record.facts_, record.tiles_.size());
    for (auto& tile : record.tiles_)
    {
        if (!tile.geometry)
            throw std::invalid_argument("New remote incidence lacks full physical verification geometry.");
        NativeRemoteFaceFacts facts;
        facts.centroid = tile.geometry->centroid();
        facts.area_vector = tile.geometry->area_vector();
        record.metadata_.owner->reserve(facts.points, tile.geometry->points().size());
        for (const auto& point : tile.geometry->points())
            facts.points.push_back(PLIC::details::GeometryIdentityAccess::point(point));
        facts.native_is_second = tile.first != source.source().key;
        auto physical = facts.native_is_second ? Planar::reversed(*tile.geometry, admission_) : tile.geometry;
        if (facts.native_is_second && !tile.second_image_identity.is_zero())
            physical = Planar::translated(*physical,
                PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image_identity), admission_);
        for (std::size_t face = 0; face < originals.size(); ++face)
            if (originals[face]->contains(*physical) && !originals[face]->oppositely_oriented(*physical))
            {
                if (facts.native_face)
                    throw std::invalid_argument("Remote incidence has ambiguous native source-face containment.");
                if (face < native_faces.size())
                    facts.native_face = native_faces[face];
                const auto& original_points = originals[face]->points();
                const auto& physical_points = physical->points();
                // Geometric containment also accepts an enriched collinear
                // edge. Drop the completed loop only for the exact same nodes.
                facts.whole = physical->contains(*originals[face]) &&
                    std::is_permutation(original_points.begin(), original_points.end(), physical_points.begin(),
                        physical_points.end(), [](const auto& original, const auto& completed)
                        {
                            using Identity = PLIC::details::GeometryIdentityAccess;
                            return Identity::point(original) == Identity::point(completed);
                        });
            }
        if (!facts.native_face)
            throw std::invalid_argument("Remote incidence has no authenticated native source face.");
        // The resident first side already supplies the canonical metrics and
        // loop. A periodic resident second side needs its first-image override.
        if (facts.whole && (!facts.native_is_second || tile.second_image_identity.is_zero()))
            tile.geometry.reset();
        record.facts_.push_back(std::move(facts));
    }
}
void NativePublicationSession::verify_incidence(
    const NativeCellSourceProof& source, std::span<const Face> outward) const
{
    auto scratch = geometry_detail::native_constructor_storage(admission_);
    const auto original = source_faces(source, scratch);
    certify_adaptive_physical_boundary_faces_native(original, outward, inputs_.byte_budget, admission_);
}
bool NativePublicationSession::same_occurrence(const CanonicalSubfaceView& a, const CanonicalSubfaceView& b) noexcept
{
    return same_control(a.page_, b.page_) && a.slot_ == b.slot_ && a.side_ == b.side_;
}
std::shared_ptr<const NativeCellIncidenceProof> NativePublicationSession::incidence(
    const std::shared_ptr<const NativeCellSourceProof>& source, std::span<const CanonicalSubfaceView> pieces)
{
    if (!source || !source->allocation_ || !source->allocation_->authentic_owner(source->allocation_) ||
        !source->allocation_->matches(source) ||
        !same_control(source->authority_.lock(), NativeKernelAuthority::observation(inputs_.kernel).lock()) ||
        source->controls_ != geometry_construction_controls())
        throw std::invalid_argument("Foreign or stale native source proof.");
    // A complete physical shell alone cannot authenticate current labels or
    // source occurrence membership. Match every expected control/slot/side
    // directly against the current completed immutable source index first.
    std::size_t expected = 0;
    for (const auto& face : source->faces())
        inputs_.subfaces->visit_face_pieces(source->source().key, face.surface,
            [&](const CanonicalSubfaceView& current)
            {
                ++expected;
                if (std::none_of(pieces.begin(), pieces.end(),
                        [&](const auto& supplied) { return same_occurrence(current, supplied); }))
                    throw std::invalid_argument("Native incidence omits a current exact completed occurrence.");
            });
    if (expected != pieces.size())
        throw std::invalid_argument("Native incidence has foreign or repeated completed occurrences.");
    if (const auto previous = indexes_->find(source->source().key);
        previous && same_control(previous->source, source) && previous->incidence)
    {
        const auto& accepted = previous->incidence;
        if (accepted->allocation_ && accepted->allocation_->authentic_owner(accepted->allocation_) &&
            accepted->allocation_->matches(accepted) && accepted->pieces_.size() == pieces.size() &&
            std::equal(pieces.begin(), pieces.end(), accepted->pieces_.begin(), same_occurrence))
            return accepted;
    }
    auto scratch = geometry_detail::native_constructor_storage(admission_);
    std::vector<Face> outward;
    scratch.owner->reserve(outward, pieces.size());
    Budget geometry_budget{inputs_.byte_budget};
    IdentityBudget identities;
    identities.bind_native(admission_, inputs_.byte_budget);
    for (const auto& piece : pieces)
    {
        const auto first = piece.first();
        const auto second = piece.second();
        const bool is_first = piece.outward_sign() > 0;
        if ((is_first && first.key() != source->source().key) ||
            (!is_first && (!second || second->key() != source->source().key)) ||
            (is_first ? first.revision() : second->revision()) != source->source().revision)
            throw std::invalid_argument("Completed subface occurrence has a foreign source revision.");
        auto face = piece.materialize_native(geometry_budget, identities, admission_).geometry;
        if (!is_first)
        {
            face = Planar::reversed(*face, admission_);
            if (!piece.second_image().is_zero())
                face = Planar::translated(*face,
                    PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, piece.second_image()), admission_);
        }
        outward.push_back(std::move(face));
    }
    verify_incidence(*source, outward);
    auto controlled = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeCellIncidenceProof>(
        admission_, NativeCellIncidenceProof::ConstructionKey{});
    controlled.owner->source_ = source;
    controlled.owner->metadata_ = geometry_detail::native_constructor_storage(admission_);
    controlled.owner->metadata_.owner->copy(controlled.owner->pieces_, pieces);
    controlled.owner->ticket_ = {inputs_.comm->getRank(), issuance_, next_ordinal_++};
    controlled.owner->allocation_ = controlled.envelope;
    auto result = std::shared_ptr<const NativeCellIncidenceProof>(std::move(controlled.owner));
    replace({source, result});
    return result;
}
std::shared_ptr<NativePublicationCache> NativePublicationSession::edited(
    const std::shared_ptr<NativePublicationCache>& source, std::span<const AdaptiveLeafPages::EditView> edits,
    const Admission& admission)
{
    auto result = make_cache(admission);
    if (!source)
        return result;
    {
        const std::lock_guard lock(source->mutex);
        result->indexes = source->indexes;
    }
    if (!result->indexes || edits.empty())
        return result;
    auto next = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationIndexes>(admission);
    *next.owner = *result->indexes;
    next.owner->allocation = next.envelope;
    NativePublicationIndexes::Pages editor(
        std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(admission.state()->counter_owner)), nullptr, {},
        admission.state());
    for (const auto& edit : edits)
    {
        const bool held = next.owner->owned_keys.ordinal_of(edit.key).has_value();
        if (!edit.leaf && held)
            next.owner->owned_keys = next.owner->owned_keys.without_key(edit.key, admission);
        else if (edit.leaf && !held)
            next.owner->owned_keys = next.owner->owned_keys.with_key(edit.key, admission);
        if (const auto old = next.owner->find(edit.key))
        {
            const auto key = *old;
            edit_fragment(*next.owner, *old->source, false, admission);
            next.owner->sources = editor.edit(next.owner->sources, key, nullptr, true);
        }
    }
    result->indexes = std::move(next.owner);
    return result;
}
void NativePublicationSession::stage_result(std::shared_ptr<NativePublicationViewData> result,
    Teuchos::RCP<const DefaultTpetraTypes::map_type> map, std::uint64_t collective_identity)
{
    if (!result || !result->payload || !result->payload->partition || !result->payload->mesh ||
        map.is_null() || map->getComm().getRawPtr() != inputs_.comm.getRawPtr() ||
        map->getGlobalNumElements() != inputs_.global_cells || map->getLocalNumElements() != inputs_.layout->size() ||
        !result->payload->partition->geometry_unchanged() || !result->payload->partition->geometry()->sparse_native() ||
        !result->payload->partition->plan().communicator().shares_resource(inputs_.comm) ||
        result->payload->partition->plan().communicator().getRawPtr() != inputs_.comm.getRawPtr() ||
        !collective_identity || result->payload->partition->plan().collective_identity() != collective_identity ||
        !result->payload->mesh->owned_cell_map().shares_resource(map) ||
        result->payload->mesh->owned_cell_map().getRawPtr() != map.getRawPtr())
        throw std::invalid_argument("Incomplete exact native publication result.");
    if (result->payload->owned.size() != inputs_.layout->size() ||
        result->payload->cell_begin != inputs_.cell_begin || !same_control(result->payload->remote.root_, indexes_->remote.root_) ||
        result->payload->keys.size() != inputs_.layout->size() + result->payload->remote.size())
        throw std::invalid_argument("Native result routes have foreign current membership.");
    for (std::size_t ordinal = 0; ordinal < inputs_.layout->size(); ++ordinal)
    {
        const auto gid = inputs_.cell_begin + ordinal;
        const auto route = std::lower_bound(result->payload->keys.begin(), result->payload->keys.end(), gid,
            [](const auto& route, std::uint64_t gid) { return route.gid < gid; });
        if (result->payload->owned.at_ordinal(ordinal) != inputs_.layout->at_ordinal(ordinal).key ||
            route == result->payload->keys.end() || route->gid != gid || route->remote || route->ordinal != ordinal ||
            map->getGlobalElement(ordinal) != DefaultTpetraTypes::global_ordinal_type(gid) ||
            result->payload->partition->plan().owned_cells()[ordinal] != gid)
            throw std::invalid_argument("Native result owned routes differ from current typed-key map order.");
    }
    for (std::size_t ordinal = 0; ordinal < result->payload->keys.size(); ++ordinal)
    {
        const auto& route = result->payload->keys[ordinal];
        if (route.gid >= inputs_.global_cells || (ordinal && result->payload->keys[ordinal - 1].gid >= route.gid) ||
            (route.remote && route.ordinal >= result->payload->remote.size()))
            throw std::invalid_argument("Native result has invalid, repeated or unsorted current routes.");
    }
    auto receipt = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationReceipt>(admission_);
    auto& value = *receipt.owner;
    value.source = inputs_.fixed_observations;
    if (!value.source.layout.matches_live(inputs_.layout) ||
        !value.source.identity.matches_live(inputs_.observation.identity) ||
        !value.source.topology.matches_live(inputs_.observation.topology_identity) ||
        !value.source.ownership.matches_live(inputs_.ownership_owner) ||
        !value.source.authority.matches_live(NativeKernelAuthority::observation(inputs_.kernel).lock()) ||
        !value.source.subfaces.matches_live(inputs_.subfaces))
        throw std::invalid_argument("Native receipt lacks actual source layout and observation controls.");
    value.partition = PLIC::details::NativeWeakOwnerObservation::native(result->payload->partition,
        geometry_detail::NativePartitionAccess::allocation(result->payload->partition));
    value.view = PLIC::details::NativeWeakOwnerObservation::native(result, result->allocation);
    value.result = result;
    value.map = map.create_weak();
    value.controls = geometry_construction_controls();
    value.source_topology = inputs_.observation.topology;
    value.source_partition = inputs_.observation.partition;
    value.source_geometry = inputs_.observation.geometry;
    value.source_native_geometry = inputs_.observation.native_geometry;
    value.source_ownership_address = inputs_.observation.partition_identity.get();
    value.output_geometry = result->payload->partition->geometry()->geometry_epoch();
    value.transaction = collective_identity;
    value.cell_begin = inputs_.cell_begin;
    value.cells = inputs_.global_cells;
    value.faces = result->payload->partition->plan().global_faces();
    value.nodes = result->payload->partition->plan().global_nodes();
    value.local_cells = inputs_.layout->size();
    value.allocation = receipt.envelope;
    receipt_ = std::move(receipt.owner);
    result_ = std::move(result);
}
void NativePublicationSession::commit_locked() noexcept
{
    if (!result_ || !receipt_ || inputs_.cache->revision != cache_revision_ ||
        cache_revision_ == std::numeric_limits<std::uint64_t>::max())
        std::terminate();
    inputs_.cache->indexes = indexes_;
    inputs_.cache->receipt = receipt_;
    ++inputs_.cache->revision;
}
void NativePublicationSession::commit() noexcept
{
    const std::lock_guard lock(inputs_.cache->mutex);
    if (inputs_.cache->commit_token)
        std::terminate();
    commit_locked();
}
AdaptiveNativePublicationView NativePublicationSession::view() const
{
    if (!result_)
        throw std::invalid_argument("Native publication has no staged result.");
    return AdaptiveNativePublicationView(result_);
}
bool NativePublicationSession::receipt_matches(const NativePublicationInputs& inputs,
    const std::shared_ptr<const CompositePartition<>>& partition,
    const Teuchos::RCP<const DefaultTpetraTypes::map_type>& map)
{
    if (!inputs.cache || !partition || map.is_null())
        return false;
    std::shared_ptr<const NativePublicationReceipt> receipt;
    {
        const std::lock_guard lock(inputs.cache->mutex);
        receipt = inputs.cache->receipt;
    }
    if (!receipt || !receipt->allocation || !receipt->allocation->authentic_owner(receipt->allocation) ||
        !receipt->allocation->matches(receipt) || !receipt->source.layout.matches_live(inputs.layout) ||
        !receipt->source.identity.matches_live(inputs.observation.identity) ||
        !receipt->source.topology.matches_live(inputs.observation.topology_identity) ||
        !receipt->source.ownership.matches_live(inputs.ownership_owner) ||
        !receipt->source.authority.matches_live(NativeKernelAuthority::observation(inputs.kernel).lock()) ||
        !receipt->source.subfaces.matches_live(inputs.subfaces) ||
        receipt->source_ownership_address != inputs.observation.partition_identity.get() ||
        inputs.ownership_owner.owner_before(inputs.observation.partition_identity) ||
        inputs.observation.partition_identity.owner_before(inputs.ownership_owner) ||
        !receipt->partition.matches_live(partition) || receipt->map.is_null() || !receipt->map.is_valid_ptr() ||
        !receipt->map.shares_resource(map) || receipt->map.getRawPtr() != map.getRawPtr() ||
        !partition->plan().communicator().shares_resource(inputs.comm) ||
        partition->plan().communicator().getRawPtr() != inputs.comm.getRawPtr() ||
        receipt->controls != geometry_construction_controls() || !inputs.subfaces ||
        receipt->controls != inputs.subfaces->controls() ||
        receipt->source_topology != inputs.observation.topology ||
        receipt->source_partition != inputs.observation.partition ||
        receipt->source_geometry != inputs.observation.geometry ||
        receipt->source_native_geometry != inputs.observation.native_geometry ||
        receipt->cell_begin != inputs.cell_begin || receipt->cells != inputs.global_cells ||
        receipt->local_cells != inputs.layout->size() || !partition->geometry_unchanged() ||
        receipt->output_geometry != partition->geometry()->geometry_epoch() ||
        receipt->transaction != partition->plan().collective_identity() ||
        receipt->faces != partition->plan().global_faces() || receipt->nodes != partition->plan().global_nodes())
        return false;
    const auto result = receipt->result.lock();
    return receipt->view.matches_live(result) && result->payload && result->payload->owned.size() == inputs.layout->size() &&
        same_control(result->payload->partition, partition);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
