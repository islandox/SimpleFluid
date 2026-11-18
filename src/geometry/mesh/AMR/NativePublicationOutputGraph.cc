#include "geometry/mesh/AMR/NativePublicationOutputGraph.hh"
#include "geometry/details/NativeMeshHandleAccess.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Graph = PLIC::details::GeometryOwnerGraphEditor;
using Accounting = PLIC::details::NativeGeometryAccounting;
using Context = NativePublicationOwnerGraph::Context;
using Kind = geometry_detail::NativeConstructorOwnerKind;
template<class Pages, class Emit> void pages(const typename Pages::Ptr& node, Context& context, const Emit& emit)
{
    if (!node || !Pages::visit_native_node(node, context.sink))
        return;
    for (const auto& row : node->rows)
        emit(row);
    for (const auto& child : node->children)
        pages<Pages>(child, context, emit);
}
template<class Vector> std::size_t bytes(const Vector& vector)
{
    return persistent_ordered_detail::checked_product(vector.capacity(), sizeof(typename Vector::value_type));
}
std::size_t numeric_bytes(const NativeSparseNumericIndex& v)
{
    // faces and boundary rows contain borrowed physical/name pointers. They
    // remain metadata, even though their other columns happen to be scalar.
    auto n = persistent_ordered_detail::checked_sum(bytes(v.cells), bytes(v.nodes));
    n = persistent_ordered_detail::checked_sum(n, bytes(v.native_cells));
    n = persistent_ordered_detail::checked_sum(n, bytes(v.native_nodes));
    n = persistent_ordered_detail::checked_sum(n, bytes(v.logical_faces));
    n = persistent_ordered_detail::checked_sum(n, bytes(v.flux_signs));
    return persistent_ordered_detail::checked_sum(n, bytes(v.loops));
}
} // namespace
void NativePublicationOutputGraph::catalog(
    const NativeRemoteKeyCatalog& catalog, Graph& graph, bool retain, Accounting* accounting)
{
    Context context(graph, retain, bool(accounting), accounting);
    pages<NativeRemoteKeyCatalog::Pages>(catalog.root_, context,
        [&](const auto& row)
        {
            const auto& key = row.binding;
            if (context.sink.semantic(key, Kind::Topology, key->allocation_))
            {
                context.sink.fixed(key->allocation_);
                context.sink.storage(key->metadata_);
            }
        });
}
void NativePublicationOutputGraph::fragments(
    const std::shared_ptr<const NativeSparseFragments>& fragments, Graph& graph, bool retain, Accounting* accounting)
{
    if (!fragments)
        return;
    Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(fragments, Kind::Topology, fragments->allocation))
        return;
    context.sink.fixed(fragments->allocation);
    vector(fragments, fragments->allocation, fragments->route, graph, retain, accounting);
    pages<NativeSparseFragments::Pages>(fragments->root, context, [&](const auto& row)
        { NativePublicationOwnerGraph::fragment(row.fragment, graph, retain, bool(accounting), accounting); });
}
void NativePublicationOutputGraph::numbering(const std::shared_ptr<const NativePublicationNumbering>& numbering,
    Graph& graph, bool retain, Accounting* accounting)
{
    if (!numbering)
        return;
    Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(numbering, Kind::Topology, numbering->allocation))
        return;
    context.sink.fixed(numbering->allocation);
    vector(numbering, numbering->allocation, numbering->pieces, graph, retain, accounting);
    vector(numbering, numbering->allocation, numbering->points, graph, retain, accounting);
    vector(numbering, numbering->allocation, numbering->loops, graph, retain, accounting);
    pages<NativePublicationNumbering::PiecePages>(numbering->piece_root, context,
        [&](const auto& row)
        {
            const auto& piece = row.record;
            if (!context.sink.semantic(piece, Kind::Topology, piece->allocation))
                return;
            context.sink.fixed(piece->allocation);
            vector(piece, piece->allocation, piece->points, graph, retain, accounting);
            vector(piece, piece->allocation, piece->second_points, graph, retain, accounting);
            vector(piece, piece->allocation, piece->first_native_nodes, graph, retain, accounting);
            vector(piece, piece->allocation, piece->second_native_nodes, graph, retain, accounting);
            graph.weak_observation(piece->page_observation, retain);
            NativePublicationOwnerGraph::fragment(piece->first_fragment, graph, retain, bool(accounting), accounting);
            NativePublicationOwnerGraph::fragment(piece->second_fragment, graph, retain, bool(accounting), accounting);
            graph.displacement_identity(piece->physical.second_image, retain);
            graph.planar(piece->physical.override_geometry, retain);
            const auto point = [&](const auto& value)
            {
                if (!context.sink.semantic(value, Kind::Geometry, value->allocation))
                    return;
                context.sink.fixed(value->allocation);
                vector(value, value->allocation, value->key, graph, retain, accounting);
                graph.point_identity(value->point, retain);
            };
            for (const auto& value : piece->points)
                point(value);
            for (const auto& value : piece->second_points)
                point(value);
        });
    pages<NativePublicationNumbering::PointPages>(numbering->point_root, context,
        [&](const auto& row)
        {
            const auto& point = row.record;
            if (context.sink.semantic(point, Kind::Geometry, point->allocation))
            {
                context.sink.fixed(point->allocation);
                vector(point, point->allocation, point->key, graph, retain, accounting);
                graph.point_identity(point->point, retain);
            }
        });
    pages<NativePublicationNumbering::RemotePages>(numbering->remote_root, context,
        [&](const auto& row)
        {
            const auto& piece = row.record;
            if (!context.sink.semantic(piece, Kind::Geometry, piece->allocation))
                return;
            context.sink.fixed(piece->allocation);
            vector(piece, piece->allocation, piece->native_nodes, graph, retain, accounting);
            NativePublicationSession::project_remote_physical(piece->physical_record, graph, retain, accounting);
            NativePublicationOwnerGraph::fragment(piece->fragment, graph, retain, bool(accounting), accounting);
            graph.displacement_identity(piece->physical.second_image, retain);
            graph.planar(piece->physical.override_geometry, retain);
        });
    pages<NativePublicationNumbering::BoundaryPages>(numbering->boundary_root, context,
        [&](const auto& row)
        {
            if (context.sink.semantic(row.record, Kind::Topology, row.record->allocation))
            {
                context.sink.fixed(row.record->allocation);
                context.sink.storage(row.record->metadata);
            }
        });
    fragments(numbering->fragments, graph, retain, accounting);
}
void NativePublicationOutputGraph::numeric(
    const std::shared_ptr<const NativeSparseNumericIndex>& numeric, Graph& graph, bool retain, Accounting* accounting)
{
    if (!numeric)
        return;
    Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(numeric, Kind::Topology, numeric->allocation))
        return;
    context.sink.fixed(numeric->allocation);
    vector(numeric, numeric->allocation, numeric->cells, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->faces, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->nodes, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->native_cells, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->native_nodes, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->logical_faces, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->flux_signs, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->loops, graph, retain, accounting);
    vector(numeric, numeric->allocation, numeric->boundaries, graph, retain, accounting);
    numbering(std::static_pointer_cast<const NativePublicationNumbering>(numeric->physical_owners), graph, retain,
        accounting);
}
void NativePublicationOutputGraph::payload(
    const std::shared_ptr<const NativePublicationPayload>& payload, Graph& graph, bool retain, Accounting* accounting)
{
    if (!payload)
        return;
    Context context(graph, retain, bool(accounting), accounting);
    if (!context.sink.semantic(payload, Kind::Topology, payload->allocation))
        return;
    context.sink.fixed(payload->allocation);
    context.sink.storage(payload->route_storage);
    catalog(payload->owned, graph, retain, accounting);
    catalog(payload->remote, graph, retain, accounting);
    numbering(payload->numbering, graph, retain, accounting);
    fragments(payload->fragments, graph, retain, accounting);
    numeric(payload->numeric, graph, retain, accounting);
    geometry_detail::NativePartitionAccess::visit(payload->partition, context.sink, graph, retain, accounting);
    geometry_detail::NativeMeshHandleAccess::visit(payload->mesh, context.sink, graph, retain, accounting);
}
std::shared_ptr<const NativePublicationRoot> NativePublicationOutputGraph::make(
    std::shared_ptr<NativePublicationPayload> payload,
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> view,
    const std::shared_ptr<const NativePublicationRoot>& previous, Accounting& accounting)
{
    if (!payload || !payload->partition || !payload->mesh || !view || !view->object_alive())
        throw std::invalid_argument("Native output subset requires actual accepted payload/view allocations.");
    auto root = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationRoot>(accounting.admission());
    root.owner->payload_ = payload;
    root.owner->allocation_ = root.envelope;
    root.owner->view_allocation_ = view;
    // The union adds the root header as a zero-byte semantic edge, outside
    // its payload manifest. The actual fixed envelope is projected below.
    accounting.authorize_semantic(0, Graph::external_kind(), root.owner, root.envelope);
    const auto old = previous ? previous->owners() : PLIC::details::OwnerManifestPair{};
    auto overlap = old.overlap_bytes();
    Graph ordinary(old.owners[0], 0, accounting.cursor(), accounting.limit(), &accounting.ordinary(), nullptr,
        accounting.admission().state(), previous ? &previous->paid_receipts() : nullptr);
    Graph bounded(old.owners[1], 1, accounting.cursor(), accounting.limit(), &accounting.bounded(), nullptr,
        accounting.admission().state());
    ordinary.pair_with(bounded, overlap);
    NativePublicationOutputGraph::payload(payload, ordinary, true, &accounting);
    ordinary.native_fixed(root.envelope, true, true);
    ordinary.native_fixed(view, true, true);
    if (previous)
    {
        NativePublicationOutputGraph::payload(previous->payload(), ordinary, false, &accounting);
        ordinary.native_fixed(previous->allocation_, false, true);
        ordinary.native_fixed(previous->view_allocation_, false, true);
    }
    root.owner->owners_ = accounting.finish_pair(ordinary, bounded, overlap);
    root.owner->paid_ = ordinary.paid_receipts();
    // Cached manifest totals are maintained by changed physical first/last
    // transitions. No diagnostic full mesh census or arena-used snapshot.
    auto total = persistent_ordered_detail::checked_sum(
        root.owner->owners_.payload_bytes(), root.owner->owners_.index_admission_bytes());
    payload->summary.scalar_numeric = numeric_bytes(*payload->numeric);
    payload->summary.scalar_numeric = persistent_ordered_detail::checked_sum(
        payload->summary.scalar_numeric, payload->keys.capacity() * sizeof(NativePublicationKeyRoute));
    payload->summary.scalar_numeric = persistent_ordered_detail::checked_sum(
        payload->summary.scalar_numeric, payload->numbering->loops.capacity() * sizeof(std::uint64_t));
    const auto& plan = payload->partition->plan();
    payload->summary.scalar_numeric =
        persistent_ordered_detail::checked_sum(payload->summary.scalar_numeric, bytes(plan.cell_ids()));
    payload->summary.scalar_numeric =
        persistent_ordered_detail::checked_sum(payload->summary.scalar_numeric, bytes(plan.face_ids()));
    payload->summary.scalar_numeric =
        persistent_ordered_detail::checked_sum(payload->summary.scalar_numeric, bytes(plan.node_ids()));
    payload->summary.scalar_numeric =
        persistent_ordered_detail::checked_sum(payload->summary.scalar_numeric, bytes(plan.cell_owner_ranks()));
    payload->summary.scalar_numeric = persistent_ordered_detail::checked_sum(
        payload->summary.scalar_numeric, geometry_detail::NativeMeshHandleAccess::scalar_bytes(payload->mesh));
    payload->summary.library_upper = geometry_detail::NativeMeshHandleAccess::library_upper(payload->mesh);
    const auto excluded =
        persistent_ordered_detail::checked_sum(payload->summary.scalar_numeric, payload->summary.library_upper);
    if (excluded > total)
        throw std::logic_error("Native scalar/library output exceeds its complete physical subset.");
    payload->summary.metadata = total - excluded;
    return std::move(root.owner);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
