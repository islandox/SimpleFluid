#include "geometry/mesh/NativeSparseAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/mesh/AMR/NativePublicationOutputGraph.hh"
#include <cmath>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
NativeSourceCellFragment NativeSparseAccess::source_cell_fragment(
    const AdaptiveLeaf& leaf, const AdaptiveCellMetrics& metrics, const Admission& admission,
    std::shared_ptr<const NativeSourceSubsetCoverage> coverage)
{
    if (leaf.provider != AdaptiveProviderKind::Polyhedral || !leaf.geometry || !(metrics.volume > 0) ||
        !std::isfinite(metrics.volume))
        throw std::invalid_argument("Native source-cell fragment requires immutable original polyhedral geometry.");
    auto source =
        geometry_detail::NativeAllocationEnvelopeAccess::make<ExplicitRegionProvider::NativeSourceCell>(admission);
    auto& data = *source.owner;
    data.metadata = geometry_detail::native_constructor_storage(admission);
    data.coverage = std::move(coverage);
    data.geometry = leaf.geometry;
    data.volume = metrics.volume;
    data.centroid = {metrics.centroid[0], metrics.centroid[1], metrics.centroid[2]};
    const auto original_faces = leaf.geometry->faces();
    const auto faces = original_faces.size();
    std::size_t incidence = 0;
    for (std::size_t f = 0; f < faces; ++f)
    {
        const auto count = original_faces.at(f).size();
        if (count > std::numeric_limits<std::size_t>::max() - incidence)
            throw std::invalid_argument("Native source-cell topology count overflows.");
        incidence += count;
    }
    if (faces == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Native source-cell face count overflows.");
    data.metadata.owner->reserve(data.offsets, faces + 1);
    data.metadata.owner->reserve(data.loops, incidence);
    // The incidence bound is exact before uniqueness. Read the original
    // stored loops directly, without constructing planar wrapper/point buffers.
    data.metadata.owner->reserve(data.points, incidence);
    data.offsets.push_back(0);
    for (std::size_t f = 0; f < faces; ++f)
        data.offsets.push_back(data.offsets.back() + original_faces.at(f).size());
    PLIC::details::NativeCellGeometrySourceAccess::visit_original_convex_face_points(*leaf.geometry, &data,
        [](void* raw, const auto& point)
        {
            auto& data = *static_cast<ExplicitRegionProvider::NativeSourceCell*>(raw);
            const auto at = std::find(data.points.begin(), data.points.end(), point);
            const auto ordinal = static_cast<std::uint64_t>(at - data.points.begin());
            if (at == data.points.end())
                data.points.push_back(point);
            data.loops.push_back(ordinal);
        });
    data.allocation = source.envelope;
    auto provider = geometry_detail::NativeAllocationEnvelopeAccess::make<ExplicitRegionProvider>(admission,
        ExplicitRegionProvider::NativeSourceKey{},
        std::shared_ptr<const ExplicitRegionProvider::NativeSourceCell>(source.owner));
    provider.owner->d_native_allocation = provider.envelope;
    auto metadata = geometry_detail::native_constructor_storage(admission);
    std::string name;
    metadata.owner->string(name, "native_source_cell");
    std::shared_ptr<const ExplicitRegionProvider> retained = provider.owner;
    return {DistributedExplicitRegion(std::move(name), retained, retained), std::move(metadata)};
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> NativeSparseAccess::source_provider_geometry(
    const std::shared_ptr<const ExplicitRegionProvider>& provider)
{
    if (!provider || !provider->d_native_source)
        throw std::invalid_argument("Not an admitted source-cell fragment.");
    return provider->d_native_source->geometry;
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> NativeSparseAccess::source_provider_geometry(
    const ExplicitRegionProvider& provider)
{
    const auto& source = provider.d_native_source;
    if (!source) return {};
    geometry_detail::NativeConstructorOwnerSink::require_envelope(source->allocation, source);
    return source->geometry;
}
const std::shared_ptr<const NativeSourceSubsetCoverage>& NativeSparseAccess::source_provider_coverage(
    const std::shared_ptr<const ExplicitRegionProvider>& provider)
{
    if (!provider || !provider->d_native_source)
        throw std::invalid_argument("Not an admitted source-cell fragment.");
    return provider->d_native_source->coverage;
}
bool NativeSparseAccess::visit_source_provider(const std::shared_ptr<const ExplicitRegionProvider>& provider,
    const geometry_detail::NativeConstructorOwnerSink& sink)
{
    using Kind = geometry_detail::NativeConstructorOwnerKind;
    if (!provider || !provider->d_native_source)
        throw std::invalid_argument("Explicit provider is not an admitted single-source native fragment.");
    geometry_detail::NativeConstructorOwnerSink::require_envelope(provider->d_native_allocation, provider);
    if (!sink.semantic(provider, Kind::NativeMesh, provider->d_native_allocation))
        return false;
    sink.fixed(provider->d_native_allocation);
    const auto& source = provider->d_native_source;
    geometry_detail::NativeConstructorOwnerSink::require_envelope(source->allocation, source);
    if (sink.semantic(source, Kind::Topology, source->allocation))
    {
        sink.fixed(source->allocation);
        sink.storage(source->metadata);
        return true;
    }
    return false;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

#include "geometry/mesh/AMR/NativePublicationNumbering.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
template<class Rows, class Key> void strictly_sorted(const Rows& rows, Key key)
{
    for (std::size_t i = 1; i < rows.size(); ++i)
        if (!(key(rows[i - 1]) < key(rows[i])))
            throw std::invalid_argument("Duplicate or unordered private sparse numeric route.");
}
} // namespace
std::shared_ptr<MultiRegionMesh> NativeSparseAccess::make(NativePublicationSession& session,
    std::shared_ptr<const NativePublicationNumbering> numbering, std::shared_ptr<const NativeSparseFragments> fragments,
    std::shared_ptr<const NativeSparseNumericIndex> numeric, std::span<const NativePublicationKeyRoute> routes,
    std::size_t full_cells,
    std::uint64_t global_faces, std::uint64_t global_nodes)
{
    const auto& admission = session.admission();
    const auto& inputs = session.inputs();
    if (!session.qualified() || !fragments || !numeric || !numbering || !numeric->allocation ||
        !numeric->allocation->authentic_owner(numeric->allocation) || !numeric->allocation->matches(numeric) ||
        routes.size() != numeric->cells.size())
        throw std::invalid_argument("Unsealed private sparse publication.");
    strictly_sorted(numeric->cells, [](const auto& r) { return r.gid; });
    strictly_sorted(numeric->faces, [](const auto& r) { return r.gid; });
    strictly_sorted(numeric->nodes, [](const auto& r) { return r.gid; });
    strictly_sorted(numeric->native_cells, [](const auto& r) { return r.first; });
    strictly_sorted(numeric->native_nodes, [](const auto& r) { return r.first; });
    strictly_sorted(numeric->logical_faces, [](const auto& r) { return r.native; });
    strictly_sorted(numeric->flux_signs, [](const auto& r) { return std::pair{r.native, r.gid}; });
    strictly_sorted(numeric->boundaries, [](const auto& r) { return r.first; });
    auto index = geometry_detail::NativeAllocationEnvelopeAccess::make<SparseNativeIndex>(admission);
    index.owner->provenance = NativeSparseProof{inputs.lineage, session.issuance_};
    index.owner->numeric = numeric;
    index.owner->allocation = index.envelope;
    index.owner->schema_owner = NativeKernelAuthority::schema_observation(inputs.kernel);
    auto mesh = geometry_detail::NativeAllocationEnvelopeAccess::make<MultiRegionMesh>(
        admission, MultiRegionMesh::NativeConstructionKey{});
    auto& result = *mesh.owner;
    result.d_native_storage = geometry_detail::native_constructor_storage(admission);
    result.d_native_allocation = mesh.envelope;
    result.d_native_fragments = std::move(fragments);
    result.d_comm = inputs.comm;
    result.d_partition_resident = true;
    result.d_num_cells = inputs.global_cells;
    result.d_num_faces = global_faces;
    result.d_num_nodes = global_nodes;
    result.d_indexer = {inputs.global_cells, global_faces, global_nodes};
    result.d_native_storage.owner->reserve(result.d_reference_axial_edges, 2);
    result.d_reference_axial_edges.insert(result.d_reference_axial_edges.end(), {0, 1});
    result.d_native_storage.owner->reserve(result.d_axial_edges, 2);
    result.d_axial_edges.insert(result.d_axial_edges.end(), {0, 1});
    // Sparse execution requires only the encoded native-face prefix. Empty
    // per-region interface directories and vector-of-vector rows are absent.
    result.d_native_storage.owner->reserve(result.d_native_faces, result.region_count() + 1);
    result.d_native_faces.push_back(0);
    for (std::size_t r = 0; r < result.region_count(); ++r)
    {
        const auto faces = result.region_layout(r).faces;
        if (faces >= std::numeric_limits<std::uint64_t>::max() - result.d_native_faces.back())
            throw std::invalid_argument("Private sparse native-face prefix overflows.");
        result.d_native_faces.push_back(result.d_native_faces.back() + faces);
    }
    result.d_sparse_native = index.owner;
    result.d_num_local_cells = result.d_num_owned_cells = result.d_num_owned_faces = 0;
    std::size_t full = 0;
    const auto& source_indexes = *session.indexes();
    for (std::size_t ordinal = 0; ordinal < routes.size(); ++ordinal)
    {
        const auto& cell = numeric->cells[ordinal];
        const auto& route = routes[ordinal];
        const bool owned = cell.owner == inputs.comm->getRank();
        if (cell.gid >= inputs.global_cells || cell.owner < 0 || cell.owner >= inputs.comm->getSize() ||
            cell.gid != route.gid || route.remote == owned ||
            (!route.remote && (route.ordinal >= inputs.layout->size() || cell.gid != inputs.cell_begin + route.ordinal)) ||
            (route.remote && route.ordinal >= source_indexes.remote.size()) ||
            !(cell.volume > 0) || !std::isfinite(cell.volume) ||
            !std::all_of(cell.centroid.begin(), cell.centroid.end(), [](double x) { return std::isfinite(x); }))
            throw std::invalid_argument("Invalid private sparse cell route.");
        if (!cell.full)
        {
            if (owned)
                throw std::invalid_argument("Owned private sparse cell is metric-only.");
            continue;
        }
        ++full;
        const auto& key = route.remote ? source_indexes.remote.at_ordinal(route.ordinal)
                                       : inputs.layout->at_ordinal(route.ordinal).key;
        const auto* row = source_indexes.find(key);
        if (!row)
            throw std::invalid_argument("Private sparse cell lacks its Session seal.");
        const auto seal = session.seal(row->source, row->incidence);
        const auto& source = seal.source();
        if (source.ticket().issuer != cell.owner || !source.fragment() ||
            &source.fragment()->region() != &result.region(cell.region) || source.native_cell() != cell.native_cell ||
            source.metrics().volume != cell.volume || source.metrics().centroid != cell.centroid ||
            cell.native_cell >= result.region_layout(cell.region).cells ||
            index.owner->native_cell({cell.region, cell.native_cell}) != cell.gid)
            throw std::invalid_argument("Private sparse cell/source controls or metrics differ.");
        ++result.d_num_local_cells;
        result.d_num_owned_cells += owned;
    }
    if (full != full_cells || result.d_num_owned_cells != inputs.layout->size())
        throw std::invalid_argument("Private sparse sealed membership differs from current owned layout.");
    for (const auto& node : numeric->nodes)
    {
        if (node.gid >= global_nodes)
            throw std::invalid_argument("Private sparse node exceeds global numbering.");
        if (node.native)
        {
            const auto [r, n] = *node.native;
            if (r >= result.region_count() || n >= result.region_layout(r).nodes ||
                index.owner->native_node({r, n}) != node.gid)
                throw std::invalid_argument("Private sparse primary native-node alias differs.");
        }
        else if (!std::all_of(node.introduced_coordinate.begin(), node.introduced_coordinate.end(),
                     [](double x) { return std::isfinite(x); }))
            throw std::invalid_argument("Private sparse introduced node is non-finite.");
    }
    for (const auto& alias : numeric->native_nodes)
    {
        const auto [r, n] = alias.first;
        if (r >= result.region_count() || n >= result.region_layout(r).nodes)
            throw std::invalid_argument("Private sparse native-node alias exceeds fragment.");
        (void) index.owner->node(alias.second);
    }
    for (const auto& face : numeric->faces)
    {
        if (face.gid >= global_faces || !face.physical || face.native.region >= result.region_count() ||
            face.native.face >= result.region_layout(face.native.region).faces ||
            face.loop_begin > numeric->loops.size() || face.loop_size > numeric->loops.size() - face.loop_begin ||
            face.second_loop_begin > numeric->loops.size() ||
            face.second_loop_size > numeric->loops.size() - face.second_loop_begin)
            throw std::invalid_argument("Private sparse canonical face route is incomplete.");
        const auto& first = index.owner->cell(face.first);
        if (face.second != MultiRegionMesh::invalid_cell_id())
            (void) index.owner->cell(face.second);
        if (face.boundary != MultiRegionMesh::invalid_boundary_id)
        {
            if (face.second != MultiRegionMesh::invalid_cell_id())
                throw std::invalid_argument("Private sparse interior has exterior label.");
            (void) index.owner->boundary(face.boundary);
        }
        for (auto gid : std::span<const std::uint64_t>(numeric->loops).subspan(face.loop_begin, face.loop_size))
            (void) index.owner->node(gid);
        for (auto gid :
            std::span<const std::uint64_t>(numeric->loops).subspan(face.second_loop_begin, face.second_loop_size))
            (void) index.owner->node(gid);
        if (!face.physical->second_image.is_zero() &&
            (face.second == MultiRegionMesh::invalid_cell_id() ||
                (face.loop_size ? face.loop_size
                                : result.topology(face.native.region, [&](const auto& t)
                                      { return t.face_nodes(face.native.face).size(); })) != face.second_loop_size))
            throw std::invalid_argument("Private sparse periodic incidence lacks exact second loop.");
        result.d_num_owned_faces += first.owner == inputs.comm->getRank();
    }
    for (const auto& row : numeric->logical_faces)
    {
        if (row.native.region >= result.region_count() ||
            row.native.face >= result.region_layout(row.native.region).faces || row.begin > numeric->loops.size() ||
            !row.size || row.size > numeric->loops.size() - row.begin)
            throw std::invalid_argument("Private sparse logical face route is invalid.");
        for (const auto gid : std::span<const std::uint64_t>(numeric->loops).subspan(row.begin, row.size))
        {
            const auto face = index.owner->face(gid);
            if (face.native != row.native && (!face.second_native || *face.second_native != row.native))
                throw std::invalid_argument("Private sparse logical/native face membership differs.");
            const auto sign = index.owner->sign(row.native, gid);
            if (sign != 1 && sign != -1)
                throw std::invalid_argument("Private sparse owner-oriented flux sign is invalid.");
        }
    }
    result.validate_static();
    return std::move(mesh.owner);
}

std::optional<NativeSparseProof> NativeSparseAccess::output_provenance(
    const std::shared_ptr<const MultiRegionMesh>& mesh)
{
    if (!mesh || !mesh->d_sparse_native) return {};
    const auto& index = mesh->d_sparse_native;
    const auto* proof = std::get_if<NativeSparseProof>(&index->provenance);
    if (!index->allocation && (!proof || (!proof->lineage[0] && !proof->lineage[1] && !proof->publication)))
        return {};
    using Sink = geometry_detail::NativeConstructorOwnerSink;
    Sink::require_envelope(index->allocation, index);
    Sink::require_envelope(mesh->d_native_allocation, mesh);
    if (!proof || !proof->lineage[0] || !proof->lineage[1] || !proof->publication ||
        index->allocation->state() != mesh->d_native_allocation->state())
        throw std::invalid_argument("Sparse output lacks its original admitted publication provenance.");
    return *proof;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
void NativeSparseAccess::visit_geometry(const std::shared_ptr<const MultiRegionMesh>& mesh,
    const geometry_detail::NativeConstructorOwnerSink& sink,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!mesh || !sink.semantic(mesh, geometry_detail::NativeConstructorOwnerKind::NativeMesh,
                     mesh->d_native_allocation)) return;
    sink.fixed(mesh->d_native_allocation);
    sink.storage(mesh->d_native_storage);
    NativePublicationOutputGraph::fragments(mesh->d_native_fragments, graph, retain, accounting);
    const auto& index = mesh->d_sparse_native;
    if (!index || !index->numeric)
        throw std::invalid_argument("Native result geometry lacks its exact sparse numeric owner.");
    if (sink.semantic(index, geometry_detail::NativeConstructorOwnerKind::Topology, index->allocation))
    {
        sink.fixed(index->allocation);
        NativePublicationOutputGraph::numeric(index->numeric, graph, retain, accounting);
        if (accounting) accounting->authorize_schema(index->schema_owner);
        NativePublicationOwnerGraph::schema(index->schema_owner, graph, retain, bool(accounting));
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
