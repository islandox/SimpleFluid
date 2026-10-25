/** @file DistributedAdaptiveNative.cc
 * @brief Procedural native fragments and certified explicit local poly shards.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include <algorithm>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using ID = std::uint64_t;
MeshUtils::Vec3 vector(PLIC::GeometryPoint p)
{
    return {p[0], p[1], p[2]};
}
std::map<int, std::string> physical_schema(
    const AdaptiveNativeDraft& draft, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t limit)
{
    std::set<std::string> local, names;
    for (const auto& tile : draft.tiles)
        if (tile.exterior && !tile.boundary.empty())
            local.insert(tile.boundary);
    adaptive_detail::Writer wire;
    adaptive_detail::collective(*comm,
        [&]
        {
            wire.word(local.size());
            for (const auto& name : local)
                wire.bytes(std::span(name));
            if (wire.data.size() > limit || wire.data.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Adaptive physical patch schema exceeds byte budget.");
        });
    adaptive_detail::Budget allocation{limit};
    for (int rank = 0; rank < comm->getSize(); ++rank)
    {
        int size = rank == comm->getRank() ? static_cast<int>(wire.data.size()) : 0;
        Teuchos::broadcast(*comm, rank, 1, &size);
        adaptive_detail::collective(*comm,
            [&]
            {
                if (size < 0 || std::size_t(size) > limit)
                    throw std::invalid_argument("Invalid adaptive patch schema size.");
            });
        std::vector<char> packet(static_cast<std::size_t>(size));
        if (rank == comm->getRank())
            packet = wire.data;
        Teuchos::broadcast(*comm, rank, size, packet.data());
        adaptive_detail::collective(*comm,
            [&]
            {
                adaptive_detail::Reader r{packet};
                const auto count = r.word();
                if (count > (packet.size() - r.offset) / 8)
                    throw std::invalid_argument("Invalid adaptive patch schema count.");
                for (ID i = 0; i < count; ++i)
                {
                    auto bytes = r.bytes();
                    std::string name(bytes.begin(), bytes.end());
                    if (name.empty())
                        throw std::invalid_argument("Empty named adaptive physical patch.");
                    if (!names.contains(name))
                    {
                        allocation.add(name.size() + sizeof(std::string) + 4 * sizeof(void*));
                        names.insert(std::move(name));
                    }
                }
                r.done();
            });
    }
    std::map<int, std::string> result;
    int next = 0;
    for (const auto& name : names)
        result.emplace(next++, name);
    return result;
}
PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> native_face(const MultiRegionMesh::Region& region, ID face)
{
    return std::visit(
        [&](const auto& provider) -> PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>
        {
            const auto& geometry = provider.geometry();
            if constexpr (requires { geometry.certified_face_geometry(face); })
                if (auto exact = geometry.certified_face_geometry(face))
                    return exact;
            if (provider.layout().family == RegionLayout::Family::Cylindrical)
                throw std::invalid_argument("Analytic native geometry lacks a planar adaptive certificate.");
            std::vector<PLIC::GeometryPoint> points;
            for (auto node : provider.topology().face_nodes(face))
            {
                const auto point = geometry.node_coordinates(node);
                points.push_back({point.x, point.y, point.z});
            }
            const auto area = geometry.face_area_vector(face);
            return PLIC::PlanarFace<ArithmeticMode::Ordinary>::make(points)->oriented_like({area.x, area.y, area.z});
        },
        region);
}
} // namespace
AdaptiveNativePublication assemble_adaptive_native(AdaptiveNativeDraft draft, std::span<const char> root_schema,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t limit)
{
    if (comm.is_null())
        throw std::invalid_argument("Adaptive native assembly requires communicator.");
    SparseNativeMeshInput input;
    input.byte_budget = limit;
    input.global_cells = draft.global_cells;
    input.global_faces = draft.global_faces;
    input.global_nodes = draft.global_nodes;
    input.root_schema.assign(root_schema.begin(), root_schema.end());
    input.boundaries = physical_schema(draft, comm, limit);
    std::map<std::string, int> boundary_ids;
    for (const auto& [id, name] : input.boundaries)
        boundary_ids.emplace(name, id);
    AdaptiveNativePublication result;
    NativeFragmentCache cache;
    std::map<AdaptiveCellKey, const AdaptiveNativeCellBinding*> bindings;
    std::map<AdaptiveCellKey, const AdaptiveLeaf*> leaves;
    std::map<AdaptiveCellKey, std::pair<std::size_t, ID>> addresses;
    std::map<ID, const AdaptiveNode*> nodes;
    adaptive_detail::collective(*comm,
        [&]
        {
            adaptive_detail::Budget allocation{limit};
            for (const auto& binding : draft.cell_bindings)
            {
                allocation.add(sizeof(binding) + encode_adaptive_key(binding.key).size() + 8 * sizeof(void*));
                if (!bindings.emplace(binding.key, &binding).second || binding.gid >= draft.global_cells)
                    throw std::invalid_argument("Duplicate/invalid adaptive native cell binding.");
                if (!result.local_keys.emplace(binding.gid, binding.key).second)
                    throw std::invalid_argument("Duplicate adaptive native numeric cell address.");
            }
            for (const auto& leaf : draft.owned)
            {
                leaves.emplace(leaf.key, &leaf);
                const auto& binding = *bindings.at(leaf.key);
                if (!binding.full || binding.owner != comm->getRank())
                    throw std::invalid_argument("Owned adaptive native binding is incomplete.");
                result.owned_gids.emplace(leaf.key, binding.gid);
            }
            for (const auto& leaf : draft.ghosts)
                if (!leaves.emplace(leaf.key, &leaf).second)
                    throw std::invalid_argument("Duplicate adaptive native full ghost.");
            for (const auto& node : draft.nodes)
            {
                allocation.add(
                    sizeof(node) + node.identity.size() + node.coordinate_identity.size() + 8 * sizeof(void*));
                if (node.coordinate_identity.empty() ||
                    node.coordinate != PLIC::point_display<ArithmeticMode::Ordinary>(node.coordinate_identity) ||
                    !nodes.emplace(node.gid, &node).second)
                    throw std::invalid_argument("Adaptive native node lacks its exact physical point proof.");
                input.nodes.push_back({node.gid, {}, node.coordinate});
                input.node_certificates.emplace(node.gid, node.coordinate_identity);
            }
            // Native procedural descriptors remain factor-shared; the cache interns
            // parts/views across owned and full ghost records for this publication.
            std::map<std::pair<const void*, const void*>, std::size_t> fragment_ids;
            for (const auto& [key, binding] : bindings)
                if (binding->full && binding->provider != AdaptiveProviderKind::Polyhedral)
                {
                    if (!binding->native_recipe)
                        throw std::invalid_argument("Full procedural adaptive cell lacks native recipe.");
                    auto region = materialize_native_fragment(*binding->native_recipe, cache);
                    const auto identity = std::visit([](const auto& r)
                        { return std::pair{r.topology().storage_identity(), r.geometry().storage_identity()}; },
                        region);
                    auto [it, inserted] = fragment_ids.emplace(identity, input.fragments.size());
                    if (inserted)
                        input.fragments.push_back(std::move(region));
                    addresses.emplace(key, std::pair{it->second, binding->native_cell});
                }
            // Explicit polyhedra use one local fragment, with dense local native
            // ordinals. Canonical global IDs exist solely in the sparse routes.
            std::map<AdaptiveCellKey, ID> poly_cells;
            std::map<ID, ID> poly_faces, poly_nodes;
            std::set<AdaptiveCellKey> poly_full;
            std::map<AdaptiveCellKey, std::vector<const AdaptiveTile*>> incident;
            for (const auto& tile : draft.tiles)
            {
                incident[tile.first].push_back(&tile);
                if (tile.second)
                    incident[*tile.second].push_back(&tile);
            }
            for (const auto& [key, binding] : bindings)
                if (binding->full && binding->provider == AdaptiveProviderKind::Polyhedral)
                {
                    poly_cells.emplace(key, poly_cells.size());
                    poly_full.insert(key);
                }
            if (!poly_cells.empty())
            {
                for (const auto& tile : draft.tiles)
                    if (poly_full.contains(tile.first) || (tile.second && poly_full.contains(*tile.second)))
                    {
                        if (!tile.second_image_identity.empty())
                            throw std::invalid_argument(
                                "Periodic explicit polyhedral native fragments are not qualified.");
                        poly_faces.emplace(tile.face_gid, poly_faces.size());
                        if (!poly_cells.contains(tile.first))
                            poly_cells.emplace(tile.first, poly_cells.size());
                        if (tile.second && !poly_cells.contains(*tile.second))
                            poly_cells.emplace(*tile.second, poly_cells.size());
                        for (auto gid : tile.node_gids)
                            poly_nodes.emplace(gid, poly_nodes.size());
                    }
                std::vector<ExplicitRegionProvider::Cell> cells;
                std::vector<ExplicitRegionProvider::Face> faces;
                std::vector<ExplicitRegionProvider::Node> points;
                std::map<ID, PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>> cell_proofs;
                std::map<ID, PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> face_proofs;
                std::map<ID, std::vector<char>> point_proofs;
                for (const auto& [key, id] : poly_cells)
                {
                    const auto& binding = *bindings.at(key);
                    std::vector<ID> incidence;
                    std::set<ID> vertices;
                    const bool full = binding.full && binding.provider == AdaptiveProviderKind::Polyhedral;
                    if (full)
                    {
                        for (const auto* tile : incident.at(key))
                        {
                            incidence.push_back(poly_faces.at(tile->face_gid));
                            for (auto node : tile->node_gids)
                                vertices.insert(poly_nodes.at(node));
                        }
                        const auto* leaf = leaves.at(key);
                        if (!leaf->geometry)
                            throw std::invalid_argument(
                                "Explicit polyhedral full cell requires canonical exact geometry.");
                        cell_proofs.emplace(id, leaf->geometry);
                        addresses.emplace(key, std::pair{input.fragments.size(), id});
                    }
                    cells.push_back({id, MeshUtils::CellType::POLYHEDRON, std::move(incidence),
                        std::vector<ID>(vertices.begin(), vertices.end()), vector(binding.centroid), binding.volume});
                }
                for (const auto& tile : draft.tiles)
                    if (poly_faces.contains(tile.face_gid))
                    {
                        std::vector<ID> loop;
                        for (auto node : tile.node_gids)
                            loop.push_back(poly_nodes.at(node));
                        const int boundary = tile.exterior && !tile.boundary.empty()
                                                 ? boundary_ids.at(tile.boundary)
                                                 : MultiRegionMesh::invalid_boundary_id;
                        faces.push_back({poly_faces.at(tile.face_gid), poly_cells.at(tile.first),
                            tile.second ? poly_cells.at(*tile.second) : MultiRegionMesh::invalid_cell_id(),
                            std::move(loop), tile.geometry->centroid(), tile.geometry->area_vector(), boundary});
                        face_proofs.emplace(poly_faces.at(tile.face_gid), tile.geometry);
                    }
                for (const auto& [gid, id] : poly_nodes)
                {
                    points.push_back({id, vector(nodes.at(gid)->coordinate)});
                    point_proofs.emplace(id, nodes.at(gid)->coordinate_identity);
                }
                RegionLayout layout{};
                layout.family = RegionLayout::Family::Explicit;
                layout.cells = poly_cells.size();
                layout.faces = poly_faces.size();
                layout.nodes = poly_nodes.size();
                auto provider = ExplicitRegionProvider::from_certified_records(layout, std::move(cells),
                    std::move(faces), std::move(points), input.boundaries, std::move(cell_proofs),
                    std::move(face_proofs), point_proofs);
                const auto region_index = input.fragments.size();
                input.fragments.emplace_back(DistributedExplicitRegion("local_polyhedral", provider, provider));
                for (const auto& [gid, id] : poly_nodes)
                    input.native_nodes.emplace(std::pair{region_index, id}, gid);
            }
            for (const auto& [key, binding] : bindings)
            {
                const auto address = binding->full ? addresses.at(key) : std::pair<std::size_t, ID>{0, 0};
                input.cells.push_back({binding->gid, binding->owner, address.first, address.second, binding->full,
                    binding->volume, binding->centroid});
            }
            // Bind exact native node identities to authority GIDs, using the domain
            // namespace rather than coordinate welding across physical contacts.
            std::map<AdaptiveBytes, ID> point_ids;
            for (const auto& node : draft.nodes)
                point_ids.emplace(node.identity, node.gid);
            for (const auto& [key, address] : addresses)
            {
                const auto* leaf = leaves.at(key);
                const auto& region = input.fragments.at(address.first);
                std::visit(
                    [&](const auto& r)
                    {
                        for (auto face : r.topology().cell_faces(address.second))
                        {
                            const auto certificate = native_face(region, face);
                            for (const auto& point : certificate->vertex_identities())
                            {
                                const auto display = PLIC::point_display<ArithmeticMode::Ordinary>(point);
                                std::optional<ID> native;
                                for (auto node : r.topology().face_nodes(face))
                                {
                                    const auto p = r.geometry().node_coordinates(node);
                                    if (p.x == display[0] && p.y == display[1] && p.z == display[2])
                                    {
                                        if (native)
                                            throw std::invalid_argument(
                                                "Ambiguous native display point in adaptive fragment.");
                                        native = node;
                                    }
                                }
                                if (!native)
                                    throw std::invalid_argument("Native adaptive face lacks exact display node.");
                                adaptive_detail::Writer identity;
                                identity.bytes(leaf->domain);
                                identity.bytes(point);
                                const auto gid = point_ids.at(identity.data);
                                auto [it, inserted] =
                                    input.native_nodes.emplace(std::pair{address.first, *native}, gid);
                                if (!inserted && it->second != gid)
                                    throw std::invalid_argument(
                                        "Native node aliases cross connected-domain identities.");
                            }
                        }
                    },
                    region);
            }
            std::map<ID, std::pair<std::size_t, ID>> primary_nodes;
            for (const auto& [native, gid] : input.native_nodes)
                primary_nodes.emplace(gid, native);
            for (auto& node : input.nodes)
                if (auto found = primary_nodes.find(node.gid); found != primary_nodes.end())
                {
                    node.native = found->second;
                    node.introduced_coordinate = {};
                }
            for (const auto& tile : draft.tiles)
            {
                allocation.add(sizeof(SparseNativeFace) + tile.node_gids.size() * sizeof(ID));
                const auto& first = *bindings.at(tile.first);
                const bool first_full = first.full;
                if (!first_full && (!tile.second || !bindings.at(*tile.second)->full))
                    throw std::invalid_argument("Adaptive native tile has no full local incident cell.");
                const auto primary_key = first_full ? tile.first : *tile.second;
                auto face_address = [&](const AdaptiveCellKey& key, bool second) -> RegionFace
                {
                    const auto address = addresses.at(key);
                    const auto& binding = *bindings.at(key);
                    auto physical = tile.geometry;
                    if (second)
                        physical = physical->reversed()->translated_packet(
                            PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image_identity));
                    // Support identity belongs to physical ancestry. Native face
                    // ordinals in tile transport belong to the owner's recipe,
                    // which may differ from this fetched/pruned local recipe.
                    bool supported = false;
                    for (const auto& support : leaves.at(key)->surfaces)
                        if (support.key == tile.surface && support.exterior == tile.exterior &&
                            support.boundary == tile.boundary && support.geometry->contains(*physical) &&
                            !support.geometry->oppositely_oriented(*physical))
                            supported = true;
                    if (!supported)
                        throw std::invalid_argument("Adaptive tile does not match actual local stable support.");
                    if (binding.provider == AdaptiveProviderKind::Polyhedral)
                        return {address.first, poly_faces.at(tile.face_gid)};
                    const auto& region = input.fragments.at(address.first);
                    std::optional<ID> found;
                    std::visit(
                        [&](const auto& provider)
                        {
                            for (auto face : provider.topology().cell_faces(address.second))
                            {
                                auto certificate = native_face(region, face);
                                if (provider.topology().owner_cell(face) != address.second)
                                    certificate = certificate->reversed();
                                if (certificate->contains(*physical) && !certificate->oppositely_oriented(*physical))
                                {
                                    if (found)
                                        throw std::invalid_argument("Ambiguous local native face for adaptive tile.");
                                    found = face;
                                }
                            }
                        },
                        region);
                    if (!found)
                        throw std::invalid_argument(
                            "Actual retained native recipe does not contain adaptive incidence.");
                    return {address.first, *found};
                };
                SparseNativeFace face;
                face.gid = tile.face_gid;
                face.first = first.gid;
                if (tile.second)
                    face.second = bindings.at(*tile.second)->gid;
                face.native = face_address(primary_key, !first_full);
                if (first_full && tile.second && bindings.at(*tile.second)->full)
                    face.second_native = face_address(*tile.second, true);
                face.boundary = tile.exterior && !tile.boundary.empty() ? boundary_ids.at(tile.boundary)
                                                                        : MultiRegionMesh::invalid_boundary_id;
                face.override_geometry = tile.geometry;
                face.override_nodes = tile.node_gids;
                face.second_image = tile.second_image;
                face.second_image_identity = tile.second_image_identity;
                face.second_node_gids = tile.second_node_gids;
                face.native_is_second = !first_full;
                input.faces.push_back(std::move(face));
            }
        });
    auto geometry = MultiRegionMesh::from_sparse_native(std::move(input), comm);
    std::vector<ID> owned, ghosts;
    for (const auto& leaf : draft.owned)
        owned.push_back(bindings.at(leaf.key)->gid);
    for (const auto& leaf : draft.ghosts)
        ghosts.push_back(bindings.at(leaf.key)->gid);
    auto partition =
        std::make_shared<const CompositePartition<>>(CompositePartition<>::from_sparse_native(geometry, owned, ghosts));
    result.partition = partition;
    result.mesh = std::make_shared<const MeshHandle<>>(*partition);
    result.storage = result.mesh->storage_report();
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
