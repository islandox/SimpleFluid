/** @file MultiRegionMeshSparse.cc
 * @brief Sparse owner-local routes retaining procedural native interiors.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
namespace SimpleFluid::Meshes
{
namespace
{
MeshUtils::Vec3 vector(PLIC::GeometryPoint p)
{
    return {p[0], p[1], p[2]};
}
bool second_occurrence(const SparseNativeFace& face, RegionFace address, std::uint64_t cell)
{
    if (cell != face.second)
        return false;
    if (cell != face.first)
        return true;
    if (address == face.native)
        return face.native_is_second;
    if (face.second_native && address == *face.second_native)
        return !face.native_is_second;
    throw std::invalid_argument("Periodic self incidence lacks a native image address.");
}
PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> physical_incidence(const SparseNativeFace& face, bool second)
{
    return second ? face.override_geometry->reversed()->translated(
                        PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, face.second_image_identity))
                  : face.override_geometry;
}
template<class Map> std::size_t map_bytes(const Map& map)
{
    return map.size() * (sizeof(typename Map::value_type) + 4 * sizeof(void*));
}
} // namespace
std::size_t SparseNativeIndex::storage_bytes() const
{
    std::size_t total = common_signature.capacity() + sizeof(*this) + map_bytes(cells) + map_bytes(faces) +
                        map_bytes(nodes) + map_bytes(native_cells) + map_bytes(native_nodes) +
                        map_bytes(logical_faces) + map_bytes(boundaries) + map_bytes(flux_signs);
    for (const auto& [face, row] : logical_faces)
        total += row.capacity() * sizeof(std::uint64_t);
    for (const auto& [gid, face] : faces)
    {
        total += (face.override_nodes.capacity() + face.second_node_gids.capacity()) * sizeof(std::uint64_t) +
                 face.second_image_identity.payload_bytes();
        if (face.override_geometry)
            total += face.override_geometry->encode_incidence().capacity();
    }
    for (const auto& [id, name] : boundaries)
        total += name.capacity();
    return total;
}
std::shared_ptr<MultiRegionMesh> MultiRegionMesh::from_sparse_native(
    SparseNativeMeshInput input, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    if (comm.is_null())
        throw std::invalid_argument("Sparse native geometry requires a communicator.");
    auto mesh = std::shared_ptr<MultiRegionMesh>(new MultiRegionMesh(PartitionTag{}));
    auto index = std::make_shared<SparseNativeIndex>();
    ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer schema;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            schema.word(input.global_cells);
            schema.word(input.global_faces);
            schema.word(input.global_nodes);
            schema.bytes(input.root_schema);
            schema.word(input.boundaries.size());
            for (const auto& [id, name] : input.boundaries)
            {
                if (id < 0 || name.empty())
                    throw std::invalid_argument("Invalid sparse global boundary schema.");
                schema.word(static_cast<std::uint64_t>(id));
                schema.bytes(std::span(name));
            }
            if (schema.data.size() > input.byte_budget ||
                schema.data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Sparse common schema exceeds packet budget.");
        });
    int schema_length = comm->getRank() == 0 ? static_cast<int>(schema.data.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &schema_length);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (schema_length < 0 || static_cast<std::size_t>(schema_length) > input.byte_budget)
                throw std::invalid_argument("Sparse common schema exceeds receiver budget.");
        });
    std::vector<char> common(static_cast<std::size_t>(schema_length));
    if (comm->getRank() == 0)
        common = schema.data;
    Teuchos::broadcast(*comm, 0, schema_length, common.data());
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (common != schema.data)
                throw std::invalid_argument("Sparse native root/count/boundary schema differs across ranks.");
            index->common_signature.assign(common.begin(), common.end());
        });
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{input.byte_budget};
            budget.add(sizeof(*mesh) + sizeof(*index));
            for (const auto& cell : input.cells)
                budget.add(sizeof(cell) + 8 * sizeof(void*));
            for (const auto& face : input.faces)
            {
                budget.add(sizeof(face) + 16 * sizeof(void*) +
                           (face.override_nodes.size() + face.second_node_gids.size()) * sizeof(ID) +
                           face.second_image_identity.payload_bytes());
                if (face.override_geometry)
                    budget.add(face.override_geometry->encode_incidence().size());
            }
            for (const auto& node : input.nodes)
                budget.add(sizeof(node) + 8 * sizeof(void*));
            for (const auto& [gid, proof] : input.node_certificates)
                budget.add(
                    proof.payload_bytes() + sizeof(decltype(input.node_certificates)::value_type) + 4 * sizeof(void*));
            budget.add(
                input.native_nodes.size() * (sizeof(decltype(input.native_nodes)::value_type) + 4 * sizeof(void*)));
            mesh->d_regions = std::move(input.fragments);
            mesh->d_comm = comm;
            mesh->d_partition_resident = true;
            mesh->d_num_cells = input.global_cells;
            mesh->d_num_faces = input.global_faces;
            mesh->d_num_nodes = input.global_nodes;
            mesh->d_num_local_cells = mesh->d_num_owned_cells = mesh->d_num_owned_faces = 0;
            mesh->d_indexer = {input.global_cells, input.global_faces, input.global_nodes};
            mesh->d_reference_axial_edges = mesh->d_axial_edges = {0, 1};
            mesh->d_region_interfaces.resize(mesh->d_regions.size());
            mesh->d_cells.resize(mesh->d_regions.size() + 1);
            mesh->d_faces.resize(mesh->d_regions.size() + 1);
            mesh->d_nodes.resize(mesh->d_regions.size() + 1);
            mesh->d_native_faces.resize(mesh->d_regions.size() + 1);
            for (std::size_t r = 0; r < mesh->d_regions.size(); ++r)
            {
                const auto& layout = mesh->region_layout(r);
                if (layout.faces >= std::numeric_limits<ID>::max() - mesh->d_native_faces[r])
                    throw std::overflow_error("Sparse native fragment address overflow.");
                mesh->d_native_faces[r + 1] = mesh->d_native_faces[r] + layout.faces;
            }
            mesh->validate_static();
            for (auto& cell : input.cells)
            {
                if (cell.gid >= input.global_cells || cell.owner < 0 || cell.owner >= comm->getSize() ||
                    !(cell.volume > 0) || !std::isfinite(cell.volume) ||
                    !std::all_of(cell.centroid.begin(), cell.centroid.end(), [](double x) { return std::isfinite(x); }))
                    throw std::invalid_argument("Invalid sparse native cell metric/binding.");
                if (!index->cells.emplace(cell.gid, cell).second)
                    throw std::invalid_argument("Duplicate sparse native global cell route.");
                if (cell.full)
                {
                    if (cell.native_cell >= mesh->region_layout(cell.region).cells ||
                        !index->native_cells.emplace(std::pair{cell.region, cell.native_cell}, cell.gid).second)
                        throw std::invalid_argument("Invalid or duplicate sparse native ordinal route.");
                    const auto volume =
                        mesh->geometry(cell.region, [&](const auto& g) { return g.cell_volume(cell.native_cell); });
                    const auto center =
                        mesh->geometry(cell.region, [&](const auto& g) { return g.cell_centroid(cell.native_cell); });
                    if (volume != cell.volume || center.x != cell.centroid[0] || center.y != cell.centroid[1] ||
                        center.z != cell.centroid[2])
                        throw std::invalid_argument("Sparse native cell metric differs from retained provider.");
                    ++mesh->d_num_local_cells;
                    if (cell.owner == comm->getRank())
                        ++mesh->d_num_owned_cells;
                }
                else if (cell.owner == comm->getRank())
                    throw std::invalid_argument("Owned sparse native cell requires full residency.");
            }
            index->native_nodes = std::move(input.native_nodes);
            index->boundaries = std::move(input.boundaries);
            if (input.node_certificates.size() != input.nodes.size())
                throw std::invalid_argument("Sparse nodes require one exact authority point proof per route.");
            for (auto& node : input.nodes)
            {
                const auto& proof = input.node_certificates.at(node.gid);
                const auto display = PLIC::point_display<ArithmeticMode::Ordinary>(proof);
                if (node.native)
                {
                    const auto [r, n] = *node.native;
                    if (n >= mesh->region_layout(r).nodes)
                        throw std::invalid_argument("Invalid sparse native node ordinal.");
                    const auto p = mesh->geometry(r, [&](const auto& g) { return g.node_coordinates(n); });
                    if (p.x != display[0] || p.y != display[1] || p.z != display[2])
                        throw std::invalid_argument(
                            "Sparse native node coordinate differs from exact authority point.");
                }
                else if (node.introduced_coordinate != display)
                    throw std::invalid_argument(
                        "Sparse introduced node coordinate differs from exact authority point.");
                if (node.gid >= input.global_nodes || !index->nodes.emplace(node.gid, node).second)
                    throw std::invalid_argument("Invalid or duplicate sparse native node.");
                if (node.native)
                {
                    const auto [r, n] = *node.native;
                    if (n >= mesh->region_layout(r).nodes)
                        throw std::invalid_argument("Invalid sparse native node ordinal.");
                }
                else if (!std::all_of(node.introduced_coordinate.begin(), node.introduced_coordinate.end(),
                             [](double x) { return std::isfinite(x); }))
                    throw std::invalid_argument("Invalid sparse introduced node coordinate.");
            }
            for (const auto& [native, gid] : index->native_nodes)
                if (native.second >= mesh->region_layout(native.first).nodes || !index->nodes.contains(gid))
                    throw std::invalid_argument("Sparse native node alias lacks a resident route.");
            for (auto& face : input.faces)
            {
                if (face.gid >= input.global_faces || !face.override_geometry ||
                    face.override_nodes.size() != face.override_geometry->vertex_identities().size() ||
                    !index->cells.contains(face.first) ||
                    (face.second != invalid_cell_id() && !index->cells.contains(face.second)) ||
                    (face.second != invalid_cell_id() && face.boundary != invalid_boundary_id) ||
                    (face.boundary != invalid_boundary_id && !index->boundaries.contains(face.boundary)))
                    throw std::invalid_argument("Incomplete sparse native canonical face proof.");
                if (face.second_image !=
                        PLIC::displacement_display<ArithmeticMode::Ordinary>(face.second_image_identity) ||
                    (!face.second_image_identity.is_zero() && face.second == invalid_cell_id()) ||
                    (face.first == face.second && face.second_image_identity.is_zero()))
                    throw std::invalid_argument(
                        "Sparse periodic face has inconsistent exact/display image provenance.");
                const auto vertices = face.override_geometry->vertex_identities();
                std::vector<PLIC::PointIdentity<ArithmeticMode::Ordinary>> routed_points;
                for (auto node : face.override_nodes)
                {
                    if (!index->nodes.contains(node))
                        throw std::invalid_argument("Sparse native face references a nonresident node.");
                    routed_points.push_back(input.node_certificates.at(node));
                }
                bool ordered = false;
                std::size_t node_shift = 0;
                for (std::size_t shift = 0; shift < vertices.size() && !ordered; ++shift)
                {
                    bool equal = true;
                    for (std::size_t i = 0; i < vertices.size(); ++i)
                        equal = equal && routed_points[i] == vertices[(i + shift) % vertices.size()];
                    ordered = equal;
                    if (equal)
                        node_shift = shift;
                }
                if (!ordered)
                    throw std::invalid_argument("Sparse face node routes differ from exact ordered incidence.");
                if (!face.second_image_identity.is_zero() && face.second_node_gids.size() != vertices.size())
                    throw std::invalid_argument("Sparse periodic face lacks second physical node incidence.");
                if (!face.second_node_gids.empty())
                {
                    if (face.second == invalid_cell_id() || face.second_node_gids.size() != vertices.size())
                        throw std::invalid_argument("Sparse second incidence has invalid extent.");
                    for (std::size_t i = 0; i < vertices.size(); ++i)
                    {
                        auto node = face.second_node_gids[i];
                        if (!index->nodes.contains(node) ||
                            input.node_certificates.at(node) !=
                                PLIC::point_translated<ArithmeticMode::Ordinary>(
                                    vertices[(i + node_shift) % vertices.size()],
                                    PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                        {}, face.second_image_identity)))
                            throw std::invalid_argument(
                                "Sparse second physical node proof disagrees with exact image.");
                    }
                }
                const auto add_native = [&](RegionFace address, bool address_is_second)
                {
                    if (address.face >= mesh->region_layout(address.region).faces)
                        throw std::invalid_argument("Invalid sparse native face ordinal.");
                    const auto owner =
                        mesh->topology(address.region, [&](const auto& t) { return t.owner_cell(address.face); });
                    const auto neighbor =
                        mesh->topology(address.region, [&](const auto& t) { return t.neighbor_cell(address.face); });
                    bool incident = false;
                    for (auto gid : {face.first, face.second})
                        if (gid != invalid_cell_id())
                        {
                            const auto& cell = index->cells.at(gid);
                            const bool role_matches = face.second_image_identity.is_zero() ||
                                                      face.first == face.second ||
                                                      ((gid == face.second) == address_is_second);
                            incident = incident || (role_matches && cell.full && cell.region == address.region &&
                                                       (cell.native_cell == owner || cell.native_cell == neighbor));
                        }
                    if (!incident)
                        throw std::invalid_argument("Sparse face has no full native incident provider.");
                    const auto certificate = mesh->native_face_certificate(address);
                    for (auto gid : {face.first, face.second})
                        if (gid != invalid_cell_id())
                        {
                            const auto& cell = index->cells.at(gid);
                            if (!cell.full || cell.region != address.region ||
                                (cell.native_cell != owner && cell.native_cell != neighbor) ||
                                (!face.second_image_identity.is_zero() && ((gid == face.second) != address_is_second) &&
                                    face.first != face.second))
                                continue;
                            const auto outward = cell.native_cell == owner ? certificate : certificate->reversed();
                            const auto tile = physical_incidence(
                                face, !face.second_image_identity.is_zero() ? address_is_second : gid == face.second);
                            if (!outward->contains(*tile) || outward->oppositely_oriented(*tile))
                                throw std::invalid_argument(
                                    "Sparse canonical face orientation disagrees with its incident cell.");
                        }
                    const auto in_address = address_is_second
                                                ? face.override_geometry->translated(
                                                      PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                                          {}, face.second_image_identity))
                                                : face.override_geometry;
                    const int sign = certificate->oppositely_oriented(*in_address) ? -1 : 1;
                    index->logical_faces[address].push_back(face.gid);
                    index->flux_signs.emplace(std::pair{address, face.gid}, sign);
                };
                add_native(face.native, face.native_is_second);
                if (face.second_native && *face.second_native != face.native)
                    add_native(*face.second_native, !face.native_is_second);
                if (index->cells.at(face.first).owner == comm->getRank())
                    ++mesh->d_num_owned_faces;
                if (!index->faces.emplace(face.gid, std::move(face)).second)
                    throw std::invalid_argument("Duplicate sparse canonical face route.");
            }
            // Validate signed outward shells and exact alias identities before
            // discarding authority certificates for uncut procedural interiors.
            std::set<std::pair<std::size_t, ID>> authenticated_aliases;
            for (const auto& [gid, cell] : index->cells)
                if (cell.full)
                {
                    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> canonical_shell, native_shell;
                    for (auto native :
                        mesh->topology(cell.region, [&](const auto& t) { return t.cell_faces(cell.native_cell); }))
                    {
                        const RegionFace address{cell.region, native};
                        const auto found = index->logical_faces.find(address);
                        if (found == index->logical_faces.end())
                            throw std::invalid_argument("Incomplete sparse full-cell incidence.");
                        const auto certificate = mesh->native_face_certificate(address);
                        const auto native_owner =
                            mesh->topology(cell.region, [&](const auto& t) { return t.owner_cell(native); });
                        const auto outward = cell.native_cell == native_owner ? certificate : certificate->reversed();
                        native_shell.push_back(outward);
                        std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> tiles;
                        for (auto face_id : found->second)
                        {
                            const auto& face = index->faces.at(face_id);
                            if (face.first != gid && face.second != gid)
                                throw std::invalid_argument("Sparse native face row names a nonincident global cell.");
                            const auto tile = physical_incidence(face, second_occurrence(face, address, gid));
                            if (outward->oppositely_oriented(*tile))
                                throw std::invalid_argument("Sparse full-cell tile is inward oriented.");
                            tiles.push_back(tile);
                            canonical_shell.push_back(tile);
                        }
                        certificate->certify_partition(tiles);
                        const auto native_nodes =
                            mesh->topology(cell.region, [&](const auto& t) { return t.face_nodes(native); });
                        for (const auto& point : certificate->vertex_identities())
                        {
                            const auto display = PLIC::point_display<ArithmeticMode::Ordinary>(point);
                            std::optional<ID> matching;
                            for (auto node : native_nodes)
                            {
                                const auto p = mesh->geometry(
                                    cell.region, [&](const auto& g) { return g.node_coordinates(node); });
                                if (p.x == display[0] && p.y == display[1] && p.z == display[2])
                                {
                                    if (matching)
                                        throw std::invalid_argument(
                                            "Sparse native face has ambiguous display-node provenance.");
                                    matching = node;
                                }
                            }
                            if (!matching)
                                throw std::invalid_argument("Sparse native exact face lacks its display-node route.");
                            const auto alias = std::pair{cell.region, *matching};
                            const auto global = index->native_nodes.at(alias);
                            if (input.node_certificates.at(global) != point)
                                throw std::invalid_argument(
                                    "Sparse native node alias differs from exact provider provenance.");
                            authenticated_aliases.insert(alias);
                        }
                    }
                    if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(canonical_shell)
                            ->equivalent(*PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(native_shell)))
                        throw std::invalid_argument(
                            "Sparse canonical outward shell differs from native cell geometry.");
                }
            for (const auto& [alias, gid] : index->native_nodes)
                if (!authenticated_aliases.contains(alias))
                    throw std::invalid_argument("Sparse native node alias lacks a full native incidence proof.");
            for (const auto& [gid, node] : index->nodes)
                if (node.native &&
                    (!authenticated_aliases.contains(*node.native) || index->native_nodes.at(*node.native) != gid))
                    throw std::invalid_argument("Sparse primary native node route lacks exact alias authentication.");
            for (auto& [gid, face] : index->faces)
            {
                const auto native = mesh->native_face_certificate(face.native);
                const auto sign = index->flux_signs.at({face.native, gid});
                const auto oriented = sign == 1 ? native : native->reversed();
                const auto actual_in_native =
                    face.native_is_second
                        ? face.override_geometry->translated(
                              PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, face.second_image_identity))
                        : face.override_geometry;
                auto native_vertices = oriented->vertex_identities(),
                     actual_vertices = actual_in_native->vertex_identities();
                std::sort(native_vertices.begin(), native_vertices.end());
                std::sort(actual_vertices.begin(), actual_vertices.end());
                face.cut_incidence = !(oriented->equivalent(*actual_in_native) && native_vertices == actual_vertices);
                if (!face.cut_incidence && (!face.native_is_second || face.second_image_identity.is_zero()))
                {
                    const auto nodes = mesh->topology(
                        face.native.region, [&](const auto& t) { return t.face_nodes(face.native.face); });
                    std::set<ID> expected, actual(face.override_nodes.begin(), face.override_nodes.end());
                    for (auto node : nodes)
                        expected.insert(index->native_nodes.at({face.native.region, node}));
                    if (expected != actual)
                        throw std::invalid_argument("Sparse native uncut node routes disagree with face proof.");
                    face.override_geometry.reset();
                    face.override_nodes.clear();
                    face.override_nodes.shrink_to_fit();
                    if (face.second_image_identity.is_zero())
                    {
                        face.second_node_gids.clear();
                        face.second_node_gids.shrink_to_fit();
                    }
                }
            }
            mesh->d_sparse_native = std::move(index);
        });
    return mesh;
}
int MultiRegionMesh::sparse_cell_owner(ID cell) const
{
    if (!d_sparse_native)
        throw std::invalid_argument("Sparse ownership requires a sparse native shard.");
    return d_sparse_native->cells.at(cell).owner;
}
PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> MultiRegionMesh::certified_sparse_face_geometry(ID face) const
{
    check_face_id(face);
    if(frozen_native())throw std::invalid_argument("Frozen native accepted-copy geometry has no exact adaptation certificate.");
    const auto& route = d_sparse_native->faces.at(face);
    if (route.override_geometry)
        return route.override_geometry;
    auto certificate = native_face_certificate(route.native);
    return native_flux_orientation(route.native, face) == 1 ? certificate : certificate->reversed();
}
std::shared_ptr<const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>>
MultiRegionMesh::certified_sparse_cell_geometry(ID cell) const
{
    check_cell_id(cell);
    if(frozen_native())throw std::invalid_argument("Frozen native accepted-copy geometry has no exact adaptation certificate.");
    const auto& route = d_sparse_native->cells.at(cell);
    if (!route.full)
        throw std::invalid_argument("Metric-only sparse neighbor has no cell certificate.");
    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
    for (auto native : topology(route.region, [&](const auto& t) { return t.cell_faces(route.native_cell); }))
    {
        const RegionFace address{route.region, native};
        for (auto face : d_sparse_native->logical_faces.at(address))
        {
            const auto& incidence = d_sparse_native->faces.at(face);
            auto geometry = certified_sparse_face_geometry(face);
            if (second_occurrence(incidence, address, cell))
                geometry = geometry->reversed()->translated(
                    PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, incidence.second_image_identity));
            faces.push_back(std::move(geometry));
        }
    }
    return PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces);
}
MultiRegionMesh::ResolvedFace MultiRegionMesh::resolve_sparse_face(ID face) const
{
    check_face_id(face);
    const auto& route = d_sparse_native->faces.at(face);
    ResolvedFace result;
    result.face = face;
    result.owner = route.first;
    result.neighbor = route.second;
    result.centroid = face_centroid(face);
    result.area_vector = face_normal(face) * face_area(face);
    result.owner_centroid = cell_centroid(route.first);
    result.owner_to_face = result.centroid - result.owner_centroid;
    if (result.interior())
    {
        result.neighbor_centroid = cell_centroid(route.second);
        const auto shift = vector(route.second_image);
        result.owner_to_neighbor = result.neighbor_centroid + shift - result.owner_centroid;
        result.neighbor_to_owner = result.owner_centroid - shift - result.neighbor_centroid;
        result.neighbor_to_face = result.centroid - shift - result.neighbor_centroid;
    }
    return result;
}
EntityRange<MultiRegionMesh::ID> MultiRegionMesh::sparse_cell_nodes(ID cell) const
{
    const auto [region, id] = native_cell(cell);
    bool changed = false;
    for (auto face : cell_faces(cell))
        changed = changed || d_sparse_native->faces.at(face).cut_incidence;
    if (!changed)
        return {this, cell, topology(region, [&](const auto& t) { return t.cell_nodes(id).size(); }),
            [](const void* p, size_t cell, size_t i) -> ID
            {
                const auto& mesh = *static_cast<const MultiRegionMesh*>(p);
                const auto [r, id] = mesh.native_cell(cell);
                const auto node = mesh.topology(r, [&](const auto& t) { return t.cell_nodes(id)[i]; });
                return mesh.d_sparse_native->native_nodes.at({r, node});
            }};
    // Only tiled cells need temporary first-occurrence uniqueness. Traverse
    // native face addresses so a self-periodic face retains both physical
    // occurrences; no cell-face or cell-node CSR is retained.
    const auto physical_nodes = [&](ID requested)
    {
        const auto [r, native_cell] = this->native_cell(requested);
        std::vector<ID> result;
        std::set<ID> seen;
        for (auto native : topology(r, [&](const auto& t) { return t.cell_faces(native_cell); }))
        {
            const RegionFace address{r, native};
            for (auto logical : d_sparse_native->logical_faces.at(address))
            {
                const auto& face = d_sparse_native->faces.at(logical);
                if (second_occurrence(face, address, requested) && !face.second_node_gids.empty())
                {
                    for (auto node : face.second_node_gids)
                        if (seen.insert(node).second)
                            result.push_back(node);
                }
                else
                    for (auto node : face_nodes(logical))
                        if (seen.insert(node).second)
                            result.push_back(node);
            }
        }
        return result;
    };
    const auto count = physical_nodes(cell).size();
    return {this, cell, count, [](const void* p, size_t cell, size_t index) -> ID
        {
            const auto& mesh = *static_cast<const MultiRegionMesh*>(p);
            const auto [r, native_cell] = mesh.native_cell(cell);
            std::set<ID> seen;
            for (auto native : mesh.topology(r, [&](const auto& t) { return t.cell_faces(native_cell); }))
            {
                const RegionFace address{r, native};
                for (auto logical : mesh.d_sparse_native->logical_faces.at(address))
                {
                    const auto& face = mesh.d_sparse_native->faces.at(logical);
                    if (second_occurrence(face, address, cell) && !face.second_node_gids.empty())
                    {
                        for (auto node : face.second_node_gids)
                            if (seen.insert(node).second && !index--)
                                return node;
                    }
                    else
                        for (auto node : mesh.face_nodes(logical))
                            if (seen.insert(node).second && !index--)
                                return node;
                }
            }
            throw std::out_of_range("Sparse physical cell-node range index out of bounds.");
        }};
}
EntityRange<MultiRegionMesh::ID> MultiRegionMesh::boundary_face_batch_impl(int boundary) const
{
    std::size_t count = 0;
    if (d_sparse_native)
        for (const auto& [id, face] : d_sparse_native->faces)
            count += face.boundary == boundary;
    else
        for (ID face = 0; face < num_faces(); ++face)
            count += boundary_id(face) == boundary;
    // Preserve independent by-value ranges. Sparse enumeration touches only
    // locally routed faces, never the complete global ID interval.
    return {this, static_cast<std::size_t>(boundary), count, [](const void* p, size_t key, size_t index) -> ID
        {
            const auto& mesh = *static_cast<const MultiRegionMesh*>(p);
            const int boundary = static_cast<int>(key);
            if (mesh.d_sparse_native)
            {
                for (const auto& [id, face] : mesh.d_sparse_native->faces)
                    if (face.boundary == boundary)
                    {
                        if (!index--)
                            return id;
                    }
            }
            else
                for (ID face = 0; face < mesh.num_faces(); ++face)
                    if (mesh.boundary_id(face) == boundary)
                    {
                        if (!index--)
                            return face;
                    }
            throw std::out_of_range("Boundary face range index out of bounds.");
        }};
}
real_t MultiRegionMesh::native_flux_orientation(RegionFace native, ID logical) const
{
    if (d_sparse_native)
        return d_sparse_native->flux_signs.at({native, logical});
    return native_face(logical) == native ? 1.0 : -1.0;
}
} // namespace SimpleFluid::Meshes
