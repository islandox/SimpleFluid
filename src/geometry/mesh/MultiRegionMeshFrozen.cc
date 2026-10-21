/** @file MultiRegionMeshFrozen.cc
 * @brief Separate accepted-native copy factory without convex or planar admission claims.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/FrozenNativeDetails.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include <algorithm>
#include <cmath>
namespace SimpleFluid::Meshes
{
namespace
{
using ID = std::uint64_t;
GeometryPoint coordinates(MeshUtils::Vec3 p)
{
    return {p.x, p.y, p.z};
}
bool finite(GeometryPoint p)
{
    return std::all_of(p.begin(), p.end(), [](double x) { return std::isfinite(x); });
}
bool same(const ExplicitRegionProvider::Cell& a, const ExplicitRegionProvider::Cell& b)
{
    return a.id == b.id && a.type == b.type && coordinates(a.centroid) == coordinates(b.centroid) &&
           a.volume == b.volume && (a.faces.empty() || b.faces.empty() || (a.faces == b.faces && a.nodes == b.nodes));
}
bool same(const ExplicitRegionProvider::Face& a, const ExplicitRegionProvider::Face& b)
{
    return a.id == b.id && a.owner == b.owner && a.neighbor == b.neighbor && a.nodes == b.nodes &&
           coordinates(a.centroid) == coordinates(b.centroid) &&
           coordinates(a.area_vector) == coordinates(b.area_vector) && a.boundary == b.boundary;
}
struct Parts
{
    std::optional<ID> node_prefix;
    std::shared_ptr<const NativeFragmentDescriptor> procedural;
    RegionLayout layout{};
    std::map<ID, ExplicitRegionProvider::Cell> cells;
    std::map<ID, ExplicitRegionProvider::Face> faces;
    std::map<ID, ExplicitRegionProvider::Node> nodes;
    std::map<int, std::string> boundaries;
};
} // namespace
bool MultiRegionMesh::frozen_native() const noexcept
{
    return d_sparse_native && std::holds_alternative<FrozenSourceCopyProof>(d_sparse_native->provenance);
}
std::shared_ptr<MultiRegionMesh> MultiRegionMesh::from_frozen_native(
    FrozenNativeMeshInput input, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    if (comm.is_null())
        throw std::invalid_argument("Frozen native publication requires communicator.");
    std::shared_ptr<MultiRegionMesh> mesh;
    std::shared_ptr<SparseNativeIndex> index;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (!input.data_ || !input.data_->root)
                throw std::invalid_argument("Missing opaque accepted-native copy witness.");
            const auto& root = *input.data_->root;
            if (!root.uid || root.observation.geometry == std::numeric_limits<ID>::max() ||
                root.observation.communicator != comm.getRawPtr() || !std::isfinite(root.frame[0]) ||
                !std::isfinite(root.frame[1]) || root.frame[0] >= root.frame[1] || root.schema.empty())
                throw std::invalid_argument("Invalid frozen source observation/frame.");
            mesh = std::shared_ptr<MultiRegionMesh>(new MultiRegionMesh(PartitionTag{}));
            index = std::make_shared<SparseNativeIndex>();
            index->provenance = FrozenSourceCopyProof{input.data_->root};
        });
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            const auto& data = *input.data_;
            const auto& root = *data.root;
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{data.byte_budget};
            budget.add(sizeof(*mesh) + sizeof(*index) + root.schema.size());
            index->common_signature.assign(root.schema.begin(), root.schema.end());
            index->boundaries = root.boundaries;
            std::map<ID, Parts> parts;
            std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> seen_parts;
            std::set<const ::SimpleFluid::Meshes::AMR::FrozenPart*> seen_addresses;
            auto add_part = [&](const std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenPart>& p)
            {
                if (!p)
                    throw std::invalid_argument("Frozen route lacks copied native provider.");
                if (!seen_addresses.insert(p.get()).second)
                    return;
                auto encoded = ::SimpleFluid::Meshes::AMR::encode_frozen_part(*p);
                if (!seen_parts.insert(encoded).second)
                    return;
                budget.add(encoded.size() + sizeof(Parts));
                auto& merged = parts[p->source];
                if (merged.node_prefix && *merged.node_prefix != p->node_prefix)
                    throw std::invalid_argument("Frozen native node prefixes disagree.");
                merged.node_prefix = p->node_prefix;
                if (p->procedural)
                {
                    if (!merged.cells.empty() || !merged.faces.empty() || !merged.nodes.empty())
                        throw std::invalid_argument("Frozen provider changed family.");
                    if (merged.procedural && encode_native_fragment_descriptor(*merged.procedural) !=
                                                 encode_native_fragment_descriptor(*p->procedural))
                        throw std::invalid_argument("Frozen procedural provider descriptions disagree.");
                    merged.procedural = p->procedural;
                    return;
                }
                if (merged.procedural)
                    throw std::invalid_argument("Frozen provider changed family.");
                if (!merged.cells.empty() || !merged.faces.empty() || !merged.nodes.empty())
                {
                    if (merged.layout.cells != p->layout.cells || merged.layout.faces != p->layout.faces ||
                        merged.layout.nodes != p->layout.nodes || merged.boundaries != p->boundaries)
                        throw std::invalid_argument("Frozen explicit source layouts disagree.");
                }
                merged.layout = p->layout;
                merged.boundaries = p->boundaries;
                for (const auto& c : p->cells)
                {
                    auto [it, added] = merged.cells.emplace(c.id, c);
                    if (!added)
                    {
                        if (!same(it->second, c))
                            throw std::invalid_argument("Frozen explicit cell copies disagree.");
                        if (it->second.faces.empty() && !c.faces.empty())
                            it->second = c;
                    }
                }
                for (const auto& f : p->faces)
                {
                    auto [it, added] = merged.faces.emplace(f.id, f);
                    if (!added && !same(it->second, f))
                        throw std::invalid_argument("Frozen explicit face copies disagree.");
                }
                for (const auto& n : p->nodes)
                {
                    auto [it, added] = merged.nodes.emplace(n.id, n);
                    if (!added && coordinates(it->second.point) != coordinates(n.point))
                        throw std::invalid_argument("Frozen explicit node copies disagree.");
                }
            };
            for (const auto& b : data.full)
            {
                if (!b || b->source_uid != root.uid || !b->part || b->metric || b->source_gid >= root.global_cells)
                    throw std::invalid_argument("Frozen full cell lacks original accepted-copy provenance.");
                add_part(b->part);
                for (const auto& row : b->rows)
                    for (const auto& [face, sign] : row.faces)
                    {
                        add_part(face->part);
                    }
            }
            std::map<ID, std::size_t> regions;
            NativeFragmentCache cache;
            for (auto& [source, p] : parts)
            {
                regions.emplace(source, mesh->d_regions.size());
                if (p.procedural)
                    mesh->d_regions.push_back(materialize_native_fragment(*p.procedural, cache));
                else
                {
                    std::vector<ExplicitRegionProvider::Cell> cells;
                    std::vector<ExplicitRegionProvider::Face> faces;
                    std::vector<ExplicitRegionProvider::Node> nodes;
                    for (auto& [id, c] : p.cells)
                        cells.push_back(std::move(c));
                    for (auto& [id, f] : p.faces)
                        faces.push_back(std::move(f));
                    for (auto& [id, n] : p.nodes)
                        nodes.push_back(std::move(n));
                    auto provider = std::make_shared<const ExplicitRegionProvider>(
                        p.layout, std::move(cells), std::move(faces), std::move(nodes), p.boundaries);
                    mesh->d_regions.emplace_back(
                        DistributedExplicitRegion("frozen_" + std::to_string(source), provider, provider));
                }
            }
            mesh->d_comm = comm;
            mesh->d_partition_resident = true;
            mesh->d_num_cells = root.global_cells;
            mesh->d_num_faces = root.global_faces;
            mesh->d_num_nodes = root.global_nodes;
            mesh->d_num_local_cells = mesh->d_num_owned_cells = mesh->d_num_owned_faces = 0;
            mesh->d_indexer = {root.global_cells, root.global_faces, root.global_nodes};
            mesh->d_reference_axial_edges = mesh->d_axial_edges = {root.frame[0], root.frame[1]};
            mesh->d_region_interfaces.resize(mesh->d_regions.size());
            mesh->d_cells.resize(mesh->d_regions.size() + 1);
            mesh->d_faces.resize(mesh->d_regions.size() + 1);
            mesh->d_nodes.resize(mesh->d_regions.size() + 1);
            mesh->d_native_faces.resize(mesh->d_regions.size() + 1);
            for (std::size_t r = 0; r < mesh->d_regions.size(); ++r)
            {
                const auto count = mesh->region_layout(r).faces;
                if (count > std::numeric_limits<ID>::max() - mesh->d_native_faces[r])
                    throw std::overflow_error("Frozen native address overflow.");
                mesh->d_native_faces[r + 1] = mesh->d_native_faces[r] + count;
            }
            mesh->validate_static();
            std::map<ID, ID> node_intervals;
            for (const auto& [source, region] : regions)
            {
                const auto prefix = *parts.at(source).node_prefix;
                const auto count = mesh->region_layout(region).nodes;
                if (prefix > root.global_nodes || count > root.global_nodes - prefix)
                    throw std::invalid_argument("Frozen native node prefix exceeds source counts.");
                auto next = node_intervals.lower_bound(prefix);
                if ((next != node_intervals.end() && prefix + count > next->first) ||
                    (next != node_intervals.begin() && std::prev(next)->second > prefix))
                    throw std::invalid_argument("Frozen provider node-prefix ranges overlap.");
                node_intervals.emplace(prefix, prefix + count);
            }
            std::set<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> full_keys;
            for (const auto& b : data.full)
                if (!full_keys.insert(b->key).second)
                    throw std::invalid_argument("Duplicate full frozen cell.");
            for (const auto& [key, b] : data.metrics)
            {
                const auto binding = data.bindings.at(key);
                if (b->source_uid != root.uid || binding.owner < 0 || binding.owner >= comm->getSize() ||
                    binding.revision != b->revision || binding.address >= root.global_cells)
                    throw std::invalid_argument("Frozen native metric binding lacks an authenticated source.");
                SparseNativeCell cell;
                cell.gid = binding.address;
                cell.owner = binding.owner;
                cell.full = full_keys.contains(key);
                if (cell.full)
                {
                    cell.region = regions.at(b->part->source);
                    cell.native_cell = b->native;
                    if (cell.native_cell >= mesh->region_layout(cell.region).cells ||
                        !index->native_cells.emplace(std::pair{cell.region, cell.native_cell}, cell.gid).second)
                        throw std::invalid_argument("Frozen native cell ordinal is invalid or duplicated.");
                    auto volume = mesh->axial_scale() * mesh->geometry(cell.region, [&](const auto& g)
                                                            { return g.cell_volume(cell.native_cell); });
                    auto center = mesh->transformed_point(
                        mesh->geometry(cell.region, [&](const auto& g) { return g.cell_centroid(cell.native_cell); }));
                    if (b->metric)
                        throw std::invalid_argument("Frozen full native cell retains redundant metric scalars.");
                    cell.volume = volume;
                    cell.centroid = coordinates(center);
                    ++mesh->d_num_local_cells;
                    if (cell.owner == comm->getRank())
                        ++mesh->d_num_owned_cells;
                }
                else
                {
                    if (cell.owner == comm->getRank() || b->part || !b->metric)
                        throw std::invalid_argument("Frozen outer metric reply has invalid native residency.");
                    cell.volume = b->metric->volume;
                    cell.centroid = b->metric->center;
                }
                if (!(cell.volume > 0) || !std::isfinite(cell.volume) || !finite(cell.centroid))
                    throw std::invalid_argument("Invalid frozen native cell metric.");
                if (cell.full)
                {
                    cell.volume = 0;
                    cell.centroid = {};
                }
                if (!index->cells.emplace(cell.gid, cell).second)
                    throw std::invalid_argument("Duplicate frozen current cell GID.");
            }
            auto add_node = [&](const std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenPart>& part, ID native)
            {
                const auto r = regions.at(part->source);
                if (native >= mesh->region_layout(r).nodes)
                    throw std::invalid_argument("Frozen native node ordinal is invalid.");
                const auto gid = part->node_prefix + native;
                const auto coordinate = mesh->transformed_point(
                    mesh->geometry(r, [&](const auto& g) { return g.node_coordinates(native); }));
                if (gid >= root.global_nodes || !finite(coordinates(coordinate)))
                    throw std::invalid_argument("Invalid derived frozen physical node.");
                auto [it, added] = index->nodes.emplace(gid, SparseNativeNode{gid, std::pair{r, native}, {}});
                if (!added && it->second.native != std::optional<std::pair<std::size_t, ID>>(std::pair{r, native}))
                    throw std::invalid_argument("Frozen source node prefix has conflicting routes.");
                auto [alias, inserted] = index->native_nodes.emplace(std::pair{r, native}, gid);
                if (!inserted && alias->second != gid)
                    throw std::invalid_argument("Frozen native node alias conflicts.");
            };
            for (const auto& b : data.full)
            {
                const auto r = regions.at(b->part->source);
                for (auto native : mesh->topology(r, [&](const auto& t) { return t.cell_nodes(b->native); }))
                    add_node(b->part, native);
                std::vector<ID> actual_faces;
                for (auto f : mesh->topology(r, [&](const auto& t) { return t.cell_faces(b->native); }))
                    actual_faces.push_back(f);
                if (actual_faces.size() != b->rows.size())
                    throw std::invalid_argument("Frozen cell face incidence differs from source.");
                for (std::size_t i = 0; i < b->rows.size(); ++i)
                {
                    const auto& row = b->rows[i];
                    if (row.native != actual_faces[i] || row.faces.empty())
                        throw std::invalid_argument("Frozen native face row differs from source.");
                    const RegionFace address{r, row.native};
                    for (const auto& [f, sign] : row.faces)
                    {
                        if (!f || f->gid >= root.global_faces || !f->part ||
                            (f->first != b->key && (!f->second || *f->second != b->key)) ||
                            (f->second && *f->second == f->first) || (sign != 1 && sign != -1))
                            throw std::invalid_argument("Frozen canonical incidence is invalid or self-periodic.");
                        const auto primary = regions.at(f->part->source);
                        if (f->native >= mesh->region_layout(primary).faces)
                            throw std::invalid_argument("Frozen canonical native face ordinal is invalid.");
                        auto center = mesh->transformed_point(
                            mesh->geometry(primary, [&](const auto& g) { return g.face_centroid(f->native); }));
                        auto vector = mesh->transformed_area_vector(
                            mesh->geometry(primary, [&](const auto& g) { return g.face_area_vector(f->native); }));
                        auto area = mesh->area_magnitude(vector);
                        if (!finite(coordinates(center)) || !(area > 0) || !std::isfinite(area) ||
                            !finite(coordinates(vector / area)) || !finite(f->second_image))
                            throw std::invalid_argument("Invalid derived frozen canonical face metric.");
                        SparseNativeFace route;
                        route.gid = f->gid;
                        route.first = data.bindings.at(f->first).address;
                        if (f->second)
                            route.second = data.bindings.at(*f->second).address;
                        route.native = {primary, f->native};
                        route.second_image = f->second_image;
                        route.boundary = f->boundary;
                        route.cut_incidence = false;
                        for (auto native :
                            mesh->topology(primary, [&](const auto& t) { return t.face_nodes(f->native); }))
                            add_node(f->part, native);
                        auto [entry, added] = index->faces.emplace(f->gid, route);
                        if (!added &&
                            (entry->second.first != route.first || entry->second.second != route.second ||
                                entry->second.native != route.native || entry->second.boundary != route.boundary ||
                                entry->second.second_image != route.second_image))
                            throw std::invalid_argument("Frozen canonical face copies disagree.");
                        auto& list = index->logical_faces[address];
                        if (std::find(list.begin(), list.end(), f->gid) == list.end())
                            list.push_back(f->gid);
                        auto [orientation, inserted] = index->flux_signs.emplace(std::pair{address, f->gid}, sign);
                        if (!inserted && orientation->second != sign)
                            throw std::invalid_argument("Frozen signed native occurrences disagree.");
                    }
                }
            }
            // Canonical geometry may belong to a provider with no local full cell.
            // Its native nodes still need source-authenticated aliases for face queries.
            for (const auto& [gid, n] : index->nodes)
                if (n.native)
                {
                    auto [it, added] = index->native_nodes.emplace(*n.native, gid);
                    if (!added && it->second != gid)
                        throw std::invalid_argument("Frozen primary native node alias conflicts.");
                }
            for (const auto& [gid, face] : index->faces)
            {
                if (face.boundary != invalid_boundary_id && !index->boundaries.contains(face.boundary))
                    throw std::invalid_argument("Frozen physical boundary schema differs.");
                if (index->cells.at(face.first).owner == comm->getRank())
                    ++mesh->d_num_owned_faces;
                if (!index->flux_signs.contains({face.native, gid}))
                    index->flux_signs.emplace(std::pair{face.native, gid}, 1);
            }
            mesh->d_sparse_native = index;
        });
    return mesh;
}
} // namespace SimpleFluid::Meshes
