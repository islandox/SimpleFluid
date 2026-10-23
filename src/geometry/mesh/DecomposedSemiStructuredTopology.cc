/** @file DecomposedSemiStructuredTopology.cc
 * @brief Factored source/local topology translation for exact XY/Z partition halos.
 */
#include "geometry/mesh/DecomposedSemiStructuredTopology.hh"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
DecomposedSemiStructuredTopology::DecomposedSemiStructuredTopology(std::shared_ptr<const SemiStructMeshTopo> source,
    std::span<const size_t> xy_owners, std::span<const size_t> xy_order, size_t rank, size_t xy_parts, size_t z_parts,
    size_t halo_depth)
    : d_source(std::move(source))
{
    if (!d_source || xy_parts == 0 || z_parts == 0 || halo_depth == 0)
        throw std::invalid_argument("Invalid decomposed XY/Z topology parameters.");
    const auto& indexer = d_source->indexer();
    const size_t nx = indexer.num_cells_per_layer, nz = indexer.num_layers;
    if (xy_owners.size() != nx || xy_order.size() != nx || rank / xy_parts >= z_parts)
        throw std::invalid_argument("Decomposed XY/Z topology ownership dimensions differ.");
    std::vector<unsigned char> visited(nx);
    for (const auto cell : xy_order)
    {
        if (cell >= nx || visited[cell] || xy_owners[cell] >= xy_parts)
            throw std::invalid_argument("Decomposed XY order or partition label is invalid.");
        visited[cell] = 1;
    }
    const size_t xy_rank = rank % xy_parts, z_rank = rank / xy_parts;
    const size_t z_begin = nz / z_parts * z_rank + std::min(nz % z_parts, z_rank);
    const size_t z_count = nz / z_parts + (z_rank < nz % z_parts ? 1 : 0), z_end = z_begin + z_count;
    std::unordered_map<Ordinal, size_t> distances;
    std::vector<Ordinal> frontier;
    for (const auto original : xy_order)
        if (xy_owners[original] == xy_rank)
        {
            distances.emplace(static_cast<Ordinal>(original), 0);
            frontier.push_back(static_cast<Ordinal>(original));
        }
    if (frontier.empty() || z_count == 0)
        return;
    for (size_t distance = 0; distance < halo_depth && !frontier.empty(); ++distance)
    {
        std::vector<Ordinal> next;
        for (const auto cell : frontier)
            for (const auto side : d_source->cell_side_faces(cell))
            {
                const auto& edge = d_source->side_face(side);
                const auto other = edge.owner == cell ? edge.neighbor : edge.owner;
                if (other != SemiStructMeshTopo::invalid_ordinal && distances.emplace(other, distance + 1).second)
                    next.push_back(other);
            }
        frontier = std::move(next);
    }
    size_t maximum_distance = 0;
    for (const auto original : xy_order)
    {
        const auto found = distances.find(static_cast<Ordinal>(original));
        if (found == distances.end())
            continue;
        d_cell_position.emplace(static_cast<Ordinal>(original), d_xy_cells.size());
        d_xy_cells.push_back(static_cast<Ordinal>(original));
        d_cell_distance.push_back(found->second);
        maximum_distance = std::max(maximum_distance, found->second);
    }
    std::unordered_map<Ordinal, size_t> edge_distance, node_distance;
    for (const auto cell : d_xy_cells)
    {
        const auto distance = distances.at(cell);
        for (const auto edge : d_source->cell_side_faces(cell))
        {
            const auto [position, inserted] = edge_distance.emplace(edge, distance);
            if (!inserted)
                position->second = std::min(position->second, distance);
            for (const auto node : d_source->side_face(edge).nodes)
            {
                const auto [entry, new_node] = node_distance.emplace(node, distance);
                if (!new_node)
                    entry->second = std::min(entry->second, distance);
            }
        }
    }
    for (const auto& [edge, distance] : edge_distance)
    {
        static_cast<void>(distance);
        d_xy_sides.push_back(edge);
    }
    for (const auto& [node, distance] : node_distance)
    {
        static_cast<void>(distance);
        d_xy_nodes.push_back(node);
    }
    std::sort(d_xy_sides.begin(), d_xy_sides.end());
    std::sort(d_xy_nodes.begin(), d_xy_nodes.end());
    for (const auto edge : d_xy_sides)
    {
        d_side_position.emplace(edge, d_side_distance.size());
        d_side_distance.push_back(edge_distance.at(edge));
        const auto found = distances.find(d_source->side_face(edge).owner);
        d_side_owned.push_back(found != distances.end() && found->second == 0);
    }
    for (const auto node : d_xy_nodes)
    {
        d_node_position.emplace(node, d_node_distance.size());
        d_node_distance.push_back(node_distance.at(node));
    }
    const auto layer_radius = [&](size_t layer)
    {
        if (layer >= nz)
            return invalid_local_id();
        size_t distance = 0;
        if (layer < z_begin || layer >= z_end)
        {
            const size_t first = layer > z_begin ? layer - z_begin : z_begin - layer;
            const size_t last = layer > z_end - 1 ? layer - (z_end - 1) : (z_end - 1) - layer;
            distance = std::min(first, last);
            if (indexer.axial_periodic)
                distance = std::min({distance, nz - first, nz - last});
        }
        return distance > halo_depth ? invalid_local_id() : std::min(halo_depth - distance, maximum_distance);
    };
    const auto plane_radius = [&](size_t plane)
    {
        size_t lower = invalid_local_id(), upper = invalid_local_id();
        if (plane > 0)
            lower = layer_radius(plane - 1);
        else if (indexer.axial_periodic)
            lower = layer_radius(nz - 1);
        if (plane < nz)
            upper = layer_radius(plane);
        if (lower == invalid_local_id())
            return upper;
        if (upper == invalid_local_id())
            return lower;
        return std::max(lower, upper);
    };
    const auto append = [&](std::vector<Band>& bands, size_t z, size_t view_id, unsigned orientation, bool owned)
    {
        const auto count = xy_view_size(view_id);
        if (count == 0)
            return;
        if (!bands.empty() && bands.back().z_end == z && bands.back().view == view_id &&
            bands.back().orientation == orientation && bands.back().owned == owned)
            ++bands.back().z_end;
        else
            bands.push_back({z, z + 1, view_id, 0, count, orientation, owned});
    };
    std::array<std::vector<Band>, 2> cells;
    std::array<std::vector<Band>, 4> faces;
    std::array<std::vector<Band>, 1> nodes;
    const auto owned_xy = view(Kind::Cell, 0);
    const auto owned_sides = view(Kind::Side, 0, 2);
    for (size_t layer = 0; layer < nz; ++layer)
    {
        const auto radius = layer_radius(layer);
        if (radius == invalid_local_id())
            continue;
        const bool owned_z = layer >= z_begin && layer < z_end;
        if (owned_z)
        {
            append(cells[0], layer, owned_xy, 0, true);
            append(faces[1], layer, owned_sides, SemiStructuredIndexer::SIDE, true);
        }
        append(cells[1], layer, view(Kind::Cell, radius, owned_z ? 1 : 0), 0, false);
        append(faces[3], layer, view(Kind::Side, radius, owned_z ? 1 : 0), SemiStructuredIndexer::SIDE, false);
    }
    for (size_t plane = 0; plane < indexer.num_node_layers; ++plane)
    {
        const auto radius = plane_radius(plane);
        if (radius == invalid_local_id())
            continue;
        const auto owner = d_source->owner_cell({0, static_cast<Ordinal>(plane), SemiStructuredIndexer::AXIAL});
        const bool owned_z = owner.k >= z_begin && owner.k < z_end;
        if (owned_z)
            append(faces[0], plane, owned_xy, SemiStructuredIndexer::AXIAL, true);
        append(faces[2], plane, view(Kind::Cell, radius, owned_z ? 1 : 0), SemiStructuredIndexer::AXIAL, false);
        append(nodes[0], plane, view(Kind::Node, radius), 0, false);
    }
    const auto finish = [](const auto& temporary, auto& bands, auto& groups, size_t& owned, size_t& total)
    {
        for (size_t group = 0; group < temporary.size(); ++group)
        {
            groups[group].begin = bands.size();
            for (auto band : temporary[group])
            {
                band.offset = total;
                total += band.count();
                if (band.owned)
                    owned += band.count();
                bands.push_back(band);
            }
            groups[group].end = bands.size();
        }
    };
    finish(cells, d_cells, d_cell_groups, d_owned_cells, d_local_cells);
    finish(faces, d_faces, d_face_groups, d_owned_faces, d_local_faces);
    size_t unused_owned = 0;
    finish(nodes, d_nodes, d_node_groups, unused_owned, d_local_nodes);
}

size_t DecomposedSemiStructuredTopology::view(Kind kind, size_t radius, unsigned ownership)
{
    for (size_t index = 0; index < d_views.size(); ++index)
        if (d_views[index].kind == kind && d_views[index].radius == radius && d_views[index].ownership == ownership)
            return index;
    View result{kind, radius, ownership, {}};
    const auto& distances = kind == Kind::Cell   ? d_cell_distance
                            : kind == Kind::Side ? d_side_distance
                                                 : d_node_distance;
    for (size_t position = 0; position < distances.size(); ++position)
    {
        const bool owned = kind == Kind::Side ? d_side_owned[position] : distances[position] == 0;
        if (distances[position] <= radius && (ownership == 0 || (ownership == 1 ? !owned : owned)))
            result.positions.push_back(position);
    }
    d_views.push_back(std::move(result));
    return d_views.size() - 1;
}
DecomposedSemiStructuredTopology::Ordinal DecomposedSemiStructuredTopology::source_entity(
    size_t view_id, size_t entry) const
{
    const auto& selected = d_views.at(view_id);
    const auto position = selected.positions.at(entry);
    return selected.kind == Kind::Cell   ? d_xy_cells[position]
           : selected.kind == Kind::Side ? d_xy_sides[position]
                                         : d_xy_nodes[position];
}
size_t DecomposedSemiStructuredTopology::position_in_view(size_t view_id, Ordinal source) const noexcept
{
    const auto& selected = d_views[view_id];
    const auto& positions = selected.kind == Kind::Cell   ? d_cell_position
                            : selected.kind == Kind::Side ? d_side_position
                                                          : d_node_position;
    const auto found = positions.find(source);
    if (found == positions.end())
        return invalid_local_id();
    const auto entry = std::lower_bound(selected.positions.begin(), selected.positions.end(), found->second);
    return entry == selected.positions.end() || *entry != found->second
               ? invalid_local_id()
               : static_cast<size_t>(entry - selected.positions.begin());
}
const DecomposedSemiStructuredTopology::Band& DecomposedSemiStructuredTopology::band_for_local(
    const std::vector<Band>& bands, size_t local)
{
    const auto entry = std::upper_bound(
        bands.begin(), bands.end(), local, [](size_t id, const Band& band) { return id < band.offset; });
    if (entry == bands.begin())
        throw std::out_of_range("Decomposed topology local ID is absent.");
    const auto& band = *(entry - 1);
    if (local - band.offset >= band.count())
        throw std::out_of_range("Decomposed topology local ID is out of range.");
    return band;
}
size_t DecomposedSemiStructuredTopology::lookup(const std::vector<Band>& bands, std::span<const Group> groups,
    Ordinal source, size_t z, unsigned orientation) const noexcept
{
    for (const auto group : groups)
    {
        if (group.begin == group.end || bands[group.begin].orientation != orientation)
            continue;
        const auto first = bands.begin() + group.begin, last = bands.begin() + group.end;
        const auto next =
            std::upper_bound(first, last, z, [](size_t layer, const Band& band) { return layer < band.z_begin; });
        if (next == first)
            continue;
        const auto& band = *(next - 1);
        if (z >= band.z_end)
            continue;
        const auto xy = position_in_view(band.view, source);
        if (xy != invalid_local_id())
            return band.offset + (z - band.z_begin) * band.xy_size + xy;
    }
    return invalid_local_id();
}
auto DecomposedSemiStructuredTopology::source_cell(size_t local) const -> CellID
{
    const auto& band = band_for_local(d_cells, local);
    const auto entry = local - band.offset;
    return {source_entity(band.view, entry % band.xy_size), static_cast<Ordinal>(band.z_begin + entry / band.xy_size)};
}
auto DecomposedSemiStructuredTopology::source_face(size_t local) const -> FaceID
{
    const auto& band = band_for_local(d_faces, local);
    const auto entry = local - band.offset;
    return {source_entity(band.view, entry % band.xy_size), static_cast<Ordinal>(band.z_begin + entry / band.xy_size),
        band.orientation};
}
auto DecomposedSemiStructuredTopology::source_node(size_t local) const -> NodeID
{
    const auto& band = band_for_local(d_nodes, local);
    const auto entry = local - band.offset;
    return {source_entity(band.view, entry % band.xy_size), static_cast<Ordinal>(band.z_begin + entry / band.xy_size)};
}
size_t DecomposedSemiStructuredTopology::cell_local_id(CellID source) const noexcept
{
    return lookup(d_cells, d_cell_groups, source.ij, source.k, 0);
}
size_t DecomposedSemiStructuredTopology::face_local_id(FaceID source) const noexcept
{
    return lookup(d_faces, d_face_groups, source.ij, source.k, source.orientation);
}
size_t DecomposedSemiStructuredTopology::node_local_id(NodeID source) const noexcept
{
    return lookup(d_nodes, d_node_groups, source.ij, source.k, 0);
}
size_t DecomposedSemiStructuredTopology::owner_cell(size_t face) const
{
    return cell_local_id(d_source->owner_cell(source_face(face)));
}
size_t DecomposedSemiStructuredTopology::neighbor_cell(size_t face) const
{
    return cell_local_id(d_source->neighbor_cell(source_face(face)));
}
int DecomposedSemiStructuredTopology::boundary_id(size_t face) const
{
    return d_source->boundary_id(source_face(face));
}
EntityRange<size_t> DecomposedSemiStructuredTopology::cell_faces(size_t cell) const
{
    return {this, cell, d_source->cell_faces(source_cell(cell)).size(),
        [](const void* source, size_t local, size_t entry)
        {
            const auto& topology = *static_cast<const DecomposedSemiStructuredTopology*>(source);
            const auto face = topology.d_source->cell_faces(topology.source_cell(local))[entry];
            return topology.face_local_id(face);
        }};
}
EntityRange<DecomposedSemiStructuredTopology::Ordinal> DecomposedSemiStructuredTopology::view_source_ids(
    size_t view_id) const
{
    return {this, view_id, xy_view_size(view_id), [](const void* source, size_t view, size_t entry)
        { return static_cast<const DecomposedSemiStructuredTopology*>(source)->source_entity(view, entry); }};
}
size_t DecomposedSemiStructuredTopology::storage_bytes() const noexcept
{
    size_t bytes =
        sizeof(*this) + (d_xy_cells.capacity() + d_xy_sides.capacity() + d_xy_nodes.capacity()) * sizeof(Ordinal) +
        (d_cell_distance.capacity() + d_side_distance.capacity() + d_node_distance.capacity()) * sizeof(size_t) +
        d_side_owned.capacity() * sizeof(unsigned char) + d_views.capacity() * sizeof(View) +
        (d_cells.capacity() + d_faces.capacity() + d_nodes.capacity()) * sizeof(Band);
    for (const auto& view : d_views)
        bytes += view.positions.capacity() * sizeof(size_t);
    return bytes;
}
size_t DecomposedSemiStructuredTopology::lookup_storage_estimate() const noexcept
{
    size_t bytes = 0;
    for (const auto* map : {&d_cell_position, &d_side_position, &d_node_position})
        bytes += map->bucket_count() * sizeof(void*) +
                 map->size() * (sizeof(std::pair<const Ordinal, size_t>) + 2 * sizeof(void*));
    return bytes;
}
} // namespace SimpleFluid::Meshes
