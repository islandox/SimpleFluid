/** @file FilteredSurface.cc
 * @brief Conservative binary graph diffusion and continuous tetrahedral isosurfaces.
 */
#include "FVM/details/GeneralVoF/FilteredSurface.hh"
#include "FVM/details/GeneralVoF/FilteredSurfaceGeometry.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include <bit>
#include <iomanip>

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
using namespace filtered_surface_kernel;
int phase(const Cell& c)
{
    // Geometry declarations have precedence over permitted physical traces.
    if (c.declaration == GeneralVoFPhaseGeometry::PureGas)
        return 1;
    if (c.declaration == GeneralVoFPhaseGeometry::PureCarrier)
        return -1;
    if (c.retained_interface)
    {
        if (c.retained_interface->pure_gas)
            return 1;
        if (c.retained_interface->pure_carrier)
            return -1;
        return 0;
    }
    if (c.phase.carrier_volume == 0 && c.phase.gas_volume > 0)
        return 1;
    if (c.phase.gas_volume == 0 && c.phase.carrier_volume > 0)
        return -1;
    return 0;
}
struct Edge
{
    std::size_t a = 0, b = 0;
    long double conductance = 0;
};
struct MeshCell
{
    const Cell* source = nullptr;
    long double volume = 0, h = 0;
    Point spans{};
    double fit_scale = 0;
    int label = -1;
    std::vector<std::size_t> neighbors, faces;
    bool frontier = false, physical_boundary = false;
};
template<class Value> void reserve_next(std::vector<Value>& values, wire::Budget& budget)
{
    if (values.size() < values.capacity())
        return;
    const auto old = values.capacity();
    if (old > std::numeric_limits<std::size_t>::max() / 2)
        reject(Reason::IncompleteSupport, "Filtered geometry capacity overflows its finite resource policy.");
    const auto next = old ? 2 * old : 1;
    if (next > std::numeric_limits<std::size_t>::max() / sizeof(Value))
        reject(Reason::IncompleteSupport, "Filtered geometry storage size overflows.");
    // Full new capacity is charged: old and new buffers coexist in reserve.
    budget.add(next * sizeof(Value));
    values.reserve(next);
}
template<class Value> void reserve_exact(std::vector<Value>& values, std::size_t count, wire::Budget& budget)
{
    if (count <= values.capacity())
        return;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value))
        reject(Reason::IncompleteSupport, "Filtered geometry storage size overflows.");
    budget.add(count * sizeof(Value));
    values.reserve(count);
}
std::size_t key_storage(const Key& key)
{
    return std::visit(
        [](const auto& ancestry) -> std::size_t
        {
            if constexpr (requires { ancestry.path; })
                return ancestry.path.size() * sizeof(std::uint64_t);
            else if constexpr (requires {
                                   ancestry.base.path;
                                   ancestry.axial.path;
                               })
                return (ancestry.base.path.size() + ancestry.axial.path.size()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
template<class Tree> void charge_tree_node(wire::Budget& budget)
{
    budget.add(sizeof(typename Tree::value_type) + 64);
}
template<class Set> bool tracked_insert(Set& values, const typename Set::value_type& value, wire::Budget& budget)
{
    if (values.contains(value))
        return false;
    charge_tree_node<Set>(budget);
    if constexpr (std::is_same_v<typename Set::value_type, Key>)
        budget.add(key_storage(value));
    values.insert(value);
    return true;
}
template<class Map> auto& tracked_row(Map& values, const typename Map::key_type& key, wire::Budget& budget)
{
    const auto found = values.find(key);
    if (found != values.end())
        return found->second;
    charge_tree_node<Map>(budget);
    if constexpr (std::is_same_v<typename Map::key_type, Key>)
        budget.add(key_storage(key));
    return values.try_emplace(key).first->second;
}
template<class Value> void tracked_push(std::vector<Value>& values, const Value& value, wire::Budget& budget)
{
    reserve_next(values, budget);
    if constexpr (std::is_same_v<Value, Key>)
        budget.add(key_storage(value));
    values.push_back(value);
}
template<class VertexKey>
void check_vertex_links(const std::map<VertexKey, std::vector<std::pair<std::size_t, std::size_t>>>& links,
    const char* message, wire::Budget& budget)
{
    for (const auto& [vertex, pairs] : links)
    {
        std::map<std::size_t, std::vector<std::size_t>> neighbors;
        for (const auto& [a, b] : pairs)
        {
            tracked_push(tracked_row(neighbors, a, budget), b, budget);
            tracked_push(tracked_row(neighbors, b, budget), a, budget);
        }
        for (const auto& [patch, adjacent] : neighbors)
            if (adjacent.size() != 2)
                reject(Reason::AmbiguousTopology, message);
        std::set<std::size_t> seen;
        std::vector<std::size_t> visit;
        tracked_insert(seen, neighbors.begin()->first, budget);
        reserve_exact(visit, neighbors.size(), budget);
        visit.push_back(neighbors.begin()->first);
        for (std::size_t j = 0; j < visit.size(); ++j)
            for (auto p : neighbors.at(visit[j]))
                if (tracked_insert(seen, p, budget))
                    visit.push_back(p);
        if (seen.size() != neighbors.size())
            reject(Reason::AmbiguousTopology, message);
    }
}
void published_metrics(GeneralVoFSurfacePatch& patch, const Tetra& tetra, long double iso)
{
    const auto fail = [&](const char* message)
    {
        std::ostringstream detail;
        detail << std::setprecision(std::numeric_limits<long double>::max_digits10) << message
               << " component=" << patch.component << " cell=" << patch.first_cell << " tetra=" << tetra.ordinal
               << " iso=" << iso << " q={";
        for (const auto& vertex : tetra.v)
            detail << vertex.q << ',';
        detail << "} wide={";
        Wide normal{};
        const auto cuts = section(tetra, iso, normal);
        for (const auto& cut : cuts)
            detail << '(' << cut.p[0] << ',' << cut.p[1] << ',' << cut.p[2] << ')';
        detail << "} published={";
        for (const auto& point : patch.vertices)
            detail << '(' << point[0] << ',' << point[1] << ',' << point[2] << ')';
        detail << '}';
        throw GeneralVoFSurfaceResolutionError(Reason::VolumeConstraint, detail.str());
    };
    const auto anchor = wide(patch.vertices.front());
    const auto original_normal = wide(patch.normal);
    Wide area_vector{}, centroid{};
    SimpleFluid::detail::CompensatedSum<> area;
    for (const auto& point : patch.vertices)
        for (unsigned k = 0; k < 3; ++k)
            if (!std::isfinite(point[k]))
                fail("Filtered polygon cannot publish finite binary64 vertices.");
    for (std::size_t j = 1; j + 1 < patch.vertices.size(); ++j)
    {
        const auto b = sub(wide(patch.vertices[j]), anchor);
        const auto c = sub(wide(patch.vertices[j + 1]), anchor);
        const auto triangle = cross(b, c);
        const auto triangle_area = norm(triangle) / 2;
        if (!(triangle_area > 0) || !std::isfinite(triangle_area) || !(dot(triangle, original_normal) > 0))
            fail("Filtered binary64 polygon collapses or reverses during publication.");
        area += triangle_area;
        for (unsigned k = 0; k < 3; ++k)
        {
            area_vector[k] += triangle[k] / 2;
            centroid[k] += triangle_area * (b[k] + c[k]) / 3;
        }
    }
    const auto area_norm = norm(area_vector);
    if (!(area.value() > 0) || !(area_norm > 0) || !std::isfinite(area_norm))
        fail("Filtered binary64 polygon has unresolved physical area.");
    patch.area = static_cast<double>(area.value());
    if (!(patch.area > 0) || !std::isfinite(patch.area))
        fail("Filtered binary64 polygon area cannot be represented.");
    for (unsigned k = 0; k < 3; ++k)
    {
        patch.normal[k] = static_cast<double>(area_vector[k] / area_norm);
        patch.centroid[k] = static_cast<double>(anchor[k] + centroid[k] / area.value());
        if (!std::isfinite(patch.normal[k]) || !std::isfinite(patch.centroid[k]) || !(patch.physical_spans[k] > 0) ||
            !std::isfinite(patch.physical_spans[k]))
            fail("Filtered binary64 polygon metrics cannot be represented.");
    }
    if (!(dot(wide(patch.normal), original_normal) > 0))
        fail("Filtered binary64 polygon normal lost its outward orientation.");
}
} // namespace

GeneralVoFSurfaceCatalog filtered_binary_surface(const GeneralVoF::Data& data, const GeneralVoFSurfaceOptions& options,
    const CapillaryNodeFans& imported_fans, std::size_t remaining_budget)
{
    if (imported_fans.source_control != data.control)
        throw std::invalid_argument("Filtered capillary node fans belong to a different General VoF source control.");
    if (data.options.arithmetic_mode != ArithmeticMode::Ordinary)
        throw std::invalid_argument("Filtered binary capillary geometry requires Ordinary arithmetic.");
    if (!(options.filter_width > 0) || !std::isfinite(options.filter_width))
        throw std::invalid_argument("Filtered capillary width must be finite and positive.");
    GeneralVoFSurfaceCatalog result;
    result.source_control = data.control;
    result.model = GeneralVoFCapillaryModel::FilteredBinary;
    wire::Budget budget{
        remaining_budget ? std::min(remaining_budget, data.options.byte_budget) : data.options.byte_budget};
    PLIC::GeometryAllocationLedger point_exports;
    std::size_t point_export_bytes = 0;
    std::vector<MeshCell> cells;
    reserve_exact(cells, data.cells.size(), budget);
    std::map<Key, std::size_t> index;
    for (const auto& [key, c] : data.cells)
    {
        budget.add(sizeof(PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>) + 64);
        const auto geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(c.finite_geometry);
        const auto vol = geometry->volume().value;
        if (!(vol > 0) || !std::isfinite(vol))
            reject(Reason::VolumeConstraint, "Filtered cell volume is unresolved.");
        tracked_row(index, key, budget) = cells.size();
        MeshCell row;
        row.source = &c;
        row.volume = vol;
        row.h = std::cbrt(vol);
        const auto spans = geometry->axis_spans();
        row.spans = {spans[0].as_double(), spans[1].as_double(), spans[2].as_double()};
        row.fit_scale = std::min({row.spans[0], row.spans[1], row.spans[2]});
        long double surface_area = 0;
        const auto before_export = budget.used;
        const auto exported = c.finite_geometry->export_points(point_exports, budget.used, budget.limit);
        point_export_bytes += budget.used - before_export;
        const auto points = exported.points();
        for (const auto& face : geometry->faces())
        {
            Wide area_vector{};
            for (std::size_t j = 1; j + 1 < face.size(); ++j)
            {
                const auto triangle = cross(sub(points[face[j]].value, points[face[0]].value),
                    sub(points[face[j + 1]].value, points[face[0]].value));
                for (unsigned k = 0; k < 3; ++k)
                    area_vector[k] += triangle[k] / 2;
            }
            surface_area += norm(area_vector);
        }
        if (!(surface_area > 0) || !std::isfinite(surface_area))
            reject(Reason::IncompleteSupport, "Filtered support has no resolved physical cell surface.");
        row.fit_scale = std::min(row.fit_scale, static_cast<double>(2 * vol / surface_area));
        if (!(row.fit_scale > 0) || !std::isfinite(row.fit_scale))
            reject(Reason::IncompleteSupport, "Filtered support has no resolved physical fit scale.");
        cells.push_back(std::move(row));
    }
    std::vector<const AMR::AdaptiveTile*> faces;
    std::vector<Edge> edges;
    std::vector<long double> rates;
    reserve_exact(rates, cells.size(), budget);
    rates.resize(cells.size());
    std::vector<const AMR::AdaptiveTile*> canonical_faces;
    if (data.topology->frontier_tiles().size() >
        std::numeric_limits<std::size_t>::max() - data.topology->surface_tiles().size())
        reject(Reason::IncompleteSupport, "Filtered canonical face count overflows.");
    const auto face_capacity = data.topology->surface_tiles().size() + data.topology->frontier_tiles().size();
    reserve_exact(canonical_faces, face_capacity, budget);
    for (const auto& tile : data.topology->surface_tiles())
        canonical_faces.push_back(&tile);
    for (const auto& tile : data.topology->frontier_tiles())
        if (tile.exterior)
            canonical_faces.push_back(&tile);
    std::sort(canonical_faces.begin(), canonical_faces.end(),
        [](const auto* a, const auto* b) { return a->face_gid < b->face_gid; });
    reserve_exact(faces, canonical_faces.size(), budget);
    reserve_exact(edges, canonical_faces.size(), budget);
    std::vector<std::size_t> neighbor_counts, face_counts;
    reserve_exact(neighbor_counts, cells.size(), budget);
    reserve_exact(face_counts, cells.size(), budget);
    neighbor_counts.resize(cells.size());
    face_counts.resize(cells.size());
    for (const auto* tile : canonical_faces)
    {
        if (tile->exterior && index.contains(tile->first))
            ++face_counts[index.at(tile->first)];
        else if (tile->second && index.contains(tile->first) && index.contains(*tile->second))
            for (auto i : {index.at(tile->first), index.at(*tile->second)})
            {
                ++neighbor_counts[i];
                ++face_counts[i];
            }
    }
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        reserve_exact(cells[i].neighbors, neighbor_counts[i], budget);
        reserve_exact(cells[i].faces, face_counts[i], budget);
    }
    for (const auto* source_tile : canonical_faces)
    {
        const auto& tile = *source_tile;
        if (tile.exterior)
        {
            // Physical boundaries supply authenticated fan tetrahedra and a
            // no-flux scalar boundary, rather than an unknown remote row.
            if (!index.contains(tile.first) || tile.second)
                reject(Reason::IncompleteSupport, "Filtered exterior face has unresolved source incidence.");
            const auto a = index.at(tile.first);
            cells[a].physical_boundary = true;
            cells[a].faces.push_back(faces.size());
            faces.push_back(&tile);
            continue;
        }
        if (!tile.second || !index.contains(tile.first) || !index.contains(*tile.second))
            continue;
        if (!tile.second_image_identity.is_zero())
            reject(Reason::IncompleteSupport, "Periodic filtered support is unqualified.");
        const auto a = index.at(tile.first), b = index.at(*tile.second);
        const auto distance = norm(sub(wide(cells[a].source->centroid), wide(cells[b].source->centroid)));
        const auto area = tile.geometry->area().value;
        if (!(distance > 0) || !(area > 0))
            reject(Reason::IncompleteSupport, "Filtered graph has unresolved physical metrics.");
        const auto conductance = (cells[a].h * cells[a].h + cells[b].h * cells[b].h) * area / (2 * distance);
        if (!(conductance > 0) || !std::isfinite(conductance))
            reject(Reason::IncompleteSupport, "Filtered graph diffusion scale is unresolved.");
        edges.push_back({a, b, conductance});
        cells[a].neighbors.push_back(b);
        cells[b].neighbors.push_back(a);
        cells[a].faces.push_back(faces.size());
        cells[b].faces.push_back(faces.size());
        faces.push_back(&tile);
        rates[a] += conductance / cells[a].volume;
        rates[b] += conductance / cells[b].volume;
    }
    // canonical_face_count is an owned-cell quota field and is not populated
    // for every full ghost. Frontier tiles authenticate the actual boundary of
    // retained full geometry. Only unknown interior neighbors truncate support;
    // authenticated exterior faces are retained as a no-flux physical boundary.
    for (const auto& tile : data.topology->frontier_tiles())
        if (!tile.exterior)
            for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                if (key && index.contains(*key))
                    cells[index.at(*key)].frontier = true;
    if (options.filter_width > 4096)
        reject(Reason::IncompleteSupport, "Filtered diffusion exceeds the bounded support/work policy.");
    const auto steps = static_cast<std::size_t>(std::ceil(options.filter_width));
    // Width is the finite graph support, in local cell lengths, rather than a
    // Gaussian variance. Each of ceil(width) simultaneous exchanges uses the
    // symmetric coefficient K_ab = width/ceil(width) * C_ab /
    // (2 max(sum_b C_ab/V_a, sum_a C_ab/V_b)). Thus every row sum K/V is at
    // most 1/2, every update is a convex combination, and pairwise amounts
    // cancel exactly before division by unequal cell volumes. Default width2
    // propagates the indicator over precisely two face-neighbor graph steps.
    for (auto& edge : edges)
    {
        const auto rate = std::max(rates[edge.a], rates[edge.b]);
        if (!(rate > 0) || !std::isfinite(rate))
            reject(Reason::IncompleteSupport, "Filtered graph row scale is unresolved.");
        edge.conductance *= options.filter_width / (2 * steps * rate);
    }
    // Components use face incidence. Edge/vertex contacts are rejected below
    // using the authenticated complete node fans, not coordinate proximity.
    std::vector<std::vector<std::size_t>> components;
    reserve_exact(components, cells.size(), budget);
    for (std::size_t i = 0; i < cells.size(); ++i)
        if (phase(*cells[i].source) > 0 && cells[i].label < 0)
        {
            const auto label = static_cast<int>(components.size());
            components.emplace_back();
            tracked_push(components.back(), i, budget);
            cells[i].label = label;
            for (std::size_t j = 0; j < components.back().size(); ++j)
                for (auto n : cells[components.back()[j]].neighbors)
                    if (phase(*cells[n].source) > 0 && cells[n].label < 0)
                    {
                        cells[n].label = label;
                        tracked_push(components.back(), n, budget);
                    }
        }
    std::vector<std::uint8_t> eligible;
    reserve_exact(eligible, components.size(), budget);
    eligible.resize(components.size(), 1);
    for (std::size_t component = 0; component < components.size(); ++component)
    {
        bool opposite_face = false;
        for (auto i : components[component])
            for (auto neighbor : cells[i].neighbors)
            {
                const auto neighbor_phase = phase(*cells[neighbor].source);
                if (neighbor_phase == 0)
                    eligible[component] = false;
                opposite_face = opposite_face || neighbor_phase < 0;
            }
        // A homogeneous gas component has no internal capillary surface. A
        // wall-touching component with a real gas/carrier face remains subject
        // to the enclosed-support admission below.
        eligible[component] = eligible[component] && opposite_face;
    }
    // Mixed components retain the reconstructed catalog. Only wholly binary
    // gas components enter this optional geometry model; a mixed cell outside
    // all eligible filter supports is unrelated and does not reject the model.
    if (std::none_of(eligible.begin(), eligible.end(), [](bool value) { return value; }))
    {
        result.allocation_bytes = budget.used - point_export_bytes;
        return result;
    }
    // The public control-node range only contains owned-incident nodes. Halo
    // scalar interpolation uses every authenticated finite source node instead.
    // A subscribed complete fan remains authoritative. For other nodes, a
    // retained interior star is complete only if no incident tile reaches a
    // physical or unknown frontier and its full-cell incidence is connected.
    std::map<std::uint64_t, std::set<Key>> fans;
    for (const auto& [node, fan] : data.topology->node_fans)
    {
        auto& retained = tracked_row(fans, node, budget);
        for (const auto& key : fan)
            tracked_insert(retained, key, budget);
    }
    // Sparse imported fans carry detached authority for halo nodes, including
    // known physical-wall stars. They do not imply that every member's cell
    // geometry is retained; active interpolation still checks that separately.
    for (const auto& [node, fan] : imported_fans.fans)
    {
        if (const auto retained = fans.find(node); retained != fans.end())
        {
            if (retained->second != fan)
                throw std::invalid_argument(
                    "Imported capillary node fan conflicts with the subscribed source authority.");
            continue;
        }
        auto& retained = tracked_row(fans, node, budget);
        for (const auto& key : fan)
            tracked_insert(retained, key, budget);
    }
    std::map<std::uint64_t, std::set<Key>> interior_stars;
    std::set<std::uint64_t> frontier_nodes;
    for (const auto& tile : data.topology->frontier_tiles())
    {
        for (auto node : tile.node_gids)
            tracked_insert(frontier_nodes, node, budget);
        for (auto node : tile.second_node_gids)
            tracked_insert(frontier_nodes, node, budget);
    }
    for (const auto* tile : canonical_faces)
        for (auto node : tile->node_gids)
        {
            if (fans.contains(node) || frontier_nodes.contains(node))
                continue;
            for (const auto* key : {&tile->first, tile->second ? &*tile->second : nullptr})
                if (key && index.contains(*key))
                    tracked_insert(tracked_row(interior_stars, node, budget), *key, budget);
        }
    for (const auto& [node, star] : interior_stars)
    {
        if (fans.contains(node) || frontier_nodes.contains(node) || star.empty())
            continue;
        std::set<Key> connected;
        std::vector<Key> visit;
        tracked_insert(connected, *star.begin(), budget);
        reserve_exact(visit, star.size(), budget);
        tracked_push(visit, *star.begin(), budget);
        for (std::size_t j = 0; j < visit.size(); ++j)
            for (auto f : cells[index.at(visit[j])].faces)
            {
                const auto& tile = *faces[f];
                if (!tile.second)
                    continue;
                if (std::find(tile.node_gids.begin(), tile.node_gids.end(), node) == tile.node_gids.end())
                    continue;
                const auto& key = tile.first == visit[j] ? *tile.second : tile.first;
                if (star.contains(key) && tracked_insert(connected, key, budget))
                    tracked_push(visit, key, budget);
            }
        if (connected.size() == star.size())
        {
            auto& fan = tracked_row(fans, node, budget);
            for (const auto& key : star)
                tracked_insert(fan, key, budget);
        }
    }
    for (const auto& [node, fan] : fans)
    {
        int label = -1;
        for (const auto& key : fan)
            if (index.contains(key) && cells[index.at(key)].label >= 0)
            {
                const auto next = cells[index.at(key)].label;
                if (label >= 0 && next != label && (eligible[label] || eligible[next]))
                    reject(Reason::AmbiguousTopology, "Binary components contact at an edge or vertex.");
                if (label < 0 || eligible[next])
                    label = next;
            }
    }
    std::map<std::uint64_t, Wide> coordinates;
    for (const auto& [node, coordinate] : data.topology->finite_nodes)
        tracked_row(coordinates, node, budget) = coordinate.value;
    std::vector<int> support_label;
    reserve_exact(support_label, cells.size(), budget);
    support_label.resize(cells.size(), -1);
    using PublishedIdentity = std::pair<std::uint64_t, CutId>;
    using PublishedCoordinates = std::array<double, 3>;
    using PublishedBits = std::array<std::uint64_t, 3>;
    std::map<PublishedIdentity, PublishedBits> identity_coordinates;
    std::map<PublishedCoordinates, PublishedIdentity> coordinate_identities;
    for (std::size_t component = 0; component < components.size(); ++component)
    {
        if (!eligible[component])
            continue;
        for (auto i : components[component])
            if (cells[i].physical_boundary)
                reject(Reason::IncompleteSupport, "Original binary gas component touches a physical boundary.");
        std::vector<int> distance;
        reserve_exact(distance, cells.size(), budget);
        distance.resize(cells.size(), -1);
        std::vector<std::size_t> reach;
        reserve_exact(reach, cells.size(), budget);
        reach.assign(components[component].begin(), components[component].end());
        for (auto i : reach)
            distance[i] = 0;
        for (std::size_t j = 0; j < reach.size(); ++j)
        {
            const auto i = reach[j];
            if (phase(*cells[i].source) == 0)
                reject(Reason::AmbiguousTopology, "Binary filter support reaches a mixed geometric component.");
            if (cells[i].label >= 0 && cells[i].label != static_cast<int>(component))
                reject(Reason::ComponentMerging, "Binary filter support reaches another gas component.");
            // The outer full-geometry ring supplies known metrics only. Its
            // neighbor rows are never expanded or used by N-step diffusion.
            if (static_cast<std::size_t>(distance[i]) >= steps + 2)
                continue;
            if (cells[i].frontier)
                reject(Reason::IncompleteSupport, "Filtered component lacks its complete finite-width support.");
            for (auto n : cells[i].neighbors)
                if (distance[n] < 0)
                {
                    distance[n] = distance[i] + 1;
                    reach.push_back(n);
                }
        }
        // Authenticate the original binary boundary before smoothing. Otherwise
        // a small cavity could disappear in diffusion and falsely pass the
        // extracted surface's single-component topology check.
        std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> original_edges;
        std::set<std::uint64_t> original_vertices;
        std::size_t original_faces = 0;
        for (auto i : components[component])
            for (auto f : cells[i].faces)
            {
                const auto& tile = *faces[f];
                if (tile.exterior || !tile.second)
                    reject(Reason::IncompleteSupport, "Original binary component has a physical boundary face.");
                const auto other = index.at(tile.first) == i ? index.at(*tile.second) : index.at(tile.first);
                if (cells[other].label == static_cast<int>(component))
                    continue;
                for (std::size_t j = 0; j < tile.node_gids.size(); ++j)
                {
                    const auto a = tile.node_gids[j], b = tile.node_gids[(j + 1) % tile.node_gids.size()];
                    tracked_insert(original_vertices, a, budget);
                    tracked_push(
                        tracked_row(original_edges, {std::min(a, b), std::max(a, b)}, budget), original_faces, budget);
                }
                ++original_faces;
            }
        std::vector<std::vector<std::size_t>> original_adjacency;
        reserve_exact(original_adjacency, original_faces, budget);
        original_adjacency.resize(original_faces);
        for (const auto& [edge, incident] : original_edges)
        {
            if (incident.size() != 2)
                reject(Reason::AmbiguousTopology, "Original binary component is not a closed two-manifold.");
            tracked_push(original_adjacency[incident[0]], incident[1], budget);
            tracked_push(original_adjacency[incident[1]], incident[0], budget);
        }
        std::map<std::uint64_t, std::vector<std::pair<std::size_t, std::size_t>>> original_links;
        for (const auto& [edge, incident] : original_edges)
            for (auto node : {edge.first, edge.second})
                tracked_push(tracked_row(original_links, node, budget), std::pair{incident[0], incident[1]}, budget);
        check_vertex_links(original_links, "Original binary vertex lacks a single-cycle manifold link.", budget);
        if (!original_faces)
            reject(Reason::IncompleteSupport, "Original binary component has no resolved enclosing boundary.");
        std::set<std::size_t> original_connected;
        std::vector<std::size_t> original_todo;
        tracked_insert(original_connected, std::size_t{0}, budget);
        reserve_exact(original_todo, original_faces, budget);
        original_todo.push_back(0);
        for (std::size_t j = 0; j < original_todo.size(); ++j)
            for (auto f : original_adjacency[original_todo[j]])
                if (tracked_insert(original_connected, f, budget))
                    original_todo.push_back(f);
        if (original_connected.size() != original_faces)
            reject(Reason::UnsupportedCavity, "Original binary component has a cavity or disconnected boundary.");
        if (static_cast<long long>(original_vertices.size()) - static_cast<long long>(original_edges.size()) +
                static_cast<long long>(original_faces) !=
            2)
            reject(Reason::AmbiguousTopology, "Original binary component has unsupported nonspherical topology.");
        GeneralVoFSurfaceComponent diagnostic;
        diagnostic.identity = std::numeric_limits<std::uint64_t>::max();
        diagnostic.filter_width_min = std::numeric_limits<double>::infinity();
        std::vector<long double> q, delta;
        reserve_exact(q, cells.size(), budget);
        reserve_exact(delta, cells.size(), budget);
        q.resize(cells.size());
        delta.resize(cells.size());
        SimpleFluid::detail::CompensatedSum<> target;
        for (auto i : components[component])
        {
            q[i] = 1;
            target += cells[i].volume;
            if (cells[i].source->gid < diagnostic.identity)
            {
                diagnostic.identity = cells[i].source->gid;
                diagnostic.owner = cells[i].source->owner;
            }
        }
        diagnostic.original_volume = target.value();
        for (auto i : reach)
        {
            const double width = options.filter_width * static_cast<double>(cells[i].h);
            diagnostic.filter_width_min = std::min(diagnostic.filter_width_min, width);
            diagnostic.filter_width_max = std::max(diagnostic.filter_width_max, width);
        }
        for (std::size_t step = 0; step < steps; ++step)
        {
            std::fill(delta.begin(), delta.end(), 0);
            for (const auto& e : edges)
            {
                const auto transfer = e.conductance * (q[e.b] - q[e.a]);
                delta[e.a] += transfer;
                delta[e.b] -= transfer;
            }
            for (std::size_t i = 0; i < q.size(); ++i)
            {
                q[i] += delta[i] / cells[i].volume;
                if (!(q[i] >= 0 && q[i] <= 1) || !std::isfinite(q[i]))
                    reject(Reason::VolumeConstraint, "Conservative filtered scalar left its convex bounds.");
            }
        }
        SimpleFluid::detail::CompensatedSum<> smoothed;
        for (std::size_t i = 0; i < q.size(); ++i)
            smoothed += q[i] * cells[i].volume;
        diagnostic.smoothed_volume = smoothed.value();
        const auto tolerance = data.options.reconstruction.volume_tolerance * diagnostic.original_volume;
        if (std::abs(diagnostic.smoothed_volume - diagnostic.original_volume) > tolerance)
            reject(Reason::VolumeConstraint, "Filtered component diffusion failed its separate conservation gate.");
        // Volume-weighted nodal interpolation is shared by every incident cell.
        // Requiring the complete fan whenever it contributes prevents truncation
        // at a retained halo, including vertex-only dependencies.
        std::map<std::uint64_t, long double> nodal;
        for (std::size_t i = 0; i < cells.size(); ++i)
            if (q[i] > 0)
                for (auto f : cells[i].faces)
                    for (auto node : faces[f]->node_gids)
                        if (!fans.contains(node))
                            reject(Reason::IncompleteSupport,
                                "Filtered positive scalar lacks an authenticated complete nodal fan.");
        for (const auto& [node, fan] : fans)
        {
            SimpleFluid::detail::CompensatedSum<> numerator, denominator;
            bool active = false, complete = true, anchored = false;
            long double anchor = 0;
            for (const auto& key : fan)
            {
                if (!index.contains(key))
                {
                    complete = false;
                    continue;
                }
                const auto i = index.at(key);
                if (!anchored)
                {
                    anchor = q[i];
                    anchored = true;
                }
                active = active || q[i] > 0;
                numerator += (q[i] - anchor) * cells[i].volume;
                denominator += cells[i].volume;
            }
            if (active && (!complete || !(denominator.value() > 0)))
                reject(Reason::IncompleteSupport, "Filtered nodal interpolation has an incomplete canonical fan.");
            if (active)
            {
                for (const auto& key : fan)
                {
                    const auto i = index.at(key);
                    if (phase(*cells[i].source) == 0)
                        reject(Reason::AmbiguousTopology, "Filtered nodal support touches mixed geometry.");
                    if (cells[i].label >= 0 && cells[i].label != static_cast<int>(component))
                        reject(Reason::ComponentMerging, "Filtered nodal support touches another gas component.");
                }
                tracked_row(nodal, node, budget) = anchor + numerator.value() / denominator.value();
            }
        }
        std::vector<Tetra> tetrahedra;
        std::map<std::uint64_t, long double> face_q;
        for (const auto* tile : faces)
        {
            std::vector<std::uint64_t> nodes;
            reserve_exact(nodes, tile->node_gids.size(), budget);
            nodes.assign(tile->node_gids.begin(), tile->node_gids.end());
            if (nodes.size() < 3)
                reject(Reason::IncompleteSupport, "Filtered face has fewer than three nodes.");
            std::rotate(nodes.begin(), std::min_element(nodes.begin(), nodes.end()), nodes.end());
            const auto fc = tile->geometry->centroid();
            const Wide center{fc.x, fc.y, fc.z};
            SimpleFluid::detail::CompensatedSum<> area, numerator;
            const auto scalar_anchor = nodal.contains(nodes.front()) ? nodal.at(nodes.front()) : 0;
            for (std::size_t j = 0; j < nodes.size(); ++j)
            {
                const auto a = nodes[j], b = nodes[(j + 1) % nodes.size()];
                if (!coordinates.contains(a) || !coordinates.contains(b))
                    reject(Reason::IncompleteSupport, "Filtered face lacks an authenticated native node.");
                const auto triangle = norm(cross(sub(coordinates.at(a), center), sub(coordinates.at(b), center))) / 2;
                if (!(triangle > 0) || !std::isfinite(triangle))
                    reject(Reason::AmbiguousTopology, "Filtered face fan has a collapsed triangle.");
                area += triangle;
                numerator += triangle *
                             (((nodal.contains(a) ? nodal.at(a) : 0) - scalar_anchor) +
                                 ((nodal.contains(b) ? nodal.at(b) : 0) - scalar_anchor)) /
                             2;
            }
            const auto actual_area = tile->geometry->area().value;
            if (!(actual_area > 0) || !std::isfinite(actual_area) ||
                std::abs(area.value() - actual_area) > data.options.reconstruction.volume_tolerance * actual_area)
                reject(Reason::VolumeConstraint, "Filtered face fan does not recover its actual physical area.");
            // Positive affine coordinates reproduce a linear nodal field at
            // the actual face centroid, including nonuniform polygon fans.
            const auto value = scalar_anchor + numerator.value() / area.value();
            if (!(value >= 0 && value <= 1) || !std::isfinite(value))
                reject(Reason::VolumeConstraint, "Filtered nodal face-center value left its convex bounds.");
            tracked_row(face_q, tile->face_gid, budget) = value;
        }
        std::size_t tetra_count = 0;
        for (std::size_t i = 0; i < cells.size(); ++i)
        {
            bool active = q[i] > 0;
            for (auto f : cells[i].faces)
                active = active || face_q.at(faces[f]->face_gid) > 0;
            if (active)
                for (auto f : cells[i].faces)
                {
                    const auto count = faces[f]->node_gids.size();
                    if (count > std::numeric_limits<std::size_t>::max() - tetra_count)
                        reject(Reason::IncompleteSupport, "Filtered tetrahedron count overflows.");
                    tetra_count += count;
                }
        }
        reserve_exact(tetrahedra, tetra_count, budget);
        for (std::size_t i = 0; i < cells.size(); ++i)
        {
            bool active = q[i] > 0;
            for (auto f : cells[i].faces)
                active = active || face_q.at(faces[f]->face_gid) > 0;
            if (!active)
                continue;
            if (phase(*cells[i].source) == 0)
                reject(Reason::AmbiguousTopology, "Filtered extraction support reaches mixed geometry.");
            if (cells[i].label >= 0 && cells[i].label != static_cast<int>(component))
                reject(Reason::ComponentMerging, "Filtered extraction support reaches another gas component.");
            if (support_label[i] >= 0 && support_label[i] != static_cast<int>(component))
                reject(Reason::ComponentMerging, "Filtered component interpolation supports overlap.");
            support_label[i] = static_cast<int>(component);
            if (cells[i].frontier)
                reject(Reason::IncompleteSupport, "Filtered tetrahedral support reaches an unknown frontier.");
            std::vector<std::size_t> ordered;
            reserve_exact(ordered, cells[i].faces.size(), budget);
            ordered.assign(cells[i].faces.begin(), cells[i].faces.end());
            std::sort(ordered.begin(), ordered.end(),
                [&](auto a, auto b) { return faces[a]->face_gid < faces[b]->face_gid; });
            SimpleFluid::detail::CompensatedSum<> bulk;
            SimpleFluid::detail::CompensatedSum<> center_numerator;
            const auto cell_begin = tetrahedra.size();
            const auto scalar_anchor = face_q.at(faces[ordered.front()]->face_gid);
            std::uint64_t ordinal = 0;
            for (auto f : ordered)
            {
                const auto& tile = *faces[f];
                std::vector<std::uint64_t> nodes;
                reserve_exact(nodes, tile.node_gids.size(), budget);
                nodes.assign(tile.node_gids.begin(), tile.node_gids.end());
                if (nodes.size() < 3)
                    reject(Reason::IncompleteSupport, "Filtered face has fewer than three nodes.");
                std::rotate(nodes.begin(), std::min_element(nodes.begin(), nodes.end()), nodes.end());
                const auto fc = tile.geometry->centroid();
                const Wide center{fc.x, fc.y, fc.z};
                for (std::size_t j = 0; j < nodes.size(); ++j)
                {
                    const auto a = nodes[j], b = nodes[(j + 1) % nodes.size()];
                    if (!coordinates.contains(a) || !coordinates.contains(b))
                        reject(Reason::IncompleteSupport, "Filtered tetrahedron lacks an authenticated native node.");
                    Tetra t{{Vertex{{2, cells[i].source->gid}, wide(cells[i].source->centroid), q[i]},
                                Vertex{{1, tile.face_gid}, center, face_q.at(tile.face_gid)},
                                Vertex{{0, a}, coordinates.at(a), nodal.contains(a) ? nodal.at(a) : 0},
                                Vertex{{0, b}, coordinates.at(b), nodal.contains(b) ? nodal.at(b) : 0}},
                        i, ordinal++};
                    const auto vol = volume(t.v[0].p, t.v[1].p, t.v[2].p, t.v[3].p);
                    if (!(vol > 0) || !std::isfinite(vol))
                        reject(Reason::AmbiguousTopology, "Filtered canonical fan has a collapsed tetrahedron.");
                    bulk += vol;
                    center_numerator +=
                        vol * ((t.v[1].q - scalar_anchor) + (t.v[2].q - scalar_anchor) + (t.v[3].q - scalar_anchor)) /
                        3;
                    tetrahedra.push_back(t);
                }
            }
            if (std::abs(bulk.value() - cells[i].volume) >
                data.options.reconstruction.volume_tolerance * cells[i].volume)
                reject(Reason::VolumeConstraint, "Filtered tetrahedral fan does not recover its actual cell volume.");
            // The source q_i is the separately conserved smoothing field. The
            // extraction uses only the continuous nodal field: positive fan
            // volume coordinates reproduce its affine value at the actual
            // cell centroid without an independent cell-center scalar lobe.
            const auto center_value = scalar_anchor + center_numerator.value() / bulk.value();
            if (!(center_value >= 0 && center_value <= 1) || !std::isfinite(center_value))
                reject(Reason::VolumeConstraint, "Filtered nodal cell-center value left its convex bounds.");
            for (auto t = cell_begin; t < tetrahedra.size(); ++t)
                tetrahedra[t].v[0].q = center_value;
        }
        const auto measure = [&](long double iso)
        {
            SimpleFluid::detail::CompensatedSum<> v;
            for (const auto& t : tetrahedra)
                v += clipped_volume(t, iso);
            return v.value();
        };
        std::set<long double> scalar_levels;
        std::map<long double, SimpleFluid::detail::CompensatedSum<>> plateau_volumes;
        for (const auto& t : tetrahedra)
        {
            for (const auto& v : t.v)
                if (!scalar_levels.contains(v.q))
                {
                    budget.add(sizeof(long double) + 64);
                    scalar_levels.insert(v.q);
                }
            if (std::all_of(t.v.begin(), t.v.end(), [&](const Vertex& v) { return v.q == t.v[0].q; }))
            {
                if (!plateau_volumes.contains(t.v[0].q))
                    budget.add(sizeof(long double) + sizeof(SimpleFluid::detail::CompensatedSum<>) + 64);
                plateau_volumes[t.v[0].q] += volume(t.v[0].p, t.v[1].p, t.v[2].p, t.v[3].p);
            }
        }
        const auto admit_regular_level = [&](long double candidate)
        {
            auto upper = plateau_volumes.lower_bound(candidate);
            std::array<decltype(upper), 2> nearest{upper, plateau_volumes.end()};
            if (upper != plateau_volumes.begin())
                nearest[1] = std::prev(upper);
            for (auto plateau : nearest)
            {
                if (plateau == plateau_volumes.end())
                    continue;
                const auto below = measure(plateau->first);
                const auto above = below + plateau->second.value();
                // The existing component tolerance accounts for numerical
                // endpoint uncertainty; it never converts plateau volume into
                // accepted geometry. A target in this positive-volume jump has
                // no resolved regular strict-superlevel surface, even if a
                // nearby level numerically appears to meet the volume gate.
                if (diagnostic.original_volume + tolerance >= below && diagnostic.original_volume - tolerance <= above)
                {
                    std::ostringstream detail;
                    detail << std::setprecision(std::numeric_limits<long double>::max_digits10)
                           << "Filtered component volume lies in an unresolved scalar plateau jump: component="
                           << diagnostic.identity << " level=" << plateau->first
                           << " target=" << diagnostic.original_volume << " strict_volume=" << below
                           << " inclusive_volume=" << above;
                    throw GeneralVoFSurfaceResolutionError(Reason::VolumeConstraint, detail.str());
                }
            }
        };
        long double low = 0, high = 1, iso = .5L, extracted = 0;
        bool solved = false;
        for (unsigned iteration = 0; iteration < data.options.reconstruction.maximum_offset_iterations; ++iteration)
        {
            iso = (low + high) / 2;
            // Prefer an actual field level only when it lies in the current
            // monotone bracket AND independently satisfies the same geometric
            // volume gate. This is exact scalar equality, without epsilon
            // snapping or removal of near-level positive fragments.
            auto upper_level = scalar_levels.lower_bound(iso);
            std::array<std::optional<long double>, 2> candidates;
            if (upper_level != scalar_levels.end())
                candidates[0] = *upper_level;
            if (upper_level != scalar_levels.begin())
                candidates[1] = *std::prev(upper_level);
            if (candidates[0] && candidates[1] && std::abs(*candidates[1] - iso) < std::abs(*candidates[0] - iso))
                std::swap(candidates[0], candidates[1]);
            for (const auto candidate : candidates)
            {
                if (!candidate || !(*candidate >= low && *candidate <= high))
                    continue;
                const auto candidate_volume = measure(*candidate);
                if (std::abs(candidate_volume - diagnostic.original_volume) <= tolerance)
                {
                    admit_regular_level(*candidate);
                    iso = *candidate;
                    extracted = candidate_volume;
                    solved = true;
                    break;
                }
            }
            if (solved)
                break;
            extracted = measure(iso);
            // Leave numerical headroom for binary64 publication; final public
            // geometry still consumes the unchanged full component tolerance.
            if (std::abs(extracted - diagnostic.original_volume) <= tolerance / 4)
            {
                admit_regular_level(iso);
                solved = true;
                break;
            }
            if (extracted > diagnostic.original_volume)
                low = iso;
            else
                high = iso;
            if (!(low < high))
                break;
        }
        if (!solved)
            reject(Reason::VolumeConstraint, "Filtered component has no resolved volume-preserving isovalue.");
        // A closed extracted shell alone is insufficient: q>iso could surround
        // a carrier cavity while occupying the physical domain boundary. Every
        // scalar vertex of each wall fan triangle must lie strictly in carrier.
        for (const auto* tile : faces)
            if (tile->exterior)
            {
                if (!(face_q.at(tile->face_gid) < iso))
                    reject(Reason::IncompleteSupport, "Filtered gas isovalue reaches a physical boundary face.");
                for (auto node : tile->node_gids)
                    if (nodal.contains(node) && !(nodal.at(node) < iso))
                        reject(Reason::IncompleteSupport, "Filtered gas isovalue reaches a physical boundary node.");
            }
        diagnostic.isovalue = static_cast<double>(iso);
        const auto begin = result.patches.size();
        using SurfaceEdge = std::pair<CutId, CutId>;
        struct Incidence
        {
            std::size_t patch = 0;
            int sign = 0;
        };
        std::map<SurfaceEdge, std::vector<Incidence>> incidence;
        std::set<CutId> vertices;
        std::vector<std::vector<CutId>> patch_ids;
        std::vector<const Tetra*> patch_tetrahedra;
        std::vector<std::uint8_t> cancelled;
        FullFacetMap full_facets;
        for (const auto& t : tetrahedra)
        {
            Wide normal{};
            // section's at-most-four intersections have bounded geometric
            // vector growth (1+2+4 entries); debit before the pure kernel.
            budget.add(7 * sizeof(Cut));
            auto cuts = section(t, iso, normal);
            if (cuts.empty())
                continue;
            GeneralVoFSurfacePatch patch;
            const auto& c = *cells[t.cell].source;
            patch.key = {GeneralVoFSurfaceKind::FilteredCell, c.gid, t.ordinal};
            patch.first_cell = c.gid;
            patch.owner = c.owner;
            patch.component = diagnostic.identity;
            patch.cell_scale = cells[t.cell].fit_scale;
            patch.physical_spans = cells[t.cell].spans;
            for (unsigned k = 0; k < 3; ++k)
                patch.normal[k] = static_cast<double>(normal[k]);
            reserve_exact(patch.vertices, cuts.size(), budget);
            std::vector<CutId> ids;
            reserve_exact(ids, cuts.size(), budget);
            for (std::size_t j = 0; j < cuts.size(); ++j)
            {
                const auto& cut = cuts[j];
                patch.vertices.push_back(
                    {static_cast<double>(cut.p[0]), static_cast<double>(cut.p[1]), static_cast<double>(cut.p[2])});
                ids.push_back(cut.id);
            }
            const auto local_patch = patch_ids.size();
            tracked_push(cancelled, std::uint8_t{0}, budget);
            if (ids.size() == 3 &&
                std::all_of(ids.begin(), ids.end(), [](const CutId& id) { return id.first == id.second; }))
            {
                std::array<NodeId, 3> facet{ids[0].first, ids[1].first, ids[2].first};
                std::sort(facet.begin(), facet.end());
                if (!full_facets.contains(facet))
                    charge_tree_node<FullFacetMap>(budget);
            }
            // Exact full facets cancel only by three source-node identities
            // and opposite gas-side orientations, before edge incidence exists.
            if (const auto match = match_full_facet(full_facets, ids, local_patch))
            {
                cancelled[*match] = true;
                cancelled[local_patch] = true;
            }
            reserve_next(patch_ids, budget);
            patch_ids.push_back(std::move(ids));
            reserve_next(patch_tetrahedra, budget);
            patch_tetrahedra.push_back(&t);
            reserve_next(result.patches, budget);
            result.patches.push_back(std::move(patch));
        }
        auto destination = begin;
        for (std::size_t local = 0; local < patch_ids.size(); ++local)
        {
            if (cancelled[local])
                continue;
            const auto source = begin + local;
            if (destination != source)
                result.patches[destination] = std::move(result.patches[source]);
            // Internal exact facets have already cancelled by source identity.
            // Only retained physical surface polygons must be publishable.
            published_metrics(result.patches[destination], *patch_tetrahedra[local], iso);
            const auto& ids = patch_ids[local];
            for (std::size_t j = 0; j < ids.size(); ++j)
            {
                const auto& cut = ids[j];
                const auto& point = result.patches[destination].vertices[j];
                PublishedCoordinates coordinate{};
                PublishedBits bits{};
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    if (!std::isfinite(point[axis]))
                        reject(Reason::VolumeConstraint, "Filtered published topology has a nonfinite coordinate.");
                    coordinate[axis] = point[axis];
                    bits[axis] = std::bit_cast<std::uint64_t>(point[axis]);
                }
                const PublishedIdentity identity{diagnostic.identity, cut};
                const auto by_identity = identity_coordinates.find(identity);
                if (by_identity != identity_coordinates.end() && by_identity->second != bits)
                    reject(Reason::VolumeConstraint,
                        "Filtered cut identity publishes inconsistent binary64 vertex coordinates.");
                const auto by_coordinate = coordinate_identities.find(coordinate);
                if (by_coordinate != coordinate_identities.end() && by_coordinate->second != identity)
                    reject(Reason::VolumeConstraint,
                        "Distinct filtered cut identities weld to the same binary64 vertex coordinate.");
                if (by_identity == identity_coordinates.end())
                {
                    budget.add(sizeof(PublishedIdentity) + sizeof(PublishedBits) + 64);
                    identity_coordinates.emplace(identity, bits);
                }
                if (by_coordinate == coordinate_identities.end())
                {
                    budget.add(sizeof(PublishedCoordinates) + sizeof(PublishedIdentity) + 64);
                    coordinate_identities.emplace(coordinate, identity);
                }
                const auto& other = ids[(j + 1) % ids.size()];
                tracked_insert(vertices, cut, budget);
                tracked_push(tracked_row(incidence, {std::min(cut, other), std::max(cut, other)}, budget),
                    Incidence{destination, cut < other ? 1 : -1}, budget);
            }
            tracked_row(result.adjacency, result.patches[destination].key, budget);
            ++destination;
        }
        result.patches.resize(destination);
        for (const auto& [edge, rows] : incidence)
        {
            if (rows.size() != 2 || rows[0].sign + rows[1].sign != 0)
                reject(
                    Reason::AmbiguousTopology, "Filtered surface lacks a closed oriented two-manifold edge incidence.");
            const auto& a = result.patches[rows[0].patch].key;
            const auto& b = result.patches[rows[1].patch].key;
            tracked_push(result.adjacency.at(a), b, budget);
            tracked_push(result.adjacency.at(b), a, budget);
        }
        // Edge multiplicity alone does not exclude a pinched vertex. The link
        // of every cut vertex must also be a single cycle of incident patches.
        std::map<CutId, std::vector<std::pair<std::size_t, std::size_t>>> links;
        for (const auto& [edge, rows] : incidence)
            for (const auto& vertex : {edge.first, edge.second})
                tracked_push(tracked_row(links, vertex, budget), std::pair{rows[0].patch, rows[1].patch}, budget);
        check_vertex_links(links, "Filtered cut vertex lacks a single-cycle manifold link.", budget);
        if (begin == result.patches.size())
            reject(Reason::VolumeConstraint, "Filtered component produced no surface.");
        std::set<GeneralVoFSurfaceKey> connected;
        std::vector<GeneralVoFSurfaceKey> todo;
        tracked_insert(connected, result.patches[begin].key, budget);
        reserve_exact(todo, result.patches.size() - begin, budget);
        todo.push_back(result.patches[begin].key);
        for (std::size_t j = 0; j < todo.size(); ++j)
            for (const auto& key : result.adjacency.at(todo[j]))
                if (tracked_insert(connected, key, budget))
                    todo.push_back(key);
        if (connected.size() != result.patches.size() - begin)
            reject(
                Reason::UnsupportedCavity, "Filtered component contains a cavity or disconnected extracted boundary.");
        if (static_cast<long long>(vertices.size()) - static_cast<long long>(incidence.size()) +
                static_cast<long long>(result.patches.size() - begin) !=
            2)
            reject(Reason::AmbiguousTopology, "Filtered component has unsupported nonspherical topology.");
        // Admit the actual public binary64 surface, not its pre-narrowing cut
        // geometry. Closed outward polygons are integrated in a translated
        // frame with signed tetrahedra so concavity cannot double-count volume.
        diagnostic.extracted_volume = admit_published_volume(
            std::span<const GeneralVoFSurfacePatch>(result.patches.data() + begin, result.patches.size() - begin),
            cells[components[component].front()].source->centroid, diagnostic.original_volume, tolerance);
        diagnostic.closed = true;
        reserve_next(result.components, budget);
        result.components.push_back(diagnostic);
        for (auto i : components[component])
        {
            reserve_next(result.filtered_source_cells, budget);
            result.filtered_source_cells.push_back(cells[i].source->gid);
        }
    }
    std::sort(result.filtered_source_cells.begin(), result.filtered_source_cells.end());
    result.allocation_bytes = budget.used - point_export_bytes;
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail
