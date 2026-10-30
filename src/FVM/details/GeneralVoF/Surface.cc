/** @file Surface.cc
 * @brief Backend-independent, source-pinned physical capillary patches.
 */
#include "FVM/details/GeneralVoF/CapillaryNodeFans.hh"
#include "FVM/details/GeneralVoF/CapillaryPlanes.hh"
#include "FVM/details/GeneralVoF/FilteredSurface.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace SimpleFluid::VoF
{
namespace
{
using WidePoint = std::array<long double, 3>;
using Edge = std::pair<std::uint64_t, std::uint64_t>;
WidePoint difference(const WidePoint& a, const WidePoint& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
long double dot(const WidePoint& a, const WidePoint& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
WidePoint cross(const WidePoint& a, const WidePoint& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
PLIC::GeometryPoint display(const WidePoint& p)
{
    return {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
}
WidePoint wide(const PLIC::GeometryPoint& p)
{
    return {p[0], p[1], p[2]};
}
[[noreturn]] void unresolved(const char* message)
{
    throw GeneralVoFSurfaceResolutionError(GeneralVoFSurfaceResolutionReason::AmbiguousTopology, message);
}
int geometric_phase(const Cell& cell)
{
    if (cell.declaration == GeneralVoFPhaseGeometry::PureGas)
        return 1;
    if (cell.declaration == GeneralVoFPhaseGeometry::PureCarrier)
        return -1;
    if (cell.retained_interface)
    {
        if (cell.retained_interface->pure_gas)
            return 1;
        if (cell.retained_interface->pure_carrier)
            return -1;
        return 0;
    }
    if (cell.phase.carrier_volume == 0 && cell.phase.gas_volume > 0)
        return 1;
    if (cell.phase.gas_volume == 0 && cell.phase.carrier_volume > 0)
        return -1;
    return 0;
}
void physical_scale(GeneralVoFSurfacePatch& patch, const Cell& cell)
{
    const auto geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(cell.finite_geometry);
    const auto spans = geometry->axis_spans();
    patch.physical_spans = {spans[0].as_double(), spans[1].as_double(), spans[2].as_double()};
    patch.cell_scale = std::min({patch.physical_spans[0], patch.physical_spans[1], patch.physical_spans[2]});
    long double surface = 0;
    const auto& points = geometry->points();
    for (const auto& face : geometry->faces())
    {
        WidePoint area{};
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const auto triangle = cross(difference(points[face[i]].value, points[face[0]].value),
                difference(points[face[i + 1]].value, points[face[0]].value));
            for (unsigned a = 0; a < 3; ++a)
                area[a] += triangle[a] / 2;
        }
        surface += std::sqrt(dot(area, area));
    }
    if (!(surface > 0) || !std::isfinite(surface))
        unresolved("Capillary support has no positive physical cell surface.");
    patch.cell_scale = std::min(patch.cell_scale, static_cast<double>(2 * geometry->volume().value / surface));
    if (!(patch.cell_scale > 0) || !std::isfinite(patch.cell_scale))
        unresolved("Capillary support has no positive physical cell scale.");
}
GeneralVoFFaceInterface face_interface(const AMR::AdaptiveTile& tile, const Cell& first, const Cell& second, int a)
{
    if (!tile.second_image_identity.is_zero())
        throw std::invalid_argument("Periodic face-interface surfaces are not qualified.");
    const auto centroid = tile.geometry->centroid();
    const auto area_vector = tile.geometry->area_vector();
    const auto area = tile.geometry->area().as_double();
    const auto magnitude = std::hypot(area_vector.x, area_vector.y, area_vector.z);
    if (!(area > 0) || !(magnitude > 0) || !std::isfinite(magnitude))
        unresolved("A canonical face-interface surface has invalid physical area.");
    PLIC::GeometryPoint n{area_vector.x / magnitude, area_vector.y / magnitude, area_vector.z / magnitude};
    long double orientation = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        orientation += n[axis] * (static_cast<long double>(second.centroid[axis]) - first.centroid[axis]);
    if (orientation == 0 || !std::isfinite(orientation))
        unresolved("A canonical face-interface surface has unresolved orientation.");
    // Geometric gas-to-carrier orientation is independent of retained traces.
    if ((orientation > 0 ? 1 : -1) != a)
        for (auto& component : n)
            component = -component;
    return {tile.face_gid, first.gid, second.gid, tile.first_owner, tile.second_owner,
        {centroid.x, centroid.y, centroid.z}, n, area};
}
// A cut reaches an authenticated face in a segment exactly when it crosses its
// interior or contains an edge. Point contacts do not connect surface patches.
bool face_trace(const AMR::AdaptiveTile& tile, const PLIC::PolyhedralInterface& plane)
{
    bool negative = false, positive = false;
    std::size_t zeros = 0;
    for (const auto& p : tile.geometry->points())
    {
        long double distance = -static_cast<long double>(plane.gas.offset);
        for (unsigned a = 0; a < 3; ++a)
            distance +=
                static_cast<long double>(plane.gas.normal[a]) * (p[a] - static_cast<long double>(plane.gas.anchor[a]));
        if (!std::isfinite(distance))
            unresolved("Capillary face trace has nonfinite placement.");
        negative = negative || distance < 0;
        positive = positive || distance > 0;
        zeros += distance == 0;
    }
    return (negative && positive) || zeros >= 2;
}
// Require a positive shared boundary segment on this actual native mesh edge.
// The existing planar roundoff band admits representation error on the same
// canonical edge; it moves no geometry and excludes point-only overlap.
bool edge_trace(const std::vector<WidePoint>& polygon, const WidePoint& a, const WidePoint& b)
{
    const auto direction = difference(b, a);
    unsigned axis = 0;
    for (unsigned k = 1; k < 3; ++k)
        if (std::abs(direction[k]) > std::abs(direction[axis]))
            axis = k;
    if (direction[axis] == 0)
        unresolved("Capillary native mesh edge has zero physical length.");
    for (std::size_t i = 0; i < polygon.size(); ++i)
    {
        const auto& p = polygon[i];
        const auto& q = polygon[(i + 1) % polygon.size()];
        const auto length = std::sqrt(dot(direction, direction));
        const auto cp = cross(difference(p, a), direction), cq = cross(difference(q, a), direction);
        const auto dp = std::sqrt(dot(cp, cp)) / length, dq = std::sqrt(dot(cq, cq)) / length;
        const auto allowance = 64 * std::numeric_limits<double>::epsilon() * length;
        const auto lower = std::max(std::min(a[axis], b[axis]), std::min(p[axis], q[axis]));
        const auto upper = std::min(std::max(a[axis], b[axis]), std::max(p[axis], q[axis]));
        if (dp <= allowance && dq <= allowance && upper - lower > 2 * allowance)
            return true;
    }
    return false;
}
} // namespace

std::vector<GeneralVoFFaceInterface> GeneralVoF::face_interfaces() const
{
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    std::vector<GeneralVoFFaceInterface> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            if (arithmetic_mode() != ArithmeticMode::Ordinary)
                throw std::invalid_argument("Explicit pure-cell face surfaces require Ordinary arithmetic.");
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& tile : data_->topology->surface_tiles)
            {
                if (tile.exterior || !tile.second)
                    continue;
                const auto& first = data_->cells.at(tile.first);
                const auto& second = data_->cells.at(*tile.second);
                const auto a = geometric_phase(first), b = geometric_phase(second);
                if (a * b != -1)
                    continue;
                budget.add(sizeof(GeneralVoFFaceInterface));
                result.push_back(face_interface(tile, first, second, a));
            }
        });
    return result;
}

GeneralVoFSurfaceCatalog GeneralVoF::capillary_surface(GeneralVoFSurfaceOptions options) const
{
    const auto comm = data_->hierarchy->communicator();
    agree_transaction(comm, data_->control);
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            if (arithmetic_mode() != ArithmeticMode::Ordinary)
                throw std::invalid_argument("Capillary surface catalogs require Ordinary arithmetic.");
            if ((options.model != GeneralVoFCapillaryModel::Reconstructed &&
                    options.model != GeneralVoFCapillaryModel::FilteredBinary) ||
                !std::isfinite(options.filter_width) || options.filter_width <= 0)
                throw std::invalid_argument("Invalid capillary surface model controls.");
        });
    // Agree before selecting any collective reconstruction path, including on
    // empty owners. A rank-local model choice must not change collective order.
    const int model = static_cast<int>(options.model);
    int model_min = 0, model_max = 0;
    double width_min = 0, width_max = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &model, &model_min);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &model, &model_max);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &options.filter_width, &width_min);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &options.filter_width, &width_max);
    wire::collective(*comm,
        [&]
        {
            if (model_min != model_max || width_min != width_max)
                throw std::invalid_argument("Capillary surface model controls differ across ranks.");
        });
    GeneralVoFSurfaceCatalog result;
    result.source_control = data_->control;
    result.model = options.model;
    // The catalog reconstructs its own owner-authenticated physical support;
    // it does not request unused transport planes as a numerical prerequisite.
    auto planes = import_capillary_planes(*data_);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& [key, plane] : planes)
                budget.add(
                    sizeof(decltype(planes)::value_type) + 4 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
            const auto grow = [&](auto& values)
            {
                if (values.size() == values.capacity())
                {
                    const auto n = values.capacity() ? 2 * values.capacity() : 1;
                    if (n < values.capacity() || n > (budget.limit - budget.used) / sizeof(values.front()))
                        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                            "Capillary catalog vector capacity exceeds its byte budget.");
                    budget.add(n * sizeof(values.front()));
                    values.reserve(n);
                }
            };
            std::map<Key, GeneralVoFSurfaceKey> mixed;
            std::map<Key, std::vector<WidePoint>> polygons;
            std::map<Edge, std::set<GeneralVoFSurfaceKey>> edge_patches;
            const auto append = [&](GeneralVoFSurfacePatch patch)
            {
                if (!(patch.area > 0) || !std::isfinite(patch.area) || patch.owner < 0 ||
                    patch.owner >= comm->getSize())
                    unresolved("Capillary patch has invalid physical area or canonical owner.");
                for (unsigned axis = 0; axis < 3; ++axis)
                    if (!std::isfinite(patch.centroid[axis]) || !std::isfinite(patch.normal[axis]))
                        unresolved("Capillary patch has nonfinite physical geometry.");
                budget.add(patch.vertices.capacity() * sizeof(PLIC::GeometryPoint) + sizeof(GeneralVoFSurfaceKey) +
                           sizeof(std::vector<GeneralVoFSurfaceKey>) + 4 * sizeof(void*));
                if (!result.adjacency.try_emplace(patch.key).second)
                    unresolved("Capillary patch canonical identity is duplicated.");
                grow(result.patches);
                result.patches.push_back(std::move(patch));
            };
            const auto connect = [&](const GeneralVoFSurfaceKey& a, const GeneralVoFSurfaceKey& b)
            {
                if (a == b)
                    return;
                auto& row = result.adjacency.at(a);
                if (std::find(row.begin(), row.end(), b) != row.end())
                    return;
                grow(row);
                row.push_back(b);
                auto& other = result.adjacency.at(b);
                grow(other);
                other.push_back(a);
            };
            for (const auto& [key, cell] : data_->cells)
            {
                const auto& plane = planes.at(key);
                if (std::getenv("SIMPLEFLUID_CAPILLARY_REPLAY"))
                {
                    std::cerr << std::setprecision(21) << "CAP_CELL control=" << data_->control
                              << " rank=" << comm->getRank() << " id=" << cell.gid
                              << " declaration=" << int(cell.declaration) << " gas=" << cell.phase.gas_volume
                              << " carrier=" << cell.phase.carrier_volume << " pure=" << plane.pure_gas << ','
                              << plane.pure_carrier << " normal=" << plane.gas.normal[0] << ',' << plane.gas.normal[1]
                              << ',' << plane.gas.normal[2] << " anchor=" << plane.gas.anchor[0] << ','
                              << plane.gas.anchor[1] << ',' << plane.gas.anchor[2] << " offset=" << plane.gas.offset
                              << " method=" << int(plane.orientation_method) << " iterations=" << plane.fit_iterations
                              << " converged=" << plane.fit_converged << " residual=" << plane.target_volume_residual
                              << " numerical_error=" << plane.target_volume_numerical_error_estimate << '\n';
                    const auto geometry =
                        PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(cell.finite_geometry);
                    std::cerr << "CAP_GEOMETRY control=" << data_->control << " rank=" << comm->getRank()
                              << " id=" << cell.gid << " centroid=" << cell.centroid[0] << ',' << cell.centroid[1]
                              << ',' << cell.centroid[2] << " volume=" << geometry->volume().value << " points=";
                    for (const auto& point : geometry->points())
                        std::cerr << point.value[0] << ',' << point.value[1] << ',' << point.value[2] << ';';
                    std::cerr << " faces=";
                    for (const auto& face : geometry->faces())
                    {
                        for (auto vertex : face)
                            std::cerr << vertex << ',';
                        std::cerr << ';';
                    }
                    std::cerr << '\n';
                }
                if (plane.pure_gas || plane.pure_carrier)
                    continue;
                const auto geometry =
                    PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(cell.finite_geometry);
                // Surface measurements require an actual closed-shell clip;
                // phase-volume shortcuts cannot provide curvature geometry.
                const auto section = geometry->clip(plane.gas, true).section;
                if (section.vertices.size() < 3 || !(section.area.value > 0) || !std::isfinite(section.area.value))
                    unresolved("A mixed capillary patch has no positive physical PLIC section.");
                GeneralVoFSurfacePatch patch;
                patch.key = {GeneralVoFSurfaceKind::MixedCell, cell.gid};
                patch.first_cell = cell.gid;
                patch.owner = cell.owner;
                patch.centroid = display(section.centroid.value);
                patch.area = section.area.as_double();
                auto n = wide(plane.gas.normal);
                const auto magnitude = std::sqrt(dot(n, n));
                if (!(magnitude > 0) || !std::isfinite(magnitude))
                    unresolved("A mixed capillary patch has no finite normal.");
                for (auto& v : n)
                    v /= magnitude;
                patch.normal = display(n);
                physical_scale(patch, cell);
                auto& polygon = polygons[key];
                budget.add(section.vertices.size() * sizeof(WidePoint) + sizeof(Key) + sizeof(std::vector<WidePoint>) +
                           4 * sizeof(void*));
                polygon.reserve(section.vertices.size());
                for (const auto& vertex : section.vertices)
                    polygon.push_back(vertex.value);
                unsigned axis = 0;
                for (unsigned a = 1; a < 3; ++a)
                    if (std::abs(n[a]) < std::abs(n[axis]))
                        axis = a;
                WidePoint reference{};
                reference[axis] = 1;
                const auto tangent = cross(reference, n), bitangent = cross(n, tangent);
                const auto origin = section.centroid.value;
                std::sort(polygon.begin(), polygon.end(),
                    [&](const WidePoint& a, const WidePoint& b)
                    {
                        return std::atan2(dot(difference(a, origin), bitangent), dot(difference(a, origin), tangent)) <
                               std::atan2(dot(difference(b, origin), bitangent), dot(difference(b, origin), tangent));
                    });
                patch.vertices.reserve(polygon.size());
                for (const auto& vertex : polygon)
                    patch.vertices.push_back(display(vertex));
                budget.add(
                    sizeof(decltype(mixed)::value_type) + 4 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
                mixed.emplace(key, patch.key);
                append(std::move(patch));
            }
            for (const auto& tile : data_->topology->surface_tiles)
            {
                if (tile.exterior || !tile.second)
                    continue;
                const auto& first = data_->cells.at(tile.first);
                const auto& second = data_->cells.at(*tile.second);
                const auto a = geometric_phase(first), b = geometric_phase(second);
                std::optional<GeneralVoFSurfaceKey> aligned;
                if (a * b == -1)
                {
                    const auto face = face_interface(tile, first, second, a);
                    GeneralVoFSurfacePatch patch;
                    patch.key = {GeneralVoFSurfaceKind::AlignedFace, tile.face_gid};
                    patch.first_cell = first.gid;
                    patch.second_cell = second.gid;
                    patch.owner = tile.first_owner;
                    patch.centroid = face.centroid;
                    patch.normal = face.normal;
                    patch.area = face.area;
                    physical_scale(patch, first);
                    GeneralVoFSurfacePatch other;
                    physical_scale(other, second);
                    patch.cell_scale = std::min(patch.cell_scale, other.cell_scale);
                    for (unsigned axis = 0; axis < 3; ++axis)
                        patch.physical_spans[axis] = std::max(patch.physical_spans[axis], other.physical_spans[axis]);
                    patch.vertices.reserve(tile.geometry->points().size());
                    for (const auto& vertex : tile.geometry->points())
                        patch.vertices.push_back(display(vertex));
                    WidePoint oriented_area{};
                    const auto anchor = wide(patch.vertices.front());
                    for (std::size_t j = 1; j + 1 < patch.vertices.size(); ++j)
                    {
                        const auto triangle = cross(difference(wide(patch.vertices[j]), anchor),
                            difference(wide(patch.vertices[j + 1]), anchor));
                        for (unsigned axis = 0; axis < 3; ++axis)
                            oriented_area[axis] += triangle[axis];
                    }
                    const auto orientation = dot(oriented_area, wide(patch.normal));
                    if (orientation == 0 || !std::isfinite(orientation))
                        unresolved("Aligned capillary patch has unresolved outward winding.");
                    if (orientation < 0)
                        std::reverse(patch.vertices.begin(), patch.vertices.end());
                    aligned = patch.key;
                    append(std::move(patch));
                }
                const auto left = mixed.find(tile.first), right = mixed.find(*tile.second);
                if (left != mixed.end() && right != mixed.end() && face_trace(tile, planes.at(tile.first)) &&
                    face_trace(tile, planes.at(*tile.second)))
                    connect(left->second, right->second);
                const auto& points = tile.geometry->points();
                if (tile.node_gids.size() != points.size())
                    unresolved("Capillary native face node loop differs from its physical polygon.");
                for (std::size_t i = 0; i < tile.node_gids.size(); ++i)
                {
                    const auto next = (i + 1) % tile.node_gids.size();
                    const Edge edge = std::minmax(tile.node_gids[i], tile.node_gids[next]);
                    const auto add_edge = [&](const GeneralVoFSurfaceKey& key)
                    {
                        budget.add(sizeof(Edge) + sizeof(GeneralVoFSurfaceKey) + 8 * sizeof(void*));
                        edge_patches[edge].insert(key);
                    };
                    if (aligned)
                        add_edge(*aligned);
                    if (left != mixed.end() && edge_trace(polygons.at(tile.first), points[i], points[next]))
                        add_edge(left->second);
                    if (right != mixed.end() && edge_trace(polygons.at(*tile.second), points[i], points[next]))
                        add_edge(right->second);
                }
            }
            for (const auto& [edge, patches] : edge_patches)
            {
                if (patches.size() > 2)
                    unresolved("Capillary mesh edge has ambiguous multiple surface sheets.");
                if (patches.size() == 2)
                    connect(*patches.begin(), *patches.rbegin());
            }
            std::sort(result.patches.begin(), result.patches.end(),
                [](const auto& a, const auto& b) { return a.key < b.key; });
            for (auto& [key, neighbors] : result.adjacency)
                std::sort(neighbors.begin(), neighbors.end());
            if (std::getenv("SIMPLEFLUID_CAPILLARY_REPLAY"))
                for (const auto& patch : result.patches)
                {
                    std::cerr << std::setprecision(21) << "CAP_PATCH control=" << data_->control
                              << " rank=" << comm->getRank() << " id=" << int(patch.key.kind) << ':'
                              << patch.key.identity << " cell=" << patch.first_cell << " area=" << patch.area
                              << " cell_scale=" << patch.cell_scale << " spans=" << patch.physical_spans[0] << ','
                              << patch.physical_spans[1] << ',' << patch.physical_spans[2]
                              << " normal=" << patch.normal[0] << ',' << patch.normal[1] << ',' << patch.normal[2]
                              << " centroid=" << patch.centroid[0] << ',' << patch.centroid[1] << ','
                              << patch.centroid[2] << " adjacency=";
                    for (const auto& neighbor : result.adjacency.at(patch.key))
                        std::cerr << int(neighbor.kind) << ':' << neighbor.identity << ';';
                    std::cerr << " vertices=";
                    for (const auto& vertex : patch.vertices)
                        std::cerr << vertex[0] << ',' << vertex[1] << ',' << vertex[2] << ';';
                    std::cerr << '\n';
                }
            result.allocation_bytes = budget.used;
        });
    planes.clear();
    if (options.model == GeneralVoFCapillaryModel::FilteredBinary)
    {
        const auto available = data_->options.byte_budget - result.allocation_bytes;
        wire::collective(*comm,
            [&]
            {
                if (!available)
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                        "Capillary raw catalog exhausts the node-fan import budget.");
            });
        const auto node_fans = import_capillary_node_fans(*data_, available);
        wire::collective(*comm,
            [&]
            {
                wire::Budget retained{data_->options.byte_budget};
                retained.add(result.allocation_bytes);
                retained.add(node_fans.allocation_bytes);
                if (retained.used == retained.limit)
                    throw PLIC::GeometryError(
                        PLIC::GeometryErrorCode::ResourceLimit, "Capillary raw catalog exhausts the filtering budget.");
                auto filtered = general_vof_detail::filtered_binary_surface(
                    *data_, options, node_fans, retained.limit - retained.used);
                retained.add(filtered.allocation_bytes);
                retained.add(filtered.filtered_source_cells.size() * (sizeof(std::uint64_t) + 4 * sizeof(void*)));
                const std::set<std::uint64_t> replaced(
                    filtered.filtered_source_cells.begin(), filtered.filtered_source_cells.end());
                std::set<GeneralVoFSurfaceKey> removed;
                for (const auto& patch : result.patches)
                    if (patch.key.kind == GeneralVoFSurfaceKind::AlignedFace &&
                        (replaced.contains(patch.first_cell) || replaced.contains(patch.second_cell)))
                    {
                        retained.add(sizeof(GeneralVoFSurfaceKey) + 4 * sizeof(void*));
                        removed.insert(patch.key);
                    }
                std::erase_if(result.patches, [&](const auto& p) { return removed.contains(p.key); });
                for (const auto& key : removed)
                    result.adjacency.erase(key);
                for (auto& [key, adjacent] : result.adjacency)
                    std::erase_if(adjacent, [&](const auto& p) { return removed.contains(p); });
                const auto total = result.patches.size() + filtered.patches.size();
                if (total > result.patches.capacity())
                {
                    if (total > (retained.limit - retained.used) / sizeof(GeneralVoFSurfacePatch))
                        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                            "Merged capillary catalog exceeds its byte budget.");
                    retained.add(total * sizeof(GeneralVoFSurfacePatch));
                    result.patches.reserve(total);
                }
                for (auto& patch : filtered.patches)
                    result.patches.push_back(std::move(patch));
                result.adjacency.merge(filtered.adjacency);
                if (!filtered.adjacency.empty())
                    unresolved("Filtered and reconstructed capillary patch identities collide.");
                result.components = std::move(filtered.components);
                result.filtered_source_cells = std::move(filtered.filtered_source_cells);
                result.allocation_bytes = retained.used;
                std::sort(result.patches.begin(), result.patches.end(),
                    [](const auto& a, const auto& b) { return a.key < b.key; });
            });
    }
    return result;
}
} // namespace SimpleFluid::VoF
