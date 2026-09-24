#pragma once

#include "fields/details/MeshTransferPolygon.hh"
#include "fields/details/RZAngularOverlap.hh"
#include "geometry/mesh/SweptRZRegionProviders.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <variant>
#include <vector>

namespace SimpleFluid::mesh_transfer_detail
{

// Cartesian centroid, native cell volume, then lower/upper bounds for either
// x,y,z or r,theta,z. Theta lower bounds are reduced modulo 2 pi; upper bounds
// may cross that seam. All records follow the owned field-map order.
enum class GeometryKind { Interpolation, Cartesian, Cylindrical, PolygonPrism, RZAngular, Composite };

struct Geometry
{
    std::array<double, 10> values{};
    GeometryKind kind = GeometryKind::Interpolation;
    // CCW physical XY vertices. Polygon records use entries 4/5 for enclosing
    // radii, 6 for polygon area, and 8/9 for the actual axial bounds.
    // RZAngular instead stores CCW (apothem,physical Z), enclosing radii in
    // 4/5, angular bounds in 6/7, and enclosing axial bounds in 8/9.
    std::vector<PolygonPoint> polygon;
    double& operator[](size_t i) { return values[i]; }
    double operator[](size_t i) const { return values[i]; }
    auto begin() { return values.begin(); }
    auto end() { return values.end(); }
    auto begin() const { return values.begin(); }
    auto end() const { return values.end(); }
};

template<class Mesh>
GeometryKind geometry_kind(const Mesh& mesh)
{
    if (std::holds_alternative<typename Mesh::CartesianPtr>(mesh.variant()))
        return GeometryKind::Cartesian;
    if (std::holds_alternative<typename Mesh::CylindricalPtr>(mesh.variant()))
        return GeometryKind::Cylindrical;
    if (std::holds_alternative<typename Mesh::SemiStructuredPtr>(mesh.variant()))
        return GeometryKind::PolygonPrism;
    if (std::holds_alternative<typename Mesh::MultiRegionPtr>(mesh.variant()))
        return GeometryKind::Composite;
    return GeometryKind::Interpolation;
}

inline constexpr double angular_period = 2.0 * std::numbers::pi_v<double>;

inline double normalized_angle(double angle)
{
    double result = std::fmod(angle, angular_period);
    if (result < 0) result += angular_period;
    // Adding the period to a tiny negative remainder can round to the period.
    return result >= angular_period ? 0.0 : result;
}

template<class Mesh>
std::vector<Geometry> geometry(const Mesh& mesh, bool conservative, bool& valid)
{
    const auto* cartesian =
        std::get_if<typename Mesh::CartesianPtr>(&mesh.variant());
    const auto* cylindrical =
        std::get_if<typename Mesh::CylindricalPtr>(&mesh.variant());
    const auto* semi_structured =
        std::get_if<typename Mesh::SemiStructuredPtr>(&mesh.variant());
    const auto* composite =
        std::get_if<typename Mesh::MultiRegionPtr>(&mesh.variant());
    valid = valid && (!conservative || cartesian || cylindrical || semi_structured || composite);
    const auto execution = mesh.acquire_execution_view();
    if (conservative && composite)
    {
        // A cell reported as a hexahedron may still have curved cylindrical
        // faces. Each allowed provider supplies one explicitly supported
        // geometry contract; cell type alone is not sufficient recognition.
        for (const auto& region : (**composite).regions())
            valid = valid && (std::holds_alternative<Meshes::NativeIsoRegion<Meshes::SemiStructuredXY_Z>>(region)
                || std::holds_alternative<Meshes::ExtrudedRegion>(region)
                || std::holds_alternative<Meshes::SweptRZRegion>(region));
        if (!valid) return {};
    }

    // Native cylindrical topology tolerates a roundoff-scale excess over one
    // turn. Its stored volumes still include that excess. For overlap geometry
    // only, represent the first physical turn once and leave the extra native
    // volume uncovered. In particular, never normalize weights or volumes to
    // hide this seam residual. A cell wholly inside the excess has zero overlap.
    bool trim_repeated_seam = false;
    double angular_origin = 0;
    if (conservative && cylindrical)
    {
        const auto& angles = (**cylindrical).cell_edges()[1];
        angular_origin = angles.front();
        const double span = angles.back() - angular_origin;
        constexpr double native_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() * angular_period;
        valid = valid && std::isfinite(span) && span > 0
            && span <= angular_period + native_tolerance;
        trim_repeated_seam = span > angular_period;
    }

    std::vector<Geometry> result(mesh.num_owned_cells());
    for (std::size_t c = 0; c < result.size(); ++c)
    {
        const auto lid = static_cast<typename Mesh::local_ordinal_type>(c);
        auto& g = result[c];
        g.kind = conservative ? geometry_kind(mesh) : GeometryKind::Interpolation;
        const auto centroid = mesh.cell_centroid(lid);
        for (std::size_t axis = 0; axis < 3; ++axis)
            g[axis] = centroid.component(axis);
        g[3] = mesh.cell_volume(lid);
        if (conservative)
        {
            const auto read_bounds = [&](const auto& native)
            {
                // Map IDs may change when a handle is reordered. Geometry IDs
                // remain attached to the original cell in the native backend.
                const auto cell = native.cell_id(
                    static_cast<std::size_t>(mesh.cell_geometry_global_id(lid)));
                const std::array<std::size_t, 3> indices{cell.i, cell.j, cell.k};
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    g[4 + 2 * axis] = native.cell_edges()[axis][indices[axis]];
                    g[5 + 2 * axis] = native.cell_edges()[axis][indices[axis] + 1];
                }
            };
            if (cartesian) read_bounds(**cartesian);
            else if (cylindrical)
            {
                read_bounds(**cylindrical);
                double angular_width = g[7] - g[6];
                if (trim_repeated_seam)
                {
                    const double lower =
                        std::min(angular_period, g[6] - angular_origin);
                    const double upper =
                        std::min(angular_period, g[7] - angular_origin);
                    angular_width = upper - lower;
                }
                g[6] = normalized_angle(g[6]);
                g[7] = g[6] + angular_width;
                valid = valid && g[4] > 0 && angular_width >= 0;
            }
            else if (semi_structured)
            {
                const auto& native = **semi_structured;
                const auto cell = native.cell_id(static_cast<size_t>(mesh.cell_geometry_global_id(lid)));
                for (const auto node : native.xy_cell_nodes()[cell.ij])
                {
                    const auto point = native.xy_nodes()[node];
                    g.polygon.push_back({point.x, point.y});
                }
                g[8] = native.z_edges()[cell.k];
                g[9] = native.z_edges()[cell.k + 1];
            }
            else if (composite)
            {
                const auto& native = **composite;
                const auto id = static_cast<typename Mesh::MultiRegion::ID>(mesh.cell_geometry_global_id(lid));
                const auto nodes = native.cell_nodes(id);
                if (nodes.size() < 6 || nodes.size() % 2 != 0) { valid = false; continue; }
                const auto count = nodes.size() / 2;
                const auto [region_id, region_cell] = native.native_cell(id);
                if (const auto* swept = std::get_if<Meshes::SweptRZRegion>(&native.regions()[region_id]))
                {
                    g.kind = GeometryKind::RZAngular;
                    const auto cell = swept->topology().indexer().cell_id(region_cell);
                    const auto base_nodes = swept->topology().base_cell_nodes(cell.ij);
                    const auto& angles = swept->geometry().theta_edges();
                    const auto width = angles[cell.k + 1] - angles[cell.k];
                    valid = valid && base_nodes.size() == count && width > 0 && width < std::numbers::pi_v<double>;
                    if (!valid) continue;
                    const auto cosine = std::cos(width / 2);
                    g[6] = normalized_angle(angles[cell.k]);
                    g[7] = g[6] + width;
                    g[8] = std::numeric_limits<double>::infinity();
                    g[9] = -g[8];
                    g[4] = std::numeric_limits<double>::infinity();
                    g[5] = 0;
                    for (size_t i = 0; i < count; ++i)
                    {
                        const auto first = native.node_coordinates(nodes[i]);
                        const auto second = native.node_coordinates(nodes[count + i]);
                        valid = valid && first.z == second.z;
                        const auto radius = swept->geometry().rz_nodes()[base_nodes[i]].x;
                        const auto apothem = radius * cosine;
                        g.polygon.push_back({apothem, first.z});
                        g[4] = std::min(g[4], apothem);
                        g[5] = std::max(g[5], radius);
                        g[8] = std::min(g[8], first.z);
                        g[9] = std::max(g[9], first.z);
                    }
                    // Swept native templates are clockwise in radius/Z;
                    // overlap moment integration uses CCW apothem/Z loops.
                    std::reverse(g.polygon.begin(), g.polygon.end());
                    valid = valid && g[9] > g[8] && g[4] > 0;
                    if (!valid) continue;
                    rz_overlap_detail::validate(g.polygon, g[6], g[7]);
                    const auto volume = rz_angular_volume_validated(g.polygon, g[6], g[7]);
                    valid = valid && std::abs(volume - g[3]) <= 1e-10L * g[3];
                    g[4] = std::max(0.0, std::nextafter(g[4], -std::numeric_limits<double>::infinity()));
                    g[5] = std::nextafter(g[5], std::numeric_limits<double>::infinity());
                }
                else
                {
                    g.kind = GeometryKind::PolygonPrism;
                    g[8] = native.node_coordinates(nodes[0]).z;
                    g[9] = native.node_coordinates(nodes[count]).z;
                    g.polygon.reserve(count);
                    for (size_t i = 0; i < count; ++i)
                    {
                        // Canonical queries include the current composite affine
                        // transform; region coordinates alone remain reference Z.
                        const auto bottom = native.node_coordinates(nodes[i]);
                        const auto top = native.node_coordinates(nodes[count + i]);
                        valid = valid && bottom.z == g[8] && top.z == g[9]
                            && bottom.x == top.x && bottom.y == top.y;
                        g.polygon.push_back({bottom.x, bottom.y});
                    }
                }
            }
            if (g.kind == GeometryKind::PolygonPrism)
            {
                for (const auto point : g.polygon)
                    valid = valid && std::isfinite(point[0]) && std::isfinite(point[1]);
                valid = valid && convex_polygon(g.polygon) && g[9] > g[8];
                if (!valid) continue;
                const auto area = polygon_area(g.polygon);
                const auto geometric_volume = area * (static_cast<long double>(g[9]) - g[8]);
                valid = valid && std::abs(geometric_volume - g[3]) <= 1e-10L * g[3];
                g[6] = static_cast<double>(area);
                const auto radial = polygon_radial_bounds(g.polygon);
                g[4] = radial[0];
                g[5] = radial[1];
            }
        }
        for (double value : g) valid = valid && std::isfinite(value);
        valid = valid && g[3] > 0;
    }
    return result;
}

inline double angular_intersection(double a_lower, double a_upper,
                                   double b_lower, double b_upper)
{
    // Each arc is represented once and intersects at most the central image
    // and one adjacent periodic image. Use extended precision for seam sums;
    // no angular quadrature or symmetry multiplier is involved.
    const long double period = angular_period;
    const long double a0 = normalized_angle(a_lower);
    const long double b0 = normalized_angle(b_lower);
    const long double aw = std::min(period,
        static_cast<long double>(a_upper) - a_lower);
    const long double bw = std::min(period,
        static_cast<long double>(b_upper) - b_lower);
    if (aw <= 0 || bw <= 0) return 0;
    long double overlap = 0;
    for (int image = -1; image <= 1; ++image)
    {
        const long double start = b0 + image * period;
        overlap += std::max(0.0L,
            std::min(a0 + aw, start + bw) - std::max(a0, start));
    }
    return static_cast<double>(overlap);
}

inline double intersection(const Geometry& a, const Geometry& b)
{
    if (a.kind == GeometryKind::Cartesian && b.kind == GeometryKind::Cartesian)
    {
        double volume = 1;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto low = 4 + 2 * axis;
            volume *= std::max(0.0, std::min(a[low + 1], b[low + 1])
                                       - std::max(a[low], b[low]));
        }
        return volume;
    }

    const double rlo = std::max(a[4], b[4]);
    const double rhi = std::min(a[5], b[5]);
    const double zlo = std::max(a[8], b[8]);
    const double zhi = std::min(a[9], b[9]);
    if (rhi <= rlo || zhi <= zlo) return 0;
    if (a.kind == GeometryKind::RZAngular || b.kind == GeometryKind::RZAngular)
    {
        const auto& swept = a.kind == GeometryKind::RZAngular ? a : b;
        const auto& cylinder = a.kind == GeometryKind::Cylindrical ? a : b;
        if (cylinder.kind != GeometryKind::Cylindrical)
            throw std::invalid_argument("RZ angular conservative overlap requires a cylindrical partner.");
        return rz_angular_sector_volume_validated(swept.polygon, swept[6], swept[7],
            cylinder[4], cylinder[5], cylinder[6], cylinder[7], zlo, zhi);
    }
    if (a.kind == GeometryKind::PolygonPrism || b.kind == GeometryKind::PolygonPrism)
    {
        const auto& polygon = a.kind == GeometryKind::PolygonPrism ? a : b;
        const auto& sector = a.kind == GeometryKind::Cylindrical ? a : b;
        const double width = sector[7] - sector[6];
        if (polygon[4] >= sector[4] && polygon[5] <= sector[5]
            && width <= std::numbers::pi_v<double>)
        {
            const long double lower = sector[6], upper = sector[7];
            const WidePoint first{std::cos(lower), std::sin(lower)}, last{std::cos(upper), std::sin(upper)};
            const bool inside = std::all_of(polygon.polygon.begin(), polygon.polygon.end(),
                [&](PolygonPoint p) { return cross(first, wide(p)) >= 0 && cross(last, wide(p)) <= 0; });
            if (inside) return polygon[6] * (zhi - zlo);
        }
        const auto area = polygon_sector_area(polygon.polygon, sector[4], sector[5],
                                               sector[6], width);
        return area * (zhi - zlo);
    }
    return 0.5 * (rhi - rlo) * (rhi + rlo)
        * angular_intersection(a[6], a[7], b[6], b[7]) * (zhi - zlo);
}

inline size_t serialized_size(const Geometry& g) { return 12 + 2 * g.polygon.size(); }

inline void append_geometry(const Geometry& g, std::vector<double>& data)
{
    data.insert(data.end(), g.begin(), g.end());
    data.push_back(static_cast<double>(g.kind));
    data.push_back(static_cast<double>(g.polygon.size()));
    for (const auto p : g.polygon) { data.push_back(p[0]); data.push_back(p[1]); }
}

inline Geometry read_geometry(const std::vector<double>& data, size_t& offset)
{
    Geometry g;
    std::copy_n(data.begin() + offset, 10, g.begin());
    offset += 10;
    g.kind = static_cast<GeometryKind>(static_cast<int>(data[offset++]));
    const auto count = static_cast<size_t>(data[offset++]);
    g.polygon.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        g.polygon.push_back({data[offset], data[offset + 1]});
        offset += 2;
    }
    return g;
}

} // namespace SimpleFluid::mesh_transfer_detail
