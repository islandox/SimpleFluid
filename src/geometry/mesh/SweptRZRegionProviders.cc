/** @file SweptRZRegionProviders.cc
 * @brief Factored radius-height topology with exact planar angular-sweep metrics.
 */
#include "geometry/mesh/SweptRZRegionProviders.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace SimpleFluid::Meshes
{
SweptRZGeometry::SweptRZGeometry(std::shared_ptr<const ExtrudedTopology> topology,
    Arr<Vec3> nodes, ArrReal theta)
    : d_topology(std::move(topology)), d_nodes(std::move(nodes)), d_theta(std::move(theta)),
      d_bounds{std::numeric_limits<real_t>::infinity(), -std::numeric_limits<real_t>::infinity()}
{
    if (!d_topology || d_nodes.size() != d_topology->indexer().num_nodes_per_layer
        || d_theta.size() != static_cast<size_t>(d_topology->indexer().num_layers) + 1)
        throw std::invalid_argument("Swept RZ geometry and template extents mismatch.");
    for (const auto& p : d_nodes)
    {
        if (!(p.x > 0) || !std::isfinite(p.x) || !std::isfinite(p.y) || p.z != 0)
            throw std::invalid_argument("Swept RZ nodes require positive finite radius, finite height, and zero third coordinate.");
        d_bounds[0] = std::min(d_bounds[0], p.y);
        d_bounds[1] = std::max(d_bounds[1], p.y);
    }
    for (size_t k = 0; k < d_theta.size(); ++k)
    {
        const auto width = k ? d_theta[k] - d_theta[k - 1] : 0.0;
        if (!std::isfinite(d_theta[k])
            || (k && (!(width > 0) || !(width < std::numbers::pi_v<real_t>))))
            throw std::invalid_argument("Swept angular edges must be finite and increase by less than pi per cell.");
    }
    if (d_theta.back() - d_theta.front() > 2 * std::numbers::pi_v<real_t>
        * (1 + 64 * std::numeric_limits<real_t>::epsilon()))
        throw std::invalid_argument("Swept angular geometry cannot cover more than one full turn.");

    d_metrics.reserve(d_topology->indexer().num_cells_per_layer);
    for (size_t cell = 0; cell < d_topology->indexer().num_cells_per_layer; ++cell)
    {
        const auto loop = d_topology->base_cell_nodes(cell);
        if (loop.size() != 3 && loop.size() != 4)
            throw std::invalid_argument("Swept RZ cells must be triangles or quadrilaterals.");
        const auto origin = d_nodes.at(loop[0]);
        long double area2 = 0, mu6 = 0, mv6 = 0, muu12 = 0, muv24 = 0;
        for (size_t edge = 0; edge < loop.size(); ++edge)
        {
            const auto a = d_nodes.at(loop[edge]);
            const auto b = d_nodes.at(loop[(edge + 1) % loop.size()]);
            for (size_t other = 0; other < loop.size(); ++other)
            {
                if (other == edge || other == (edge + 1) % loop.size()) continue;
                const auto c = d_nodes.at(loop[other]);
                const auto turn = (static_cast<long double>(b.x) - a.x) * (c.y - a.y)
                                - (static_cast<long double>(b.y) - a.y) * (c.x - a.x);
                if (!(turn < 0))
                    throw std::invalid_argument("Swept RZ base cells must be strictly convex and clockwise.");
            }
            const long double u = static_cast<long double>(a.x) - origin.x;
            const long double v = static_cast<long double>(a.y) - origin.y;
            const long double w = static_cast<long double>(b.x) - origin.x;
            const long double t = static_cast<long double>(b.y) - origin.y;
            const auto cross = -(u * t - w * v); // Convert clockwise to positive moments.
            area2 += cross;
            mu6 += (u + w) * cross;
            mv6 += (v + t) * cross;
            muu12 += (u * u + u * w + w * w) * cross;
            muv24 += (2 * u * v + u * t + w * v + 2 * w * t) * cross;
        }
        const auto area = area2 / 2;
        const auto mu = mu6 / 6;
        const auto mv = mv6 / 6;
        const auto radial = origin.x * area + mu;
        const auto radial2 = static_cast<long double>(origin.x) * origin.x * area
                           + 2 * origin.x * mu + muu12 / 12;
        // Separate the height origin to retain small-cell precision far from z=0.
        const auto weighted_height = origin.y + (origin.x * mv + muv24 / 24) / radial;
        BaseMetrics metric{static_cast<real_t>(area), static_cast<real_t>(radial),
            static_cast<real_t>(radial2 / radial), static_cast<real_t>(weighted_height),
            static_cast<real_t>(origin.x + mu / area), static_cast<real_t>(origin.y + mv / area)};
        if (!(metric.area > 0) || !(metric.radial_moment > 0)
            || !std::isfinite(metric.area) || !std::isfinite(metric.radial_moment)
            || !std::isfinite(metric.radial_second_over_first)
            || !std::isfinite(metric.radial_height_over_first)
            || !std::isfinite(metric.centroid_radius) || !std::isfinite(metric.centroid_height))
            throw std::invalid_argument("Swept RZ moments must be positive and finite.");
        d_metrics.push_back(metric);
    }
    // Extremal angular sine factors suffice for finite positive volume checks.
    real_t smallest = 1, largest = 0;
    for (size_t k = 1; k < d_theta.size(); ++k)
    {
        const auto factor = std::sin(d_theta[k] - d_theta[k - 1]);
        smallest = std::min(smallest, factor);
        largest = std::max(largest, factor);
    }
    for (const auto& metric : d_metrics)
        if (!(metric.radial_moment * smallest > 0)
            || !std::isfinite(metric.radial_moment * largest))
            throw std::invalid_argument("Swept RZ cell volumes are not representable.");
}

SweptRZGeometry::Vec3 SweptRZGeometry::physical_node(unsigned base, unsigned angle) const
{
    const auto& p = d_nodes.at(base);
    const auto theta = d_theta.at(angle);
    return {p.x * std::cos(theta), p.x * std::sin(theta), p.y};
}

real_t SweptRZGeometry::cell_volume(size_t cell) const
{
    if (cell >= layout().cells) throw std::out_of_range("Swept cell ordinal out of bounds.");
    const auto id = d_topology->indexer().cell_id(cell);
    return d_metrics.at(id.ij).radial_moment * std::sin(d_theta[id.k + 1] - d_theta[id.k]);
}

SweptRZGeometry::Vec3 SweptRZGeometry::cell_centroid(size_t cell) const
{
    if (cell >= layout().cells) throw std::out_of_range("Swept cell ordinal out of bounds.");
    const auto id = d_topology->indexer().cell_id(cell);
    const auto& m = d_metrics.at(id.ij);
    const auto a = d_theta[id.k], b = d_theta[id.k + 1];
    return {0.5 * (std::cos(a) + std::cos(b)) * m.radial_second_over_first,
            0.5 * (std::sin(a) + std::sin(b)) * m.radial_second_over_first,
            m.radial_height_over_first};
}

SweptRZGeometry::Vec3 SweptRZGeometry::face_area_vector(size_t face) const
{
    if (face >= layout().faces) throw std::out_of_range("Swept face ordinal out of bounds.");
    const auto id = d_topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
    {
        const auto area = d_metrics.at(id.ij).area * (id.k == 0 ? -1.0 : 1.0);
        return {-area * std::sin(d_theta[id.k]), area * std::cos(d_theta[id.k]), 0};
    }
    const auto& edge = d_topology->native_topology().side_face(id.ij);
    const auto a = d_nodes.at(edge.nodes[0]), b = d_nodes.at(edge.nodes[1]);
    const auto t0 = d_theta[id.k], t1 = d_theta[id.k + 1];
    const auto radial_factor = -0.5 * (b.y - a.y) * (a.x + b.x);
    return {radial_factor * (std::sin(t1) - std::sin(t0)),
            radial_factor * (std::cos(t0) - std::cos(t1)),
            0.5 * (b.x - a.x) * (a.x + b.x) * std::sin(t1 - t0)};
}

real_t SweptRZGeometry::face_area(size_t face) const
{
    return face_area_vector(face).norm();
}

SweptRZGeometry::Vec3 SweptRZGeometry::face_centroid(size_t face) const
{
    if (face >= layout().faces) throw std::out_of_range("Swept face ordinal out of bounds.");
    const auto id = d_topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
    {
        const auto& m = d_metrics.at(id.ij);
        return {m.centroid_radius * std::cos(d_theta[id.k]),
                m.centroid_radius * std::sin(d_theta[id.k]), m.centroid_height};
    }
    const auto& edge = d_topology->native_topology().side_face(id.ij);
    const std::vector<Vec3> points{physical_node(edge.nodes[0], id.k),
        physical_node(edge.nodes[1], id.k), physical_node(edge.nodes[1], id.k + 1),
        physical_node(edge.nodes[0], id.k + 1)};
    return MeshUtils::face_centroid(points);
}

SweptRZGeometry::Vec3 SweptRZGeometry::node_coordinates(size_t node) const
{
    if (node >= layout().nodes) throw std::out_of_range("Swept node ordinal out of bounds.");
    const auto id = d_topology->indexer().node_id(node);
    return physical_node(id.ij, id.k);
}

MeshStorageReport SweptRZGeometry::storage_report() const
{
    return {.geometry = d_nodes.capacity() * sizeof(Vec3) + d_theta.capacity() * sizeof(real_t)
                      + d_metrics.capacity() * sizeof(BaseMetrics)};
}

EntityRange<EntityRange<uint64_t>> SweptRZGeometry::rz_cell_nodes() const
{
    return {d_topology.get(), 0, d_topology->indexer().num_cells_per_layer,
        [](const void* topology, size_t, size_t cell)
        { return static_cast<const ExtrudedTopology*>(topology)->base_cell_nodes(cell); }};
}

SweptRZRegion swept_rz_region(std::string name, std::shared_ptr<const ExtrudedTopology> topology,
    Arr<MeshUtils::Vec3> nodes, ArrReal theta)
{
    auto geometry = std::make_shared<const SweptRZGeometry>(topology, std::move(nodes), std::move(theta));
    return {std::move(name), std::move(topology), std::move(geometry)};
}
} // namespace SimpleFluid::Meshes
