#include "geometry/details/ConvexCellGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
std::vector<char> topology_descriptor(const ExtrudedTopology& topology)
{
    WireWriter w;
    const auto& indexer = topology.indexer();
    w.word(indexer.num_nodes_per_layer);
    w.word(indexer.num_layers);
    w.word(indexer.num_cells_per_layer);
    for (std::size_t c = 0; c < indexer.num_cells_per_layer; ++c)
    {
        auto nodes = topology.base_cell_nodes(c);
        w.word(nodes.size());
        for (auto n : nodes)
            w.word(n);
    }
    const auto ids = topology.boundary_batch_ids();
    w.word(ids.size());
    for (auto id : ids)
    {
        const auto& name = topology.boundary_batch_name(id);
        w.word(name.size());
        for (unsigned char ch : name)
            w.word(ch);
    }
    for (const auto& e : topology.native_topology().side_faces())
    {
        w.word(e.nodes[0]);
        w.word(e.nodes[1]);
        w.word(e.owner);
        w.word(e.neighbor);
        w.word(e.boundary_id);
    }
    return std::move(w.data);
}

MeshUtils::Vec3 rounded_point(const P& p)
{
    return {measured(p[0]).as_double(), measured(p[1]).as_double(), measured(p[2]).as_double()};
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
GeometryPoint ConvexCellGeometry::centroid() const
{
    const auto origin = point(impl_->origin);
    const auto moments = exact_moments(impl_->poly, origin);
    const auto center = rounded_point(add(origin, mul(moments.first, Q(1) / moments.volume)));
    return {center.x, center.y, center.z};
}

std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::make(
    std::shared_ptr<const ExtrudedTopology> topology, std::vector<ExactPointRecipe> recipes,
    std::vector<std::size_t> active, std::vector<ExactScalarRecipe> zrecipes, std::vector<std::size_t> zactive)
{
    if (!topology || active.size() != topology->indexer().num_nodes_per_layer ||
        zactive.size() != std::size_t(topology->indexer().num_layers) + 1)
        throw std::invalid_argument("Exact extrusion construction and retained topology differ.");
    auto impl = std::make_shared<Impl>();
    impl->topology = std::move(topology);
    impl->base->topology = impl->topology->encode_base();
    impl->base->recipes = std::move(recipes);
    impl->axis->values->zrecipes = std::move(zrecipes);
    impl->base->active = std::move(active);
    impl->axis->zactive = std::move(zactive);
    for (std::size_t i = 0; i < impl->base->recipes.size(); ++i)
    {
        const auto& recipe = impl->base->recipes[i];
        P p{};
        (void) point(recipe.root);
        if (recipe.parents.empty())
            p = point(recipe.root);
        else
        {
            for (auto parent : recipe.parents)
            {
                if (parent >= i)
                    throw std::invalid_argument("Exact point construction is not an acyclic ancestry DAG.");
                p = add(p, impl->base->points[parent]);
            }
            p = mul(p, Q(1, recipe.parents.size()));
        }
        if (p[2] != 0)
            throw std::invalid_argument("Exact extrusion base must lie in XY.");
        impl->base->points.push_back(std::move(p));
    }
    std::set<std::size_t> unique;
    std::set<P> unique_points;
    std::set<GeometryPoint> display;
    for (auto id : impl->base->active)
    {
        if (id >= impl->base->points.size() || !unique.insert(id).second ||
            !unique_points.insert(impl->base->points[id]).second)
            throw std::invalid_argument("Exact extrusion has invalid/duplicate active points.");
        const auto p = rounded_point(impl->base->points[id]);
        if (!display.insert({p.x, p.y, p.z}).second)
            throw std::invalid_argument("Exact extrusion display vertices are indistinguishable.");
    }
    for (std::size_t i = 0; i < impl->axis->values->zrecipes.size(); ++i)
    {
        const auto& recipe = impl->axis->values->zrecipes[i];
        Q z = 0;
        (void) exact(recipe.root);
        if (recipe.parents.empty())
            z = exact(recipe.root);
        else
        {
            for (auto parent : recipe.parents)
            {
                if (parent >= i)
                    throw std::invalid_argument("Exact axial construction is not acyclic.");
                z += impl->axis->values->z[parent];
            }
            z /= recipe.parents.size();
        }
        impl->axis->values->z.push_back(z);
    }
    Q previous = 0;
    double previous_display = 0;
    for (std::size_t k = 0; k < impl->axis->zactive.size(); ++k)
    {
        const auto id = impl->axis->zactive[k];
        if (id >= impl->axis->values->z.size())
            throw std::invalid_argument("Exact axial point index is invalid.");
        const auto value = measured(impl->axis->values->z[id]).as_double();
        if (k && (!(impl->axis->values->z[id] > previous) || !(value > previous_display)))
            throw std::invalid_argument("Exact axial/display edges must be distinct and increasing.");
        previous = impl->axis->values->z[id];
        previous_display = value;
        if (k)
        {
            const auto lo = measured(impl->axis->values->z[impl->axis->zactive[k - 1]]).as_double();
            const auto center =
                measured(Q((impl->axis->values->z[id] + impl->axis->values->z[impl->axis->zactive[k - 1]]) / 2))
                    .as_double();
            if (!(center > lo && center < value))
                throw std::invalid_argument("Exact extrusion has unrepresentable signed axial metric distances.");
        }
    }
    for (std::size_t c = 0; c < impl->topology->indexer().num_cells_per_layer; ++c)
    {
        Product product;
        product.lower = 0;
        product.upper = 1;
        for (auto n : impl->topology->base_cell_nodes(c))
            product.base.push_back(impl->base->points[impl->base->active[n]]);
        for (std::size_t i = 0; i < product.base.size(); ++i)
            if (cross(sub(product.base[(i + 1) % product.base.size()], product.base[i]),
                    sub(product.base[(i + 2) % product.base.size()], product.base[(i + 1) % product.base.size()]))[2] <=
                0)
                throw std::invalid_argument("Exact extrusion base must be strictly convex and counterclockwise.");
        (void) product_poly(product);
        std::vector<P> display_polygon;
        for (const auto& p : product.base)
        {
            auto v = rounded_point(p);
            display_polygon.push_back(point({v.x, v.y, 0}));
        }
        for (std::size_t i = 0; i < display_polygon.size(); ++i)
        {
            const auto a = display_polygon[i], b = display_polygon[(i + 1) % display_polygon.size()],
                       c = display_polygon[(i + 2) % display_polygon.size()];
            if (cross(sub(b, a), sub(c, b))[2] <= 0)
                throw std::invalid_argument(
                    "Exact extrusion display polygon is not a distinct strictly convex native loop.");
            for (const auto& p : display_polygon)
                if (cross(sub(b, a), sub(p, a))[2] < 0)
                    throw std::invalid_argument("Exact extrusion display polygon is nonsimple.");
        }
        const auto m = product_moments(product, P{});
        impl->base->areas.push_back(m.volume);
        impl->base->centers.push_back({m.first[0] / m.volume, m.first[1] / m.volume, 0});
    }
    for (const auto& edge : impl->topology->native_topology().side_faces())
    {
        const auto a = impl->base->points[impl->base->active[edge.nodes[0]]],
                   b = impl->base->points[impl->base->active[edge.nodes[1]]];
        impl->base->deltas.push_back(sub(b, a));
        impl->base->edge_centers.push_back(mul(add(a, b), Q(1, 2)));
    }
    for (std::size_t e = 0; e < impl->base->deltas.size(); ++e)
    {
        const auto& edge = impl->topology->native_topology().side_face(e);
        const auto face = rounded_point(impl->base->edge_centers[e]);
        const auto owner = rounded_point(impl->base->centers.at(edge.owner));
        const auto& d = impl->base->deltas[e];
        const auto normal = rounded_point({d[1], -d[0], 0});
        const auto owner_distance = (face - owner).dot(normal);
        if (!std::isfinite(owner_distance) || !(owner_distance > 0))
            throw std::invalid_argument("Exact extrusion has unrepresentable signed base metric distances.");
        if (edge.neighbor != SemiStructMeshTopo::invalid_ordinal)
        {
            const auto neighbor = rounded_point(impl->base->centers.at(edge.neighbor));
            const auto neighbor_distance = (face - neighbor).dot(normal);
            if (!std::isfinite(neighbor_distance) || !(neighbor_distance < 0))
                throw std::invalid_argument("Exact extrusion neighbor metric distance is unresolved.");
        }
    }
    impl->layers = impl->axis->zactive.size() - 1;
    for (auto id : impl->base->active)
        impl->base->display.push_back(rounded_point(impl->base->points[id]));
    for (auto id : impl->axis->zactive)
        impl->axis->display.push_back(measured(impl->axis->values->z[id]).as_double());
    return std::shared_ptr<const ExactExtrudedConstruction>(new ExactExtrudedConstruction(std::move(impl)));
}

bool ExactExtrudedConstruction::matches_topology(const std::shared_ptr<const ExtrudedTopology>& topology) const noexcept
{
    return impl_->topology == topology;
}

std::vector<MeshUtils::Vec3> ExactExtrudedConstruction::display_points() const
{
    return impl_->base->display;
}

std::vector<double> ExactExtrudedConstruction::display_z() const
{
    const auto span = z_view();
    return {span.begin(), span.end()};
}

const std::vector<MeshUtils::Vec3>& ExactExtrudedConstruction::display_points_ref() const noexcept
{
    return impl_->base->display;
}

std::span<const double> ExactExtrudedConstruction::z_view() const noexcept
{
    return std::span<const double>(impl_->axis->display).subspan(impl_->begin, impl_->layers + 1);
}

std::size_t ExactExtrudedConstruction::first_layer() const noexcept
{
    return impl_->begin;
}

double ExactExtrudedConstruction::cell_volume(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    return measured(
        Q(impl_->base->areas.at(id.ij) * (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                                             impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)])))
        .as_double();
}

MeshUtils::Vec3 ExactExtrudedConstruction::cell_centroid(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    auto p = impl_->base->centers.at(id.ij);
    p[2] = (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)] +
               impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) /
           2;
    return rounded_point(p);
}

double ExactExtrudedConstruction::face_area(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return measured(impl_->base->areas.at(id.ij)).as_double();
    const Q height = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                     impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    return square_root(Q(dot(impl_->base->deltas.at(id.ij), impl_->base->deltas.at(id.ij)) * height * height))
        .as_double();
}

MeshUtils::Vec3 ExactExtrudedConstruction::face_centroid(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    auto p = id.orientation == SemiStructuredIndexer::AXIAL ? impl_->base->centers.at(id.ij)
                                                            : impl_->base->edge_centers.at(id.ij);
    p[2] = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    if (id.orientation != SemiStructuredIndexer::AXIAL)
        p[2] = (p[2] + impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) / 2;
    return rounded_point(p);
}

MeshUtils::Vec3 ExactExtrudedConstruction::face_area_vector(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return {0, 0, (id.k == 0 ? -1 : 1) * measured(impl_->base->areas.at(id.ij)).as_double()};
    const auto& d = impl_->base->deltas.at(id.ij);
    const Q h = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    return rounded_point({d[1] * h, -d[0] * h, 0});
}

ConvexGeometryPtr ExactExtrudedConstruction::certified_cell_geometry(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    Product product;
    for (auto n : impl_->topology->base_cell_nodes(id.ij))
        product.base.push_back(impl_->base->points[impl_->base->active[n]]);
    product.lower = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    product.upper = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)];
    auto geometry = std::make_shared<ConvexCellGeometry::Impl>();
    geometry->poly = product_poly(product);
    geometry->product = std::move(product);
    const auto first = rounded_point(geometry->poly.vertices[0]);
    geometry->origin = {first.x, first.y, first.z};
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(geometry)));
}

std::vector<char> ExactExtrudedConstruction::encode() const
{
    WireWriter w;
    w.word(0x5346455854524331ULL);
    w.word(1);
    const auto signature = topology_descriptor(*impl_->topology);
    w.word(signature.size());
    for (auto byte : signature)
        w.data.push_back(byte);
    w.word(impl_->base->recipes.size());
    for (const auto& p : impl_->base->recipes)
    {
        for (auto x : p.root)
            w.real(x);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(impl_->base->active.size());
    for (auto id : impl_->base->active)
        w.word(id);
    w.word(impl_->axis->values->zrecipes.size());
    for (const auto& p : impl_->axis->values->zrecipes)
    {
        w.real(p.root);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(impl_->layers + 1);
    for (std::size_t k = 0; k <= impl_->layers; ++k)
        w.word(impl_->axis->zactive.at(impl_->begin + k));
    return std::move(w.data);
}

std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::decode(
    std::shared_ptr<const ExtrudedTopology> topology, std::span<const char> packet)
{
    if (!topology)
        throw std::invalid_argument("Exact extrusion decode needs retained topology.");
    WireReader r{packet};
    if (r.word() != 0x5346455854524331ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact extrusion packet.");
    auto n = r.word();
    if (n > r.data.size() - r.position)
        throw std::invalid_argument("Invalid exact extrusion topology descriptor.");
    const auto expected = topology_descriptor(*topology);
    if (n != expected.size() || !std::equal(expected.begin(), expected.end(), r.data.begin() + r.position))
        throw std::invalid_argument("Exact extrusion packet belongs to different topology/order.");
    r.position += n;
    std::vector<ExactPointRecipe> points;
    std::vector<ExactScalarRecipe> z;
    std::vector<std::size_t> active, zactive;
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ExactPointRecipe p;
        for (auto& x : p.root)
            x = r.real();
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(r.word());
        points.push_back(std::move(p));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        active.push_back(r.word());
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ExactScalarRecipe p;
        p.root = r.real();
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(r.word());
        z.push_back(std::move(p));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        zactive.push_back(r.word());
    r.finish();
    return make(std::move(topology), std::move(points), std::move(active), std::move(z), std::move(zactive));
}

std::size_t ExactExtrudedConstruction::storage_bytes() const
{
    return base_storage_bytes() + axis_storage_bytes() + view_storage_bytes();
}

double ExactExtrudedConstruction::scaled_base_area(std::size_t base_cell) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Q largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->base->points[impl_->base->active[loop[i]]],
            impl_->base->points[impl_->base->active[loop[(i + 1) % loop.size()]]]);
        const Q length = dot(d, d);
        largest = std::max(largest, length);
    }
    return measured(Q(impl_->base->areas.at(base_cell) / largest)).as_double();
}

bool ExactExtrudedConstruction::base_quality_above(std::size_t base_cell, double minimum) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Q largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->base->points[impl_->base->active[loop[i]]],
            impl_->base->points[impl_->base->active[loop[(i + 1) % loop.size()]]]);
        const Q length = dot(d, d);
        largest = std::max(largest, length);
    }
    return impl_->base->areas.at(base_cell) > exact(minimum) * largest;
}

void ExactExtrudedConstruction::certify_base_embedding() const
{
    struct Candidate
    {
        std::size_t cell;
        Q xmin, xmax, ymin, ymax;
    };
    std::vector<Candidate> candidates;
    auto polygon = [&](std::size_t cell)
    {
        Product product;
        product.lower = 0;
        product.upper = 1;
        for (auto n : impl_->topology->base_cell_nodes(cell))
            product.base.push_back(impl_->base->points[impl_->base->active[n]]);
        return product;
    };
    for (std::size_t c = 0; c < impl_->base->areas.size(); ++c)
    {
        const auto p = polygon(c);
        Candidate candidate{c, p.base[0][0], p.base[0][0], p.base[0][1], p.base[0][1]};
        for (const auto& v : p.base)
        {
            candidate.xmin = std::min(candidate.xmin, v[0]);
            candidate.xmax = std::max(candidate.xmax, v[0]);
            candidate.ymin = std::min(candidate.ymin, v[1]);
            candidate.ymax = std::max(candidate.ymax, v[1]);
        }
        candidates.push_back(std::move(candidate));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.xmin < b.xmin; });
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        const auto& a = candidates[i];
        std::erase_if(active, [&](auto j) { return candidates[j].xmax <= a.xmin; });
        for (auto j : active)
        {
            const auto& b = candidates[j];
            if (std::max(a.ymin, b.ymin) < std::min(a.ymax, b.ymax) &&
                product_intersection(polygon(a.cell), polygon(b.cell)))
                throw std::invalid_argument("Extruded root base cells have positive overlapping interiors.");
        }
        active.push_back(i);
    }
    // A vertex strictly inside another native whole edge is an unresolved
    // T-junction/overlapping seam, even when all cell interiors are disjoint.
    std::vector<std::size_t> vertices = impl_->base->active;
    std::sort(vertices.begin(), vertices.end(),
        [&](auto a, auto b) { return impl_->base->points[a][0] < impl_->base->points[b][0]; });
    for (const auto& edge : impl_->topology->native_topology().side_faces())
    {
        const auto a = impl_->base->points[impl_->base->active[edge.nodes[0]]],
                   b = impl_->base->points[impl_->base->active[edge.nodes[1]]], delta = sub(b, a);
        const Q length = dot(delta, delta);
        const Q xmin = std::min(a[0], b[0]), xmax = std::max(a[0], b[0]), ymin = std::min(a[1], b[1]),
                ymax = std::max(a[1], b[1]);
        auto it = std::lower_bound(vertices.begin(), vertices.end(), xmin,
            [&](auto id, const Q& x) { return impl_->base->points[id][0] < x; });
        for (; it != vertices.end() && impl_->base->points[*it][0] <= xmax; ++it)
        {
            const auto& p = impl_->base->points[*it];
            if (p[1] < ymin || p[1] > ymax)
                continue;
            const auto offset = sub(p, a);
            const Q along = dot(offset, delta);
            if (zero(cross(delta, offset)) && along > 0 && along < length)
                throw std::invalid_argument("Extruded root embedding has an unresolved whole-edge T-junction.");
        }
    }
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
P polygon_vector(const Polygon& polygon)
{
    P area{};
    if (polygon.size() < 3)
        return area;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
        area = add(area, cross(sub(polygon[i], polygon[0]), sub(polygon[i + 1], polygon[0])));
    return mul(area, Q(1, 2));
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
