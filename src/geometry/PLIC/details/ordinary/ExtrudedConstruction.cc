#include "geometry/PLIC/details/ordinary/ExtrudedConstructionInternal.hh"

namespace SimpleFluid::PLIC
{
namespace ordinary_extrusion_detail
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
    return std::move(w.bytes);
}

MeshUtils::Vec3 rounded_point(const P& p)
{
    return {measured(p[0]).as_double(), measured(p[1]).as_double(), measured(p[2]).as_double()};
}

// A source edge is identified by construction ancestry, not by fitting a plane
// to its rounded descendants. Each selected recipe is serialized once, using a
// canonical topological numbering independent of the provider's compact IDs.
PlaneSupportIdentity<ArithmeticMode::Ordinary> segment_recipe_support(
    const std::vector<PointRecipe>& recipes, std::size_t first, std::size_t second)
{
    std::set<std::size_t> selected;
    std::vector<std::size_t> pending{first, second};
    while (!pending.empty())
    {
        const auto id = pending.back();
        pending.pop_back();
        if (selected.insert(id).second)
            pending.insert(pending.end(), recipes.at(id).parents.begin(), recipes.at(id).parents.end());
    }
    std::map<std::size_t, std::size_t> depth, canonical;
    std::map<std::size_t, std::vector<std::size_t>> levels;
    for (auto id : selected)
    {
        std::size_t level = 0;
        for (auto parent : recipes.at(id).parents)
            level = std::max(level, depth.at(parent) + 1);
        depth[id] = level;
        levels[level].push_back(id);
    }
    std::vector<details::OrdinarySupportRecipe> records;
    for (const auto& [level, ids] : levels)
    {
        struct Entry
        {
            std::vector<std::size_t> ids;
            details::OrdinarySupportRecipe recipe;
        };
        std::map<std::vector<char>, Entry> unique;
        for (auto id : ids)
        {
            const auto& recipe = recipes.at(id);
            WireWriter signature;
            details::OrdinarySupportRecipe native;
            native.original = recipe.parents.empty();
            signature.byte(static_cast<std::uint8_t>(
                recipe.parents.empty() ? SupportRecipeKind::Original : SupportRecipeKind::Derived));
            if (recipe.parents.empty())
            {
                native.root = details::ordinary_native::wide(recipe.root);
                for (double value : recipe.root)
                    signature.real(value);
            }
            else
            {
                std::vector<std::size_t> parents;
                for (auto parent : recipe.parents)
                    parents.push_back(canonical.at(parent));
                std::sort(parents.begin(), parents.end());
                signature.word(parents.size());
                for (auto parent : parents)
                    signature.word(parent);
                native.parents = std::move(parents);
            }
            auto [entry, fresh] = unique.try_emplace(std::move(signature.bytes));
            if (fresh)
                entry->second.recipe = std::move(native);
            entry->second.ids.push_back(id);
        }
        for (auto& [signature, entry] : unique)
        {
            for (auto id : entry.ids)
                canonical[id] = records.size();
            records.push_back(std::move(entry.recipe));
        }
    }
    auto a = canonical.at(first), b = canonical.at(second);
    if (a > b)
        std::swap(a, b);
    return details::GeometryIdentityAccess::support_dag(std::move(records), a, b);
}
std::pair<PlaneSupportIdentity<ArithmeticMode::Ordinary>, P> ancestral_edge_support(
    const ExtrudedBaseData& base, std::size_t first, std::size_t second)
{
    std::set<std::size_t> ancestry;
    std::vector<std::size_t> pending{first, second};
    std::set<std::pair<std::size_t, std::size_t>> candidates{std::minmax(first, second)};
    while (!pending.empty())
    {
        const auto id = pending.back();
        pending.pop_back();
        if (!ancestry.insert(id).second)
            continue;
        const auto& recipe = base.recipes.at(id);
        std::set<std::size_t> parents(recipe.parents.begin(), recipe.parents.end());
        if (parents.size() == 2)
            candidates.emplace(*parents.begin(), *parents.rbegin());
        pending.insert(pending.end(), parents.begin(), parents.end());
    }
    const auto membership = [&](std::size_t a, std::size_t b)
    {
        std::vector<unsigned char> contained(base.recipes.size(), 0);
        for (std::size_t i = 0; i < base.recipes.size(); ++i)
        {
            const auto& parents = base.recipes[i].parents;
            contained[i] = i == a || i == b ||
                           (!parents.empty() && std::all_of(parents.begin(), parents.end(),
                                                    [&](auto parent) { return contained[parent]; }));
        }
        return contained;
    };
    auto source = std::minmax(first, second);
    std::pair<std::size_t, std::size_t> selected{source.first, source.second};
    for (const auto& [a, b] : candidates)
    {
        const auto contained = membership(a, b);
        if (contained[first] && contained[second] && contained[selected.first] && contained[selected.second])
            selected = {a, b};
    }
    const auto edge = sub(base.points.at(selected.second), base.points.at(selected.first));
    const P normal{edge[1], -edge[0], 0};
    if (zero(normal))
        details::ordinary_native::reject("native ancestral edge is degenerate.");
    if (edge[0] == 0 || edge[1] == 0)
    {
        auto a = base.points.at(selected.first), b = base.points.at(selected.second), upper = a;
        upper[2] = 1;
        const std::array source{a, b, upper};
        return {details::ordinary_native::source_support_packet(source), normal};
    }
    return {segment_recipe_support(base.recipes, selected.first, selected.second), normal};
}

} // namespace ordinary_extrusion_detail
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{
std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>
ExtrudedConstruction<ArithmeticMode::Ordinary>::make(std::shared_ptr<const ExtrudedTopology> topology,
    std::vector<PointRecipe> recipes, std::vector<std::size_t> active, std::vector<ScalarRecipe> zrecipes,
    std::vector<std::size_t> zactive)
{
    if (!topology || active.size() != topology->indexer().num_nodes_per_layer ||
        zactive.size() != std::size_t(topology->indexer().num_layers) + 1)
        throw std::invalid_argument("Ordinary extrusion construction and retained topology differ.");
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
                    throw std::invalid_argument("Ordinary point construction is not an acyclic ancestry DAG.");
                p = add(p, impl->base->points[parent]);
            }
            p = mul(p, (1.L / static_cast<long double>(recipe.parents.size())));
        }
        if (p[2] != 0)
            throw std::invalid_argument("Ordinary extrusion base must lie in XY.");
        impl->base->points.push_back(std::move(p));
    }
    std::set<std::size_t> unique;
    std::set<P> unique_points;
    std::set<GeometryPoint> display;
    for (auto id : impl->base->active)
    {
        if (id >= impl->base->points.size() || !unique.insert(id).second ||
            !unique_points.insert(impl->base->points[id]).second)
            throw std::invalid_argument("Ordinary extrusion has invalid/duplicate active points.");
        const auto p = rounded_point(impl->base->points[id]);
        if (!display.insert({p.x, p.y, p.z}).second)
            throw std::invalid_argument("Ordinary extrusion display vertices are indistinguishable.");
    }
    for (std::size_t i = 0; i < impl->axis->values->zrecipes.size(); ++i)
    {
        const auto& recipe = impl->axis->values->zrecipes[i];
        Scalar z = 0;
        (void) represented(recipe.root);
        if (recipe.parents.empty())
            z = represented(recipe.root);
        else
        {
            for (auto parent : recipe.parents)
            {
                if (parent >= i)
                    throw std::invalid_argument("Ordinary axial construction is not acyclic.");
                z += impl->axis->values->z[parent];
            }
            z /= recipe.parents.size();
        }
        impl->axis->values->z.push_back(z);
    }
    Scalar previous = 0;
    double previous_display = 0;
    for (std::size_t k = 0; k < impl->axis->zactive.size(); ++k)
    {
        const auto id = impl->axis->zactive[k];
        if (id >= impl->axis->values->z.size())
            throw std::invalid_argument("Ordinary axial point index is invalid.");
        const auto value = measured(impl->axis->values->z[id]).as_double();
        if (k && (!(impl->axis->values->z[id] > previous) || !(value > previous_display)))
            throw std::invalid_argument("Ordinary axial/display edges must be distinct and increasing.");
        previous = impl->axis->values->z[id];
        previous_display = value;
        if (k)
        {
            const auto lo = measured(impl->axis->values->z[impl->axis->zactive[k - 1]]).as_double();
            const auto center =
                measured(Scalar((impl->axis->values->z[id] + impl->axis->values->z[impl->axis->zactive[k - 1]]) / 2))
                    .as_double();
            if (!(center > lo && center < value))
                throw std::invalid_argument("Ordinary extrusion has unrepresentable signed axial metric distances.");
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
                throw std::invalid_argument("Ordinary extrusion base must be strictly convex and counterclockwise.");
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
                    "Ordinary extrusion display polygon is not a distinct strictly convex native loop.");
            for (const auto& p : display_polygon)
                if (cross(sub(b, a), sub(p, a))[2] < 0)
                    throw std::invalid_argument("Ordinary extrusion display polygon is nonsimple.");
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
        impl->base->edge_centers.push_back(mul(add(a, b), .5L));
        auto [support, normal] =
            ancestral_edge_support(*impl->base, impl->base->active[edge.nodes[0]], impl->base->active[edge.nodes[1]]);
        impl->base->edge_supports.push_back(std::move(support));
        impl->base->support_normals.push_back(normal);
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
            throw std::invalid_argument("Ordinary extrusion has unrepresentable signed base metric distances.");
        if (edge.neighbor != SemiStructMeshTopo::invalid_ordinal)
        {
            const auto neighbor = rounded_point(impl->base->centers.at(edge.neighbor));
            const auto neighbor_distance = (face - neighbor).dot(normal);
            if (!std::isfinite(neighbor_distance) || !(neighbor_distance < 0))
                throw std::invalid_argument("Ordinary extrusion neighbor metric distance is unresolved.");
        }
    }
    impl->layers = impl->axis->zactive.size() - 1;
    for (auto id : impl->base->active)
        impl->base->display.push_back(rounded_point(impl->base->points[id]));
    for (auto id : impl->axis->zactive)
        impl->axis->display.push_back(measured(impl->axis->values->z[id]).as_double());
    return std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>(
        new ExtrudedConstruction<ArithmeticMode::Ordinary>(std::move(impl)));
}

bool ExtrudedConstruction<ArithmeticMode::Ordinary>::matches_topology(
    const std::shared_ptr<const ExtrudedTopology>& topology) const noexcept
{
    return impl_->topology == topology;
}

std::vector<MeshUtils::Vec3> ExtrudedConstruction<ArithmeticMode::Ordinary>::display_points() const
{
    return impl_->base->display;
}

std::vector<double> ExtrudedConstruction<ArithmeticMode::Ordinary>::display_z() const
{
    const auto span = z_view();
    return {span.begin(), span.end()};
}

const std::vector<MeshUtils::Vec3>& ExtrudedConstruction<ArithmeticMode::Ordinary>::display_points_ref() const noexcept
{
    return impl_->base->display;
}

std::span<const double> ExtrudedConstruction<ArithmeticMode::Ordinary>::z_view() const noexcept
{
    return std::span<const double>(impl_->axis->display).subspan(impl_->begin, impl_->layers + 1);
}

std::size_t ExtrudedConstruction<ArithmeticMode::Ordinary>::first_layer() const noexcept
{
    return impl_->begin;
}

double ExtrudedConstruction<ArithmeticMode::Ordinary>::cell_volume(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    return positive_measure(Scalar(impl_->base->areas.at(id.ij) *
                                   (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                                       impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)])))
        .as_double();
}

MeshUtils::Vec3 ExtrudedConstruction<ArithmeticMode::Ordinary>::cell_centroid(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    auto p = impl_->base->centers.at(id.ij);
    p[2] = (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)] +
               impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) /
           2;
    return rounded_point(p);
}

double ExtrudedConstruction<ArithmeticMode::Ordinary>::face_area(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return measured(impl_->base->areas.at(id.ij)).as_double();
    const Scalar height = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                          impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    return square_root(Scalar(dot(impl_->base->deltas.at(id.ij), impl_->base->deltas.at(id.ij)) * height * height))
        .as_double();
}

MeshUtils::Vec3 ExtrudedConstruction<ArithmeticMode::Ordinary>::face_centroid(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    auto p = id.orientation == SemiStructuredIndexer::AXIAL ? impl_->base->centers.at(id.ij)
                                                            : impl_->base->edge_centers.at(id.ij);
    p[2] = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    if (id.orientation != SemiStructuredIndexer::AXIAL)
        p[2] = (p[2] + impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) / 2;
    return rounded_point(p);
}

MeshUtils::Vec3 ExtrudedConstruction<ArithmeticMode::Ordinary>::face_area_vector(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return {0, 0, (id.k == 0 ? -1 : 1) * measured(impl_->base->areas.at(id.ij)).as_double()};
    const auto& d = impl_->base->deltas.at(id.ij);
    const Scalar h = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                     impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    return rounded_point({d[1] * h, -d[0] * h, 0});
}

ConvexGeometryPtr<ArithmeticMode::Ordinary> ExtrudedConstruction<ArithmeticMode::Ordinary>::certified_cell_geometry(
    std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    Product product;
    for (auto n : impl_->topology->base_cell_nodes(id.ij))
        product.base.push_back(impl_->base->points[impl_->base->active[n]]);
    product.lower = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    product.upper = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)];
    auto geometry = product_geometry(product);
    std::vector<PlaneSupportIdentity<ArithmeticMode::Ordinary>> supports;
    for (const auto z : {product.lower, product.upper})
    {
        auto cap = product.base;
        for (auto& point : cap)
            point[2] = z;
        supports.push_back(details::ordinary_native::source_support_packet(cap));
    }
    for (auto edge : impl_->topology->native_topology().cell_side_faces(id.ij))
        supports.push_back(impl_->base->edge_supports.at(edge));
    return details::NativeSupportAccess::cell_supports(*geometry, supports);
}

std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::encode() const
{
    WireWriter w;
    w.tag(GeometryTag::ExtrudedConstruction);
    w.byte(recipe_packet_version);
    const auto signature = topology_descriptor(*impl_->topology);
    w.word(signature.size());
    for (auto byte : signature)
        w.bytes.push_back(byte);
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
    return std::move(w.bytes);
}

std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>
ExtrudedConstruction<ArithmeticMode::Ordinary>::decode(
    std::shared_ptr<const ExtrudedTopology> topology, std::span<const char> packet)
{
    if (!topology)
        throw std::invalid_argument("Ordinary extrusion decode needs retained topology.");
    WireReader r{packet};
    if (r.tag() != GeometryTag::ExtrudedConstruction || r.byte() != recipe_packet_version)
        throw std::invalid_argument("Unknown ordinary extrusion packet.");
    auto n = r.word();
    if (n > r.bytes.size() - r.cursor)
        throw std::invalid_argument("Invalid ordinary extrusion topology descriptor.");
    const auto expected = topology_descriptor(*topology);
    if (n != expected.size() || !std::equal(expected.begin(), expected.end(), r.bytes.begin() + r.cursor))
        throw std::invalid_argument("Ordinary extrusion packet belongs to different topology/order.");
    r.cursor += n;
    std::vector<PointRecipe> points;
    std::vector<ScalarRecipe> z;
    std::vector<std::size_t> active, zactive;
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        PointRecipe p;
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
        ScalarRecipe p;
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

std::size_t ExtrudedConstruction<ArithmeticMode::Ordinary>::storage_bytes() const
{
    return base_storage_bytes() + axis_storage_bytes() + view_storage_bytes();
}

double ExtrudedConstruction<ArithmeticMode::Ordinary>::scaled_base_area(std::size_t base_cell) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Scalar largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->base->points[impl_->base->active[loop[i]]],
            impl_->base->points[impl_->base->active[loop[(i + 1) % loop.size()]]]);
        const Scalar length = dot(d, d);
        largest = std::max(largest, length);
    }
    return measured(Scalar(impl_->base->areas.at(base_cell) / largest)).as_double();
}

bool ExtrudedConstruction<ArithmeticMode::Ordinary>::base_quality_above(std::size_t base_cell, double minimum) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Scalar largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->base->points[impl_->base->active[loop[i]]],
            impl_->base->points[impl_->base->active[loop[(i + 1) % loop.size()]]]);
        const Scalar length = dot(d, d);
        largest = std::max(largest, length);
    }
    return impl_->base->areas.at(base_cell) > represented(minimum) * largest;
}

void ExtrudedConstruction<ArithmeticMode::Ordinary>::certify_base_embedding() const
{
    struct Candidate
    {
        std::size_t cell;
        Scalar xmin, xmax, ymin, ymax;
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
        const Scalar length = dot(delta, delta);
        const Scalar xmin = std::min(a[0], b[0]), xmax = std::max(a[0], b[0]), ymin = std::min(a[1], b[1]),
                     ymax = std::max(a[1], b[1]);
        auto it = std::lower_bound(vertices.begin(), vertices.end(), xmin,
            [&](auto id, const Scalar& x) { return impl_->base->points[id][0] < x; });
        for (; it != vertices.end() && impl_->base->points[*it][0] <= xmax; ++it)
        {
            const auto& p = impl_->base->points[*it];
            if (p[1] < ymin || p[1] > ymax)
                continue;
            const auto offset = sub(p, a);
            const Scalar along = dot(offset, delta);
            if (zero(cross(delta, offset)) && along > 0 && along < length)
                throw std::invalid_argument("Extruded root embedding has an unresolved whole-edge T-junction.");
        }
    }
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
PlanarFacePtr<ArithmeticMode::Ordinary> ExtrudedConstruction<ArithmeticMode::Ordinary>::certified_face_geometry(
    std::size_t face) const
{
    std::vector<P> points;
    for (auto node : impl_->topology->face_nodes(face))
    {
        const auto n = impl_->topology->indexer().node_id(node);
        auto p = impl_->base->points.at(impl_->base->active.at(n.ij));
        p[2] = impl_->axis->values->z.at(impl_->axis->zactive.at(impl_->begin + n.k));
        points.push_back(p);
    }
    const auto outward = face_area_vector(face);
    const auto id = impl_->topology->indexer().face_id(face);
    P normal{outward.x, outward.y, outward.z};
    std::optional<PlaneSupportIdentity<ArithmeticMode::Ordinary>> support;
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        support = details::ordinary_native::source_support_packet(points);
    else
    {
        normal = impl_->base->support_normals.at(id.ij);
        if (dot(normal, P{outward.x, outward.y, outward.z}) < 0)
            normal = mul(normal, -1);
        support = impl_->base->edge_supports.at(id.ij);
    }
    if (dot(details::ordinary_native::polygon_vector(points), normal) < 0)
        std::reverse(points.begin(), points.end());
    return details::NativeSupportAccess::face_support(points, normal, std::move(*support));
}
ConvexGeometryPtr<ArithmeticMode::Ordinary> ExtrudedConstruction<ArithmeticMode::Ordinary>::atomic_cell_geometry(
    std::size_t cell) const
{
    std::vector<PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
    for (auto face : impl_->topology->cell_faces(cell))
    {
        auto geometry = certified_face_geometry(face);
        if (impl_->topology->owner_cell(face) != cell)
            geometry = geometry->reversed();
        faces.push_back(std::move(geometry));
    }
    return PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces);
}
} // namespace SimpleFluid::PLIC
