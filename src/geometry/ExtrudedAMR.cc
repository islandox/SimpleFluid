/** @file ExtrudedAMR.cc
 * @brief Collective conforming factor adaptation with retained exact constructions.
 */
#include "geometry/ExtrudedAMR.hh"
#include "geometry/LocalProductAMR.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
namespace
{
using ID = std::uint64_t;
struct BaseNode
{
    std::vector<std::size_t> vertices;
    std::vector<BaseEdgeKey> edges;
    std::size_t reference = 0;
    std::vector<BaseCellKey> children;
};
struct EdgeNode
{
    std::size_t a = 0, b = 0;
    std::string boundary;
    bool physical = false;
};
struct ZNode
{
    std::size_t lower = 0, upper = 0;
    std::vector<AxialIntervalKey> children;
};
template<class F> void collectively(F&& f)
{
    std::exception_ptr error;
    try
    {
        f();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = error ? 1 : 0, failed = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &failed);
    if (failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("Extruded AMR validation failed on another rank.");
    }
}
void same(const std::string& local)
{
    auto comm = Tpetra::getDefaultComm();
    collectively(
        [&]
        {
            if (local.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::overflow_error("Extruded AMR description too large.");
        });
    int n = comm->getRank() == 0 ? static_cast<int>(local.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &n);
    std::string reference = comm->getRank() == 0 ? local : std::string(n, '\0');
    Teuchos::broadcast(*comm, 0, n, reference.data());
    collectively(
        [&]
        {
            if (local != reference)
                throw std::invalid_argument("Extruded AMR descriptions or requests differ across ranks.");
        });
}
ID identity()
{
    static std::atomic<ID> counter{0};
    auto comm = Tpetra::getDefaultComm();
    ID id = 0;
    if (comm->getRank() == 0)
    {
        auto previous = counter.load();
        while (previous != std::numeric_limits<ID>::max())
        {
            if (counter.compare_exchange_weak(previous, previous + 1))
            {
                id = previous + 1;
                break;
            }
        }
    }
    Teuchos::broadcast(*comm, 0, 1, &id);
    if (!id)
        throw std::overflow_error("Extruded AMR collective identity exhausted.");
    return id;
}
void key_text(std::ostream& out, const BaseCellKey& key)
{
    out << key.root << ' ' << key.root_template << ' ' << key.path.size() << ' ';
    for (auto p : key.path)
        out << p << ' ';
}
void key_text(std::ostream& out, const AxialIntervalKey& key)
{
    out << key.root << ' ' << key.path.size() << ' ';
    for (auto p : key.path)
        out << p << ' ';
}
template<class Key> Key child(Key parent, ID ordinal)
{
    parent.path.push_back(ordinal);
    return parent;
}
} // namespace

struct ExtrudedAMR::Data
{
    std::shared_ptr<const ExtrudedAMRGrid> grid;
    std::map<BaseCellKey, BaseNode> base;
    std::set<BaseCellKey> leaves;
    std::map<BaseEdgeKey, EdgeNode> edges;
    std::map<AxialIntervalKey, ZNode> axial;
    std::set<AxialIntervalKey> intervals;
    std::vector<ExactPointRecipe> points;
    std::vector<ExactScalarRecipe> z;
    std::vector<BaseCellKey> base_keys;
    std::vector<AxialIntervalKey> interval_keys;
    std::shared_ptr<const ExtrudedTopology> topology;
    std::shared_ptr<const ExactExtrudedConstruction> construction;
    SP<const MultiRegionMesh> geometry;
    std::shared_ptr<const CompositePartition<>> partition;
    ID topology_generation = 0, partition_generation = 0, collective_id = 0;
    std::shared_ptr<const void> token = std::make_shared<const char>(0), topology_token = token,
                                partition_token = token;
};

namespace
{
using Data = ExtrudedAMR::Data;
GeometryPoint displayed(const Data& data, std::size_t point)
{
    const auto& recipe = data.points.at(point);
    if (recipe.parents.empty())
        return recipe.root;
    GeometryPoint result{};
    for (auto parent : recipe.parents)
    {
        auto p = displayed(data, parent);
        for (std::size_t a = 0; a < 3; ++a)
            result[a] += p[a] / recipe.parents.size();
    }
    return result;
}
std::size_t longest(const Data& data, const std::vector<std::size_t>& vertices)
{
    long double largest = -1;
    std::size_t selected = 0;
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        auto a = displayed(data, vertices[i]), b = displayed(data, vertices[(i + 1) % vertices.size()]);
        long double x = static_cast<long double>(a[0]) - b[0], y = static_cast<long double>(a[1]) - b[1],
                    q = x * x + y * y;
        if (q > largest)
        {
            largest = q;
            selected = i;
        }
    }
    return selected;
}
std::string description(const Data& data, bool roots_only = false)
{
    std::ostringstream out;
    out << std::hexfloat << data.grid->region_id << ' ' << data.grid->material_id << ' ' << data.grid->max_depth << ' '
        << data.grid->minimum_scaled_area << ' ';
    for (const auto& p : data.grid->xy_nodes)
        for (auto x : p)
            out << x << ' ';
    out << " base ";
    for (const auto& c : data.grid->base_cells)
    {
        out << c.size() << ' ';
        for (auto n : c)
            out << n << ' ';
    }
    for (const auto& b : data.grid->boundaries)
        out << b.node0 << ' ' << b.node1 << ' ' << b.batch_name.size() << ':' << b.batch_name << ' ';
    for (const auto& [source, target] : data.grid->boundary_aliases)
        out << source.size() << ':' << source << ' ' << target.size() << ':' << target << ' ';
    for (auto z : data.grid->z_edges)
        out << z << ' ';
    out << " refs ";
    for (auto r : data.grid->triangle_reference_edges)
        out << r << ' ';
    if (!roots_only)
    {
        out << " leaves ";
        for (const auto& key : data.leaves)
            key_text(out, key);
        out << " axial ";
        for (const auto& key : data.intervals)
            key_text(out, key);
    }
    return out.str();
}
BaseEdgeKey new_edge(Data& data, const BaseCellKey& parent, ID ordinal, std::size_t a, std::size_t b,
    BaseEdgeKey::Kind kind = BaseEdgeKey::Kind::TemplateInterior)
{
    // Roots supply canonical original-ID order; internal templates supply an
    // invariant direction (NVB opposite->midpoint, quad midpoint->center,
    // polygon fan anchor->vertex). Compacted recipe ordinals never orient keys.
    BaseEdgeKey key{parent, ordinal, {}, kind};
    auto [it, inserted] = data.edges.emplace(key, EdgeNode{a, b, {}});
    if (!inserted && std::set{it->second.a, it->second.b} != std::set{a, b})
        throw std::logic_error("Extruded edge construction collision.");
    return key;
}
std::size_t midpoint(Data& data, const BaseEdgeKey& key)
{
    const auto e = data.edges.at(key);
    auto left = key;
    left.path.push_back(0);
    auto right = key;
    right.path.push_back(1);
    if (data.edges.contains(left))
        return data.edges.at(left).b;
    const auto m = data.points.size();
    data.points.push_back({{}, {e.a, e.b}});
    data.edges[left] = {e.a, m, e.boundary, e.physical};
    data.edges[right] = {m, e.b, e.boundary, e.physical};
    return m;
}
BaseEdgeKey half_edge(const Data& data, BaseEdgeKey key, std::size_t endpoint)
{
    key.path.push_back(data.edges.at(key).a == endpoint ? 0 : 1);
    return key;
}
void split_base(Data& data, const BaseCellKey& key)
{
    if (!data.leaves.contains(key))
        throw std::invalid_argument("XY refinement key is not a leaf.");
    if (key.path.size() >= data.grid->max_depth)
        throw std::invalid_argument("XY refinement/closure exceeds maximum depth.");
    const auto parent = data.base.at(key);
    std::vector<BaseNode> children;
    const auto n = parent.vertices.size();
    if (n == 3)
    {
        const auto i = parent.reference, a = parent.vertices[i], b = parent.vertices[(i + 1) % 3],
                   c = parent.vertices[(i + 2) % 3];
        const auto m = midpoint(data, parent.edges[i]);
        const auto seam = new_edge(data, key, 0, c, m);
        children.push_back({{c, a, m}, {parent.edges[(i + 2) % 3], half_edge(data, parent.edges[i], a), seam}, 0, {}});
        children.push_back({{c, m, b}, {seam, half_edge(data, parent.edges[i], b), parent.edges[(i + 1) % 3]}, 2, {}});
    }
    else if (n == 4)
    {
        std::vector<std::size_t> mids;
        for (const auto& edge : parent.edges)
            mids.push_back(midpoint(data, edge));
        const auto center = data.points.size();
        data.points.push_back({{}, parent.vertices});
        std::vector<BaseEdgeKey> seams;
        for (std::size_t i = 0; i < 4; ++i)
            seams.push_back(new_edge(data, key, i, mids[i], center));
        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto prev = (i + 3) % 4;
            children.push_back({{parent.vertices[i], mids[i], center, mids[prev]},
                {half_edge(data, parent.edges[i], parent.vertices[i]), seams[i], seams[prev],
                    half_edge(data, parent.edges[prev], parent.vertices[i])},
                0, {}});
        }
    }
    else
    {
        // Canonical fan starts at the smallest persistent vertex recipe ID.
        auto start = std::min_element(parent.vertices.begin(), parent.vertices.end()) - parent.vertices.begin();
        auto vertex = [&](std::size_t i) { return parent.vertices[(start + i) % n]; };
        auto edge = [&](std::size_t i) { return parent.edges[(start + i) % n]; };
        std::vector<BaseEdgeKey> diagonals;
        for (std::size_t i = 2; i + 1 < n; ++i)
            diagonals.push_back(new_edge(data, key, i, vertex(0), vertex(i)));
        for (std::size_t i = 1; i + 1 < n; ++i)
        {
            BaseNode node{{vertex(0), vertex(i), vertex(i + 1)},
                {i == 1 ? edge(0) : diagonals[i - 2], edge(i), i + 1 == n - 1 ? edge(n - 1) : diagonals[i - 1]}, 0, {}};
            node.reference = longest(data, node.vertices);
            children.push_back(std::move(node));
        }
    }
    data.leaves.erase(key);
    data.base[key].children.clear();
    for (std::size_t i = 0; i < children.size(); ++i)
    {
        auto k = child(key, i);
        data.base[key].children.push_back(k);
        data.base[k] = std::move(children[i]);
        data.leaves.insert(k);
    }
}
void close_base(Data& data)
{
    for (;;)
    {
        std::set<BaseEdgeKey> subdivided;
        for (const auto& leaf : data.leaves)
            for (auto edge : data.base.at(leaf).edges)
                while (!edge.path.empty())
                {
                    edge.path.pop_back();
                    subdivided.insert(edge);
                }
        std::optional<BaseCellKey> refine;
        for (const auto& leaf : data.leaves)
        {
            for (const auto& edge : data.base.at(leaf).edges)
                if (subdivided.contains(edge))
                {
                    refine = leaf;
                    break;
                }
            if (refine)
                break;
        }
        if (!refine)
            return;
        split_base(data, *refine);
    }
}
void compact(Data& data)
{
    std::set<BaseCellKey> retained;
    for (auto key : data.leaves)
    {
        retained.insert(key);
        while (!key.path.empty())
        {
            key.path.pop_back();
            retained.insert(key);
        }
    }
    std::erase_if(data.base, [&](const auto& item) { return !retained.contains(item.first); });
    std::set<AxialIntervalKey> zretained;
    for (auto key : data.intervals)
    {
        zretained.insert(key);
        while (!key.path.empty())
        {
            key.path.pop_back();
            zretained.insert(key);
        }
    }
    std::erase_if(data.axial, [&](const auto& item) { return !zretained.contains(item.first); });
    std::set<BaseEdgeKey> edge_keys;
    for (const auto& [key, node] : data.base)
        for (auto edge : node.edges)
        {
            edge_keys.insert(edge);
            while (!edge.path.empty())
            {
                edge.path.pop_back();
                edge_keys.insert(edge);
            }
        }
    std::erase_if(data.edges, [&](const auto& item) { return !edge_keys.contains(item.first); });
    std::set<std::size_t> used;
    std::function<void(std::size_t)> visit = [&](auto id)
    {
        if (used.insert(id).second)
            for (auto parent : data.points.at(id).parents)
                visit(parent);
    };
    for (const auto& [key, node] : data.base)
        for (auto id : node.vertices)
            visit(id);
    for (const auto& [key, edge] : data.edges)
    {
        visit(edge.a);
        visit(edge.b);
    }
    std::map<std::size_t, std::size_t> remap;
    std::vector<ExactPointRecipe> points;
    for (auto id : used)
    {
        auto recipe = data.points[id];
        for (auto& p : recipe.parents)
            p = remap.at(p);
        remap[id] = points.size();
        points.push_back(std::move(recipe));
    }
    data.points = std::move(points);
    for (auto& [key, node] : data.base)
        for (auto& id : node.vertices)
            id = remap.at(id);
    for (auto& [key, edge] : data.edges)
    {
        edge.a = remap.at(edge.a);
        edge.b = remap.at(edge.b);
    }
    used.clear();
    std::function<void(std::size_t)> zvisit = [&](auto id)
    {
        if (used.insert(id).second)
            for (auto parent : data.z.at(id).parents)
                zvisit(parent);
    };
    for (const auto& [key, node] : data.axial)
    {
        zvisit(node.lower);
        zvisit(node.upper);
    }
    remap.clear();
    std::vector<ExactScalarRecipe> z;
    for (auto id : used)
    {
        auto recipe = data.z[id];
        for (auto& p : recipe.parents)
            p = remap.at(p);
        remap[id] = z.size();
        z.push_back(std::move(recipe));
    }
    data.z = std::move(z);
    for (auto& [key, node] : data.axial)
    {
        node.lower = remap.at(node.lower);
        node.upper = remap.at(node.upper);
    }
}
void prepare_geometry(Data& data)
{
    compact(data);
    data.base_keys.assign(data.leaves.begin(), data.leaves.end());
    data.interval_keys.assign(data.intervals.begin(), data.intervals.end());
    std::set<std::size_t> used;
    for (const auto& key : data.base_keys)
        for (auto id : data.base.at(key).vertices)
            used.insert(id);
    std::vector<std::size_t> active(used.begin(), used.end());
    std::map<std::size_t, unsigned> ids;
    for (std::size_t i = 0; i < active.size(); ++i)
        ids[active[i]] = i;
    Arr<Arr<unsigned>> cells;
    for (const auto& key : data.base_keys)
    {
        Arr<unsigned> loop;
        for (auto id : data.base.at(key).vertices)
            loop.push_back(ids.at(id));
        cells.push_back(std::move(loop));
    }
    Arr<SemiStructMeshTopo::BoundaryEdge> boundaries;
    std::set<BaseEdgeKey> boundary_edges;
    for (const auto& key : data.base_keys)
        for (const auto& edgekey : data.base.at(key).edges)
        {
            const auto& edge = data.edges.at(edgekey);
            if (!edge.boundary.empty() && boundary_edges.insert(edgekey).second)
                boundaries.push_back({ids.at(edge.a), ids.at(edge.b), edge.boundary});
        }
    std::vector<std::size_t> zactive;
    for (const auto& key : data.interval_keys)
        zactive.push_back(data.axial.at(key).lower);
    zactive.push_back(data.axial.at(data.interval_keys.back()).upper);
    if (active.size() > std::numeric_limits<unsigned>::max() || cells.size() > std::numeric_limits<unsigned>::max() ||
        data.interval_keys.size() > std::numeric_limits<unsigned>::max())
        throw std::overflow_error("Extruded AMR native ordinal extent overflow.");
    data.topology = std::make_shared<const ExtrudedTopology>(
        active.size(), cells, data.interval_keys.size(), boundaries, data.grid->boundary_aliases);
    data.construction = ExactExtrudedConstruction::make(data.topology, data.points, active, data.z, zactive);
    auto geometry = std::make_shared<const ExtrudedGeometry>(data.topology, data.construction);
    for (std::size_t c = 0; c < cells.size(); ++c)
        if (!data.construction->base_quality_above(c, data.grid->minimum_scaled_area))
            throw std::invalid_argument("Extruded AMR base child violates scaled-area quality admission.");
    data.partition.reset();
}
void publish_geometry(Data& data)
{
    auto g = std::make_shared<const ExtrudedGeometry>(data.topology, data.construction);
    data.geometry = std::make_shared<const MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{ExtrudedRegion("product", data.topology, std::move(g))},
        std::vector<MultiRegionMesh::Interface>{});
}
void initialize(Data& data)
{
    const auto& grid = *data.grid;
    if (grid.xy_nodes.empty() || grid.base_cells.empty() || grid.z_edges.size() < 2 || !grid.max_depth ||
        grid.max_depth > 48 || !std::isfinite(grid.minimum_scaled_area) || grid.minimum_scaled_area < 0)
        throw std::invalid_argument(
            "Extruded AMR needs finite nonempty nonperiodic roots and bounded refinement depth.");
    if (!grid.triangle_reference_edges.empty() && grid.triangle_reference_edges.size() != grid.base_cells.size())
        throw std::invalid_argument("Root triangle reference edge count mismatch.");
    for (const auto& p : grid.xy_nodes)
        data.points.push_back({p, {}});
    for (auto z : grid.z_edges)
        data.z.push_back({z, {}});
    std::map<std::pair<std::size_t, std::size_t>, BaseEdgeKey> shared;
    std::map<std::pair<std::size_t, std::size_t>, std::string> labels;
    for (const auto& b : grid.boundaries)
    {
        if (b.node0 == b.node1 || b.batch_name.empty() ||
            !labels.emplace(std::minmax(std::size_t(b.node0), std::size_t(b.node1)), b.batch_name).second)
            throw std::invalid_argument("Extruded AMR boundary labels are invalid/duplicate.");
    }
    for (std::size_t c = 0; c < grid.base_cells.size(); ++c)
    {
        const auto& root = grid.base_cells[c];
        if (root.size() < 3)
            throw std::invalid_argument("Extruded root polygon has fewer than three nodes.");
        BaseCellKey key{c, root.size() == 3 ? 0ULL : root.size() == 4 ? 1ULL : 2ULL, {}};
        BaseNode node;
        for (auto id : root)
        {
            if (id >= grid.xy_nodes.size())
                throw std::invalid_argument("Extruded root node is outside catalog.");
            node.vertices.push_back(id);
        }
        for (std::size_t e = 0; e < root.size(); ++e)
        {
            std::pair<std::size_t, std::size_t> endpoints{
                std::min(root[e], root[(e + 1) % root.size()]), std::max(root[e], root[(e + 1) % root.size()])};
            auto found = shared.find(endpoints);
            BaseEdgeKey edge;
            if (found == shared.end())
            {
                edge = new_edge(data, key, e, endpoints.first, endpoints.second, BaseEdgeKey::Kind::Root);
                shared.emplace(endpoints, edge);
                if (labels.contains(endpoints))
                    data.edges[edge].boundary = labels.at(endpoints);
            }
            else
                edge = found->second;
            node.edges.push_back(edge);
        }
        node.reference =
            grid.triangle_reference_edges.empty() ? longest(data, node.vertices) : grid.triangle_reference_edges[c];
        if (root.size() == 3 && node.reference >= 3)
            throw std::invalid_argument("Triangle reference edge is invalid.");
        data.base[key] = std::move(node);
        data.leaves.insert(key);
    }
    for (const auto& [edge, label] : labels)
        if (!shared.contains(edge))
            throw std::invalid_argument("Extruded boundary label is not a root edge.");
    std::map<BaseEdgeKey, std::vector<std::pair<BaseCellKey, bool>>> incident;
    for (const auto& key : data.leaves)
    {
        const auto& node = data.base.at(key);
        for (std::size_t e = 0; e < node.edges.size(); ++e)
            incident[node.edges[e]].push_back({key, node.vertices.size() == 3 && node.reference == e});
    }
    for (const auto& [edge, cells] : incident)
    {
        data.edges[edge].physical = cells.size() == 1;
        if (cells.size() == 2 && data.base.at(cells[0].first).vertices.size() == 3 &&
            data.base.at(cells[1].first).vertices.size() == 3 && cells[0].second != cells[1].second)
            throw std::invalid_argument("Root triangle refinement edges are incompatible across a shared edge.");
    }
    for (std::size_t k = 0; k + 1 < grid.z_edges.size(); ++k)
    {
        AxialIntervalKey key{k, {}};
        data.axial[key] = {k, k + 1, {}};
        data.intervals.insert(key);
    }
    prepare_geometry(data);
    data.construction->certify_base_embedding();
}
} // namespace

ExtrudedAMR::ExtrudedAMR(ExtrudedAMRGrid grid)
{
    auto data = std::make_shared<Data>();
    data->grid = std::make_shared<const ExtrudedAMRGrid>(std::move(grid));
    collectively([&] { initialize(*data); });
    same(description(*data));
    publish_geometry(*data);
    data->collective_id = identity();
    data_ = std::move(data);
}
ExtrudedAMR::ExtrudedAMR(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
HierarchyObservation ExtrudedAMR::observation() const noexcept
{
    const auto* g = data_->partition ? data_->partition->geometry().get() : data_->geometry.get();
    ID epoch = std::numeric_limits<ID>::max();
    try
    {
        epoch = g->geometry_epoch();
    }
    catch (...)
    {
    }
    return {data_->token, data_->topology_token, data_->partition_token, data_->topology_generation,
        data_->partition_generation, epoch, g, Tpetra::getDefaultComm().getRawPtr()};
}
bool ExtrudedAMR::geometry_unchanged() const noexcept
{
    try
    {
        return data_->geometry->geometry_epoch() == 0 && (!data_->partition || data_->partition->geometry_unchanged());
    }
    catch (...)
    {
        return false;
    }
}
const SP<const MultiRegionMesh>& ExtrudedAMR::native_geometry() const noexcept
{
    return data_->geometry;
}
const SP<const MultiRegionMesh>& ExtrudedAMR::geometry() const noexcept
{
    return data_->geometry;
}
const std::shared_ptr<const CompositePartition<>>& ExtrudedAMR::native_partition() const noexcept
{
    return data_->partition;
}
const std::vector<BaseCellKey>& ExtrudedAMR::base_keys() const noexcept
{
    return data_->base_keys;
}
const std::vector<BaseEdgeKey>& ExtrudedAMR::base_edge_keys(std::size_t cell) const
{
    return data_->base.at(data_->base_keys.at(cell)).edges;
}
const std::vector<AxialIntervalKey>& ExtrudedAMR::interval_keys() const noexcept
{
    return data_->interval_keys;
}
AdaptiveCellKey ExtrudedAMR::cell_key(ID cell) const
{
    if (cell >= data_->geometry->num_cells())
        throw std::out_of_range("Extruded AMR cell GID outside hierarchy.");
    return {data_->grid->region_id, ProductCellAncestry{data_->base_keys[cell % data_->base_keys.size()],
                                        data_->interval_keys[cell / data_->base_keys.size()]}};
}
ID ExtrudedAMR::cell_material(ID cell) const
{
    (void) cell_key(cell);
    return data_->grid->material_id;
}
std::size_t ExtrudedAMR::cell_fragment(ID cell) const
{
    (void) cell_key(cell);
    return 0;
}
ConvexGeometryPtr ExtrudedAMR::cell_geometry(ID cell) const
{
    (void) cell_key(cell);
    if (!geometry_unchanged())
        throw std::invalid_argument("Extruded hierarchy geometry is stale.");
    return data_->construction->certified_cell_geometry(cell);
}
ExtrudedAMR::StorageCounts ExtrudedAMR::storage_counts() const noexcept
{
    return {data_->leaves.size(), data_->topology->indexer().num_nodes_per_layer,
        data_->topology->indexer().num_side_faces_per_layer, data_->intervals.size() + 1, data_->points.size(),
        data_->z.size(), data_->base.size(), data_->axial.size()};
}
MeshStorageReport ExtrudedAMR::storage_report() const
{
    auto report = data_->geometry->storage_report();
    std::size_t bytes = sizeof(Data) + sizeof(ExtrudedAMRGrid);
    auto basekey = [](const BaseCellKey& k) { return k.path.capacity() * sizeof(ID); };
    auto zkey = [](const AxialIntervalKey& k) { return k.path.capacity() * sizeof(ID); };
    for (const auto& [k, n] : data_->base)
    {
        bytes += sizeof(decltype(data_->base)::value_type) + 4 * sizeof(void*) + basekey(k) +
                 n.vertices.capacity() * sizeof(std::size_t) + n.edges.capacity() * sizeof(BaseEdgeKey) +
                 n.children.capacity() * sizeof(BaseCellKey);
        for (const auto& e : n.edges)
            bytes += basekey(e.creator) + e.path.capacity() * sizeof(ID);
        for (const auto& c : n.children)
            bytes += basekey(c);
    }
    for (const auto& [k, e] : data_->edges)
        bytes += sizeof(decltype(data_->edges)::value_type) + 4 * sizeof(void*) + basekey(k.creator) +
                 k.path.capacity() * sizeof(ID) + e.boundary.capacity();
    for (const auto& [k, n] : data_->axial)
    {
        bytes += sizeof(decltype(data_->axial)::value_type) + 4 * sizeof(void*) + zkey(k) +
                 n.children.capacity() * sizeof(AxialIntervalKey);
        for (const auto& c : n.children)
            bytes += zkey(c);
    }
    for (const auto& k : data_->leaves)
        bytes += sizeof(BaseCellKey) + 4 * sizeof(void*) + basekey(k);
    for (const auto& k : data_->intervals)
        bytes += sizeof(AxialIntervalKey) + 4 * sizeof(void*) + zkey(k);
    bytes +=
        data_->base_keys.capacity() * sizeof(BaseCellKey) + data_->interval_keys.capacity() * sizeof(AxialIntervalKey);
    for (const auto& k : data_->base_keys)
        bytes += basekey(k);
    for (const auto& k : data_->interval_keys)
        bytes += zkey(k);
    bytes += data_->points.capacity() * sizeof(ExactPointRecipe) + data_->z.capacity() * sizeof(ExactScalarRecipe);
    for (const auto& p : data_->points)
        bytes += p.parents.capacity() * sizeof(std::size_t);
    for (const auto& z : data_->z)
        bytes += z.parents.capacity() * sizeof(std::size_t);
    bytes += data_->grid->xy_nodes.capacity() * sizeof(GeometryPoint) +
             data_->grid->base_cells.capacity() * sizeof(std::vector<unsigned>) +
             data_->grid->z_edges.capacity() * sizeof(double) +
             data_->grid->boundaries.capacity() * sizeof(SemiStructMeshTopo::BoundaryEdge) +
             data_->grid->triangle_reference_edges.capacity() * sizeof(std::size_t);
    for (const auto& c : data_->grid->base_cells)
        bytes += c.capacity() * sizeof(unsigned);
    for (const auto& b : data_->grid->boundaries)
        bytes += b.batch_name.capacity();
    for (const auto& [a, b] : data_->grid->boundary_aliases)
        bytes += sizeof(std::pair<const std::string, std::string>) + 4 * sizeof(void*) + a.capacity() + b.capacity();
    report.objects += bytes;
    return report;
}
std::shared_ptr<const ExtrudedAMR> ExtrudedAMR::with_partition(
    std::shared_ptr<const CompositePartition<>> partition) const
{
    collectively(
        [&]
        {
            auto comm = Tpetra::getDefaultComm();
            if (!geometry_unchanged() || !partition || !partition->geometry_unchanged() ||
                partition->plan().communicator().getRawPtr() != comm.getRawPtr() ||
                (comm->getRank() == partition->plan().source_rank() &&
                    !partition->plan().matches_source(data_->geometry)))
                throw std::invalid_argument("Extruded partition source/revision/communicator mismatch.");
        });
    same(std::to_string(data_->collective_id) + " " + std::to_string(partition->plan().collective_identity()));
    same(data_->partition == partition ? "unchanged" : "replace");
    if (data_->partition == partition)
        return std::shared_ptr<const ExtrudedAMR>(new ExtrudedAMR(data_));
    auto data = std::make_shared<Data>(*data_);
    collectively(
        [&]
        {
            if (data->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Extruded partition generation exhausted.");
        });
    data->partition = std::move(partition);
    ++data->partition_generation;
    data->token = std::make_shared<const char>(0);
    data->partition_token = data->token;
    data->collective_id = identity();
    return std::shared_ptr<const ExtrudedAMR>(new ExtrudedAMR(std::move(data)));
}
std::shared_ptr<const ExtrudedAMR> ExtrudedAMR::adapt(const std::vector<std::size_t>& refine_base,
    const std::vector<std::size_t>& refine_z, const std::vector<BaseCellKey>& coarsen_base,
    const std::vector<AxialIntervalKey>& coarsen_z) const
{
    auto data = std::make_shared<Data>(*data_);
    std::set<BaseCellKey> refine;
    std::set<AxialIntervalKey> zrefine;
    collectively(
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Extruded hierarchy is stale.");
            for (auto id : refine_base)
                refine.insert(data_->base_keys.at(id));
            for (auto id : refine_z)
                zrefine.insert(data_->interval_keys.at(id));
        });
    std::ostringstream marks;
    marks << data_->collective_id << " XY ";
    for (const auto& k : refine)
        key_text(marks, k);
    marks << " Z ";
    for (const auto& k : zrefine)
        key_text(marks, k);
    marks << " XYcoarsen ";
    for (const auto& k : coarsen_base)
        key_text(marks, k);
    marks << " Zcoarsen ";
    for (const auto& k : coarsen_z)
        key_text(marks, k);
    same(marks.str());
    collectively(
        [&]
        {
            std::set<BaseCellKey> seen;
            for (const auto& key : coarsen_base)
            {
                if (!seen.insert(key).second || !data_->base.contains(key) || data_->base.at(key).children.empty())
                    throw std::invalid_argument("XY coarsening parent is invalid.");
                for (const auto& c : data_->base.at(key).children)
                    if (!data_->leaves.contains(c) || refine.contains(c))
                        throw std::invalid_argument("XY coarsening needs a complete unmarked immediate source family.");
            }
            std::set<AxialIntervalKey> zseen;
            for (const auto& key : coarsen_z)
            {
                if (!zseen.insert(key).second || !data_->axial.contains(key) ||
                    data_->axial.at(key).children.size() != 2)
                    throw std::invalid_argument("Z coarsening parent is invalid.");
                for (const auto& c : data_->axial.at(key).children)
                    if (!data_->intervals.contains(c) || zrefine.contains(c))
                        throw std::invalid_argument("Z coarsening needs complete unmarked immediate source siblings.");
            }
            for (const auto& key : coarsen_base)
            {
                for (const auto& c : data_->base.at(key).children)
                    data->leaves.erase(c);
                data->leaves.insert(key);
                data->base[key].children.clear();
            }
            for (const auto& key : coarsen_z)
            {
                for (const auto& c : data_->axial.at(key).children)
                    data->intervals.erase(c);
                data->intervals.insert(key);
                data->axial[key].children.clear();
            }
            for (const auto& key : refine)
                split_base(*data, key);
            close_base(*data);
            for (const auto& key : zrefine)
            {
                if (key.path.size() >= data->grid->max_depth)
                    throw std::invalid_argument("Z refinement exceeds maximum depth.");
                auto node = data->axial.at(key);
                const auto midpoint = data->z.size();
                data->z.push_back({0, {node.lower, node.upper}});
                auto a = child(key, 0), b = child(key, 1);
                data->axial[key].children = {a, b};
                data->axial[a] = {node.lower, midpoint, {}};
                data->axial[b] = {midpoint, node.upper, {}};
                data->intervals.erase(key);
                data->intervals.insert(a);
                data->intervals.insert(b);
            }
            // Coarsening cannot be silently undone by conformity closure.
            for (const auto& key : coarsen_base)
                if (!data->leaves.contains(key))
                    throw std::invalid_argument("XY coarsening would violate conformity.");
            if (data->leaves != data_->leaves || data->intervals != data_->intervals)
                prepare_geometry(*data);
        });
    if (data->leaves == data_->leaves && data->intervals == data_->intervals)
        return std::shared_ptr<const ExtrudedAMR>(new ExtrudedAMR(data_));
    collectively(
        [&]
        {
            if (data->topology_generation == std::numeric_limits<ID>::max() ||
                data->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Extruded generation exhausted.");
        });
    ++data->topology_generation;
    ++data->partition_generation;
    data->token = std::make_shared<const char>(0);
    data->topology_token = data->token;
    data->partition_token = data->token;
    same(description(*data));
    publish_geometry(*data);
    data->collective_id = identity();
    return std::shared_ptr<const ExtrudedAMR>(new ExtrudedAMR(std::move(data)));
}
ExtrudedAMR::AdaptationPlan ExtrudedAMR::plan_adaptation(const std::vector<std::size_t>& b,
    const std::vector<std::size_t>& z, const std::vector<BaseCellKey>& bc,
    const std::vector<AxialIntervalKey>& zc) const
{
    AdaptationPlan p;
    p.source_ = observation();
    p.candidate_ = adapt(b, z, bc, zc);
    p.collective_id_ = identity();
    return p;
}
std::shared_ptr<const ExtrudedAMR> ExtrudedAMR::apply_adaptation(const AdaptationPlan& p) const
{
    collectively(
        [&]
        {
            if (!geometry_unchanged() || p.source_ != observation() || !p.candidate_)
                throw std::invalid_argument("Extruded adaptation plan is stale/foreign.");
        });
    same(std::to_string(p.collective_id_));
    return p.candidate_;
}

namespace
{
template<class Key> std::vector<std::size_t> comparable(const std::map<Key, std::size_t>& index, const Key& key)
{
    std::vector<std::size_t> found;
    auto ancestor = key;
    for (;;)
    {
        auto it = index.find(ancestor);
        if (it != index.end())
            found.push_back(it->second);
        if (ancestor.path.empty())
            break;
        ancestor.path.pop_back();
    }
    auto it = index.lower_bound(key);
    for (; it != index.end(); ++it)
    {
        const auto& k = it->first;
        if (k.root != key.root)
            break;
        if constexpr (requires { k.root_template; })
            if (k.root_template != key.root_template)
                break;
        if (k.path.size() < key.path.size() || !std::equal(key.path.begin(), key.path.end(), k.path.begin()))
            break;
        if (k.path.size() > key.path.size())
            found.push_back(it->second);
    }
    return found;
}
template<class Key> bool prefix(const Key& a, const Key& b)
{
    return a.path.size() <= b.path.size() && std::equal(a.path.begin(), a.path.end(), b.path.begin());
}
} // namespace
std::vector<AdaptiveAncestryRelation> ExtrudedAMR::overlap_relations(const ExtrudedAMR& target) const
{
    std::vector<ID> cells(target.geometry()->num_cells());
    std::iota(cells.begin(), cells.end(), 0);
    return overlap_relations(target, cells);
}
std::vector<AdaptiveAncestryRelation> ExtrudedAMR::overlap_relations(
    const ExtrudedAMR& target, const std::vector<ID>& selected) const
{
    if (description(*data_, true) != description(*target.data_, true))
        throw std::invalid_argument("Product overlap endpoints have different roots/templates/patch labels/material.");
    std::map<BaseCellKey, std::size_t> bases;
    std::map<AxialIntervalKey, std::size_t> axial;
    for (std::size_t i = 0; i < base_keys().size(); ++i)
        bases[base_keys()[i]] = i;
    for (std::size_t i = 0; i < interval_keys().size(); ++i)
        axial[interval_keys()[i]] = i;
    std::vector<AdaptiveAncestryRelation> records;
    std::set<ID> unique;
    for (auto gid : selected)
    {
        if (!unique.insert(gid).second)
            throw std::invalid_argument("Duplicate queried product target.");
        const auto p = std::get<ProductCellAncestry>(target.cell_key(gid).ancestry);
        for (auto b : comparable(bases, p.base))
            for (auto z : comparable(axial, p.axial))
            {
                const auto& oldb = base_keys()[b];
                const auto& oldz = interval_keys()[z];
                AdaptiveAncestryRelation::Kind kind;
                if (oldb == p.base && oldz == p.axial)
                    kind = AdaptiveAncestryRelation::Kind::Unchanged;
                else if (prefix(oldb, p.base) && prefix(oldz, p.axial))
                    kind = AdaptiveAncestryRelation::Kind::Refinement;
                else if (prefix(p.base, oldb) && prefix(p.axial, oldz))
                    kind = AdaptiveAncestryRelation::Kind::Coarsening;
                else
                    kind = AdaptiveAncestryRelation::Kind::ProductIntersection;
                records.push_back({kind, z * base_keys().size() + b, gid});
            }
    }
    return records;
}
struct LocalProductAMR::Data
{
    ExtrudedAMR::Data pool;
    std::map<BaseCellKey, std::shared_ptr<const std::vector<AxialIntervalKey>>> columns;
    std::vector<LocalProductRun> runs;
    std::shared_ptr<const void> token = std::make_shared<const char>(0);
    std::uint64_t generation = 0;
    std::uint64_t collective_id = 0;
};
namespace
{
using LocalData = LocalProductAMR::Data;
using ZKeys = std::vector<AxialIntervalKey>;
std::string edge_name(const BaseEdgeKey& key)
{
    std::ostringstream s;
    s << "__amr_side_" << unsigned(key.kind) << '_' << key.creator.root << '_' << key.creator.root_template;
    for (auto p : key.creator.path)
        s << '_' << p;
    s << "_e" << key.ordinal;
    for (auto p : key.path)
        s << '_' << p;
    return s.str();
}
LocalProductPlaneKey plane_key(AxialIntervalKey key, bool upper)
{
    if (key.path.empty())
        return {LocalProductPlaneKey::Kind::RootPoint, key.root + upper, {}};
    const auto side = key.path.back();
    key.path.pop_back();
    if ((upper && side == 0) || (!upper && side == 1))
        return {LocalProductPlaneKey::Kind::IntervalMidpoint, 0, key};
    return plane_key(std::move(key), upper);
}
std::string cap_name(const LocalProductPlaneKey& key)
{
    std::ostringstream out;
    out << "__amr_cap_" << unsigned(key.kind) << '_' << key.root_point << '_' << key.creator.root;
    for (auto p : key.creator.path)
        out << '_' << p;
    return out.str();
}
std::string local_description(const LocalData& state)
{
    std::ostringstream s;
    s << description(state.pool, true) << " columnsets ";
    std::map<ZKeys, std::size_t> patterns;
    for (const auto& [base, z] : state.columns)
    {
        key_text(s, base);
        auto [it, inserted] = patterns.emplace(*z, patterns.size());
        s << it->second << ' ';
        if (inserted)
        {
            s << z->size() << ' ';
            for (const auto& key : *z)
                key_text(s, key);
        }
    }
    return s.str();
}
void intern_columns(LocalData& state)
{
    std::map<ZKeys, std::shared_ptr<const ZKeys>> patterns;
    for (auto it = state.columns.begin(); it != state.columns.end();)
    {
        if (it->second->empty())
        {
            it = state.columns.erase(it);
            continue;
        }
        auto [p, inserted] = patterns.emplace(*it->second, it->second);
        if (!inserted)
            it->second = p->second;
        ++it;
    }
    state.pool.leaves.clear();
    state.pool.intervals.clear();
    for (const auto& [base, z] : state.columns)
    {
        state.pool.leaves.insert(base);
        for (const auto& interval : *z)
            state.pool.intervals.insert(interval);
    }
    compact(state.pool);
    for (auto& [key, node] : state.pool.base)
        if (!node.children.empty() && std::none_of(node.children.begin(), node.children.end(),
                                          [&](const auto& child) { return state.pool.base.contains(child); }))
            std::vector<BaseCellKey>().swap(node.children);
    for (auto& [key, node] : state.pool.axial)
        if (!node.children.empty() && std::none_of(node.children.begin(), node.children.end(),
                                          [&](const auto& child) { return state.pool.axial.contains(child); }))
            std::vector<AxialIntervalKey>().swap(node.children);
}
struct BaseTemplate
{
    std::shared_ptr<const ExtrudedTopology> topology;
    std::shared_ptr<const ExactExtrudedConstruction> construction;
    std::shared_ptr<const std::vector<BaseCellKey>> keys;
    std::vector<std::pair<BaseEdgeKey, std::string>> sides;
};
std::vector<std::size_t> axis_points(const ExtrudedAMR::Data& pool, const ZKeys& keys)
{
    std::vector<std::size_t> points;
    for (const auto& key : keys)
    {
        const auto& node = pool.axial.at(key);
        if (points.empty() || points.back() != node.lower)
            points.push_back(node.lower);
        points.push_back(node.upper);
    }
    return points;
}
void build_local_runs(LocalData& state, const std::shared_ptr<const LocalData>& previous = {})
{
    intern_columns(state);
    auto& pool = state.pool;
    std::map<std::vector<BaseCellKey>, BaseTemplate> bases;
    std::map<std::vector<BaseCellKey>, BaseTemplate> old_bases;
    if (previous)
        for (const auto& run : previous->runs)
        {
            std::vector<BaseCellKey> keys(run.base_keys.begin(), run.base_keys.end());
            auto topology = std::shared_ptr<const ExtrudedTopology>(previous, &run.region.topology());
            old_bases.emplace(keys, BaseTemplate{topology, run.region.geometry().construction(),
                                        std::make_shared<const std::vector<BaseCellKey>>(keys), {}});
        }
    std::map<ZKeys, std::vector<BaseCellKey>> groups;
    std::map<ZKeys, std::shared_ptr<const ZKeys>> selections;
    for (const auto& [base, z] : state.columns)
    {
        groups[*z].push_back(base);
        selections[*z] = z;
    }
    std::vector<std::size_t> global_axis(pool.z.size());
    std::iota(global_axis.begin(), global_axis.end(), 0);
    std::vector<long double> zvalues;
    for (const auto& recipe : pool.z)
    {
        long double value = recipe.root;
        if (!recipe.parents.empty())
        {
            value = 0;
            for (auto parent : recipe.parents)
                value += zvalues.at(parent) / recipe.parents.size();
        }
        zvalues.push_back(value);
    }
    std::sort(global_axis.begin(), global_axis.end(), [&](auto a, auto b) { return zvalues[a] < zvalues[b]; });
    std::shared_ptr<const ExactExtrudedConstruction> coordinates;
    std::map<std::vector<std::size_t>, std::shared_ptr<const ExactExtrudedConstruction>> axes;
    struct Planned
    {
        BaseTemplate base;
        std::shared_ptr<const ZKeys> zkeys;
        std::size_t first, count, axis_first;
        std::shared_ptr<const ExactExtrudedConstruction> axis;
    };
    std::vector<Planned> plans;
    for (const auto& [zkeys, basekeys] : groups)
    {
        std::map<BaseEdgeKey, std::vector<BaseCellKey>> owners;
        for (const auto& key : basekeys)
            for (const auto& edge : pool.base.at(key).edges)
                owners[edge].push_back(key);
        std::map<BaseCellKey, std::vector<BaseCellKey>> neighbors;
        for (const auto& [edge, cells] : owners)
        {
            if (cells.size() > 2)
                throw std::invalid_argument("Local product base patch is nonmanifold.");
            if (cells.size() == 2)
            {
                neighbors[cells[0]].push_back(cells[1]);
                neighbors[cells[1]].push_back(cells[0]);
            }
        }
        std::set<BaseCellKey> pending(basekeys.begin(), basekeys.end());
        while (!pending.empty())
        {
            std::vector<BaseCellKey> component{*pending.begin()};
            pending.erase(component[0]);
            for (std::size_t i = 0; i < component.size(); ++i)
                for (const auto& next : neighbors[component[i]])
                    if (pending.erase(next))
                        component.push_back(next);
            std::sort(component.begin(), component.end());
            BaseTemplate base;
            auto found = bases.find(component);
            if (found != bases.end())
                base = found->second;
            else
            {
                std::map<BaseEdgeKey, unsigned> edge_counts;
                for (const auto& key : component)
                    for (const auto& edge : pool.base.at(key).edges)
                        ++edge_counts[edge];
                for (const auto& [edge, count] : edge_counts)
                    if (count == 1)
                    {
                        const auto& native = pool.edges.at(edge);
                        base.sides.push_back({edge,
                            native.physical ? (native.boundary.empty() ? "side" : native.boundary) : edge_name(edge)});
                    }
                auto old = old_bases.find(component);
                if (old != old_bases.end())
                {
                    base.topology = old->second.topology;
                    base.construction = old->second.construction;
                    base.keys = old->second.keys;
                }
                else
                {
                    auto patch = pool;
                    patch.leaves = std::set<BaseCellKey>(component.begin(), component.end());
                    patch.intervals = {zkeys.front()};
                    auto grid = std::make_shared<ExtrudedAMRGrid>(*pool.grid);
                    grid->boundary_aliases.clear();
                    std::set<std::string> names{"zmin", "zmax"};
                    for (auto& [edge, node] : patch.edges)
                        node.boundary.clear();
                    for (const auto& [edge, name] : base.sides)
                    {
                        patch.edges.at(edge).boundary = name;
                        names.insert(name);
                    }
                    for (const auto& [source, target] : pool.grid->boundary_aliases)
                        if (names.contains(source))
                            grid->boundary_aliases[source] = target;
                    patch.grid = std::move(grid);
                    prepare_geometry(patch);
                    base.topology = patch.topology;
                    base.keys = std::make_shared<const std::vector<BaseCellKey>>(component);
                    if (!coordinates)
                    {
                        auto all_topology = base.topology->with_layers(global_axis.size() - 1);
                        std::vector<std::size_t> active;
                        std::set<std::size_t> used;
                        for (const auto& k : patch.base_keys)
                            for (auto v : patch.base.at(k).vertices)
                                used.insert(v);
                        active.assign(used.begin(), used.end());
                        coordinates =
                            ExactExtrudedConstruction::make(all_topology, patch.points, active, pool.z, global_axis);
                        if (previous)
                            for (const auto& run : previous->runs)
                                if (run.region.geometry().construction()->encode_axis_values() ==
                                    coordinates->encode_axis_values())
                                {
                                    coordinates = run.region.geometry().construction();
                                    break;
                                }
                    }
                    base.construction = patch.construction;
                }
                bases.emplace(component, base);
            }
            if (!coordinates)
            {
                // A retained template still needs the current common axial DAG.
                // Decode its compact point recipes, supplying the new global Z
                // construction through the existing validated standalone format.
                auto one = pool;
                one.leaves = std::set<BaseCellKey>(component.begin(), component.end());
                one.intervals = {zkeys.front()};
                auto grid = std::make_shared<ExtrudedAMRGrid>(*pool.grid);
                grid->boundary_aliases.clear();
                for (auto& [edge, node] : one.edges)
                    node.boundary.clear();
                std::set<std::string> names{"zmin", "zmax"};
                for (const auto& [edge, name] : base.sides)
                {
                    one.edges.at(edge).boundary = name;
                    names.insert(name);
                }
                for (const auto& [source, target] : pool.grid->boundary_aliases)
                    if (names.contains(source))
                        grid->boundary_aliases[source] = target;
                one.grid = grid;
                prepare_geometry(one);
                std::set<std::size_t> used;
                for (const auto& k : one.base_keys)
                    for (auto v : one.base.at(k).vertices)
                        used.insert(v);
                std::vector<std::size_t> active(used.begin(), used.end());
                coordinates = ExactExtrudedConstruction::make(
                    one.topology->with_layers(global_axis.size() - 1), one.points, active, pool.z, global_axis);
                if (previous)
                    for (const auto& run : previous->runs)
                        if (run.region.geometry().construction()->encode_axis_values() ==
                            coordinates->encode_axis_values())
                        {
                            coordinates = run.region.geometry().construction();
                            break;
                        }
            }
            const auto points = axis_points(pool, zkeys);
            auto axis = axes.find(points);
            if (axis == axes.end())
            {
                auto topology = base.topology->with_layers(points.size() - 1);
                auto construction =
                    ExactExtrudedConstruction::shared_axis_selection(topology, base.construction, coordinates, points);
                if (previous)
                    for (const auto& run : previous->runs)
                    {
                        const auto& old = run.region.geometry().construction();
                        if (old->encode_axis_values() == construction->encode_axis_values() &&
                            old->encode_axis_selection() == construction->encode_axis_selection())
                        {
                            construction = old;
                            break;
                        }
                    }
                axis = axes.emplace(points, std::move(construction)).first;
            }
            std::size_t first = 0;
            while (first < zkeys.size())
            {
                std::size_t last = first + 1;
                while (last < zkeys.size() && pool.axial.at(zkeys[last - 1]).upper == pool.axial.at(zkeys[last]).lower)
                    ++last;
                const auto lower = pool.axial.at(zkeys[first]).lower;
                const auto axis_first = std::find(points.begin(), points.end(), lower) - points.begin();
                plans.push_back(
                    {base, selections.at(zkeys), first, last - first, std::size_t(axis_first), axis->second});
                first = last;
            }
        }
    }
    std::sort(plans.begin(), plans.end(),
        [](const auto& a, const auto& b)
        {
            if (*a.base.keys != *b.base.keys)
                return *a.base.keys < *b.base.keys;
            return std::lexicographical_compare(a.zkeys->begin() + a.first, a.zkeys->begin() + a.first + a.count,
                b.zkeys->begin() + b.first, b.zkeys->begin() + b.first + b.count);
        });
    state.runs.clear();
    for (const auto& plan : plans)
    {
        auto low = plane_key((*plan.zkeys)[plan.first], false),
             high = plane_key((*plan.zkeys)[plan.first + plan.count - 1], true);
        const bool physical_low = low.kind == LocalProductPlaneKey::Kind::RootPoint && low.root_point == 0,
                   physical_high = high.kind == LocalProductPlaneKey::Kind::RootPoint &&
                                   high.root_point + 1 == pool.grid->z_edges.size();
        auto physical_name = [&](const std::string& source)
        {
            auto alias = pool.grid->boundary_aliases.find(source);
            return alias == pool.grid->boundary_aliases.end() ? source : alias->second;
        };
        std::map<std::string, std::string> aliases{{"zmin", physical_low ? physical_name("zmin") : cap_name(low)},
            {"zmax", physical_high ? physical_name("zmax") : cap_name(high)}};
        auto topology = plan.base.topology->with_layers(plan.count, std::move(aliases));
        auto construction = ExactExtrudedConstruction::shared_view(
            topology, plan.base.construction, plan.axis, plan.axis_first, plan.count);
        auto geometry = std::make_shared<const ExtrudedGeometry>(topology, std::move(construction));
        LocalProductRun run{
            ExtrudedRegion("product_run_" + std::to_string(state.runs.size()), topology, std::move(geometry)),
            {plan.base.keys, 0, plan.base.keys->size()}, {plan.zkeys, plan.first, plan.count}, {}};
        for (const auto& [edge, name] : plan.base.sides)
        {
            auto resolved = topology->boundary_aliases();
            auto alias = resolved.find(name);
            const auto& actual = alias == resolved.end() ? name : alias->second;
            int batch = -1;
            for (auto id : topology->boundary_batch_ids())
                if (topology->boundary_batch_name(id) == actual)
                {
                    batch = id;
                    break;
                }
            if (batch < 0)
                throw std::logic_error("Local seam boundary batch is missing.");
            run.boundaries.push_back({LocalProductBoundary::Kind::Side, batch, pool.edges.at(edge).physical, edge, {}});
        }
        run.boundaries.push_back({LocalProductBoundary::Kind::LowerCap, 0, physical_low, {}, low});
        run.boundaries.push_back({LocalProductBoundary::Kind::UpperCap, 1, physical_high, {}, high});
        state.runs.push_back(std::move(run));
    }
    pool.geometry.reset();
    pool.topology.reset();
    pool.construction.reset();
    pool.partition.reset();
    pool.base_keys.clear();
    pool.interval_keys.clear();
}
std::vector<BaseCellKey> local_base_children(ExtrudedAMR::Data& pool, const BaseCellKey& key)
{
    auto found = pool.base.find(key);
    if (found == pool.base.end())
        throw std::invalid_argument("Unknown local XY factor.");
    if (!found->second.children.empty() && std::all_of(found->second.children.begin(), found->second.children.end(),
                                               [&](const auto& child) { return pool.base.contains(child); }))
        return found->second.children;
    pool.base[key].children.clear();
    auto leaves = pool.leaves;
    pool.leaves = {key};
    split_base(pool, key);
    auto children = pool.base.at(key).children;
    pool.leaves = std::move(leaves);
    return children;
}
std::vector<AxialIntervalKey> local_z_children(ExtrudedAMR::Data& pool, const AxialIntervalKey& key)
{
    const auto node = pool.axial.at(key);
    if (node.children.size() == 2 &&
        std::all_of(node.children.begin(), node.children.end(), [&](const auto& c) { return pool.axial.contains(c); }))
        return node.children;
    if (key.path.size() >= pool.grid->max_depth)
        throw std::invalid_argument("Local Z refinement exceeds maximum depth.");
    const auto middle = pool.z.size();
    pool.z.push_back({0, {node.lower, node.upper}});
    auto a = child(key, 0), b = child(key, 1);
    pool.axial[key].children = {a, b};
    pool.axial[a] = {node.lower, middle, {}};
    pool.axial[b] = {middle, node.upper, {}};
    return {a, b};
}
} // namespace
LocalProductAMR::LocalProductAMR(ExtrudedAMRGrid grid)
{
    auto data = std::make_shared<Data>();
    data->pool.grid = std::make_shared<const ExtrudedAMRGrid>(std::move(grid));
    collectively(
        [&]
        {
            initialize(data->pool);
            auto intervals = std::make_shared<const ZKeys>(data->pool.interval_keys);
            for (const auto& base : data->pool.base_keys)
                data->columns.emplace(base, intervals);
            build_local_runs(*data);
        });
    same(local_description(*data));
    data->collective_id = SimpleFluid::Meshes::identity();
    data_ = std::move(data);
}
const std::vector<LocalProductRun>& LocalProductAMR::runs() const noexcept
{
    return data_->runs;
}
const std::shared_ptr<const void>& LocalProductAMR::identity() const noexcept
{
    return data_->token;
}
std::uint64_t LocalProductAMR::generation() const noexcept
{
    return data_->generation;
}
std::uint64_t LocalProductAMR::material_id() const noexcept
{
    return data_->pool.grid->material_id;
}
AdaptiveCellKey LocalProductAMR::cell_key(std::size_t run_id, std::uint64_t cell) const
{
    const auto& run = data_->runs.at(run_id);
    if (cell >= run.region.layout().cells)
        throw std::out_of_range("Local product cell ordinal is out of bounds.");
    return {data_->pool.grid->region_id,
        ProductCellAncestry{run.base_keys[cell % run.base_keys.size()], run.intervals[cell / run.base_keys.size()]}};
}
ConvexGeometryPtr LocalProductAMR::cell_geometry(std::size_t run, std::uint64_t cell) const
{
    (void) cell_key(run, cell);
    return data_->runs.at(run).region.geometry().certified_cell_geometry(cell);
}
std::vector<LocalProductAMR::ContactViolation> LocalProductAMR::contact_violations(
    double ratio, unsigned maximum_jump, std::size_t contact_budget) const
{
    if (!std::isfinite(ratio) || ratio < 1)
        throw std::invalid_argument("Contact span ratio must be finite and at least one.");
    std::size_t work = 0;
    auto charge = [&]
    {
        if (work >= contact_budget)
            throw std::invalid_argument("Product contact budget exhausted.");
        ++work;
    };
    using Pair = std::pair<BaseCellKey, BaseCellKey>;
    struct Contact
    {
        unsigned dimension = 0;
        std::size_t jump = 0;
    };
    std::map<Pair, Contact> pairs;
    auto add = [&](BaseCellKey a, BaseCellKey b, unsigned dimension, std::size_t jump = 0)
    {
        charge();
        if (a == b)
            return;
        if (b < a)
            std::swap(a, b);
        auto& value = pairs[{a, b}];
        value.dimension = std::max(value.dimension, dimension);
        value.jump = std::max(value.jump, jump);
    };
    std::map<BaseEdgeKey, std::vector<BaseCellKey>> edges;
    std::map<std::size_t, std::set<BaseCellKey>> vertices;
    std::map<BaseEdgeKey, std::set<std::size_t>> edge_points;
    for (const auto& [key, selection] : data_->columns)
    {
        charge();
        const auto& node = data_->pool.base.at(key);
        for (auto vertex : node.vertices)
        {
            charge();
            vertices[vertex].insert(key);
        }
        for (const auto& edge : node.edges)
        {
            charge();
            edges[edge].push_back(key);
            auto ancestor = edge;
            const auto& endpoints = data_->pool.edges.at(edge);
            for (;;)
            {
                charge();
                edge_points[ancestor].insert(endpoints.a);
                edge_points[ancestor].insert(endpoints.b);
                if (ancestor.path.empty())
                    break;
                ancestor.path.pop_back();
            }
        }
        auto ancestor = key;
        while (!ancestor.path.empty())
        {
            ancestor.path.pop_back();
            if (data_->columns.contains(ancestor))
                add(key, ancestor, 2, key.path.size() - ancestor.path.size());
        }
    }
    for (const auto& [edge, owners] : edges)
    {
        auto ancestor = edge;
        for (;;)
        {
            if (auto it = edges.find(ancestor); it != edges.end())
                for (const auto& a : owners)
                    for (const auto& b : it->second)
                        add(a, b, 1, edge.path.size() - ancestor.path.size());
            if (ancestor.path.empty())
                break;
            ancestor.path.pop_back();
        }
        for (auto point : edge_points.at(edge))
            if (auto it = vertices.find(point); it != vertices.end())
                for (const auto& a : owners)
                    for (const auto& b : it->second)
                        add(a, b, 0);
    }
    std::map<std::pair<BaseEdgeKey, std::size_t>, std::vector<std::pair<BaseCellKey, std::size_t>>> edge_endpoints;
    for (const auto& [edge, owners] : edges)
    {
        auto stem = edge;
        stem.path.clear();
        const auto& node = data_->pool.edges.at(edge);
        for (auto point : {node.a, node.b})
            for (const auto& owner : owners)
            {
                charge();
                auto& incident = edge_endpoints[{stem, point}];
                for (const auto& [other, depth] : incident)
                    add(owner, other, 0, std::max(depth, edge.path.size()) - std::min(depth, edge.path.size()));
                incident.emplace_back(owner, edge.path.size());
            }
    }
    for (const auto& [vertex, owners] : vertices)
        for (const auto& a : owners)
            for (const auto& b : owners)
                add(a, b, 0);
    // Dyadic addresses order exact axial endpoints without converting coordinates.
    using Address = std::pair<std::uint64_t, std::uint64_t>;
    const auto depth = data_->pool.grid->max_depth;
    const std::uint64_t scale = std::uint64_t(1) << depth;
    auto endpoint = [&](const AxialIntervalKey& key, bool upper) -> Address
    {
        std::uint64_t ordinal = 0;
        for (auto bit : key.path)
            ordinal = 2 * ordinal + bit;
        auto numerator = (ordinal + unsigned(upper)) << (depth - key.path.size());
        return numerator == scale ? Address{key.root + 1, 0} : Address{key.root, numerator};
    };
    auto geometry = [&](const ProductCellAncestry& key)
    {
        for (std::size_t r = 0; r < data_->runs.size(); ++r)
        {
            const auto& run = data_->runs[r];
            auto b = std::lower_bound(run.base_keys.begin(), run.base_keys.end(), key.base);
            auto z = std::lower_bound(run.intervals.begin(), run.intervals.end(), key.axial);
            if (b != run.base_keys.end() && *b == key.base && z != run.intervals.end() && *z == key.axial)
                return cell_geometry(
                    r, (z - run.intervals.begin()) * run.base_keys.size() + (b - run.base_keys.begin()));
        }
        throw std::logic_error("Contact factor is absent from native runs.");
    };
    std::vector<ContactViolation> result;
    auto check = [&](const BaseCellKey& a, const AxialIntervalKey& az, const BaseCellKey& b, const AxialIntervalKey& bz,
                     Contact contact)
    {
        charge();
        const auto lo = std::max(endpoint(az, false), endpoint(bz, false));
        const auto hi = std::min(endpoint(az, true), endpoint(bz, true));
        if (hi < lo)
            return;
        unsigned dimension = contact.dimension + unsigned(lo < hi);
        if (dimension == 3)
            throw std::logic_error("Active local products overlap.");
        bool jump = contact.jump > maximum_jump;
        if (az.root == bz.root)
            jump = jump ||
                   std::max(az.path.size(), bz.path.size()) - std::min(az.path.size(), bz.path.size()) > maximum_jump;
        ProductCellAncestry first{a, az}, second{b, bz};
        const auto ga = geometry(first), gb = geometry(second);
        bool span = false;
        for (unsigned axis = 0; axis < 3; ++axis)
            span = span || ga->axis_span_exceeds(*gb, axis, ratio) || gb->axis_span_exceeds(*ga, axis, ratio);
        if (span || jump)
            result.push_back({first, second, dimension, span, jump});
    };
    for (const auto& [pair, contact] : pairs)
    {
        const auto& first = *data_->columns.at(pair.first);
        const auto& second = *data_->columns.at(pair.second);
        for (const auto& az : first)
        {
            auto it = std::lower_bound(second.begin(), second.end(), endpoint(az, false),
                [&](const auto& z, const auto& value) { return endpoint(z, true) < value; });
            for (; it != second.end() && endpoint(*it, false) <= endpoint(az, true); ++it)
                check(pair.first, az, pair.second, *it, contact);
        }
    }
    for (const auto& [base, selection] : data_->columns)
        for (std::size_t i = 1; i < selection->size(); ++i)
            check(base, (*selection)[i - 1], base, (*selection)[i], {2, 0});
    return result;
}
std::vector<ProductContactViolation> ExtrudedAMR::contact_violations(
    double ratio, unsigned maximum_jump, std::size_t budget) const
{
    if (data_->base.size() + data_->edges.size() + data_->axial.size() + data_->points.size() + data_->z.size() >
        budget)
        throw std::invalid_argument("Product contact metadata exceeds budget.");
    auto local = std::make_shared<LocalProductAMR::Data>();
    local->pool = *data_;
    auto bases = std::make_shared<const std::vector<BaseCellKey>>(data_->base_keys);
    auto intervals = std::make_shared<const std::vector<AxialIntervalKey>>(data_->interval_keys);
    for (const auto& base : *bases)
        local->columns.emplace(base, intervals);
    auto geometry = std::make_shared<const ExtrudedGeometry>(data_->topology, data_->construction);
    local->runs.push_back({ExtrudedRegion("contact_query", data_->topology, std::move(geometry)),
        {bases, 0, bases->size()}, {intervals, 0, intervals->size()}, {}});
    return LocalProductAMR(std::move(local)).contact_violations(ratio, maximum_jump, budget);
}
std::shared_ptr<const LocalProductAMR> LocalProductAMR::adapt(const std::vector<ProductCellAncestry>& refine_xy,
    const std::vector<ProductCellAncestry>& refine_z, const std::vector<ProductCellAncestry>& coarsen_xy,
    const std::vector<ProductCellAncestry>& coarsen_z) const
{
    std::ostringstream marks;
    marks << data_->collective_id << ' ';
    for (const auto* list : {&refine_xy, &refine_z, &coarsen_xy, &coarsen_z})
    {
        marks << " marks ";
        for (const auto& p : *list)
        {
            key_text(marks, p.base);
            key_text(marks, p.axial);
        }
    }
    same(marks.str());
    auto data = std::make_shared<Data>(*data_);
    collectively(
        [&]
        {
            auto exists = [&](const ProductCellAncestry& p)
            {
                auto column = data_->columns.find(p.base);
                return column != data_->columns.end() &&
                       std::binary_search(column->second->begin(), column->second->end(), p.axial);
            };
            std::map<ProductCellAncestry, unsigned> refine;
            for (const auto& p : refine_xy)
            {
                if (!exists(p))
                    throw std::invalid_argument("Local XY mark is not an active product leaf.");
                refine[p] |= 1;
            }
            for (const auto& p : refine_z)
            {
                if (!exists(p))
                    throw std::invalid_argument("Local Z mark is not an active product leaf.");
                refine[p] |= 2;
            }
            std::set<ProductCellAncestry> consumed;
            std::set<ProductCellAncestry> unique;
            for (const auto& p : coarsen_xy)
            {
                if (!unique.insert(p).second || !data_->pool.base.contains(p.base) ||
                    data_->pool.base.at(p.base).children.empty())
                    throw std::invalid_argument("Local XY coarsening parent is invalid.");
                for (const auto& child : data_->pool.base.at(p.base).children)
                {
                    ProductCellAncestry leaf{child, p.axial};
                    if (!exists(leaf) || refine.contains(leaf) || !consumed.insert(leaf).second)
                        throw std::invalid_argument(
                            "Local XY coarsening needs an unmarked complete immediate source family.");
                }
            }
            unique.clear();
            for (const auto& p : coarsen_z)
            {
                if (!unique.insert(p).second || !data_->pool.axial.contains(p.axial) ||
                    data_->pool.axial.at(p.axial).children.size() != 2)
                    throw std::invalid_argument("Local Z coarsening parent is invalid.");
                for (const auto& child : data_->pool.axial.at(p.axial).children)
                {
                    ProductCellAncestry leaf{p.base, child};
                    if (!exists(leaf) || refine.contains(leaf) || !consumed.insert(leaf).second)
                        throw std::invalid_argument(
                            "Local Z coarsening needs an unmarked complete immediate source family.");
                }
            }
            std::map<BaseCellKey, std::set<AxialIntervalKey>> changed;
            auto column = [&](const BaseCellKey& key) -> std::set<AxialIntervalKey>&
            {
                auto [it, inserted] = changed.try_emplace(key);
                if (inserted)
                {
                    auto original = data_->columns.find(key);
                    if (original != data_->columns.end())
                        it->second.insert(original->second->begin(), original->second->end());
                }
                return it->second;
            };
            for (const auto& p : coarsen_xy)
            {
                for (const auto& child : data_->pool.base.at(p.base).children)
                    column(child).erase(p.axial);
                column(p.base).insert(p.axial);
            }
            for (const auto& p : coarsen_z)
            {
                for (const auto& child : data_->pool.axial.at(p.axial).children)
                    column(p.base).erase(child);
                column(p.base).insert(p.axial);
            }
            for (const auto& [p, flags] : refine)
            {
                auto bases = flags & 1 ? local_base_children(data->pool, p.base) : std::vector<BaseCellKey>{p.base};
                auto zkeys = flags & 2 ? local_z_children(data->pool, p.axial) : std::vector<AxialIntervalKey>{p.axial};
                column(p.base).erase(p.axial);
                for (const auto& base : bases)
                    for (const auto& z : zkeys)
                        column(base).insert(z);
            }
            for (auto& [key, z] : changed)
            {
                if (z.empty())
                    data->columns.erase(key);
                else
                    data->columns[key] = std::make_shared<const ZKeys>(z.begin(), z.end());
            }
            if (changed.empty())
                return;
            if (data->generation == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Local product generation exhausted.");
            ++data->generation;
            data->token = std::make_shared<const char>(0);
            build_local_runs(*data, data_);
        });
    if (data->generation == data_->generation)
        return std::shared_ptr<const LocalProductAMR>(new LocalProductAMR(data_));
    same(local_description(*data));
    data->collective_id = SimpleFluid::Meshes::identity();
    return std::shared_ptr<const LocalProductAMR>(new LocalProductAMR(std::move(data)));
}
LocalProductAMR::StorageCounts LocalProductAMR::storage_counts() const
{
    StorageCounts counts{};
    std::set<const void*> bases, axes, values, selections;
    counts.runs = data_->runs.size();
    counts.point_recipes = data_->pool.points.size();
    counts.z_recipes = data_->pool.z.size();
    for (const auto& [base, z] : data_->columns)
        if (selections.insert(z.get()).second)
            counts.selection_entries += z->size();
    for (const auto& run : data_->runs)
    {
        counts.cells += run.region.layout().cells;
        const auto& construction = run.region.geometry().construction();
        if (bases.insert(construction->base_storage_identity()).second)
        {
            ++counts.base_templates;
            counts.base_vertices += run.region.topology().indexer().num_nodes_per_layer;
        }
        if (axes.insert(construction->axis_storage_identity()).second)
            ++counts.axis_templates;
        if (values.insert(construction->axis_values_storage_identity()).second)
            ++counts.axial_values;
        for (const auto& b : run.boundaries)
            counts.seam_descriptors += !b.physical;
    }
    return counts;
}
MeshStorageReport LocalProductAMR::storage_report() const
{
    MeshStorageReport report;
    report.objects = sizeof(LocalProductAMR) + sizeof(Data) + data_->runs.capacity() * sizeof(LocalProductRun);
    std::set<const void*> topology, bases, axes, values, basekeys, zkeys;
    for (const auto& run : data_->runs)
    {
        report.objects += sizeof(ExtrudedTopology) + sizeof(ExtrudedGeometry) + run.region.name().capacity();
        const auto& t = run.region.topology();
        const auto& c = run.region.geometry().construction();
        auto ts = t.storage_report().topology;
        if (!topology.insert(t.base_storage_identity()).second)
            ts -= t.base_storage_bytes();
        report.topology += ts;
        report.geometry += c->view_storage_bytes();
        if (bases.insert(c->base_storage_identity()).second)
            report.geometry += c->base_storage_bytes();
        if (axes.insert(c->axis_storage_identity()).second)
        {
            report.geometry += c->axis_storage_bytes();
            if (!values.insert(c->axis_values_storage_identity()).second)
                report.geometry -= c->axis_values_storage_bytes();
        }
        if (basekeys.insert(run.base_keys.storage_identity()).second)
            report.indexing += run.base_keys.storage_bytes();
        if (zkeys.insert(run.intervals.storage_identity()).second)
            report.indexing += run.intervals.storage_bytes();
        report.interfaces += run.boundaries.capacity() * sizeof(LocalProductBoundary);
        for (const auto& b : run.boundaries)
            report.interfaces +=
                (b.edge.creator.path.capacity() + b.edge.path.capacity() + b.plane.creator.path.capacity()) *
                sizeof(std::uint64_t);
    }
    for (const auto& [base, z] : data_->columns)
    {
        report.indexing += sizeof(decltype(data_->columns)::value_type) + 4 * sizeof(void*) +
                           base.path.capacity() * sizeof(std::uint64_t);
        if (zkeys.insert(z.get()).second)
        {
            report.indexing += sizeof(ZKeys) + z->capacity() * sizeof(AxialIntervalKey);
            for (const auto& k : *z)
                report.indexing += k.path.capacity() * sizeof(std::uint64_t);
        }
    }
    // Retained factor ancestry and construction recipes are separate from the
    // shared native templates; this capacity estimate includes their payload.
    const auto& p = data_->pool;
    report.objects += sizeof(ExtrudedAMRGrid) + p.grid->xy_nodes.capacity() * sizeof(GeometryPoint) +
                      p.grid->base_cells.capacity() * sizeof(std::vector<unsigned>) +
                      p.grid->z_edges.capacity() * sizeof(double) + p.points.capacity() * sizeof(ExactPointRecipe) +
                      p.z.capacity() * sizeof(ExactScalarRecipe);
    for (const auto& c : p.grid->base_cells)
        report.objects += c.capacity() * sizeof(unsigned);
    for (const auto& q : p.points)
        report.objects += q.parents.capacity() * sizeof(std::size_t);
    for (const auto& q : p.z)
        report.objects += q.parents.capacity() * sizeof(std::size_t);
    for (const auto& [k, n] : p.base)
    {
        report.indexing += sizeof(decltype(p.base)::value_type) + 4 * sizeof(void*) +
                           k.path.capacity() * sizeof(std::uint64_t) + n.vertices.capacity() * sizeof(std::size_t) +
                           n.edges.capacity() * sizeof(BaseEdgeKey) + n.children.capacity() * sizeof(BaseCellKey);
        for (const auto& e : n.edges)
            report.indexing += (e.creator.path.capacity() + e.path.capacity()) * sizeof(std::uint64_t);
        for (const auto& c : n.children)
            report.indexing += c.path.capacity() * sizeof(std::uint64_t);
    }
    for (const auto& [k, e] : p.edges)
        report.indexing += sizeof(decltype(p.edges)::value_type) + 4 * sizeof(void*) +
                           (k.creator.path.capacity() + k.path.capacity()) * sizeof(std::uint64_t) +
                           e.boundary.capacity();
    for (const auto& [k, n] : p.axial)
    {
        report.indexing += sizeof(decltype(p.axial)::value_type) + 4 * sizeof(void*) +
                           k.path.capacity() * sizeof(std::uint64_t) + n.children.capacity() * sizeof(AxialIntervalKey);
        for (const auto& c : n.children)
            report.indexing += c.path.capacity() * sizeof(std::uint64_t);
    }
    report.objects += p.grid->boundaries.capacity() * sizeof(SemiStructMeshTopo::BoundaryEdge) +
                      p.grid->triangle_reference_edges.capacity() * sizeof(std::size_t);
    for (const auto& b : p.grid->boundaries)
        report.objects += b.batch_name.capacity();
    for (const auto& [a, b] : p.grid->boundary_aliases)
        report.objects +=
            sizeof(decltype(p.grid->boundary_aliases)::value_type) + 4 * sizeof(void*) + a.capacity() + b.capacity();
    report.indexing +=
        p.base_keys.capacity() * sizeof(BaseCellKey) + p.interval_keys.capacity() * sizeof(AxialIntervalKey);
    for (const auto& k : p.leaves)
        report.indexing += sizeof(BaseCellKey) + 4 * sizeof(void*) + k.path.capacity() * sizeof(std::uint64_t);
    for (const auto& k : p.intervals)
        report.indexing += sizeof(AxialIntervalKey) + 4 * sizeof(void*) + k.path.capacity() * sizeof(std::uint64_t);
    return report;
}

} // namespace SimpleFluid::Meshes
