/** @file ExtrudedAMR.cc
 * @brief Collective conforming factor adaptation with retained exact constructions.
 */
#include "geometry/ExtrudedAMR.hh"
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
    data.edges[left] = {e.a, m, e.boundary};
    data.edges[right] = {m, e.b, e.boundary};
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
        if (cells.size() == 2 && data.base.at(cells[0].first).vertices.size() == 3 &&
            data.base.at(cells[1].first).vertices.size() == 3 && cells[0].second != cells[1].second)
            throw std::invalid_argument("Root triangle refinement edges are incompatible across a shared edge.");
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
} // namespace SimpleFluid::Meshes
