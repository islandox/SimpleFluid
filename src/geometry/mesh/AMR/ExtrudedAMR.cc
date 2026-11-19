/** @file ExtrudedAMR.cc
 * @brief Collective conforming factor adaptation with retained exact constructions.
 */
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/PLIC/GeometryCodecTags.hh"
#include "geometry/PLIC/details/NativeConstructionWireAccess.hh"
#include "geometry/VoFWireProtocol.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedProductKernel.hh"
#include "geometry/mesh/AMR/LocalProductAMR.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/NativeFragmentAccess.hh"
#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <atomic>
#include <bit>
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

namespace SimpleFluid::Meshes::AMR
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
    std::vector<PLIC::PointRecipe> points;
    std::vector<PLIC::ScalarRecipe> z;
    std::vector<BaseCellKey> base_keys;
    std::vector<AxialIntervalKey> interval_keys;
    std::shared_ptr<const ExtrudedTopology> topology;
    std::shared_ptr<const PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>> construction;
    SP<const MultiRegionMesh> geometry;
    std::shared_ptr<const CompositePartition<>> partition;
    ID topology_generation = 0, partition_generation = 0, collective_id = 0;
    std::shared_ptr<const void> token = std::make_shared<const char>(0), topology_token = token,
                                partition_token = token;
};

namespace
{
using Data = ExtrudedAMR::Data;
PLIC::GeometryPoint displayed(const Data& data, std::size_t point)
{
    const auto& recipe = data.points.at(point);
    if (recipe.parents.empty())
        return recipe.root;
    PLIC::GeometryPoint result{};
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
    std::vector<PLIC::PointRecipe> points;
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
    std::vector<PLIC::ScalarRecipe> z;
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
    data.construction =
        PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>::make(data.topology, data.points, active, data.z, zactive);
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
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> ExtrudedAMR::cell_geometry(ID cell) const
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
    bytes += data_->points.capacity() * sizeof(PLIC::PointRecipe) + data_->z.capacity() * sizeof(PLIC::ScalarRecipe);
    for (const auto& p : data_->points)
        bytes += p.parents.capacity() * sizeof(std::size_t);
    for (const auto& z : data_->z)
        bytes += z.parents.capacity() * sizeof(std::size_t);
    bytes += data_->grid->xy_nodes.capacity() * sizeof(PLIC::GeometryPoint) +
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
    std::shared_ptr<const PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>> construction;
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
    std::shared_ptr<const PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>> coordinates;
    std::map<std::vector<std::size_t>, std::shared_ptr<const PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>>>
        axes;
    struct Planned
    {
        BaseTemplate base;
        std::shared_ptr<const ZKeys> zkeys;
        std::size_t first, count, axis_first;
        std::shared_ptr<const PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>> axis;
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
                        coordinates = PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>::make(
                            all_topology, patch.points, active, pool.z, global_axis);
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
                coordinates = PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>::make(
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
                auto construction = PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>::shared_axis_selection(
                    topology, base.construction, coordinates, points);
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
        auto construction = PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>::shared_view(
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
    data->collective_id = SimpleFluid::Meshes::AMR::identity();
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
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> LocalProductAMR::cell_geometry(
    std::size_t run, std::uint64_t cell) const
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
    data->collective_id = SimpleFluid::Meshes::AMR::identity();
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
    report.objects += sizeof(ExtrudedAMRGrid) + p.grid->xy_nodes.capacity() * sizeof(PLIC::GeometryPoint) +
                      p.grid->base_cells.capacity() * sizeof(std::vector<unsigned>) +
                      p.grid->z_edges.capacity() * sizeof(double) + p.points.capacity() * sizeof(PLIC::PointRecipe) +
                      p.z.capacity() * sizeof(PLIC::ScalarRecipe);
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

namespace product_detail
{
using adaptive_detail::Reader;
using adaptive_detail::Writer;
using Pool = ExtrudedAMR::Data;
void key(Writer& w, const BaseCellKey& k)
{
    w.bytes(encode_adaptive_key({0, ProductCellAncestry{k, {}}}));
}
BaseCellKey base(Reader& r)
{
    return std::get<ProductCellAncestry>(decode_adaptive_key(r.bytes()).ancestry).base;
}
void key(Writer& w, const AxialIntervalKey& k)
{
    w.bytes(encode_adaptive_key({0, ProductCellAncestry{{}, k}}));
}
AxialIntervalKey axial(Reader& r)
{
    return std::get<ProductCellAncestry>(decode_adaptive_key(r.bytes()).ancestry).axial;
}
void key(Writer& w, const BaseEdgeKey& k)
{
    key(w, k.creator);
    w.word(k.ordinal);
    w.word(std::uint64_t(k.kind));
    w.word(k.path.size());
    for (auto n : k.path)
        w.word(n);
}
BaseEdgeKey edge(Reader& r)
{
    BaseEdgeKey k;
    k.creator = base(r);
    k.ordinal = r.word();
    auto kind = r.word();
    if (kind > 1)
        throw std::invalid_argument("Invalid product edge creator.");
    k.kind = BaseEdgeKey::Kind(kind);
    auto n = r.word();
    if (n > 48)
        throw std::invalid_argument("Invalid product edge depth.");
    for (std::uint64_t i = 0; i < n; ++i)
    {
        auto child = r.word();
        if (child > 1)
            throw std::invalid_argument("Invalid edge half.");
        k.path.push_back(child);
    }
    return k;
}
std::size_t count(Reader& r)
{
    auto n = r.word();
    if (n > (r.data.size() - r.offset) / 8)
        throw std::invalid_argument("Invalid factor count.");
    return n;
}
AdaptiveBytes root_packet(const ExtrudedAMRGrid& grid)
{
    Writer w;
    packet_codec::write_header(w, packet_codec::Family::ProductRoot, uint8_t{7}, packet_codec::Record::Value);
    w.word(grid.region_id);
    w.word(grid.material_id);
    w.word(grid.max_depth);
    w.word(std::bit_cast<std::uint64_t>(grid.minimum_scaled_area));
    w.word(grid.z_edges.size());
    for (double z : grid.z_edges)
        w.word(std::bit_cast<std::uint64_t>(z));
    w.word(grid.boundary_aliases.size());
    for (const auto& [a, b] : grid.boundary_aliases)
    {
        w.bytes(std::span(a));
        w.bytes(std::span(b));
    }
    return std::move(w.data);
}
void root_identity(Writer& w, std::span<const char> capsule)
{
    Reader reader{capsule};
    packet_codec::read_header(
        reader, packet_codec::Family::ProductRoot, uint8_t{7}, packet_codec::Record::Value);
    const auto body = capsule.subspan(reader.offset);
    // This prefix belongs only to canonical trace identities. The transport
    // capsule has its own byte header; neither representation reads the other.
    constexpr std::uint64_t canonical_root_identity_prefix = 0x534650524f4f5436ULL;
    w.word(sizeof(canonical_root_identity_prefix) + body.size());
    w.word(canonical_root_identity_prefix);
    w.data.insert(w.data.end(), body.begin(), body.end());
}
std::shared_ptr<const ExtrudedAMRGrid> root_grid(std::span<const char> packet)
{
    Reader r{packet};
    packet_codec::read_header(r, packet_codec::Family::ProductRoot, uint8_t{7}, packet_codec::Record::Value);
    auto g = std::make_shared<ExtrudedAMRGrid>();
    g->region_id = r.word();
    g->material_id = r.word();
    g->max_depth = r.word();
    g->minimum_scaled_area = std::bit_cast<double>(r.word());
    auto nz = count(r);
    if (nz < 2 || nz > std::numeric_limits<unsigned>::max() || !g->max_depth || g->max_depth > 48 ||
        !std::isfinite(g->minimum_scaled_area) || g->minimum_scaled_area < 0)
        throw std::invalid_argument("Invalid product root admission.");
    // Only root interval count is needed by cap provenance, not a replicated
    // active axial recipe. Root coordinates live in pruned scalar capsules.
    for (std::size_t i = 0; i < nz; ++i)
    {
        const auto z = std::bit_cast<double>(r.word());
        if (!std::isfinite(z) || (!g->z_edges.empty() && z <= g->z_edges.back()))
            throw std::invalid_argument("Invalid root axial coordinates.");
        g->z_edges.push_back(z);
    }
    auto n = count(r);
    for (std::size_t i = 0; i < n; ++i)
    {
        auto a = r.bytes(), b = r.bytes();
        if (!g->boundary_aliases.emplace(std::string(a.begin(), a.end()), std::string(b.begin(), b.end())).second)
            throw std::invalid_argument("Repeated product alias.");
    }
    r.done();
    return g;
}
AdaptiveBytes base_packet(Pool pool, const BaseCellKey& selected)
{
    pool.leaves = {selected};
    pool.intervals.clear();
    compact(pool);
    Writer w;
    packet_codec::write_header(w, packet_codec::Family::ProductBase, uint8_t{7}, packet_codec::Record::Value);
    w.word(pool.points.size());
    for (const auto& recipe : pool.points)
    {
        for (double q : recipe.root)
            w.word(std::bit_cast<std::uint64_t>(q));
        w.word(recipe.parents.size());
        for (auto p : recipe.parents)
            w.word(p);
    }
    w.word(pool.base.size());
    for (const auto& [k, node] : pool.base)
    {
        key(w, k);
        w.word(node.vertices.size());
        for (auto v : node.vertices)
            w.word(v);
        for (const auto& e : node.edges)
            key(w, e);
        w.word(node.reference);
        w.word(node.children.size());
        for (const auto& c : node.children)
            key(w, c);
    }
    w.word(pool.edges.size());
    for (const auto& [k, node] : pool.edges)
    {
        key(w, k);
        w.word(node.a);
        w.word(node.b);
        w.word(node.physical);
        w.bytes(std::span(node.boundary));
    }
    return std::move(w.data);
}
void read_base(Pool& pool, std::span<const char> packet)
{
    Reader r{packet};
    packet_codec::read_header(r, packet_codec::Family::ProductBase, uint8_t{7}, packet_codec::Record::Value);
    auto n = count(r);
    for (std::size_t i = 0; i < n; ++i)
    {
        PLIC::PointRecipe recipe;
        for (auto& q : recipe.root)
        {
            q = std::bit_cast<double>(r.word());
            if (!std::isfinite(q))
                throw std::invalid_argument("Nonfinite product root point.");
        }
        auto np = count(r);
        for (std::size_t j = 0; j < np; ++j)
        {
            auto parent = r.word();
            if (parent >= i)
                throw std::invalid_argument("Cyclic product point recipe.");
            recipe.parents.push_back(parent);
        }
        pool.points.push_back(std::move(recipe));
    }
    n = count(r);
    for (std::size_t i = 0; i < n; ++i)
    {
        auto k = base(r);
        BaseNode node;
        auto nv = count(r);
        if (nv < 3)
            throw std::invalid_argument("Invalid product polygon.");
        for (std::size_t v = 0; v < nv; ++v)
        {
            auto id = r.word();
            if (id >= pool.points.size())
                throw std::invalid_argument("Invalid point reference.");
            node.vertices.push_back(id);
        }
        for (std::size_t e = 0; e < nv; ++e)
            node.edges.push_back(edge(r));
        node.reference = r.word();
        auto nc = count(r);
        for (std::size_t c = 0; c < nc; ++c)
            node.children.push_back(base(r));
        if (!pool.base.emplace(k, std::move(node)).second)
            throw std::invalid_argument("Repeated product base ancestry.");
    }
    n = count(r);
    for (std::size_t i = 0; i < n; ++i)
    {
        auto k = edge(r);
        EdgeNode node;
        node.a = r.word();
        node.b = r.word();
        auto physical = r.word();
        if (node.a >= pool.points.size() || node.b >= pool.points.size() || node.a == node.b || physical > 1)
            throw std::invalid_argument("Invalid product edge endpoints.");
        node.physical = physical;
        auto name = r.bytes();
        node.boundary.assign(name.begin(), name.end());
        if (!pool.edges.emplace(k, std::move(node)).second)
            throw std::invalid_argument("Repeated product edge identity.");
    }
    r.done();
    for (const auto& [k, node] : pool.base)
        for (const auto& edge : node.edges)
            if (!pool.edges.contains(edge))
                throw std::invalid_argument("Missing product edge ancestor.");
}
AdaptiveBytes axis_packet(Pool pool, const AxialIntervalKey& selected)
{
    pool.leaves.clear();
    pool.intervals = {selected};
    compact(pool);
    Writer w;
    packet_codec::write_header(w, packet_codec::Family::ProductAxis, uint8_t{7}, packet_codec::Record::Value);
    w.word(pool.z.size());
    for (const auto& recipe : pool.z)
    {
        w.word(std::bit_cast<std::uint64_t>(recipe.root));
        w.word(recipe.parents.size());
        for (auto p : recipe.parents)
            w.word(p);
    }
    w.word(pool.axial.size());
    for (const auto& [k, node] : pool.axial)
    {
        key(w, k);
        w.word(node.lower);
        w.word(node.upper);
        w.word(node.children.size());
        for (const auto& c : node.children)
            key(w, c);
    }
    return std::move(w.data);
}
void read_axis(Pool& pool, std::span<const char> packet)
{
    Reader r{packet};
    packet_codec::read_header(r, packet_codec::Family::ProductAxis, uint8_t{7}, packet_codec::Record::Value);
    auto n = count(r);
    for (std::size_t i = 0; i < n; ++i)
    {
        PLIC::ScalarRecipe recipe;
        recipe.root = std::bit_cast<double>(r.word());
        if (!std::isfinite(recipe.root))
            throw std::invalid_argument("Nonfinite product root Z.");
        auto np = count(r);
        for (std::size_t j = 0; j < np; ++j)
        {
            auto p = r.word();
            if (p >= i)
                throw std::invalid_argument("Cyclic scalar recipe.");
            recipe.parents.push_back(p);
        }
        pool.z.push_back(std::move(recipe));
    }
    n = count(r);
    for (std::size_t i = 0; i < n; ++i)
    {
        auto k = axial(r);
        ZNode node;
        node.lower = r.word();
        node.upper = r.word();
        if (node.lower >= pool.z.size() || node.upper >= pool.z.size() || node.lower == node.upper)
            throw std::invalid_argument("Invalid axial endpoints.");
        auto nc = count(r);
        for (std::size_t c = 0; c < nc; ++c)
            node.children.push_back(axial(r));
        if (!pool.axial.emplace(k, std::move(node)).second)
            throw std::invalid_argument("Repeated axial ancestry.");
    }
    r.done();
}
LocalData local_state(const AdaptiveLeaf& leaf)
{
    if (!leaf.factor_recipe || !leaf.factor_recipe->root || !leaf.factor_recipe->base || !leaf.factor_recipe->axis)
        throw std::invalid_argument("Product leaf requires detached independent factor recipes.");
    const auto* p = std::get_if<ProductCellAncestry>(&leaf.key.ancestry);
    if (!p)
        throw std::invalid_argument("Product kernel received another ancestry family.");
    LocalData state;
    state.pool.grid = root_grid(*leaf.factor_recipe->root);
    read_base(state.pool, *leaf.factor_recipe->base);
    read_axis(state.pool, *leaf.factor_recipe->axis);
    if (leaf.key.region != state.pool.grid->region_id || leaf.material != state.pool.grid->material_id ||
        !state.pool.base.contains(p->base) || !state.pool.axial.contains(p->axial))
        throw std::invalid_argument("Product factor/key binding differs.");
    state.columns.emplace(p->base, std::make_shared<const ZKeys>(ZKeys{p->axial}));
    intern_columns(state);
    return state;
}
using OriginalProductLeaves = std::map<AdaptiveCellKey, const AdaptiveLeaf*>;
bool same_product_base_order(const NativeFragmentDescriptor&, const NativeFragmentDescriptor&);

std::shared_ptr<const NativeFragmentDescriptor> retain_run_parts(const LocalProductRun& run,
    std::uint64_t region, std::shared_ptr<const NativeFragmentDescriptor> descriptor,
    const OriginalProductLeaves& original)
{
    if (descriptor->kind != NativeFragmentDescriptor::Kind::ConstructedProduct || descriptor->parts.size() != 4)
        return descriptor;
    std::array<std::shared_ptr<const NativeFragmentBytes>, 4> retained{};
    std::set<std::shared_ptr<const NativeFragmentDescriptor>, std::owner_less<>> checked;
    for (const auto& axis : run.intervals)
        for (const auto& base : run.base_keys)
        {
            const auto& prior = original.at({region, ProductCellAncestry{base, axis}})->native_recipe;
            if (!prior || prior->kind != descriptor->kind || prior->parts.size() != retained.size() ||
                !checked.insert(prior).second)
                continue;
            bool equivalent = prior->view == descriptor->view;
            for (std::size_t part = 0; part < retained.size(); ++part)
            {
                const auto& previous = prior->parts[part];
                const auto& current = descriptor->parts[part];
                const bool match = previous && current && (previous == current || *previous == *current);
                equivalent &= match;
                if (match && !retained[part]) retained[part] = previous;
            }
            if (equivalent) return prior;
            // Compacting independent source capsules can renumber base nodes
            // while preserving every ordered cell, edge and construction. In
            // that case keep the original paired topology/construction parts.
            if ((!retained[0] || !retained[1]) && same_product_base_order(*prior, *descriptor))
            {
                retained[0] = prior->parts[0];
                retained[1] = prior->parts[1];
            }
        }
    if (std::ranges::none_of(retained, [](const auto& part) { return bool(part); })) return descriptor;
    auto shared = std::make_shared<NativeFragmentDescriptor>();
    shared->kind = descriptor->kind;
    shared->view = descriptor->view;
    shared->parts = descriptor->parts;
    for (std::size_t i = 0; i < retained.size(); ++i)
        if (retained[i]) shared->parts[i] = std::move(retained[i]);
    return shared;
}

std::vector<AdaptiveLeaf> export_state(LocalData& state, const AdaptiveBytes& domain,
    std::shared_ptr<const AdaptiveBytes> root = {}, const OriginalProductLeaves* original = nullptr)
{
    intern_columns(state);
    build_local_runs(state);
    std::map<BaseCellKey, std::shared_ptr<const AdaptiveBytes>> bases;
    std::map<AxialIntervalKey, std::shared_ptr<const AdaptiveBytes>> axes;
    if (!original)
    {
        if (!root) root = std::make_shared<const AdaptiveBytes>(root_packet(*state.pool.grid));
        for (const auto& [base, z] : state.columns)
        {
            bases.try_emplace(base, std::make_shared<const AdaptiveBytes>(base_packet(state.pool, base)));
            for (const auto& interval : *z)
                if (!axes.contains(interval))
                    axes.emplace(interval, std::make_shared<const AdaptiveBytes>(axis_packet(state.pool, interval)));
        }
    }
    std::vector<AdaptiveLeaf> result;
    NativeFragmentCache cache;
    for (const auto& run : state.runs)
    {
        auto native = extract_native_fragment(run.region, cache);
        if (original) native = retain_run_parts(run, state.pool.grid->region_id, std::move(native), *original);
        for (std::size_t z = 0; z < run.intervals.size(); ++z)
            for (std::size_t b = 0; b < run.base_keys.size(); ++b)
            {
                AdaptiveLeaf leaf;
                leaf.domain = domain;
                leaf.key = {state.pool.grid->region_id, ProductCellAncestry{run.base_keys[b], run.intervals[z]}};
                leaf.material = state.pool.grid->material_id;
                leaf.provider = AdaptiveProviderKind::Product;
                leaf.native_recipe = native;
                leaf.native_cell = z * run.base_keys.size() + b;
                if (original)
                {
                    // local_state and merge_pool already validated the exact
                    // original capsule closure for this active typed key.
                    const auto& before = *original->at(leaf.key);
                    leaf.factor_recipe = before.factor_recipe;
                    leaf.revision = before.revision;
                    leaf.parent_family = before.parent_family;
                    leaf.work = before.work;
                }
                else
                    leaf.factor_recipe = std::make_shared<const AdaptiveFactorRecipe>(AdaptiveFactorRecipe{
                        root, bases.at(run.base_keys[b]), axes.at(run.intervals[z])});
                result.push_back(std::move(leaf));
            }
    }
    return result;
}
void merge_pool(Pool& target, const Pool& source)
{
    if (!target.grid)
        target.grid = source.grid;
    if (root_packet(*target.grid) != root_packet(*source.grid))
        throw std::invalid_argument("Product root recipes differ.");
    auto merge_recipes = [](auto& destination, const auto& incoming)
    {
        std::map<AdaptiveBytes, std::size_t> known;
        auto signature = [](const auto& recipe)
        {
            Writer w;
            if constexpr (std::is_same_v<std::decay_t<decltype(recipe.root)>, double>)
                w.word(std::bit_cast<std::uint64_t>(recipe.root));
            else
                for (double x : recipe.root)
                    w.word(std::bit_cast<std::uint64_t>(x));
            auto parents = recipe.parents;
            std::sort(parents.begin(), parents.end());
            w.word(parents.size());
            for (auto p : parents)
                w.word(p);
            return std::move(w.data);
        };
        for (std::size_t i = 0; i < destination.size(); ++i)
            known.emplace(signature(destination[i]), i);
        std::vector<std::size_t> remap;
        for (const auto& source_recipe : incoming)
        {
            auto recipe = source_recipe;
            for (auto& parent : recipe.parents)
                parent = remap.at(parent);
            auto bytes = signature(recipe);
            auto [it, inserted] = known.emplace(bytes, destination.size());
            if (inserted)
                destination.push_back(std::move(recipe));
            remap.push_back(it->second);
        }
        return remap;
    };
    const auto points = merge_recipes(target.points, source.points), z = merge_recipes(target.z, source.z);
    for (const auto& [key, source_node] : source.base)
    {
        auto node = source_node;
        for (auto& vertex : node.vertices)
            vertex = points.at(vertex);
        auto [it, inserted] = target.base.emplace(key, node);
        if (!inserted)
        {
            if (it->second.vertices != node.vertices || it->second.edges != node.edges ||
                it->second.reference != node.reference)
                throw std::invalid_argument("Shared base ancestry has conflicting exact construction.");
            if (it->second.children.empty())
                it->second.children = node.children;
            else if (!node.children.empty() && it->second.children != node.children)
                throw std::invalid_argument("Conflicting base child family.");
        }
    }
    for (const auto& [key, source_node] : source.edges)
    {
        auto node = source_node;
        node.a = points.at(node.a);
        node.b = points.at(node.b);
        auto [it, inserted] = target.edges.emplace(key, node);
        if (!inserted && (it->second.a != node.a || it->second.b != node.b || it->second.physical != node.physical ||
                             it->second.boundary != node.boundary))
            throw std::invalid_argument("Conflicting exact edge ancestry.");
    }
    for (const auto& [key, source_node] : source.axial)
    {
        auto node = source_node;
        node.lower = z.at(node.lower);
        node.upper = z.at(node.upper);
        auto [it, inserted] = target.axial.emplace(key, node);
        if (!inserted)
        {
            if (it->second.lower != node.lower || it->second.upper != node.upper)
                throw std::invalid_argument("Shared axial ancestry has conflicting exact construction.");
            if (it->second.children.empty())
                it->second.children = node.children;
            else if (!node.children.empty() && it->second.children != node.children)
                throw std::invalid_argument("Conflicting axial children.");
        }
    }
}

using ProductConstruction = PLIC::details::ExtrudedConstructionAccess;
using NativeAdmission = geometry_detail::NativeConstructionAdmission;
using NativeStorage = geometry_detail::NativeConstructorStorageOwner;
using NativeRoutes = std::span<adaptive_detail::NativeKernelSourceFace>;
using SupportVisit = adaptive_detail::NativeKernelAuthority::SupportVisit;

std::span<const char> words(Reader& reader)
{
    const auto n = count(reader);
    const auto result = reader.data.subspan(reader.offset, n * sizeof(std::uint64_t));
    reader.offset += result.size();
    return result;
}

std::string_view text(Reader& reader)
{
    const auto bytes = reader.bytes_view();
    return {bytes.data(), bytes.size()};
}

struct BorrowedProductPointRecipe
{
    std::array<double, 3> root;
    std::span<const char> parents;
};

std::vector<BorrowedProductPointRecipe> product_point_recipes(Reader& reader, std::span<const char> topology)
{
    if (reader.byte() != static_cast<std::uint8_t>(PLIC::GeometryTag::ExtrusionBase) || reader.byte() != 2 ||
        !std::ranges::equal(reader.bytes_view(), topology))
        throw std::invalid_argument("Product base constructor belongs to a different topology.");
    const auto n = count(reader);
    std::vector<BorrowedProductPointRecipe> points;
    points.reserve(n);
    for (std::size_t index = 0; index < n; ++index)
    {
        BorrowedProductPointRecipe point;
        for (auto& coordinate : point.root)
            coordinate = PLIC::details::NativeConstructionWireAccess::ordinary_scalar(reader.data, reader.offset);
        point.parents = words(reader);
        for (Reader parents{point.parents}; parents.offset < parents.data.size();)
            if (parents.word() >= index)
                throw std::invalid_argument("Product base has a cyclic point recipe.");
        points.push_back(point);
    }
    return points;
}

bool same_product_base_order(const NativeFragmentDescriptor& a, const NativeFragmentDescriptor& b)
{
    if (!a.parts[0] || !a.parts[1] || !b.parts[0] || !b.parts[1]) return false;
    Reader left{*a.parts[0]}, right{*b.parts[0]};
    for (auto* reader : {&left, &right})
        packet_codec::read_header(*reader, packet_codec::Family::ExtrudedTopologyBase, uint8_t{2},
            packet_codec::Record::Value);
    const auto nodes = left.word();
    const auto other_nodes = right.word();
    const auto cells = count(left), other_cells = count(right);
    if (nodes != other_nodes || cells != other_cells || nodes > left.data.size() / sizeof(std::uint64_t))
        return false;
    constexpr auto unassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> permutation(nodes, unassigned), inverse(nodes, unassigned);
    for (std::size_t cell = 0; cell < cells; ++cell)
    {
        Reader a_loop{words(left)}, b_loop{words(right)};
        if (a_loop.data.size() != b_loop.data.size()) return false;
        while (a_loop.offset < a_loop.data.size())
        {
            const auto from = a_loop.word(), to = b_loop.word();
            if (from >= nodes || to >= nodes || (permutation[from] != unassigned && permutation[from] != to) ||
                (inverse[to] != unassigned && inverse[to] != from))
                return false;
            permutation[from] = to;
            inverse[to] = from;
        }
    }
    if (std::ranges::find(permutation, unassigned) != permutation.end()) return false;
    const auto edges = count(left), other_edges = count(right);
    if (edges != other_edges) return false;
    for (std::size_t edge = 0; edge < edges; ++edge)
    {
        const auto from0 = left.word(), from1 = left.word(), to0 = right.word(), to1 = right.word();
        if (from0 >= nodes || from1 >= nodes || permutation[from0] != to0 || permutation[from1] != to1 ||
            text(left) != text(right))
            return false;
    }
    left.done();
    right.done();

    left = Reader{*a.parts[1]};
    right = Reader{*b.parts[1]};
    const auto old_recipe = product_point_recipes(left, *a.parts[0]);
    const auto new_recipe = product_point_recipes(right, *b.parts[0]);
    if (count(left) != nodes || count(right) != nodes) return false;
    std::vector<std::size_t> old_selection, new_selection;
    old_selection.reserve(nodes);
    new_selection.reserve(nodes);
    for (std::size_t n = 0; n < nodes; ++n)
    {
        old_selection.push_back(left.word());
        new_selection.push_back(right.word());
    }
    left.done();
    right.done();
    // Compare actual ordered DAG expressions rather than display coordinates.
    // A factor with the same points and different support ancestry cannot
    // supply an original Product base anchor.
    std::vector<std::size_t> mapped(old_recipe.size(), unassigned);
    std::vector<std::pair<std::size_t, std::size_t>> work;
    work.reserve(old_recipe.size());
    const auto match = [&](std::size_t old, std::size_t current)
    {
        if (old >= old_recipe.size() || current >= new_recipe.size()) return false;
        if (mapped[old] != unassigned) return mapped[old] == current;
        mapped[old] = current;
        work.emplace_back(old, current);
        return true;
    };
    for (std::size_t node = 0; node < nodes; ++node)
        if (!match(old_selection[node], new_selection[permutation[node]])) return false;
    for (std::size_t i = 0; i < work.size(); ++i)
    {
        const auto [old, current] = work[i];
        const auto& from = old_recipe[old];
        const auto& to = new_recipe[current];
        if (from.root != to.root || from.parents.size() != to.parents.size()) return false;
        Reader a_parents{from.parents}, b_parents{to.parents};
        while (a_parents.offset < a_parents.data.size())
            if (!match(a_parents.word(), b_parents.word())) return false;
    }
    return true;
}

struct SourceRoot
{
    std::size_t axial_points;
    double minimum_area;
    double lower, upper;
    Reader aliases;

    std::string_view boundary(std::string_view name) const
    {
        auto reader = aliases;
        auto result = name;
        bool found = false;
        const auto size = count(reader);
        for (std::size_t i = 0; i < size; ++i)
        {
            const auto original = text(reader), selected = text(reader);
            if (original.empty() || selected.empty())
                throw std::invalid_argument("Product root contains an empty boundary alias.");
            if (original == name)
            {
                if (found)
                    throw std::invalid_argument("Product root repeats a source boundary alias.");
                found = true;
                result = selected;
            }
        }
        reader.done();
        return result;
    }
};

SourceRoot source_root(const AdaptiveLeaf& leaf, const ProductCellAncestry& cell)
{
    Reader reader{*leaf.factor_recipe->root};
    packet_codec::read_header(reader, packet_codec::Family::ProductRoot, uint8_t{7}, packet_codec::Record::Value);
    if (reader.word() != leaf.key.region || reader.word() != leaf.material || leaf.domain.empty())
        throw std::invalid_argument("Product source differs from its exact region, material or domain.");
    const auto depth = reader.word();
    const auto quality = std::bit_cast<double>(reader.word());
    const auto n = count(reader);
    if (!depth || depth > 48 || cell.base.path.size() > depth || cell.axial.path.size() > depth || n < 2 ||
        n > std::numeric_limits<unsigned>::max() || cell.axial.root >= n - 1 || !(quality >= 0) ||
        !std::isfinite(quality))
        throw std::invalid_argument("Product source exceeds its original root limits.");
    double lower = 0, upper = 0, previous = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto value = std::bit_cast<double>(reader.word());
        if (!std::isfinite(value) || (i && !(previous < value)))
            throw std::invalid_argument("Product source has an invalid root axial coordinate.");
        if (i == cell.axial.root) lower = value;
        if (i == cell.axial.root + 1) upper = value;
        previous = value;
    }
    return {n, quality, lower, upper, reader};
}

template<std::size_t N> std::vector<std::array<long double, N>> source_values(
    Reader& reader, const NativeStorage& scratch)
{
    using Value = std::array<long double, N>;
    std::vector<Value> values;
    const auto n = count(reader);
    scratch.owner->reserve(values, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        Value root{};
        for (auto& x : root)
        {
            const auto q = std::bit_cast<double>(reader.word());
            if (!std::isfinite(q))
                throw std::invalid_argument("Product factor contains a nonfinite root coordinate.");
            x = q;
        }
        const auto parents = count(reader);
        if (!parents)
        {
            values.push_back(root);
            continue;
        }
        Value value{};
        for (std::size_t j = 0; j < parents; ++j)
        {
            const auto parent = reader.word();
            if (parent >= values.size())
                throw std::invalid_argument("Product factor coordinate ancestry is cyclic.");
            for (std::size_t d = 0; d < N; ++d)
                value[d] += values[parent][d];
        }
        for (auto& x : value)
        {
            // Match the original distinct ordinary XY and axial expressions.
            if constexpr (N == 1)
                x /= parents;
            else
                x *= 1.L / static_cast<long double>(parents);
            if (!std::isfinite(x))
                throw std::invalid_argument("Product factor coordinate ancestry overflows.");
        }
        values.push_back(value);
    }
    return values;
}

bool prefix(std::span<const std::uint64_t> a, std::span<const std::uint64_t> b)
{
    return a.size() <= b.size() && std::ranges::equal(a, b.first(a.size()));
}

std::span<const char> source_edge(Reader& reader)
{
    const auto start = reader.offset;
    (void) reader.bytes_view();
    (void) reader.word();
    if (reader.word() > 1)
        throw std::invalid_argument("Product source edge has an invalid creator.");
    const auto path = count(reader);
    if (path > 48)
        throw std::invalid_argument("Product source edge exceeds the admitted depth.");
    for (std::size_t i = 0; i < path; ++i)
        if (reader.word() > 1)
            throw std::invalid_argument("Product source edge has an invalid edge half.");
    return reader.data.subspan(start, reader.offset - start);
}

void emit_side_support(std::span<const char> edge, std::size_t surface, bool exterior,
    const NativeAdmission& admission, void* context, SupportVisit emit)
{
    if (!emit) return;
    Reader cursor{edge};
    (void) cursor.bytes_view();
    (void) cursor.word();
    (void) cursor.word();
    const auto stem = edge.first(cursor.offset);
    auto storage = geometry_detail::native_constructor_storage(admission);
    Writer key;
    storage.owner->reserve(key.data, 2 * sizeof(std::uint64_t) + stem.size());
    key.word(0x5346505349444536ULL);
    key.data.insert(key.data.end(), stem.begin(), stem.end());
    key.word(0);
    emit(context, surface, key.data, exterior);
}

void emit_cap_support(std::uint64_t base_root, const AxialIntervalKey& axis, bool upper, bool exterior,
    const NativeAdmission& admission, void* context, SupportVisit emit)
{
    if (!emit) return;
    auto storage = geometry_detail::native_constructor_storage(admission);
    auto depth = axis.path.size();
    while (depth && axis.path[depth - 1] == std::uint64_t(upper)) --depth;
    const auto kind = depth ? LocalProductPlaneKey::Kind::IntervalMidpoint : LocalProductPlaneKey::Kind::RootPoint;
    AxialIntervalKey creator{};
    if (depth)
    {
        creator.root = axis.root;
        storage.owner->copy(creator.path, std::span(axis.path).first(depth - 1));
    }
    AdaptiveCellKey ancestry{0, ProductCellAncestry{{}, std::move(creator)}};
    Writer key;
    storage.owner->reserve(key.data, 5 * sizeof(std::uint64_t) + encoded_adaptive_key_size(ancestry));
    key.word(0x5346504341503631ULL);
    key.word(base_root);
    key.word(std::uint64_t(kind));
    key.word(depth ? 0 : axis.root + upper);
    adaptive_detail::NativeAdaptiveKeyCodec::write(key, ancestry);
    emit(context, upper ? 1 : 0, key.data, exterior);
}

bool physical_cap(const AxialIntervalKey& axis, bool upper, std::size_t root_points)
{
    return (upper ? axis.root + 2 == root_points : axis.root == 0) &&
        std::ranges::all_of(axis.path, [upper](auto side) { return side == std::uint64_t(upper); });
}

const ExtrudedRegion& source_region(const NativeFragmentRecord& fragment, const AdaptiveLeaf& leaf)
{
    const auto* region = std::get_if<ExtrudedRegion>(&fragment.region());
    if (!region || !region->geometry().has_construction() || leaf.native_cell >= region->layout().cells ||
        !fragment.allocation() || !fragment.allocation()->authentic_owner(fragment.allocation()) ||
        !fragment.allocation()->object_alive() || fragment.allocation()->observed_target_address() != &fragment)
        throw std::invalid_argument("Product source lacks an admitted constructed fragment and exact ordinal.");
    return *region;
}

AdaptiveCellMetrics product_routes(const ExtrudedRegion& region, std::size_t cell, NativeRoutes routes)
{
    const auto& topology = region.topology();
    const auto faces = topology.cell_faces(cell);
    if (faces.size() != routes.size() || faces.size() < 5)
        throw std::invalid_argument("Product source face count differs from its native route span.");
    for (std::size_t f = 0; f < faces.size(); ++f)
    {
        const auto face = faces[f];
        const auto owner = topology.owner_cell(face);
        if (owner != cell && topology.neighbor_cell(face) != cell)
            throw std::invalid_argument("Product source face has no owner-oriented native route.");
        // The native Product loop uses the same caps-then-sides order as
        // certified_cell_geometry and Kernel::surfaces.
        routes[f] = {f, face, owner == cell ? 1 : -1};
    }
    const auto& geometry = region.geometry();
    const auto center = geometry.cell_centroid(cell);
    return {geometry.cell_volume(cell), {center.x, center.y, center.z}};
}

template<class Factor> void verify_source_children(
    Reader& reader, const Factor& parent, std::size_t count, const NativeStorage& scratch)
{
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
    {
        auto decoded = adaptive_detail::NativeAdaptiveKeyCodec::decode(reader.bytes_view(), scratch);
        auto* product = std::get_if<ProductCellAncestry>(&decoded.ancestry);
        if (!product || decoded.region)
            throw std::invalid_argument("Product source has a foreign encoded child key.");
        auto& child = [&]() -> Factor&
        {
            if constexpr (std::is_same_v<Factor, BaseCellKey>)
            {
                if (product->axial != AxialIntervalKey{} || product->base.root_template != parent.root_template)
                    throw std::invalid_argument("Product source has a foreign encoded base child.");
                return product->base;
            }
            else
            {
                if (product->base != BaseCellKey{})
                    throw std::invalid_argument("Product source has a foreign encoded axial child.");
                return product->axial;
            }
        }();
        if (child.root != parent.root || child.path.size() != parent.path.size() + 1 ||
            !prefix(parent.path, child.path) || child.path.back() != ordinal)
            throw std::invalid_argument("Product source child differs from its immediate ancestry or order.");
        scratch.owner->reset_buffer(child.path);
    }
}

void verify_source_axis(const AdaptiveLeaf& leaf, const AxialIntervalKey& selected, const SourceRoot& root,
    const ProductConstruction::CellFactors& factors, const NativeStorage& scratch)
{
    Reader reader{*leaf.factor_recipe->axis};
    packet_codec::read_header(reader, packet_codec::Family::ProductAxis, uint8_t{7}, packet_codec::Record::Value);
    const auto values = source_values<1>(reader, scratch);
    const auto levels = count(reader);
    if (levels != selected.path.size() + 1)
        throw std::invalid_argument("Product source axis capsule has a foreign ancestry closure.");
    std::array<std::array<long double, 2>, 49> spans{};
    std::array<bool, 49> found{};
    std::vector<AdaptiveCellKey> keys;
    scratch.owner->reserve(keys, levels);
    for (std::size_t i = 0; i < levels; ++i)
    {
        keys.push_back(adaptive_detail::NativeAdaptiveKeyCodec::decode(reader.bytes_view(), scratch));
        const auto& key = keys.back();
        const auto* product = std::get_if<ProductCellAncestry>(&key.ancestry);
        if (!product || key.region || product->base != BaseCellKey{} || product->axial.root != selected.root ||
            !prefix(product->axial.path, selected.path))
            throw std::invalid_argument("Product source axis capsule contains a foreign key.");
        const auto depth = product->axial.path.size();
        const auto low = reader.word(), high = reader.word();
        if (found[depth] || low >= values.size() || high >= values.size() || !(values[low][0] < values[high][0]))
            throw std::invalid_argument("Product source axis capsule contains an invalid interval.");
        found[depth] = true;
        spans[depth] = {values[low][0], values[high][0]};
        const auto children = count(reader);
        if ((children && children != 2) || (depth < selected.path.size() && children != 2))
            throw std::invalid_argument("Product source axis capsule has incomplete child ancestry.");
        verify_source_children(reader, product->axial, children, scratch);
    }
    reader.done();
    std::array<long double, 2> expected{root.lower, root.upper};
    for (std::size_t depth = 0; depth < levels; ++depth)
    {
        if (!found[depth] || spans[depth] != expected)
            throw std::invalid_argument("Product axis factor differs from its original root and dyadic selection.");
        if (depth == selected.path.size()) break;
        const auto side = selected.path[depth];
        if (side > 1)
            throw std::invalid_argument("Product source axis has an invalid child half.");
        expected[1 - side] = (expected[0] + expected[1]) / 2;
    }
    if (factors.axial != expected)
        throw std::invalid_argument("Product fragment axial selection differs from its original factor.");
}

void verify_source_base(const AdaptiveLeaf& leaf, const BaseCellKey& selected, const SourceRoot& root,
    const ExtrudedRegion& region, const ProductConstruction::CellFactors& factors, NativeRoutes routes,
    const NativeStorage& scratch, void* context, SupportVisit emit)
{
    Reader reader{*leaf.factor_recipe->base};
    packet_codec::read_header(reader, packet_codec::Family::ProductBase, uint8_t{7}, packet_codec::Record::Value);
    const auto points = source_values<3>(reader, scratch);
    const auto levels = count(reader);
    if (levels != selected.path.size() + 1)
        throw std::invalid_argument("Product source base capsule has a foreign ancestry closure.");
    const auto& topology = region.topology();
    const auto cell = topology.indexer().cell_id(leaf.native_cell);
    const auto native_nodes = topology.base_cell_nodes(cell.ij);
    std::array<bool, 49> found{};
    std::vector<AdaptiveCellKey> keys;
    std::vector<std::span<const char>> edges;
    std::vector<std::size_t> vertices;
    scratch.owner->reserve(keys, levels);
    scratch.owner->reserve(edges, routes.size() - 2);
    scratch.owner->reserve(vertices, routes.size() - 2);
    for (std::size_t i = 0; i < levels; ++i)
    {
        keys.push_back(adaptive_detail::NativeAdaptiveKeyCodec::decode(reader.bytes_view(), scratch));
        const auto& key = keys.back();
        const auto* product = std::get_if<ProductCellAncestry>(&key.ancestry);
        if (!product || key.region || product->axial != AxialIntervalKey{} || product->base.root != selected.root ||
            product->base.root_template != selected.root_template || !prefix(product->base.path, selected.path))
            throw std::invalid_argument("Product source base capsule contains a foreign key.");
        const auto depth = product->base.path.size();
        Reader loop{words(reader)};
        const auto size = loop.data.size() / sizeof(std::uint64_t);
        const auto arity = size == 3 ? 2 : size == 4 ? 4 : size > 4 ? size - 2 : 0;
        const bool active = depth == selected.path.size();
        if (found[depth] || size < 3 ||
            (!depth && selected.root_template != (size == 3 ? 0u : size == 4 ? 1u : 2u)) ||
            (!active && selected.path[depth] >= arity) ||
            (active && (routes.size() != size + 2 || native_nodes.size() != size)))
            throw std::invalid_argument("Product source has inconsistent base factor ancestry or arity.");
        found[depth] = true;
        for (std::size_t j = 0; j < size; ++j)
        {
            const auto index = loop.word();
            if (index >= points.size() || points[index][2] != 0)
                throw std::invalid_argument("Product base vertex has an invalid source coordinate.");
            if (active)
            {
                const auto native = factors.base_selection[native_nodes[j]];
                if (points[index] != factors.base_points[native])
                    throw std::invalid_argument("Product fragment base loop differs from the original factor.");
                vertices.push_back(index);
            }
        }
        for (std::size_t side = 0; side < size; ++side)
        {
            const auto edge = source_edge(reader);
            if (active) edges.push_back(edge);
        }
        const auto reference = reader.word();
        const auto children = count(reader);
        if (reference >= size || (children && children != arity) || (!active && children != arity))
            throw std::invalid_argument("Product source base has an incomplete original child family.");
        verify_source_children(reader, product->base, children, scratch);
    }
    if (edges.size() != routes.size() - 2 || std::size_t(std::count(found.begin(), found.end(), true)) != levels)
        throw std::invalid_argument("Product source base did not provide its complete selected ancestry.");

    std::vector<unsigned char> verified;
    scratch.owner->reserve(verified, edges.size());
    verified.resize(edges.size());
    const auto edge_count = count(reader);
    for (std::size_t i = 0; i < edge_count; ++i)
    {
        const auto edge = source_edge(reader);
        const auto first = reader.word(), second = reader.word(), physical = reader.word();
        const auto boundary = text(reader);
        if (first >= points.size() || second >= points.size() || first == second || physical > 1)
            throw std::invalid_argument("Product source side has an invalid original edge record.");
        const auto found = std::ranges::find_if(edges, [&](auto other) { return std::ranges::equal(other, edge); });
        if (found == edges.end()) continue;
        const auto side = static_cast<std::size_t>(found - edges.begin());
        const auto a = vertices[side], b = vertices[(side + 1) % vertices.size()];
        if (verified[side] || !((first == a && second == b) || (first == b && second == a)))
            throw std::invalid_argument("Product side support ancestry differs from its original base loop.");
        verified[side] = 1;
        routes[side + 2].physical_boundary = bool(physical);
        emit_side_support(edge, side + 2, bool(physical), scratch.owner->admission(), context, emit);
        if (physical)
        {
            const auto id = topology.boundary_id(routes[side + 2].native_face);
            const auto name = root.boundary(boundary.empty() ? std::string_view("side") : boundary);
            if (id < 0 || topology.boundary_batch_name(id) != name)
                throw std::invalid_argument("Product native side loses its original physical boundary alias.");
        }
    }
    reader.done();
    if (std::ranges::find(verified, 0) != verified.end() ||
        !region.geometry().construction()->base_quality_above(cell.ij, root.minimum_area))
        throw std::invalid_argument("Product native base fails its original edge and quality admission.");
}
class Kernel final : public AdaptiveProviderKernel
{
public:
    void visit_native_publication_children(const geometry_detail::NativeConstructorOwnerSink&,
        adaptive_detail::NativeKernelAuthority::IdentityVisit,
        adaptive_detail::NativeKernelAuthority::GeometryVisit) const {}

    static constexpr bool native_source_publication_ready = true;
    bool supports_native_source(const AdaptiveLeaf& leaf) const noexcept
    {
        const auto& factors = leaf.factor_recipe;
        return leaf.provider == AdaptiveProviderKind::Product &&
            std::holds_alternative<ProductCellAncestry>(leaf.key.ancestry) && factors && factors->root &&
            factors->base && factors->axis && leaf.native_recipe &&
            leaf.native_recipe->kind == NativeFragmentDescriptor::Kind::ConstructedProduct &&
            !leaf.geometry && leaf.surfaces.empty() && !leaf.join_recipe;
    }

    std::size_t native_source_face_count(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment) const
    {
        const auto& region = source_region(fragment, leaf);
        return region.topology().cell_faces(leaf.native_cell).size();
    }

    AdaptiveCellMetrics validate_native_source(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
        const NativeAdmission& admission, NativeRoutes routes, void* context = nullptr, SupportVisit emit = nullptr) const
    {
        if (!supports_native_source(leaf))
            throw std::invalid_argument("Product source is not its original procedural leaf.");
        const auto& ancestry = std::get<ProductCellAncestry>(leaf.key.ancestry);
        const auto root = source_root(leaf, ancestry);
        const auto& region = source_region(fragment, leaf);
        const auto metrics = product_routes(region, leaf.native_cell, routes);
        const auto factors = ProductConstruction::cell_factors(*region.geometry().construction(), leaf.native_cell);
        {
            const auto scratch = geometry_detail::native_constructor_storage(admission);
            verify_source_axis(leaf, ancestry.axial, root, factors, scratch);
        }
        {
            const auto scratch = geometry_detail::native_constructor_storage(admission);
            verify_source_base(leaf, ancestry.base, root, region, factors, routes, scratch, context, emit);
        }
        for (bool upper : {false, true})
        {
            const auto exterior = physical_cap(ancestry.axial, upper, root.axial_points);
            routes[upper ? 1 : 0].physical_boundary = exterior;
            if (exterior)
            {
                const auto& topology = region.topology();
                const auto id = topology.boundary_id(routes[upper ? 1 : 0].native_face);
                if (id < 0 || topology.boundary_batch_name(id) != root.boundary(upper ? "zmax" : "zmin"))
                    throw std::invalid_argument("Product native cap loses its original physical boundary alias.");
            }
            emit_cap_support(ancestry.base.root, ancestry.axial, upper, exterior, admission, context, emit);
        }
        return metrics;
    }

    bool rebind_native_source(const AdaptiveLeaf& previous, const NativeFragmentRecord& previous_fragment,
        std::span<const adaptive_detail::NativeKernelSourceFace> previous_routes, const AdaptiveLeaf& current,
        const NativeFragmentRecord& current_fragment, NativeRoutes routes, const NativeAdmission&) const
    {
        if (!product_native_repack_candidate(previous, current)) return false;
        const auto& old_region = source_region(previous_fragment, previous);
        const auto& new_region = source_region(current_fragment, current);
        const auto& old_topology = old_region.topology();
        const auto& new_topology = new_region.topology();
        const auto old_cell = old_topology.indexer().cell_id(previous.native_cell);
        const auto new_cell = new_topology.indexer().cell_id(current.native_cell);
        const auto old_nodes = old_topology.base_cell_nodes(old_cell.ij);
        const auto new_nodes = new_topology.base_cell_nodes(new_cell.ij);
        const auto old_values = ProductConstruction::cell_factors(*old_region.geometry().construction(), previous.native_cell);
        const auto new_values = ProductConstruction::cell_factors(*new_region.geometry().construction(), current.native_cell);
        if (old_values.axial != new_values.axial || old_nodes.size() != new_nodes.size()) return false;
        for (std::size_t i = 0; i < old_nodes.size(); ++i)
            if (old_values.base_points[old_values.base_selection[old_nodes[i]]] !=
                new_values.base_points[new_values.base_selection[new_nodes[i]]])
                return false;

        const auto metrics = product_routes(new_region, current.native_cell, routes);
        const auto old_faces = old_topology.cell_faces(previous.native_cell);
        if (old_faces.size() != routes.size() || previous_routes.size() != routes.size()) return false;
        for (std::size_t i = 0; i < routes.size(); ++i)
        {
            const auto& prior = previous_routes[i];
            if (prior.surface != i || prior.native_face != old_faces[i]) return false;
            if (i >= 2)
            {
                const auto old_side = old_topology.indexer().face_id(old_faces[i]).ij;
                const auto new_side = new_topology.indexer().face_id(routes[i].native_face).ij;
                if (old_values.side_supports[old_side] != new_values.side_supports[new_side]) return false;
            }
            routes[i].physical_boundary = prior.physical_boundary;
            if (prior.physical_boundary)
            {
                const auto old_boundary = old_topology.boundary_id(old_faces[i]);
                const auto new_boundary = new_topology.boundary_id(routes[i].native_face);
                if (old_boundary < 0 || new_boundary < 0 ||
                    old_topology.boundary_batch_name(old_boundary) != new_topology.boundary_batch_name(new_boundary))
                    return false;
            }
        }
        const auto& old_geometry = old_region.geometry();
        const auto old_center = old_geometry.cell_centroid(previous.native_cell);
        const auto same = [](double a, double b) { return std::bit_cast<ID>(a) == std::bit_cast<ID>(b); };
        return same(old_geometry.cell_volume(previous.native_cell), metrics.volume) &&
            same(old_center.x, metrics.centroid[0]) && same(old_center.y, metrics.centroid[1]) &&
            same(old_center.z, metrics.centroid[2]);
    }

    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf&) const override
    {
        return {AdaptiveContactNormalization::PhysicalIdentity, false, false};
    }
    AdaptiveBytes descriptor() const override { return AdaptiveBytes{'P', 'R', 'O', 'D', '6'}; }
    AdaptiveSplit closure_split(const AdaptiveLeaf&, std::size_t axis) const override
    {
        if (axis >= 3)
            throw std::invalid_argument("Invalid product closure axis.");
        return axis == 2 ? AdaptiveSplit::Axial : AdaptiveSplit::Base;
    }
    AdaptiveTrace traces(const AdaptiveLeaf& leaf) const override
    {
        auto state = local_state(leaf);
        const auto& product = std::get<ProductCellAncestry>(leaf.key.ancestry);
        AdaptiveTrace result;
        result.key = leaf.key;
        auto ancestor = product.base;
        for (;;)
        {
            Writer w;
            w.word(0x5346504254524336ULL);
            root_identity(w, *leaf.factor_recipe->root);
            product_detail::key(w, ancestor);
            result.faces.push_back({std::move(w.data), unsigned(product.base.path.size()), AdaptiveSplit::Base, {},
                ancestor == product.base});
            if (ancestor.path.empty())
                break;
            ancestor.path.pop_back();
        }
        {
            Writer w;
            w.word(0x5346505a54524336ULL);
            root_identity(w, *leaf.factor_recipe->root);
            w.word(product.axial.root);
            result.faces.push_back(
                {std::move(w.data), unsigned(product.axial.path.size()), AdaptiveSplit::Axial, {}, true});
        }
        const auto shape = geometry(leaf);
        const auto& base = state.pool.base.at(product.base);
        for (std::size_t i = 0; i < base.edges.size(); ++i)
        {
            auto stem = base.edges[i];
            const auto depth = stem.path.size();
            stem.path.clear();
            Writer w;
            w.word(0x5346504554524336ULL);
            root_identity(w, *leaf.factor_recipe->root);
            product_detail::key(w, stem);
            result.faces.push_back(
                {std::move(w.data), unsigned(depth), AdaptiveSplit::Base, shape->face_geometry(i + 2), true});
        }
        return result;
    }
    void validate(const AdaptiveLeaf& leaf) const override
    {
        if (leaf.provider != AdaptiveProviderKind::Product ||
            !std::holds_alternative<ProductCellAncestry>(leaf.key.ancestry))
            throw std::invalid_argument("Product provider/typed ancestry binding differs.");
        const auto state = local_state(leaf);
        const auto& ancestry = std::get<ProductCellAncestry>(leaf.key.ancestry);
        if (ancestry.base.path.size() > state.pool.grid->max_depth ||
            ancestry.axial.path.size() > state.pool.grid->max_depth ||
            std::any_of(ancestry.axial.path.begin(), ancestry.axial.path.end(), [](auto child) { return child > 1; }))
            throw std::invalid_argument("Product factor ancestry exceeds admitted depth or axial child range.");
        auto parent = ancestry.base;
        parent.path.clear();
        for (auto child : ancestry.base.path)
        {
            const auto vertices = state.pool.base.at(parent).vertices.size();
            const auto children = vertices == 3 ? 2 : vertices == 4 ? 4 : vertices - 2;
            if (child >= children)
                throw std::invalid_argument("Product base child exceeds its actual parent template arity.");
            parent.path.push_back(child);
        }
        AdaptiveProviderKernel::validate(leaf);
    }
    bool supports_leaf_local_repack() const noexcept override { return true; }
    std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const override
    {
        if (leaves.empty())
            return {};
        LocalData state;
        std::map<BaseCellKey, ZKeys> columns;
        OriginalProductLeaves original;
        for (const auto& leaf : leaves)
        {
            auto one = local_state(leaf);
            merge_pool(state.pool, one.pool);
            const auto& p = std::get<ProductCellAncestry>(leaf.key.ancestry);
            columns[p.base].push_back(p.axial);
            if (!original.emplace(leaf.key, &leaf).second || leaf.domain != leaves[0].domain)
                throw std::invalid_argument("Duplicate product leaf or mixed connected domain.");
        }
        for (auto& [base, z] : columns)
        {
            std::sort(z.begin(), z.end());
            if (std::adjacent_find(z.begin(), z.end()) != z.end())
                throw std::invalid_argument("Repeated product factor selection.");
            state.columns.emplace(base, std::make_shared<const ZKeys>(std::move(z)));
        }
        return export_state(state, leaves[0].domain, {}, &original);
    }
    std::vector<AdaptiveProductFamilySelector> ancestry_family_selectors(const AdaptiveLeaf& leaf) const override
    {
        const auto original = std::get<ProductCellAncestry>(leaf.key.ancestry);
        std::vector<AdaptiveProductFamilySelector> families;
        const auto emit = [&](const ProductCellAncestry& parent, AdaptiveSplit split)
        { families.push_back({{leaf.key.region, parent}, split}); };
        auto p = original;
        while (!p.base.path.empty())
        {
            p.base.path.pop_back();
            emit(p, AdaptiveSplit::Base);
        }
        p = original;
        while (!p.axial.path.empty())
        {
            p.axial.path.pop_back();
            emit(p, AdaptiveSplit::Axial);
        }
        if (!original.base.path.empty() && !original.axial.path.empty())
        {
            p = original;
            p.base.path.pop_back();
            p.axial.path.pop_back();
            emit(p, AdaptiveSplit::Isotropic);
        }
        return families;
    }
    bool product_family_proof_revision_invariant(const AdaptiveLeaf*) const override { return true; }

    AdaptiveFamily ancestry_family(
        const AdaptiveLeaf& leaf, const AdaptiveProductFamilySelector& selector) const override
    {
        const auto available = ancestry_family_selectors(leaf);
        if (std::ranges::find(available, selector) == available.end())
            throw std::invalid_argument("Unknown exact product family selector.");
        return make_ancestry_family(leaf, selector);
    }

    std::vector<AdaptiveFamily> ancestry_families(const AdaptiveLeaf& leaf) const override
    {
        std::vector<AdaptiveFamily> families;
        for (const auto& selector : ancestry_family_selectors(leaf))
            families.push_back(make_ancestry_family(leaf, selector));
        return families;
    }

private:
    AdaptiveFamily make_ancestry_family(const AdaptiveLeaf& leaf, const AdaptiveProductFamilySelector& selector) const
    {
        const auto& parent_key = std::get<ProductCellAncestry>(selector.parent.ancestry);
        auto state = local_state(leaf);
        state.columns.clear();
        state.columns.emplace(parent_key.base, std::make_shared<const ZKeys>(ZKeys{parent_key.axial}));
        auto records = export_state(state, leaf.domain, leaf.factor_recipe->root);
        if (records.size() != 1)
            throw std::logic_error("Product parent factor did not produce one cell.");
        auto parent = std::move(records.front());
        parent.revision = leaf.revision;
        auto result = split(parent, selector.split);
        result.family.anchor = encode_adaptive_key(leaf.key);
        return std::move(result.family);
    }

    static ExtrudedRegion native_region(const AdaptiveLeaf& leaf)
    {
        if (!leaf.native_recipe)
            throw std::invalid_argument("Missing native product descriptor.");
        NativeFragmentCache cache;
        auto region = materialize_native_fragment(*leaf.native_recipe, cache);
        auto* product = std::get_if<ExtrudedRegion>(&region);
        if (!product || leaf.native_cell >= product->layout().cells)
            throw std::invalid_argument("Product native descriptor/ordinal invalid.");
        return std::move(*product);
    }

    static PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> certified_cell(
        const AdaptiveLeaf& leaf, const ExtrudedRegion& native)
    {
        auto actual = native.geometry().certified_cell_geometry(leaf.native_cell);
        auto state = local_state(leaf);
        build_local_runs(state);
        if (!actual || state.runs.size() != 1 ||
            !actual->equivalent(*state.runs[0].region.geometry().certified_cell_geometry(0)))
            throw std::invalid_argument("Native product recipe differs from exact factor ancestry.");
        return actual;
    }

public:
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        const auto native = native_region(leaf);
        return certified_cell(leaf, native);
    }
    AdaptiveCellMetrics metrics(const AdaptiveLeaf& leaf) const override
    {
        const auto native = native_region(leaf);
        (void) certified_cell(leaf, native);
        const auto& geometry = native.geometry();
        const auto center = geometry.cell_centroid(leaf.native_cell);
        return {geometry.cell_volume(leaf.native_cell), {center.x, center.y, center.z}};
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override
    {
        const auto state = local_state(leaf);
        const auto native = native_region(leaf);
        const auto shape = certified_cell(leaf, native);
        const auto& key = std::get<ProductCellAncestry>(leaf.key.ancestry);
        const auto& base_node = state.pool.base.at(key.base);
        std::vector<AdaptiveSurface> result;
        // Native construction emits side faces in base-loop order, followed by
        // lower and upper caps; authoritative native atomic shape is checked.
        for (std::size_t i = 0; i < base_node.edges.size(); ++i)
        {
            auto edge = base_node.edges[i];
            auto stem = edge;
            stem.path.clear();
            const auto& node = state.pool.edges.at(edge);
            AdaptiveSurface s;
            Writer w;
            w.word(0x5346505349444536ULL);
            key_write(w, stem);
            s.key = std::move(w.data);
            s.orientation = base_node.vertices[i] == node.a ? 1 : -1;
            s.exterior = node.physical;
            if (s.exterior)
            {
                auto alias = state.pool.grid->boundary_aliases.find(node.boundary);
                s.boundary = alias == state.pool.grid->boundary_aliases.end() ? node.boundary : alias->second;
            }
            result.push_back(std::move(s));
        }
        for (bool upper : {false, true})
        {
            auto plane = plane_key(key.axial, upper);
            AdaptiveSurface s;
            Writer w;
            w.word(0x5346504341503631ULL);
            w.word(key.base.root);
            w.word(std::uint64_t(plane.kind));
            w.word(plane.root_point);
            product_detail::key(w, plane.creator);
            s.key = std::move(w.data);
            s.orientation = upper ? 1 : -1;
            s.exterior = plane.kind == LocalProductPlaneKey::Kind::RootPoint &&
                         (upper ? plane.root_point + 1 == state.pool.grid->z_edges.size() : plane.root_point == 0);
            if (s.exterior)
            {
                const std::string name = upper ? "zmax" : "zmin";
                auto alias = state.pool.grid->boundary_aliases.find(name);
                s.boundary = alias == state.pool.grid->boundary_aliases.end() ? name : alias->second;
            }
            result.push_back(std::move(s));
        }
        if (result.size() != shape->num_faces())
            throw std::invalid_argument("Product authoritative face mapping differs.");
        std::rotate(result.begin(), result.end() - 2, result.end());
        std::set<std::uint64_t> matched;
        for (std::size_t f = 0; f < result.size(); ++f)
        {
            auto& surface = result[f];
            surface.geometry = shape->face_geometry(f);
            surface.require_native_whole_tile = true;
            surface.native_whole_geometry = surface.geometry;
            for (auto face : native.topology().cell_faces(leaf.native_cell))
            {
                const auto certificate = native.geometry().certified_face_geometry(face);
                if (!certificate)
                    continue;
                // Identical finite vertices also prove equality of the two
                // convex faces when separately constructed support differs.
                if (!std::ranges::is_permutation(surface.geometry->points(), certificate->points()) &&
                    !surface.geometry->equivalent(*certificate))
                    continue;
                if (!matched.insert(face).second)
                    throw std::invalid_argument("Repeated procedural native face binding.");
                surface.native_face = face;
                break;
            }
            if (surface.native_face == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Native product surface lacks native face binding.");
        }
        return result;
    }
    static void key_write(Writer& w, const BaseEdgeKey& edge) { product_detail::key(w, edge); }
    AdaptiveSplitResult split(const AdaptiveLeaf& parent, AdaptiveSplit split) const override
    {
        auto state = local_state(parent);
        const auto p = std::get<ProductCellAncestry>(parent.key.ancestry);
        state.columns.clear();
        const auto bases =
            split == AdaptiveSplit::Axial ? std::vector<BaseCellKey>{p.base} : local_base_children(state.pool, p.base);
        const auto z = split == AdaptiveSplit::Base ? ZKeys{p.axial} : local_z_children(state.pool, p.axial);
        for (const auto& base : bases)
            state.columns.emplace(base, std::make_shared<const ZKeys>(z));
        AdaptiveSplitResult result;
        result.children = export_state(state, parent.domain, parent.factor_recipe->root);
        result.family.key = adaptive_family_key(parent.key, split);
        result.family.parent = parent;
        result.family.split = split;
        result.family.revision = parent.revision;
        if (parent.revision == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Product leaf revision exhausted.");
        for (auto& leaf : result.children)
        {
            leaf.revision = parent.revision + 1;
            leaf.parent_family = result.family.key;
            leaf.work = parent.work;
            result.family.children.push_back(leaf.key);
        }
        result.family.anchor =
            encode_adaptive_key(*std::min_element(result.family.children.begin(), result.family.children.end()));
        return result;
    }
};
} // namespace product_detail
std::shared_ptr<const AdaptiveProviderKernel> product_adaptive_kernel()
{
    return adaptive_detail::NativeKernelAuthority::create<product_detail::Kernel>();
}
bool product_native_repack_candidate(const AdaptiveLeaf& previous, const AdaptiveLeaf& current) noexcept
{
    const auto same = [](const auto& a, const auto& b)
    { return a && a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
    const auto& a = previous.factor_recipe;
    const auto& b = current.factor_recipe;
    return previous.provider == AdaptiveProviderKind::Product && current.provider == previous.provider &&
        previous.key == current.key && previous.revision == current.revision && previous.material == current.material &&
        previous.domain == current.domain && previous.parent_family == current.parent_family && !previous.geometry &&
        !current.geometry && previous.surfaces.empty() && current.surfaces.empty() && !previous.join_recipe &&
        !current.join_recipe && same(a, b) && same(a->root, b->root) && same(a->base, b->base) && same(a->axis, b->axis) &&
        previous.native_recipe && current.native_recipe &&
        previous.native_recipe->kind == NativeFragmentDescriptor::Kind::ConstructedProduct &&
        current.native_recipe->kind == NativeFragmentDescriptor::Kind::ConstructedProduct;
}
AdaptiveLeaf product_support_root(const AdaptiveLeaf& leaf)
{
    auto state=product_detail::local_state(leaf);auto root=std::get<ProductCellAncestry>(leaf.key.ancestry);root.base.path.clear();root.axial.path.clear();
    if(root==std::get<ProductCellAncestry>(leaf.key.ancestry))return leaf;
    state.columns.clear();state.columns.emplace(root.base,std::make_shared<const ZKeys>(ZKeys{root.axial}));auto records=product_detail::export_state(state,leaf.domain);if(records.size()!=1)throw std::logic_error("Original product support root was not one cell.");return std::move(records[0]);
}
AdaptiveBytes LocalProductAMR::root_descriptor() const
{const auto text=description(data_->pool,true);return {text.begin(),text.end()};}
AdaptiveBytes ExtrudedAMR::root_descriptor() const
{const auto text=description(*data_,true);return {text.begin(),text.end()};}
AdaptiveBootstrap LocalProductAMR::detach_owned(
    std::span<const ProductCellAncestry> selected, AdaptiveBytes domain) const
{
    LocalData state;
    state.pool = data_->pool;
    std::map<BaseCellKey, ZKeys> columns;
    for (const auto& cell : selected)
    {
        auto it = data_->columns.find(cell.base);
        if (it == data_->columns.end() || !std::binary_search(it->second->begin(), it->second->end(), cell.axial))
            throw std::invalid_argument("Detached product selection is not active.");
        columns[cell.base].push_back(cell.axial);
    }
    for (auto& [base, z] : columns)
    {
        std::sort(z.begin(), z.end());
        if (std::adjacent_find(z.begin(), z.end()) != z.end())
            throw std::invalid_argument("Repeated detached product selection.");
        state.columns.emplace(base, std::make_shared<const ZKeys>(std::move(z)));
    }
    AdaptiveBootstrap result;
    result.kernel = product_adaptive_kernel();
    if (!columns.empty())
        result.owned = product_detail::export_state(state, domain);
    return result;
}
AdaptiveBootstrap ExtrudedAMR::detach_owned(std::span<const std::uint64_t> selected, AdaptiveBytes domain) const
{
    LocalData state;
    state.pool = *data_;
    std::map<BaseCellKey, ZKeys> columns;
    for (auto id : selected)
    {
        const auto cell = std::get<ProductCellAncestry>(cell_key(id).ancestry);
        columns[cell.base].push_back(cell.axial);
    }
    for (auto& [base, z] : columns)
    {
        std::sort(z.begin(), z.end());
        if (std::adjacent_find(z.begin(), z.end()) != z.end())
            throw std::invalid_argument("Repeated detached product selection.");
        state.columns.emplace(base, std::make_shared<const ZKeys>(std::move(z)));
    }
    AdaptiveBootstrap result;
    result.kernel = product_adaptive_kernel();
    if (!columns.empty())
        result.owned = product_detail::export_state(state, domain);
    for (auto id : selected)
        result.source_cells.push_back({cell_key(id), id});
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
