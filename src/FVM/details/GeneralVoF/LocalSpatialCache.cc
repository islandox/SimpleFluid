/** @file LocalSpatialCache.cc
 * @brief Retained geometry bounds, immutable indexes and synchronized cache admission.
 */
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <numeric>
#include <sstream>
namespace SimpleFluid::VoF::general_vof_detail
{
namespace local_spatial_detail
{
void Box::add(const Wide& p)
{
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(p[axis]))
            throw std::invalid_argument("Local VoF index coordinate is nonfinite.");
        low[axis] = std::min(low[axis], std::nextafter(p[axis], -infinity));
        high[axis] = std::max(high[axis], std::nextafter(p[axis], infinity));
    }
}
void Box::add(const Box& b)
{
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        low[axis] = std::min(low[axis], b.low[axis]);
        high[axis] = std::max(high[axis], b.high[axis]);
    }
}
bool Box::overlaps(const Box& b) const
{
    for (unsigned axis = 0; axis < 3; ++axis)
        if (high[axis] < b.low[axis] || b.high[axis] < low[axis])
            return false;
    return true;
}
std::size_t Index::build(std::size_t begin, std::size_t end, SimpleFluid::geometry::swept_volume::Work& work)
{
    const auto here = nodes_.size();
    nodes_.push_back({});
    Box box;
    for (auto i = begin; i < end; ++i)
    {
        work.test();
        box.add(boxes_[order_[i]]);
    }
    nodes_[here] = {box, begin, end, 0, 0};
    if (end - begin > 4)
    {
        unsigned axis = 0;
        for (unsigned a = 1; a < 3; ++a)
            if (box.high[a] - box.low[a] > box.high[axis] - box.low[axis])
                axis = a;
        const auto middle = begin + (end - begin) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
            [&](auto a, auto b)
            {
                work.test();
                const auto x = boxes_[a].low[axis] / 2 + boxes_[a].high[axis] / 2;
                const auto y = boxes_[b].low[axis] / 2 + boxes_[b].high[axis] / 2;
                return x != y ? x < y : a < b;
            });
        const auto left = build(begin, middle, work), right = build(middle, end, work);
        nodes_[here].left = left;
        nodes_[here].right = right;
    }
    return here;
}
void Index::visit(std::size_t i, const Box& box, std::vector<std::size_t>& result,
    SimpleFluid::geometry::swept_volume::Work& work, GeneralVoFTransportDiagnostics& diagnostics) const
{
    work.test();
    ++diagnostics.index_node_visits;
    const auto& node = nodes_[i];
    if (!node.box.overlaps(box))
        return;
    if (node.left)
    {
        visit(node.left, box, result, work, diagnostics);
        visit(node.right, box, result, work, diagnostics);
    }
    else
        for (auto j = node.begin; j < node.end; ++j)
        {
            work.test();
            if (boxes_[order_[j]].overlaps(box))
                work.push(result, order_[j]);
        }
}
void Index::initialize(SimpleFluid::geometry::swept_volume::Work& work)
{
    const auto& boxes = boxes_;
    if (boxes.size() > std::numeric_limits<std::size_t>::max() / (2 * sizeof(Node) + sizeof(std::size_t)))
        throw std::invalid_argument("Local VoF index capacity overflows.");
    work.charge(boxes.size() * (2 * sizeof(Node) + sizeof(std::size_t)));
    order_.resize(boxes.size());
    std::iota(order_.begin(), order_.end(), 0);
    nodes_.reserve(2 * boxes.size());
    if (!boxes.empty())
        build(0, boxes.size(), work);
}
std::vector<std::size_t> Index::query(
    const Box& box, SimpleFluid::geometry::swept_volume::Work& work, GeneralVoFTransportDiagnostics& diagnostics) const
{
    ++diagnostics.index_queries;
    std::vector<std::size_t> result;
    if (!nodes_.empty())
        visit(0, box, result, work, diagnostics);
    std::sort(result.begin(), result.end(),
        [&](auto a, auto b)
        {
            work.test();
            return a < b;
        });
    return result;
}
} // namespace local_spatial_detail
namespace
{
using namespace local_spatial_detail;
using Geometry = PLIC::ConvexCellGeometry<mode>;
std::string identity_text(std::span<const char> bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : bytes)
    {
        const auto value = static_cast<unsigned char>(byte);
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result.empty() ? "empty" : result;
}
[[noreturn]] void reach_failure(const Frontier& frontier, const char* reason, std::uint64_t source,
    const char* role = "unclassified", std::size_t point = 0, const char* classification = "NotEvaluated")
{
    std::ostringstream message;
    // Collective failure records retain a bounded prefix. Put the recovery
    // decision and offending point before the longer lossless identity packets.
    message << "Local VoF retained reach: " << reason << "; sweep_face=" << source
            << "; frontier_face=" << frontier.face_gid << "; point_role=" << role << "; point_index=" << point
            << "; classifier=" << classification
            << "; kind=" << (frontier.kind == Frontier::Kind::PhysicalBoundary ? "PhysicalBoundary" : "RetainedHalo")
            << "; retained_owner=" << frontier.retained_owner << "; outside_owner=" << frontier.outside_owner
            << "; boundary=" << frontier.boundary << "; retained_gid=" << frontier.retained_gid << "; outside_key="
            << (frontier.outside_key ? identity_text(AMR::encode_adaptive_key(*frontier.outside_key)) : "none")
            << "; retained_key=" << identity_text(AMR::encode_adaptive_key(frontier.retained_key))
            << "; retained_side=" << (frontier.first ? "first" : "second")
            << "; tile=" << identity_text(frontier.tile_identity)
            << ". This conservative infinite supporting-plane screen is a sufficient containment proof; "
               "it does not prove intersection with the finite frontier tile.";
    throw std::invalid_argument(message.str());
}
Frontier make_frontier(
    const GeneralVoF::Data& data, const AMR::AdaptiveTile& tile, SimpleFluid::geometry::swept_volume::Work& work)
{
    if (!tile.geometry || tile.geometry->points().size() < 3)
        throw std::invalid_argument("Local VoF retained frontier has no finite polygon.");
    if (tile.exterior == bool(tile.second))
        throw std::invalid_argument("Local VoF retained frontier boundary kind is inconsistent.");
    Frontier frontier;
    frontier.first = data.cells.contains(tile.first);
    frontier.retained_key = frontier.first ? tile.first : *tile.second;
    const auto& cell = data.cells.at(frontier.retained_key);
    frontier.retained_gid = cell.gid;
    frontier.retained_owner = cell.owner;
    if (cell.key != frontier.retained_key || cell.owner < 0 ||
        cell.owner != (frontier.first ? tile.first_owner : tile.second_owner))
        throw std::invalid_argument("Local VoF retained frontier cell identity or owner differs from tile incidence.");
    frontier.face_gid = tile.face_gid;
    frontier.tile_identity = tile.identity;
    frontier.boundary = tile.boundary;
    frontier.kind = tile.exterior ? Frontier::Kind::PhysicalBoundary : Frontier::Kind::RetainedHalo;
    if (!tile.exterior)
    {
        frontier.outside_key = frontier.first ? *tile.second : tile.first;
        frontier.outside_owner = frontier.first ? tile.second_owner : tile.first_owner;
        if (frontier.outside_owner < 0)
            throw std::invalid_argument("Local VoF retained halo frontier lacks its outside owner.");
    }
    frontier.geometry = tile.geometry;
    work.charge(sizeof(Geometry) + 2 * sizeof(void*));
    frontier.retained_geometry = Geometry::from_dynamic(cell.finite_geometry);
    frontier.support_identity = PLIC::planar_support<mode>(*tile.geometry);
    bool matched = false;
    for (std::size_t face = 0; face < frontier.retained_geometry->num_faces(); ++face)
    {
        work.test();
        const auto support = PLIC::planar_support<mode>(*frontier.retained_geometry->face_geometry(face));
        work.charge(support.capacity());
        if (support != frontier.support_identity)
            continue;
        matched = true;
        frontier.retained_face = face;
        // Several coplanar shell facets may retain the same original support.
        // Any such original cell constraint supplies the same half-space.
        break;
    }
    if (!matched)
        throw std::invalid_argument("Local VoF retained frontier lacks its original immutable cell support.");
    frontier.original_boundary.emplace(PLIC::FaceConstraint<mode>{frontier.retained_geometry, frontier.retained_face});
    work.charge(frontier.original_boundary->storage_bytes() - sizeof(PLIC::OrdinaryBoundary));
    work.charge(
        frontier.tile_identity.capacity() + frontier.support_identity.capacity() + frontier.boundary.capacity());
    work.charge(AMR::encode_adaptive_key(frontier.retained_key).size());
    if (frontier.outside_key)
        work.charge(AMR::encode_adaptive_key(*frontier.outside_key).size());
    return frontier;
}
} // namespace
void prove_retained(const GeneralVoF::Data& data, const LocalSpatialCache& cache,
    const local_spatial_detail::Frontier& frontier,
    const SimpleFluid::geometry::swept_volume::Sweep<ArithmeticMode::Ordinary>& recipe,
    SimpleFluid::geometry::swept_volume::Work& work)
{
    using namespace local_spatial_detail;
    work.test();
    if (cache.observation != data.observation || cache.supplied_observation != data.supplied_observation ||
        cache.topology_identity != data.topology.get())
        reach_failure(frontier, "stale source geometry/topology/partition observation", recipe.face);
    const auto cell = data.cells.find(frontier.retained_key);
    if (cell == data.cells.end() || cell->second.key != frontier.retained_key ||
        cell->second.gid != frontier.retained_gid || cell->second.owner != frontier.retained_owner ||
        !frontier.retained_geometry || cell->second.finite_geometry != frontier.retained_geometry->dynamic_geometry() ||
        !frontier.original_boundary || (frontier.outside_key && data.cells.contains(*frontier.outside_key)))
        reach_failure(frontier, "original support lacks current retained cell/owner authority", recipe.face);
    const auto check = [&](const PLIC::Point<mode>& point, const char* role, std::size_t index)
    {
        work.test();
        switch (frontier.original_boundary->classify(point))
        {
            case PLIC::OrdinaryBoundaryClass::Inside:
            case PLIC::OrdinaryBoundaryClass::ConstructionContact:
            case PLIC::OrdinaryBoundaryClass::CoordinateContact:
                return;
            case PLIC::OrdinaryBoundaryClass::Outside:
                reach_failure(frontier,
                    frontier.kind == Frontier::Kind::PhysicalBoundary ? "possible physical wall escape"
                                                                      : "missing donor beyond retained halo",
                    recipe.face, role, index, "Outside");
            case PLIC::OrdinaryBoundaryClass::Unresolved:
                reach_failure(frontier, "boundary arithmetic uncertainty", recipe.face, role, index, "Unresolved");
        }
        reach_failure(frontier, "unknown boundary classification", recipe.face, role, index, "Unresolved");
    };
    for (std::size_t i = 0; i < recipe.current.size(); ++i)
        check(recipe.current[i], "current-node", i);
    for (std::size_t i = 0; i < recipe.departure.size(); ++i)
        check(recipe.departure[i], "departure-node", i);
    check(recipe.center, "current-center", 0);
    check(recipe.backcenter, "corrected-departure-center", 0);
}
std::shared_ptr<const LocalSpatialCache> retained_spatial_cache(const GeneralVoF::Data& data,
    SimpleFluid::geometry::swept_volume::Work& work, GeneralVoFTransportDiagnostics& diagnostics,
    SpatialCacheAdmission admit_retained)
{
    using namespace local_spatial_detail;
    std::lock_guard lock(data.topology->spatial_cache_mutex);
    const auto& previous = data.topology->spatial_cache;
    bool matches =
        previous && previous->topology_identity == data.topology.get() && previous->observation == data.observation &&
        previous->supplied_observation == data.supplied_observation && previous->cells.size() == data.cells.size();
    if (matches)
    {
        auto cell = data.cells.begin();
        for (const auto& identity : previous->cells)
        {
            work.test();
            if (identity.key != cell->first || identity.key != cell->second.key || identity.gid != cell->second.gid ||
                identity.owner != cell->second.owner || identity.geometry != cell->second.finite_geometry)
            {
                matches = false;
                break;
            }
            ++cell;
        }
    }
    if (matches)
    {
        if (previous->cells.size() > work.limits.maximum_atoms ||
            previous->frontiers.size() > work.limits.maximum_atoms)
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF cached atom capacity is exhausted.");
        work.charge(previous->bytes);
        ++diagnostics.spatial_index_reuses;
        diagnostics.spatial_index_bytes = previous->bytes;
        return previous;
    }
    const auto before = work.bytes;
    work.charge(sizeof(LocalSpatialCache) + 2 * sizeof(void*));
    auto cache = std::make_shared<LocalSpatialCache>();
    cache->observation = data.observation;
    cache->supplied_observation = data.supplied_observation;
    cache->topology_identity = data.topology.get();
    // Map order is exactly the former key/GID sort order: keys are unique.
    for (const auto& [key, cell] : data.cells)
    {
        work.test();
        if (!cell.finite_geometry || cell.finite_geometry->arithmetic_mode() != mode)
            throw std::invalid_argument("Local VoF retained native geometry is missing or has another mode.");
        work.charge(AMR::encode_adaptive_key(key).size());
        work.push(cache->cells, LocalSpatialCache::CellIdentity{key, cell.gid, cell.owner, cell.finite_geometry});
        Box box;
        for (const auto& p : cell.finite_geometry->points())
        {
            work.test();
            box.add(p.value);
        }
        work.push(cache->cell_boxes, box);
    }
    if (!data.topology->order.empty() && cache->cells.empty())
        throw std::invalid_argument("Local VoF owned cells have no retained native geometry.");
    std::vector<const AMR::AdaptiveTile*> frontier_tiles;
    for (const auto& tile : data.topology->frontier_tiles)
    {
        work.test();
        const bool first = data.cells.contains(tile.first), second = tile.second && data.cells.contains(*tile.second);
        if (first != second)
            work.push(frontier_tiles, &tile);
    }
    std::sort(frontier_tiles.begin(), frontier_tiles.end(),
        [&](auto a, auto b)
        {
            work.test();
            return std::tie(a->face_gid, a->identity) < std::tie(b->face_gid, b->identity);
        });
    for (const auto* tile : frontier_tiles)
    {
        if (!tile->second_image_identity.empty())
            throw std::invalid_argument("Local VoF retained frontier has an unsupported periodic image.");
        auto frontier = make_frontier(data, *tile, work);
        Box box;
        for (const auto& p : tile->geometry->points())
        {
            work.test();
            box.add(p);
        }
        work.push(cache->frontier_boxes, box);
        work.push(cache->frontiers, std::move(frontier));
    }
    cache->donor_index.initialize(work);
    cache->frontier_index.initialize(work);
    // Work charges capacities conservatively, including the temporary frontier
    // ordering buffer. Preserve this admission footprint for later consumers.
    cache->bytes = work.bytes - before;
    // Admit the combined source and proposed retained cache before changing the
    // accepted topology slot. A rejected trial cannot poison later retries.
    if (admit_retained)
        admit_retained(data, cache->bytes);
    ++diagnostics.spatial_index_builds;
    diagnostics.spatial_index_bytes = cache->bytes;
    data.topology->spatial_cache = cache;
    return cache;
}
} // namespace SimpleFluid::VoF::general_vof_detail
