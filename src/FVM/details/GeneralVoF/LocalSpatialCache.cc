/** @file LocalSpatialCache.cc
 * @brief Retained geometry bounds, immutable indexes and synchronized cache admission.
 */
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include <numeric>
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
void Index::visit(std::size_t i, const Box& box, std::vector<std::size_t>& result, SimpleFluid::geometry::swept_volume::Work& work,
    GeneralVoFTransportDiagnostics& diagnostics) const
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
std::vector<std::size_t> Index::query(const Box& box, SimpleFluid::geometry::swept_volume::Work& work, GeneralVoFTransportDiagnostics& diagnostics) const
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
std::shared_ptr<const LocalSpatialCache> retained_spatial_cache(const GeneralVoF::Data& data, SimpleFluid::geometry::swept_volume::Work& work,
    GeneralVoFTransportDiagnostics& diagnostics, SpatialCacheAdmission admit_retained)
{
    using namespace local_spatial_detail;
    std::lock_guard lock(data.topology->spatial_cache_mutex);
    const auto& previous = data.topology->spatial_cache;
    bool matches = previous && previous->observation == data.observation &&
                   previous->supplied_observation == data.supplied_observation &&
                   previous->cells.size() == data.cells.size();
    if (matches)
    {
        auto cell = data.cells.begin();
        for (const auto& identity : previous->cells)
        {
            work.test();
            if (identity.key != cell->first || identity.gid != cell->second.gid ||
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
            return a->face_gid < b->face_gid;
        });
    for (const auto* tile : frontier_tiles)
    {
        if (!tile->second_image_identity.empty())
            throw std::invalid_argument("Local VoF retained frontier has an unsupported periodic image.");
        Box box;
        for (const auto& p : tile->geometry->points())
        {
            work.test();
            box.add(p);
        }
        work.push(cache->frontier_boxes, box);
        work.push(cache->frontiers, Frontier{tile->geometry, data.cells.contains(tile->first)});
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
