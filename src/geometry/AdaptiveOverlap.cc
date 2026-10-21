/** @file AdaptiveOverlap.cc
 * @brief Sparse ancestry selection with exact represented-cell coverage.
 */
#include "geometry/AdaptiveOverlap.hh"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
namespace
{
bool descendant(const AdaptiveCellKey& parent, const AdaptiveCellKey& child)
{
    if (parent.region != child.region || parent.ancestry.index() != child.ancestry.index())
        return false;
    if (const auto* p = std::get_if<CartesianCellAncestry>(&parent.ancestry))
    {
        const auto& c = std::get<CartesianCellAncestry>(child.ancestry);
        if (c.level <= p->level || c.level - p->level >= 64)
            return false;
        for (std::size_t a = 0; a < 3; ++a)
            if ((c.index[a] >> (c.level - p->level)) != p->index[a])
                return false;
        return true;
    }
    const auto& p = std::get<ConvexCellAncestry>(parent.ancestry);
    const auto& c = std::get<ConvexCellAncestry>(child.ancestry);
    return p.root == c.root && p.path.size() < c.path.size() &&
           std::equal(p.path.begin(), p.path.end(), c.path.begin());
}

std::map<std::uint64_t, const AdaptiveCellBinding*> index(const std::vector<AdaptiveCellBinding>& cells)
{
    std::map<std::uint64_t, const AdaptiveCellBinding*> result;
    std::set<AdaptiveCellKey> keys;
    for (const auto& cell : cells)
        if (!cell.geometry || !result.emplace(cell.gid, &cell).second || !keys.insert(cell.key).second)
            throw std::invalid_argument("Adaptive overlap catalog has missing geometry or duplicate GIDs/ancestry.");
    // Active cells are leaves, even when a supplied geometric partition looks
    // plausible. Walk only each key's bounded ancestry, never geometric pairs.
    for (const auto& cell : cells)
    {
        auto ancestor = cell.key;
        if (auto* cartesian = std::get_if<CartesianCellAncestry>(&ancestor.ancestry))
        {
            if (cartesian->level >= 64)
                throw std::invalid_argument("Cartesian ancestry exceeds its checked coordinate depth.");
            while (cartesian->level)
            {
                *cartesian = cartesian->parent();
                if (keys.contains(ancestor))
                    throw std::invalid_argument(
                        "Adaptive overlap catalog contains both an ancestor and its active descendant.");
            }
        }
        else
        {
            auto& convex = std::get<ConvexCellAncestry>(ancestor.ancestry);
            while (!convex.path.empty())
            {
                convex.path.pop_back();
                if (keys.contains(ancestor))
                    throw std::invalid_argument(
                        "Adaptive overlap catalog contains both an ancestor and its active descendant.");
            }
        }
    }
    return result;
}
} // namespace

std::shared_ptr<const AdaptiveOverlapPlan> AdaptiveOverlapPlan::build(HierarchyObservation source,
    HierarchyObservation target, std::vector<AdaptiveCellBinding> source_cells,
    std::vector<AdaptiveCellBinding> target_cells, std::span<const AdaptiveAncestryRelation> relations)
{
    if (!source.identity || !target.identity || !source.native_geometry || !target.native_geometry ||
        source.communicator != target.communicator)
        throw std::invalid_argument(
            "Adaptive overlap needs exact retained source/target observations on one communicator.");
    auto plan = std::make_shared<AdaptiveOverlapPlan>();
    plan->source_ = std::move(source);
    plan->target_ = std::move(target);
    plan->source_cells_ = std::move(source_cells);
    plan->target_cells_ = std::move(target_cells);
    const auto sources = index(plan->source_cells_), targets = index(plan->target_cells_);
    std::map<std::uint64_t, std::vector<ConvexGeometryPtr>> source_pieces, target_pieces;
    std::set<std::pair<std::uint64_t, std::uint64_t>> pairs;
    for (const auto& relation : relations)
    {
        if (!sources.contains(relation.source_gid) || !targets.contains(relation.target_gid) ||
            !pairs.emplace(relation.source_gid, relation.target_gid).second)
            throw std::invalid_argument("Adaptive overlap relation has unknown or duplicate endpoints.");
        const auto& s = *sources.at(relation.source_gid);
        const auto& t = *targets.at(relation.target_gid);
        const bool valid =
            relation.kind == AdaptiveAncestryRelation::Kind::Unchanged ? s.key == t.key
            : relation.kind == AdaptiveAncestryRelation::Kind::Refinement
                ? descendant(s.key, t.key)
                : relation.kind == AdaptiveAncestryRelation::Kind::Coarsening && descendant(t.key, s.key);
        if (!valid || s.material != t.material)
            throw std::invalid_argument("Adaptive overlap relation violates ancestry or material membership.");
        auto intersection = s.geometry->intersect(*t.geometry);
        if (!intersection)
            throw std::invalid_argument("Declared ancestry overlap has no positive represented volume.");
        plan->records_.push_back({s.key, t.key, s.gid, t.gid, relation.kind, intersection, intersection->moments()});
        source_pieces[s.gid].push_back(intersection);
        target_pieces[t.gid].push_back(std::move(intersection));
    }
    for (const auto& cell : plan->source_cells_)
        cell.geometry->certify_partition(source_pieces[cell.gid]);
    for (const auto& cell : plan->target_cells_)
        cell.geometry->certify_partition(target_pieces[cell.gid]);
    return plan;
}

void AdaptiveOverlapPlan::validate_observations(
    const HierarchyObservation& source, const HierarchyObservation& target) const
{
    if (!(source_ == source) || !(target_ == target))
        throw std::invalid_argument("Adaptive overlap plan source/target observation is stale or foreign.");
}
} // namespace SimpleFluid::Meshes
