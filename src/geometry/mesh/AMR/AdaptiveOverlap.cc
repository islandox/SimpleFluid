/** @file AdaptiveOverlap.cc
 * @brief Sparse ancestry selection with exact represented-cell coverage.
 */
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace SimpleFluid::Meshes::AMR
{
namespace
{
bool prefix(const BaseCellKey& parent, const BaseCellKey& child)
{
    return parent.root == child.root && parent.root_template == child.root_template &&
           parent.path.size() <= child.path.size() &&
           std::equal(parent.path.begin(), parent.path.end(), child.path.begin());
}
bool prefix(const AxialIntervalKey& parent, const AxialIntervalKey& child)
{
    return parent.root == child.root && parent.path.size() <= child.path.size() &&
           std::equal(parent.path.begin(), parent.path.end(), child.path.begin());
}
bool crossed_products(const AdaptiveCellKey& source, const AdaptiveCellKey& target)
{
    if (source.region != target.region)
        return false;
    const auto* a = std::get_if<ProductCellAncestry>(&source.ancestry);
    const auto* b = std::get_if<ProductCellAncestry>(&target.ancestry);
    if (!a || !b)
        return false;
    return (prefix(a->base, b->base) && a->base != b->base && prefix(b->axial, a->axial) && a->axial != b->axial) ||
           (prefix(b->base, a->base) && a->base != b->base && prefix(a->axial, b->axial) && a->axial != b->axial);
}

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
    if (const auto* p = std::get_if<ConvexCellAncestry>(&parent.ancestry))
    {
        const auto& c = std::get<ConvexCellAncestry>(child.ancestry);
        return p->root == c.root && p->path.size() < c.path.size() &&
               std::equal(p->path.begin(), p->path.end(), c.path.begin());
    }
    if (std::holds_alternative<StaticCellAncestry>(parent.ancestry))
        return false;
    const auto& p = std::get<ProductCellAncestry>(parent.ancestry);
    const auto& c = std::get<ProductCellAncestry>(child.ancestry);
    return p != c && prefix(p.base, c.base) && prefix(p.axial, c.axial);
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
        else if (auto* value = std::get_if<ConvexCellAncestry>(&ancestor.ancestry))
        {
            auto& convex = *value;
            while (!convex.path.empty())
            {
                convex.path.pop_back();
                if (keys.contains(ancestor))
                    throw std::invalid_argument(
                        "Adaptive overlap catalog contains both an ancestor and its active descendant.");
            }
        }
    }
    // Product leaves form a partial order, including crossed-factor overlap.
    // Index bounded factor prefixes rather than comparing geometric cell pairs.
    using Address = std::tuple<std::uint64_t, BaseCellKey, AxialIntervalKey>;
    std::map<Address, std::size_t> descendants, terminals;
    for (const auto& cell : cells)
        if (const auto* product = std::get_if<ProductCellAncestry>(&cell.key.ancestry))
        {
            auto base = product->base;
            for (;;)
            {
                ++terminals[{cell.key.region, base, product->axial}];
                auto axial = product->axial;
                for (;;)
                {
                    ++descendants[{cell.key.region, base, axial}];
                    if (axial.path.empty())
                        break;
                    axial.path.pop_back();
                }
                if (base.path.empty())
                    break;
                base.path.pop_back();
            }
        }
    for (const auto& cell : cells)
        if (const auto* product = std::get_if<ProductCellAncestry>(&cell.key.ancestry))
        {
            if (descendants[{cell.key.region, product->base, product->axial}] > 1)
                throw std::invalid_argument("Adaptive product catalog contains nested active leaves.");
            auto axial = product->axial;
            while (!axial.path.empty())
            {
                axial.path.pop_back();
                if (terminals.contains({cell.key.region, product->base, axial}))
                    throw std::invalid_argument("Adaptive product catalog contains crossed overlapping active leaves.");
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
    std::map<std::uint64_t, std::vector<PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>>> source_pieces,
        target_pieces;
    std::set<std::pair<std::uint64_t, std::uint64_t>> pairs;
    for (const auto& relation : relations)
    {
        if (!sources.contains(relation.source_gid) || !targets.contains(relation.target_gid) ||
            !pairs.emplace(relation.source_gid, relation.target_gid).second)
            throw std::invalid_argument("Adaptive overlap relation has unknown or duplicate endpoints.");
        const auto& s = *sources.at(relation.source_gid);
        const auto& t = *targets.at(relation.target_gid);
        const bool valid = relation.kind == AdaptiveAncestryRelation::Kind::Unchanged    ? s.key == t.key
                           : relation.kind == AdaptiveAncestryRelation::Kind::Refinement ? descendant(s.key, t.key)
                           : relation.kind == AdaptiveAncestryRelation::Kind::Coarsening
                               ? descendant(t.key, s.key)
                               : relation.kind == AdaptiveAncestryRelation::Kind::ProductIntersection &&
                                     crossed_products(s.key, t.key);
        if (!valid || s.material != t.material)
            throw std::invalid_argument("Adaptive overlap relation violates ancestry or material membership.");
        // Refinement/coarsening already supplies the candidate overlap as an
        // immutable child. Reclipping that child against independently rounded
        // parent planes can manufacture slivers and discard its support ancestry.
        // Ancestry selects a candidate only: both complete geometric partition
        // proofs below remain mandatory, including for unchanged cells.
        PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> intersection;
        switch (relation.kind)
        {
            case AdaptiveAncestryRelation::Kind::Unchanged:
            case AdaptiveAncestryRelation::Kind::Coarsening:
                intersection = s.geometry;
                break;
            case AdaptiveAncestryRelation::Kind::Refinement:
                intersection = t.geometry;
                break;
            case AdaptiveAncestryRelation::Kind::ProductIntersection:
                intersection = s.geometry->intersect(*t.geometry);
                break;
        }
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
} // namespace SimpleFluid::Meshes::AMR
