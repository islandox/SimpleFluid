/** @file DistributedAMROverlap.cc
 * @brief Shared owner-authenticated immediate-family overlap preparation.
 */
#include "FVM/details/DistributedAMROverlap.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
namespace SimpleFluid::VoF::detail
{
using Key = Meshes::AMR::AdaptiveCellKey;
namespace
{
bool prefix(const std::vector<std::uint64_t>& first, const std::vector<std::uint64_t>& second)
{
    return first.size() <= second.size() && std::equal(first.begin(), first.end(), second.begin());
}
std::vector<Key> parents(const Key& key)
{
    std::vector<Key> result;
    if (const auto* cart = std::get_if<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(&key.ancestry))
    {
        if (cart->level)
        {
            auto parent = key;
            parent.ancestry = cart->parent();
            result.push_back(std::move(parent));
        }
    }
    else if (const auto* convex = std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&key.ancestry))
    {
        if (!convex->path.empty())
        {
            auto parent = key;
            std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(parent.ancestry).path.pop_back();
            result.push_back(std::move(parent));
        }
    }
    else if (const auto* product = std::get_if<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(&key.ancestry))
    {
        if (!product->base.path.empty())
        {
            auto parent = key;
            std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(parent.ancestry).base.path.pop_back();
            result.push_back(std::move(parent));
        }
        if (!product->axial.path.empty())
        {
            auto parent = key;
            std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(parent.ancestry).axial.path.pop_back();
            result.push_back(std::move(parent));
        }
        if (!product->base.path.empty() && !product->axial.path.empty())
        {
            auto parent = key;
            auto& p = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(parent.ancestry);
            p.base.path.pop_back();
            p.axial.path.pop_back();
            result.push_back(std::move(parent));
        }
    }
    return result;
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveSplit> splits(const Key& key)
{
    if (std::holds_alternative<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry))
        return {::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial};
    if (std::holds_alternative<::SimpleFluid::Meshes::AMR::StaticCellAncestry>(key.ancestry))
        return {};
    return {::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic};
}
std::optional<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind> relation(const Key& source, const Key& target)
{
    using Kind = ::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind;
    if (source == target)
        return Kind::Unchanged;
    if (source.region != target.region || source.ancestry.index() != target.ancestry.index())
        return {};
    const auto source_parents = parents(source), target_parents = parents(target);
    if (std::find(target_parents.begin(), target_parents.end(), source) != target_parents.end())
        return Kind::Refinement;
    if (std::find(source_parents.begin(), source_parents.end(), target) != source_parents.end())
        return Kind::Coarsening;
    if (const auto* a = std::get_if<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(&source.ancestry))
    {
        const auto& b = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(target.ancestry);
        if (a->base.root != b.base.root || a->base.root_template != b.base.root_template ||
            a->axial.root != b.axial.root ||
            std::abs(static_cast<long long>(a->base.path.size()) - static_cast<long long>(b.base.path.size())) > 1 ||
            std::abs(static_cast<long long>(a->axial.path.size()) - static_cast<long long>(b.axial.path.size())) > 1)
            return {};
        const bool base_forward = prefix(a->base.path, b.base.path), base_reverse = prefix(b.base.path, a->base.path);
        const bool z_forward = prefix(a->axial.path, b.axial.path), z_reverse = prefix(b.axial.path, a->axial.path);
        if ((base_forward && z_reverse) || (base_reverse && z_forward))
            return Kind::ProductIntersection;
    }
    return {};
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> catalog(
    const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& hierarchy, const std::vector<Key>& probes,
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes>& family_probes, std::size_t limit,
    std::size_t leaf_fetch_budget)
{
    const auto comm = hierarchy.communicator();
    const auto service_limit = std::min(limit, hierarchy.options().byte_budget);
    auto located = hierarchy.lookup_cells(probes, false, service_limit);
    auto family_locations = hierarchy.lookup_families(family_probes, false, service_limit);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> existing;
    for (std::size_t i = 0; i < family_probes.size(); ++i)
        if (family_locations[i].owner >= 0)
            existing.push_back(family_probes[i]);
    auto families = hierarchy.fetch_families(existing, service_limit);
    std::vector<Key> child_queries;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            for (const auto& family : families)
                for (const auto& child : family.children)
                {
                    budget.add(sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(child).size());
                    child_queries.push_back(child);
                }
        });
    auto children = hierarchy.lookup_cells(child_queries, false, service_limit);
    std::vector<Key> keys;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::set<Key> complete;
            const auto add = [&](const Key& key)
            {
                if (!complete.contains(key))
                {
                    budget.add(
                        sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*));
                    complete.insert(key);
                }
            };
            for (const auto& [key, leaf] : hierarchy.owned())
                add(key);
            for (std::size_t i = 0; i < probes.size(); ++i)
                if (located[i].owner >= 0)
                    add(probes[i]);
            std::size_t offset = 0;
            for (const auto& family : families)
            {
                bool active = true;
                for (std::size_t i = 0; i < family.children.size(); ++i)
                    active = active && children[offset + i].owner >= 0;
                if (active)
                    for (const auto& child : family.children)
                        add(child);
                offset += family.children.size();
            }
            keys.assign(complete.begin(), complete.end());
        });
    // These probe/family packages are consumed. Release their buffers and
    // shared recipes before lending the phase-specific leaf decode arena.
    decltype(located)().swap(located);
    decltype(family_locations)().swap(family_locations);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes>().swap(existing);
    decltype(families)().swap(families);
    std::vector<Key>().swap(child_queries);
    decltype(children)().swap(children);
    const auto locations = hierarchy.lookup_cells(keys, true, service_limit);
    const auto leaves = hierarchy.fetch_leaves(
        keys, std::min(leaf_fetch_budget ? leaf_fetch_budget : service_limit, hierarchy.options().byte_budget));
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> result;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (leaves[i].key != keys[i] || leaves[i].revision != locations[i].revision)
                    throw std::invalid_argument("Overlap family owner/revision authentication failed.");
                auto geometry = hierarchy.geometry(leaves[i]);
                if (!geometry)
                    throw std::invalid_argument(
                        "Every topology-remap participant requires certified planar convex geometry.");
                budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveCellBinding) +
                           ::SimpleFluid::Meshes::AMR::encode_adaptive_key(keys[i]).size() + geometry->storage_bytes());
                result.push_back({keys[i], locations[i].address, leaves[i].material, std::move(geometry)});
            }
        });
    return result;
}
} // namespace
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> build_adaptive_overlap(
    const Meshes::AMR::DistributedAdaptiveHierarchy& source, const Meshes::AMR::DistributedAdaptiveHierarchy& target,
    std::size_t limit, std::size_t leaf_fetch_budget)
{
    const auto comm = source.communicator();
    std::vector<Key> probes;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> family_probes;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::set<Key> roots;
            const auto add = [&](const Key& key)
            {
                if (!roots.contains(key))
                {
                    budget.add(
                        sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*));
                    roots.insert(key);
                }
            };
            for (const auto* hierarchy : {&source, &target})
                for (const auto& [key, leaf] : hierarchy->owned())
                {
                    add(key);
                    for (const auto& parent : parents(key))
                        add(parent);
                }
            probes.assign(roots.begin(), roots.end());
            std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> families;
            for (const auto& key : probes)
                for (auto split : splits(key))
                {
                    auto family = ::SimpleFluid::Meshes::AMR::adaptive_family_key(key, split);
                    if (!families.contains(family))
                    {
                        budget.add(
                            family.size() + sizeof(::SimpleFluid::Meshes::AMR::AdaptiveBytes) + 4 * sizeof(void*));
                        families.insert(std::move(family));
                    }
                }
            family_probes.assign(families.begin(), families.end());
        });
    auto from = catalog(source, probes, family_probes, limit, leaf_fetch_budget),
         to = catalog(target, probes, family_probes, limit, leaf_fetch_budget);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::map<Key, std::vector<std::size_t>> target_index;
            for (std::size_t i = 0; i < to.size(); ++i)
            {
                auto ancestors = parents(to[i].key);
                ancestors.push_back(to[i].key);
                for (const auto& ancestor : ancestors)
                {
                    budget.add(sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(ancestor).size() +
                               sizeof(std::size_t) + 4 * sizeof(void*));
                    target_index[ancestor].push_back(i);
                }
            }
            for (const auto& cell : from)
            {
                std::set<std::size_t> candidates;
                auto ancestors = parents(cell.key);
                ancestors.push_back(cell.key);
                for (const auto& ancestor : ancestors)
                    if (auto found = target_index.find(ancestor); found != target_index.end())
                        candidates.insert(found->second.begin(), found->second.end());
                for (auto i : candidates)
                    if (auto kind = relation(cell.key, to[i].key))
                    {
                        if (cell.material != to[i].material)
                            throw std::invalid_argument("Topology transfer changes physical material membership.");
                        budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation));
                        relations.push_back({*kind, cell.gid, to[i].gid});
                    }
            }
        });
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> result;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            result = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
                source.observation(), target.observation(), std::move(from), std::move(to), relations);
        });
    return result;
}
} // namespace SimpleFluid::VoF::detail
