/** @file CellCatalog.tcc
 * @brief Compiled mode-specific sparse native/fallback catalog construction.
 */
#pragma once
#include "FVM/details/GeneralVoF/CellCatalogInternal.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
{
namespace
{
template<class T> void reserve_checked(std::vector<T>& values, std::size_t count, wire::Budget& budget)
{
    if (count > values.max_size())
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Native geometry vector count exceeds its representable size.");
    charge_items(budget, count, sizeof(T));
    values.reserve(count);
    if (values.capacity() > count)
        charge_items(budget, values.capacity() - count, sizeof(T));
}
} // namespace
template<ArithmeticMode Mode>
CatalogBuild build(const GeneralVoF::Hierarchy& hierarchy, std::span<const AMR::AdaptiveLeaf* const> leaves,
    wire::Budget& budget, const GeneralVoF::Data* reuse, CatalogPurpose purpose,
    PLIC::GeometryAllocationLedger* admitted_originals, const CatalogReuseAuthority* reuse_authority,
    const void* construction_operation)
{
    constexpr auto ordinary_mode = ArithmeticMode::Ordinary;
    using Catalog = PLIC::CellGeometryCatalog<ordinary_mode>;
    using BoundedCatalog = PLIC::CellGeometryCatalog<ArithmeticMode::BoundedCertification>;
    CatalogBuild result;
    const bool state_purpose = purpose == CatalogPurpose::State;
    const auto controls = construction_controls();
    bool stable_controls = controls.supported;
    const auto observe_controls = [&] { stable_controls = stable_controls && construction_controls() == controls; };
    // Source-only callers reproduce fresh provider construction under their
    // current controls; accepted catalogs are never equivalent by recipe alone.
    if (!state_purpose)
        reuse = nullptr;
    PLIC::GeometryAllocationLedger local_ordinary_owners, bounded_owners;
    auto& ordinary_owners = admitted_originals ? *admitted_originals : local_ordinary_owners;
    if (reuse)
    {
        if (reuse->hierarchy.get() != &hierarchy || reuse->options.arithmetic_mode != Mode)
            throw std::invalid_argument("Native catalog reuse lacks its matching hierarchy/arithmetic authority.");
        ordinary_owners.retain(reuse->topology->cell_geometry, budget.used, budget.limit);
        if (state_purpose && reuse->topology->ordinary_cell_geometry)
            ordinary_owners.retain(reuse->topology->ordinary_cell_geometry, budget.used, budget.limit);
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            if (state_purpose)
                bounded_owners.retain(reuse->topology->bounded_cell_geometry, budget.used, budget.limit);
    }
    // Old roots above stay admitted even when numeric controls prohibit
    // binding reuse. Only the matching in-constructor operation may reuse.
    const bool reuse_bindings = reuse && state_purpose && construction_operation && reuse_authority &&
                                reuse_authority->stable && reuse_authority->operation == construction_operation &&
                                reuse_authority->original_catalog == reuse->topology->cell_geometry.get() &&
                                controls.supported && reuse_authority->controls == controls;
    std::vector<Catalog::Entry> entries;
    reserve_checked(entries, leaves.size(), budget);
    entries.resize(leaves.size());
    if (state_purpose)
    {
        reserve_checked(result.native_metrics, leaves.size(), budget);
        result.native_metrics.resize(leaves.size());
    }
    if (std::any_of(leaves.begin(), leaves.end(), [](const auto* leaf) { return leaf == nullptr; }))
        throw std::invalid_argument("Native geometry catalog lacks an authenticated source leaf.");
    std::vector<Catalog::Entry> finite_ordinary_entries;
    bool separate_ordinary = false;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        separate_ordinary = state_purpose && std::any_of(leaves.begin(), leaves.end(),
                                                 [&](const auto* leaf) { return !eligible(hierarchy, *leaf, Mode); });
        if (separate_ordinary)
        {
            reserve_checked(finite_ordinary_entries, leaves.size(), budget);
            finite_ordinary_entries.resize(leaves.size());
        }
    }
    std::vector<typename BoundedCatalog::Entry> bounded_entries;
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
    {
        if (state_purpose)
        {
            reserve_checked(bounded_entries, leaves.size(), budget);
            bounded_entries.resize(leaves.size());
        }
    }
    struct Group
    {
        std::size_t count = 0, source_ordinal = 0;
        std::vector<std::size_t> slots;
    };
    std::map<const Meshes::NativeFragmentDescriptor*, Group> groups;
    using ReusedSource = std::pair<std::size_t, std::shared_ptr<const PLIC::CartesianCellSource>>;
    std::map<const PLIC::CartesianCellSource*, ReusedSource> reused_sources;
    std::vector<AMR::AdaptiveCartesianGeometryBounds> raw_bounds;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        if (std::any_of(
                leaves.begin(), leaves.end(), [&](const auto* leaf) { return eligible(hierarchy, *leaf, Mode); }))
        {
            reserve_checked(raw_bounds, leaves.size(), budget);
            raw_bounds.resize(leaves.size());
        }
    }
    std::vector<unsigned char> reused_slots;
    reserve_checked(reused_slots, leaves.size(), budget);
    reused_slots.resize(leaves.size(), 0);
    std::vector<unsigned char> native_slots;
    reserve_checked(native_slots, leaves.size(), budget);
    native_slots.resize(leaves.size(), 0);
    const auto require_separate_ordinary = [&]
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            if (!separate_ordinary)
            {
                reserve_checked(finite_ordinary_entries, leaves.size(), budget);
                finite_ordinary_entries.assign(entries.begin(), entries.end());
                separate_ordinary = true;
            }
    };
    for (std::size_t slot = 0; slot < leaves.size(); ++slot)
    {
        const auto& leaf = *leaves[slot];
        if (eligible(hierarchy, leaf, Mode))
        {
            // Invoke the actual private final kernel once, in source traversal
            // order. Stored descriptor axes remain metric/ordinal authority;
            // they are not the caller-controlled geometry() coordinates.
            const auto bounds = hierarchy.native_cartesian_geometry_bounds(leaf);
            if (!bounds)
                throw std::logic_error("Trusted Cartesian geometry capability changed.");
            raw_bounds[slot] = *bounds;
            observe_controls();
            native_slots[slot] =
                !state_purpose || PLIC::details::NativeCellGeometrySourceAccess::state_compatible(bounds->lower,
                                      bounds->upper, geometry_limits(ordinary_mode, budget.limit - budget.used));
            observe_controls();
            if (state_purpose && !native_slots[slot])
                require_separate_ordinary();
        }
        const Cell* reused_cell = nullptr;
        if (reuse_bindings && stable_controls && construction_controls() == controls &&
            hierarchy.owned().contains(leaf.key))
        {
            const auto found = reuse->cells.find(leaf.key);
            const auto& original_leaf = hierarchy.owned().at(leaf.key);
            if (found != reuse->cells.end() && found->second.revision == leaf.revision &&
                found->second.owner == hierarchy.communicator()->getRank() &&
                (same_native_recipe(original_leaf, leaf) ||
                    (leaf.geometry && original_leaf.geometry == leaf.geometry && original_leaf.key == leaf.key &&
                        original_leaf.revision == leaf.revision && original_leaf.material == leaf.material &&
                        original_leaf.domain == leaf.domain && original_leaf.provider == leaf.provider &&
                        original_leaf.native_cell == leaf.native_cell &&
                        original_leaf.native_recipe == leaf.native_recipe)))
                reused_cell = &found->second;
        }
        if constexpr (Mode == ArithmeticMode::Ordinary)
            if (reused_cell && native_cell_geometry(*reuse, *reused_cell).is_procedural() != bool(native_slots[slot]))
                reused_cell = nullptr;
        if (reused_cell && native_slots[slot])
        {
            const auto ref =
                PLIC::details::NativeCellGeometrySourceAccess::source_cell(native_cell_geometry(*reuse, *reused_cell));
            auto found = reused_sources.find(ref.source.get());
            if (found == reused_sources.end())
            {
                charge_items(budget, 1, sizeof(typename decltype(reused_sources)::value_type) + 4 * sizeof(void*));
                found = reused_sources.emplace(ref.source.get(), ReusedSource{reused_sources.size(), ref.source}).first;
            }
            ordinary_owners.retain(ref.source, budget.used, budget.limit);
            reused_slots[slot] = 1;
            entries[slot] = PLIC::NativeCellGeometrySlot{found->second.first, ref.cell_slot};
            if (separate_ordinary)
                finite_ordinary_entries[slot] = entries[slot];
            if (state_purpose)
                result.native_metrics[slot] = {reused_cell->native_volume, reused_cell->centroid};
            continue;
        }
        if (reused_cell)
        {
            auto original = existing_native_geometry(*reuse, *reused_cell);
            ordinary_owners.retain(original, budget.used, budget.limit);
            entries[slot] = std::move(original);
            if (state_purpose)
                result.native_metrics[slot] = {reused_cell->native_volume, reused_cell->centroid};
            if constexpr (Mode == ArithmeticMode::Ordinary)
            {
                if (state_purpose)
                {
                    auto imported = existing_cell_geometry<Mode>(*reuse, *reused_cell);
                    ordinary_owners.retain(imported, budget.used, budget.limit);
                    finite_ordinary_entries[slot] = std::move(imported);
                }
            }
            else if constexpr (Mode == ArithmeticMode::BoundedCertification)
            {
                if (state_purpose)
                {
                    auto imported = existing_cell_geometry<Mode>(*reuse, *reused_cell);
                    bounded_owners.retain(imported, budget.used, budget.limit);
                    bounded_entries[slot] = std::move(imported);
                }
            }
            continue;
        }
        if (native_slots[slot])
        {
            const auto found = groups.find(leaf.native_recipe.get());
            if (found == groups.end())
            {
                budget.add(sizeof(typename decltype(groups)::value_type) + 4 * sizeof(void*));
                const auto ordinal = groups.size();
                auto& group = groups[leaf.native_recipe.get()];
                group.source_ordinal = ordinal;
                ++group.count;
            }
            else
                ++found->second.count;
            continue;
        }
        auto original = hierarchy.geometry(leaf);
        observe_controls();
        ordinary_owners.retain(original, budget.used, budget.limit);
        if (state_purpose)
        {
            result.native_metrics[slot] = hierarchy.metrics(leaf);
            observe_controls();
        }
        if constexpr (Mode != ArithmeticMode::Exact)
        {
            if (state_purpose)
            {
                auto imported = PLIC::ConvexCellGeometry<Mode>::from_geometry(
                    *original, geometry_limits(Mode, budget.limit - budget.used));
                observe_controls();
                if constexpr (Mode == ArithmeticMode::Ordinary)
                    ordinary_owners.retain(imported, budget.used, budget.limit);
                else
                    bounded_owners.retain(imported, budget.used, budget.limit);
                if constexpr (Mode == ordinary_mode)
                {
                    finite_ordinary_entries[slot] = imported;
                    if (!leaf.geometry && std::holds_alternative<AMR::CartesianCellAncestry>(leaf.key.ancestry) &&
                        original->same_native_representation(*imported))
                        original = imported;
                }
                else
                    bounded_entries[slot] = std::move(imported);
            }
        }
        entries[slot] = std::move(original);
    }
    for (auto& [recipe, group] : groups)
    {
        reserve_checked(group.slots, group.count, budget);
    }
    for (std::size_t slot = 0; slot < leaves.size(); ++slot)
        if (native_slots[slot] && !reused_slots[slot])
            groups.at(leaves[slot]->native_recipe.get()).slots.push_back(slot);
    typename Catalog::SourceTable sources;
    if (groups.size() > std::numeric_limits<std::size_t>::max() - reused_sources.size())
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Native source table count overflows.");
    reserve_checked(sources, groups.size() + reused_sources.size(), budget);
    sources.resize(reused_sources.size());
    for (const auto& [pointer, record] : reused_sources)
        sources.at(record.first) = record.second;
    std::vector<const Meshes::NativeFragmentDescriptor*> recipe_order;
    reserve_checked(recipe_order, groups.size(), budget);
    recipe_order.resize(groups.size());
    for (const auto& [recipe, group] : groups)
        recipe_order.at(group.source_ordinal) = recipe;
    for (const auto* recipe : recipe_order)
    {
        const auto& slots = groups.at(recipe).slots;
        const RecipeAxes axes(*recipe);
        struct BoundCoordinate
        {
            std::uint64_t index = 0;
            double value = 0;
        };
        std::array<std::vector<BoundCoordinate>, 3> selected;
        for (auto& indices : selected)
        {
            reserve_checked(indices, checked_product(slots.size(), 2), budget);
        }
        for (auto slot : slots)
        {
            const auto location = axes.location(*leaves[slot]);
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                selected[axis].push_back({location[axis], raw_bounds[slot].lower[axis]});
                selected[axis].push_back({location[axis] + 1, raw_bounds[slot].upper[axis]});
            }
        }
        PLIC::CartesianSourceDescription description;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            auto& indices = selected[axis];
            std::sort(indices.begin(), indices.end(), [](const auto& a, const auto& b) { return a.index < b.index; });
            for (std::size_t i = 1; i < indices.size(); ++i)
                if (indices[i - 1].index == indices[i].index && std::bit_cast<std::uint64_t>(indices[i - 1].value) !=
                                                                    std::bit_cast<std::uint64_t>(indices[i].value))
                    throw std::invalid_argument("Cartesian shared raw coordinate construction differs.");
            indices.erase(std::unique(indices.begin(), indices.end(),
                              [](const auto& a, const auto& b) { return a.index == b.index; }),
                indices.end());
            if (indices.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Native Cartesian compact coordinate index overflows.");
            reserve_checked(description.axes[axis], indices.size(), budget);
            for (const auto& coordinate : indices)
                description.axes[axis].push_back(coordinate.value);
        }
        reserve_checked(description.cells, slots.size(), budget);
        for (auto slot : slots)
        {
            const auto location = axes.location(*leaves[slot]);
            PLIC::CartesianSourceCellSlot cell;
            PLIC::GeometryPoint lower{}, upper{}, center{};
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                const auto& indices = selected[axis];
                cell.lower[axis] = static_cast<std::uint32_t>(
                    std::lower_bound(indices.begin(), indices.end(), location[axis],
                        [](const auto& coordinate, std::uint64_t index) { return coordinate.index < index; }) -
                    indices.begin());
                cell.upper[axis] = static_cast<std::uint32_t>(
                    std::lower_bound(indices.begin(), indices.end(), location[axis] + 1,
                        [](const auto& coordinate, std::uint64_t index) { return coordinate.index < index; }) -
                    indices.begin());
                if (state_purpose)
                {
                    lower[axis] = axes.value(axis, location[axis]);
                    upper[axis] = axes.value(axis, location[axis] + 1);
                    center[axis] = std::midpoint(lower[axis], upper[axis]);
                }
            }
            // Keep RegionProviders.cc's binary64 subtraction/multiplication order.
            if (state_purpose)
            {
                const double native_volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
                result.native_metrics[slot] = {native_volume, center};
            }
            entries[slot] = PLIC::NativeCellGeometrySlot{sources.size(), description.cells.size()};
            if (separate_ordinary)
                finite_ordinary_entries[slot] = entries[slot];
            description.cells.push_back(cell);
        }
        auto source = PLIC::details::NativeCellGeometrySourceAccess::make(std::move(description),
            geometry_limits(ordinary_mode, budget.limit - budget.used),
            state_purpose ? PLIC::details::NativeCellGeometryMoments::PrimedState
                          : PLIC::details::NativeCellGeometryMoments::DeferredRaw);
        observe_controls();
        ordinary_owners.retain(source, budget.used, budget.limit);
        sources.push_back(std::move(source));
    }
    typename Catalog::SourceTable finite_ordinary_sources;
    if (separate_ordinary)
    {
        reserve_checked(finite_ordinary_sources, sources.size(), budget);
        finite_ordinary_sources.insert(finite_ordinary_sources.end(), sources.begin(), sources.end());
    }
    result.ordinary = Catalog::make(
        std::move(sources), std::move(entries), geometry_limits(ordinary_mode, budget.limit - budget.used));
    ordinary_owners.retain(result.ordinary, budget.used, budget.limit);
    if (separate_ordinary)
    {
        result.finite_ordinary = Catalog::make(std::move(finite_ordinary_sources), std::move(finite_ordinary_entries),
            geometry_limits(ordinary_mode, budget.limit - budget.used));
        ordinary_owners.retain(result.finite_ordinary, budget.used, budget.limit);
    }
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
    {
        if (state_purpose)
        {
            result.bounded = BoundedCatalog::make(typename BoundedCatalog::SourceTable{}, std::move(bounded_entries),
                geometry_limits(Mode, budget.limit - budget.used));
            bounded_owners.retain(result.bounded, budget.used, budget.limit);
        }
    }
    observe_controls();
    result.reuse_authority = {construction_operation, result.ordinary.get(), controls, stable_controls};
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
