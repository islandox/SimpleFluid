/** @file CellCatalog.tcc
 * @brief Compiled mode-specific sparse native/fallback catalog construction.
 */
#pragma once
#include "geometry/PLIC/details/CellGeometryEvaluation.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotInternal.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
{
namespace
{
template<class T> void reserve_checked(std::vector<T>& values, std::size_t count, adaptive_detail::Budget& budget)
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
CatalogBuild Builder::build(const DistributedAdaptiveHierarchy& hierarchy, std::span<const AdaptiveLeaf* const> leaves,
    adaptive_detail::Budget& budget, const AdaptiveGeometrySnapshot* reuse, CatalogPurpose purpose,
    PLIC::GeometryAllocationLedger& ordinary_owners, PLIC::GeometryAllocationLedger& bounded_owners,
    std::span<const PublicationSource> publication, BuildFacts& facts, bool transient_catalog)
{
    constexpr auto ordinary_mode = ArithmeticMode::Ordinary;
    using Catalog = PLIC::CellGeometryCatalog<ordinary_mode>;
    using BoundedCatalog = PLIC::CellGeometryCatalog<ArithmeticMode::BoundedCertification>;
    CatalogBuild result;
    const bool state_purpose = purpose == CatalogPurpose::State;
    const auto controls = construction_controls();
    bool stable_controls = controls.supported;
    const auto observe_controls = [&] { stable_controls = stable_controls && construction_controls() == controls; };
    if (reuse)
        reuse->retain(ordinary_owners, bounded_owners, budget.used, budget.limit);
    const bool reuse_bindings = reuse && reuse->stable_controls_ && controls.supported && reuse->controls_ == controls;
    std::vector<Catalog::Entry> entries;
    reserve_checked(entries, leaves.size(), budget);
    entries.resize(leaves.size());
    reserve_checked(facts.policy, leaves.size(), budget);
    facts.policy.resize(leaves.size());
    if (state_purpose)
    {
        reserve_checked(facts.metrics, leaves.size(), budget);
        facts.metrics.resize(leaves.size());
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
    struct BoundCoordinate
    {
        double value = 0;
        std::uint32_t slot = 0;
    };
    /** Compact raw coordinate column. Entries are appended while cells are
     * traversed and sorted by axis index before slot assignment; a vector costs
     * a quarter of a map node per coordinate and keeps the description scratch
     * proportional to the actual axis endpoints. */
    using BoundAxis = std::vector<std::pair<std::uint64_t, BoundCoordinate>>;
    const auto lower_coordinate = [](const BoundAxis& values, std::uint64_t index)
    {
        return std::lower_bound(values.begin(), values.end(), index,
            [](const auto& value, std::uint64_t wanted) { return value.first < wanted; });
    };
    struct Group
    {
        explicit Group(const NativeFragmentDescriptor& recipe) : axes(recipe) {}
        RecipeAxes axes;
        std::array<BoundAxis, 3> raw_coordinates;
        std::size_t count = 0, source_ordinal = 0;
        std::vector<std::size_t> slots;
    };
    std::map<const Meshes::NativeFragmentDescriptor*, Group> groups;
    using ReusedSource = std::pair<std::size_t, std::shared_ptr<const PLIC::CartesianCellSource>>;
    struct SourceBinding
    {
        const PLIC::CartesianCellSource* source = nullptr;
        const NativeFragmentDescriptor* recipe = nullptr;
        bool operator<(const SourceBinding& other) const
        {
            if (source != other.source)
                return std::less<const PLIC::CartesianCellSource*>{}(source, other.source);
            return std::less<const NativeFragmentDescriptor*>{}(recipe, other.recipe);
        }
    };
    std::map<SourceBinding, ReusedSource> reused_sources;
    /** Raw rows whose previous context is a prepared page: they adopt a deferred
     * subset of that page's immutable base instead of rebuilding a source. */
    struct SubsetRequest
    {
        std::size_t slot = 0;
        std::shared_ptr<const PLIC::CartesianCellSource> source;
        std::uint64_t cell = 0;
    };
    std::vector<SubsetRequest> subset_requests;
    std::vector<unsigned char> reused_slots;
    reserve_checked(reused_slots, leaves.size(), budget);
    reused_slots.resize(leaves.size(), 0);
    std::vector<unsigned char> native_slots;
    reserve_checked(native_slots, leaves.size(), budget);
    native_slots.resize(leaves.size(), 0);
    std::vector<unsigned char> subset_rows;
    reserve_checked(subset_rows, leaves.size(), budget);
    subset_rows.resize(leaves.size(), 0);
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
    const auto adopt_native =
        [&](std::size_t slot, const AdaptiveLeaf& leaf, const PLIC::details::NativeCartesianCellReference& ref)
    {
        // Raw recipe contexts stay independent even when their immutable
        // source body is shared with a State page.
        const SourceBinding binding{ref.source.get(), state_purpose ? nullptr : leaf.native_recipe.get()};
        auto found = reused_sources.find(binding);
        if (found == reused_sources.end())
        {
            charge_items(budget, 1, sizeof(typename decltype(reused_sources)::value_type) + 4 * sizeof(void*));
            auto source = state_purpose ? ref.source
                                        : PLIC::CartesianCellSource::clone_deferred(
                                              ref.source, geometry_limits(ordinary_mode, budget.limit - budget.used));
            found = reused_sources.emplace(binding, ReusedSource{reused_sources.size(), std::move(source)}).first;
        }
        ordinary_owners.retain(found->second.second, budget.used, budget.limit);
        entries[slot] = PLIC::NativeCellGeometrySlot{found->second.first, ref.cell_slot};
        if (separate_ordinary)
            finite_ordinary_entries[slot] = entries[slot];
        native_slots[slot] = reused_slots[slot] = 1;
    };
    const auto prepared_source =
        [&](const AdaptiveLeaf& leaf,
            const AdaptiveCartesianGeometryBounds& bounds) -> std::optional<PLIC::details::NativeCartesianCellReference>
    {
        if (!state_purpose || !reuse_bindings || !stable_controls || !reuse->prepared_cartesian_ ||
            !reuse->matches_layout(hierarchy.leaf_layout()))
            return std::nullopt;
        const auto* owned = exact_owned_cartesian_source(hierarchy, leaf);
        if (!owned)
            return std::nullopt;
        const auto first = leaf.native_cell / prepared_cartesian_page_size * prepared_cartesian_page_size;
        const auto bit = std::uint64_t(1) << (leaf.native_cell - first);
        const auto* found = reuse->prepared_cartesian_->find(owned->native_recipe, first);
        if (!found || !(found->rows & bit))
            return std::nullopt;
        const PLIC::details::NativeCartesianCellReference ref{
            found->source, std::size_t(std::popcount(found->rows & (bit - 1)))};
        const auto [lower, upper] = PLIC::details::NativeCellGeometrySourceAccess::raw_bounds(ref);
        const auto equal = [](double a, double b)
        { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); };
        if (!std::equal(lower.begin(), lower.end(), bounds.lower.begin(), equal) ||
            !std::equal(upper.begin(), upper.end(), bounds.upper.begin(), equal))
            return std::nullopt;
        return ref;
    };
    for (std::size_t slot = 0; slot < leaves.size(); ++slot)
    {
        const auto& leaf = *leaves[slot];
        std::optional<AdaptiveCartesianGeometryBounds> raw_bounds;
        auto reused_slot =
            reuse_bindings && stable_controls && construction_controls() == controls ? reuse->find(leaf) : std::nullopt;
        const auto* published = slot < publication.size() ? &publication[slot] : nullptr;
        if (reused_slot && published && leaf.geometry)
        {
            const PLIC::CellGeometryView<ordinary_mode> previous(reuse->ordinary_, *reused_slot);
            if (previous.is_procedural() ||
                PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(previous) != leaf.geometry)
                reused_slot.reset();
        }
        if (reused_slot && reuse->purpose_ != purpose)
        {
            const PLIC::CellGeometryView<ordinary_mode> previous(reuse->ordinary_, *reused_slot);
            if (state_purpose || (!previous.is_procedural() && !leaf.geometry))
                reused_slot.reset();
            else if (previous.is_procedural() && reuse->prepared_cartesian_)
            {
                // A prepared State page can contain unselected local cells. A raw
                // context exposes only the requested rows by sharing that page's
                // immutable base through a deferred subset body.
                const auto reference = PLIC::details::NativeCellGeometrySourceAccess::source_cell(previous);
                if (reuse->prepared_cartesian_->contains_source(reference.source))
                    subset_rows[slot] = 1;
            }
        }
        if (reused_slot && !subset_rows[slot])
        {
            const PLIC::CellGeometryView<ordinary_mode> previous(reuse->ordinary_, *reused_slot);
            native_slots[slot] = previous.is_procedural();
            if (state_purpose && !native_slots[slot])
                require_separate_ordinary();
        }
        else if (!subset_rows[slot] && eligible(hierarchy, leaf, Mode))
        {
            // Invoke the actual private final kernel once, in source traversal
            // order. Stored descriptor axes remain metric/ordinal authority;
            // they are not the caller-controlled geometry() coordinates.
            const auto bounds = hierarchy.native_cartesian_geometry_bounds(leaf);
            if (!bounds)
                throw std::logic_error("Trusted Cartesian geometry capability changed.");
            raw_bounds = *bounds;
            observe_controls();
            if (const auto prepared = prepared_source(leaf, *bounds))
            {
                adopt_native(slot, leaf, *prepared);
                facts.metrics[slot] = RecipeAxes(*leaf.native_recipe).metrics(leaf);
                observe_controls();
                continue;
            }
            native_slots[slot] =
                !state_purpose || PLIC::details::NativeCellGeometrySourceAccess::state_compatible(bounds->lower,
                                      bounds->upper, geometry_limits(ordinary_mode, budget.limit - budget.used));
            observe_controls();
            if (state_purpose && !native_slots[slot])
                require_separate_ordinary();
        }
        if (reused_slot && subset_rows[slot])
        {
            const PLIC::CellGeometryView<ordinary_mode> previous(reuse->ordinary_, *reused_slot);
            const auto reference = PLIC::details::NativeCellGeometrySourceAccess::source_cell(previous);
            subset_requests.push_back({slot, reference.source, leaf.native_cell});
            facts.policy[slot] = reuse->evaluation_policy(*reused_slot);
            native_slots[slot] = 0;
            reused_slots[slot] = 0;
            continue;
        }
        if (reused_slot)
        {
            const PLIC::CellGeometryView<ordinary_mode> previous(reuse->ordinary_, *reused_slot);
            if ((state_purpose || previous.is_procedural() || leaf.geometry ||
                    reuse->purpose_ == CatalogPurpose::OriginalSources) &&
                previous.is_procedural() == bool(native_slots[slot]))
            {
                if (previous.is_procedural())
                    adopt_native(slot, leaf, PLIC::details::NativeCellGeometrySourceAccess::source_cell(previous));
                else
                {
                    auto original = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(previous);
                    ordinary_owners.retain(original, budget.used, budget.limit);
                    entries[slot] = reuse->ordinary_->explicit_entry(*reused_slot);
                    if constexpr (Mode == ArithmeticMode::Ordinary)
                    {
                        if (state_purpose)
                        {
                            const auto selected = reuse->finite_ordinary_ ? reuse->finite_ordinary_ : reuse->ordinary_;
                            auto imported = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                                PLIC::CellGeometryView<Mode>(selected, *reused_slot));
                            ordinary_owners.retain(imported, budget.used, budget.limit);
                            finite_ordinary_entries[slot] = selected->explicit_entry(*reused_slot);
                        }
                    }
                    else if constexpr (Mode == ArithmeticMode::BoundedCertification)
                    {
                        if (state_purpose)
                        {
                            auto imported = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                                PLIC::CellGeometryView<Mode>(reuse->bounded_, *reused_slot));
                            bounded_owners.retain(imported, budget.used, budget.limit);
                            bounded_entries[slot] = reuse->bounded_->explicit_entry(*reused_slot);
                        }
                    }
                }
                facts.policy[slot] = reuse->evaluation_policy(*reused_slot);
                reused_slots[slot] = 1;
                if (state_purpose)
                    facts.metrics[slot] = published && leaf.geometry ? AdaptiveCellMetrics{published->binding->volume,
                                                                           published->binding->centroid}
                                                                     : reuse->native_metrics()[*reused_slot];
                continue;
            }
        }
        if (native_slots[slot])
        {
            auto found = groups.find(leaf.native_recipe.get());
            if (found == groups.end())
            {
                budget.add(sizeof(typename decltype(groups)::value_type) + 4 * sizeof(void*));
                const auto ordinal = groups.size();
                found = groups.try_emplace(leaf.native_recipe.get(), *leaf.native_recipe).first;
                found->second.source_ordinal = ordinal;
            }
            auto& group = found->second;
            ++group.count;
            const auto location = group.axes.location(leaf);
            if (!raw_bounds)
                throw std::logic_error("New Cartesian source lacks raw bounds.");
            for (unsigned axis = 0; axis < 3; ++axis)
                for (unsigned endpoint = 0; endpoint < 2; ++endpoint)
                {
                    auto& coordinates = group.raw_coordinates[axis];
                    const auto index = location[axis] + endpoint;
                    const auto value = endpoint ? raw_bounds->upper[axis] : raw_bounds->lower[axis];
                    const auto at = lower_coordinate(coordinates, index);
                    if (at != coordinates.end() && at->first == index)
                    {
                        if (std::bit_cast<std::uint64_t>(at->second.value) != std::bit_cast<std::uint64_t>(value))
                            throw std::invalid_argument("Cartesian shared raw coordinate construction differs.");
                    }
                    else
                    {
                        const auto position = std::size_t(at - coordinates.begin());
                        budget.append_capacity(coordinates);
                        coordinates.insert(coordinates.begin() + position, {index, BoundCoordinate{value}});
                    }
                }
            continue;
        }
        // Only a private, authenticated publication row may supply the already
        // mapped original owner. Public provider-input acquisition still calls
        // the actual kernel, including providers that ignore leaf.geometry.
        auto original = published && leaf.geometry ? leaf.geometry : hierarchy.geometry(leaf);
        observe_controls();
        ordinary_owners.retain(original, budget.used, budget.limit);
        if (state_purpose)
        {
            facts.metrics[slot] = published && leaf.geometry
                                      ? AdaptiveCellMetrics{published->binding->volume, published->binding->centroid}
                                      : hierarchy.metrics(leaf);
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
                    if (!leaf.geometry && std::holds_alternative<CartesianCellAncestry>(leaf.key.ancestry) &&
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
        for (auto& coordinates : group.raw_coordinates)
            std::sort(coordinates.begin(), coordinates.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
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
    // One deferred subset body per previous prepared source, ordered by cell.
    {
        std::map<const PLIC::CartesianCellSource*, std::vector<std::size_t>> by_source;
        for (std::size_t index = 0; index < subset_requests.size(); ++index)
            by_source[subset_requests[index].source.get()].push_back(index);
        for (const auto& [address, rows] : by_source)
        {
            std::vector<std::uint64_t> cells;
            reserve_checked(cells, rows.size(), budget);
            for (const auto row : rows)
                cells.push_back(subset_requests[row].cell);
            std::sort(cells.begin(), cells.end());
            auto clone = PLIC::CartesianCellSource::clone_deferred_subset(
                subset_requests[rows.front()].source, cells, geometry_limits(ordinary_mode, budget.limit - budget.used));
            ordinary_owners.retain(clone, budget.used, budget.limit);
            const auto ordinal = sources.size();
            sources.push_back(clone);
            for (const auto row : rows)
            {
                const auto rank = std::lower_bound(cells.begin(), cells.end(), subset_requests[row].cell) -
                                  cells.begin();
                entries[subset_requests[row].slot] = PLIC::NativeCellGeometrySlot{ordinal, std::size_t(rank)};
                if (separate_ordinary)
                    finite_ordinary_entries[subset_requests[row].slot] =
                        PLIC::NativeCellGeometrySlot{ordinal, std::size_t(rank)};
            }
        }
    }

    std::vector<const Meshes::NativeFragmentDescriptor*> recipe_order;
    reserve_checked(recipe_order, groups.size(), budget);
    recipe_order.resize(groups.size());
    for (const auto& [recipe, group] : groups)
        recipe_order.at(group.source_ordinal) = recipe;
    for (const auto* recipe : recipe_order)
    {
        auto& group = groups.at(recipe);
        const auto& axes = group.axes;
        const auto& slots = group.slots;
        PLIC::CartesianSourceDescription description;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            auto& coordinates = group.raw_coordinates[axis];
            if (coordinates.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Native Cartesian compact coordinate index overflows.");
            reserve_checked(description.axes[axis], coordinates.size(), budget);
            for (auto& [index, coordinate] : coordinates)
            {
                coordinate.slot = static_cast<std::uint32_t>(description.axes[axis].size());
                description.axes[axis].push_back(coordinate.value);
            }
        }
        reserve_checked(description.cells, slots.size(), budget);
        for (auto slot : slots)
        {
            const auto location = axes.location(*leaves[slot]);
            PLIC::CartesianSourceCellSlot cell;
            PLIC::GeometryPoint lower{}, upper{}, center{};
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                const auto& coordinates = group.raw_coordinates[axis];
                const auto lower_slot = lower_coordinate(coordinates, location[axis]);
                const auto upper_slot = lower_coordinate(coordinates, location[axis] + 1);
                if (lower_slot == coordinates.end() || lower_slot->first != location[axis] ||
                    upper_slot == coordinates.end() || upper_slot->first != location[axis] + 1)
                    throw std::logic_error("Cartesian source coordinate column lost an axis endpoint.");
                cell.lower[axis] = lower_slot->second.slot;
                cell.upper[axis] = upper_slot->second.slot;
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
                facts.metrics[slot] = {native_volume, center};
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
    result.ordinary_ = Catalog::make(
        std::move(sources), std::move(entries), geometry_limits(ordinary_mode, budget.limit - budget.used));
    if (transient_catalog)
        budget.add(result.ordinary_->storage_bytes());
    else
        ordinary_owners.retain(result.ordinary_, budget.used, budget.limit);
    if (separate_ordinary)
    {
        result.finite_ordinary_ = Catalog::make(std::move(finite_ordinary_sources), std::move(finite_ordinary_entries),
            geometry_limits(ordinary_mode, budget.limit - budget.used));
        if (transient_catalog)
            budget.add(result.finite_ordinary_->storage_bytes());
        else
            ordinary_owners.retain(result.finite_ordinary_, budget.used, budget.limit);
    }
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
    {
        if (state_purpose)
        {
            result.bounded_ = BoundedCatalog::make(typename BoundedCatalog::SourceTable{}, std::move(bounded_entries),
                geometry_limits(Mode, budget.limit - budget.used));
            if (transient_catalog)
                budget.add(result.bounded_->storage_bytes());
            else
                bounded_owners.retain(result.bounded_, budget.used, budget.limit);
        }
    }
    for (std::size_t slot = 0; slot < leaves.size(); ++slot)
    {
        if (reused_slots[slot])
            continue;
        const PLIC::CellGeometryView<ordinary_mode> original(result.ordinary_, slot);
        if (!original.is_procedural())
            facts.policy[slot].original_integrals = PLIC::details::cell_evaluation_cache_ready<ordinary_mode>(
                PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(original));
        if constexpr (Mode == ArithmeticMode::Ordinary)
        {
            if (state_purpose)
            {
                const auto& selected_catalog = result.finite_ordinary_ ? result.finite_ordinary_ : result.ordinary_;
                const PLIC::CellGeometryView<ordinary_mode> selected(selected_catalog, slot);
                if (!selected.is_procedural())
                    facts.policy[slot].ordinary_integrals = PLIC::details::cell_evaluation_cache_ready<ordinary_mode>(
                        PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(selected));
            }
        }
        else if constexpr (Mode == ArithmeticMode::BoundedCertification)
            if (state_purpose)
                facts.policy[slot].bounded_integrals = PLIC::details::cell_evaluation_cache_ready<Mode>(
                    PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                        PLIC::CellGeometryView<Mode>(result.bounded_, slot)));
    }
    observe_controls();
    result.controls_ = controls;
    result.stable_controls_ = stable_controls;
    result.mode_ = Mode;
    result.purpose_ = purpose;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
