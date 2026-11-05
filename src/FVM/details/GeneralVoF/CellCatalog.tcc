/** @file CellCatalog.tcc
 * @brief FVM ordering adapters over mesh-owned immutable geometry sources.
 */
#pragma once
#include "FVM/details/GeneralVoF/CellCatalogInternal.hh"
#include <type_traits>
#include "geometry/PLIC/details/CellGeometryEvaluation.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include <map>
namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
{
namespace
{
template<class T> void reserve_checked(std::vector<T>& values, std::size_t count, wire::Budget& budget)
{
    charge_items(budget, count, sizeof(T));
    values.reserve(count);
    if (values.capacity() > count)
        charge_items(budget, values.capacity() - count, sizeof(T));
}
template<ArithmeticMode Mode>
using EvaluationOwners = std::map<const PLIC::ConvexCellGeometry<Mode>*, std::array<PLIC::ConvexGeometryPtr<Mode>, 2>>;
template<ArithmeticMode Mode>
bool same_source_owner(const PLIC::CellGeometryView<Mode>& first, const PLIC::CellGeometryView<Mode>& second)
{
    if (first.is_procedural() != second.is_procedural())
        return false;
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (first.is_procedural())
        {
            const auto a = PLIC::details::NativeCellGeometrySourceAccess::source_cell(first);
            const auto b = PLIC::details::NativeCellGeometrySourceAccess::source_cell(second);
            return a.source == b.source && a.cell_slot == b.cell_slot;
        }
    return PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(first) ==
           PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(second);
}
template<ArithmeticMode Mode>
std::vector<const Cell*> reusable_evaluations(const GeneralVoF::Hierarchy& hierarchy,
    std::span<const AMR::AdaptiveLeaf* const> leaves, std::span<const std::size_t> slots,
    const AMR::AdaptiveGeometrySnapshot& source, const GeneralVoF::Data& reuse,
    const CatalogConstructionControls& controls, wire::Budget& budget)
{
    std::vector<const Cell*> result;
    reserve_checked(result, leaves.size(), budget);
    result.resize(leaves.size(), nullptr);
    if (!reuse.topology->geometry_snapshot)
        return result;
    const auto& previous = *reuse.topology->geometry_snapshot;
    if (previous.arithmetic_mode() != Mode || source.arithmetic_mode() != Mode)
        return result;
    for (std::size_t slot = 0; slot < leaves.size(); ++slot)
    {
        if (construction_controls() != controls)
            break;
        const auto& leaf = *leaves[slot];
        const auto owned = hierarchy.owned().find(leaf.key);
        if (owned == hierarchy.owned().end())
            continue;
        const auto& original = owned->second;
        const auto found = reuse.cells.find(leaf.key);
        if (found == reuse.cells.end() || found->second.revision != leaf.revision ||
            found->second.owner != hierarchy.communicator()->getRank() ||
            !(same_native_recipe(original, leaf) ||
                (leaf.geometry && original.geometry == leaf.geometry && original.key == leaf.key &&
                    original.revision == leaf.revision && original.material == leaf.material &&
                    original.domain == leaf.domain && original.provider == leaf.provider &&
                    original.native_cell == leaf.native_cell && original.native_recipe == leaf.native_recipe)))
            continue;
        const auto old_slot = previous.find(original);
        if (!old_slot ||
            !same_source_owner(PLIC::CellGeometryView<ArithmeticMode::Ordinary>(previous.ordinary_catalog(), *old_slot),
                PLIC::CellGeometryView<ArithmeticMode::Ordinary>(source.ordinary_catalog(), slots[slot])))
            continue;
        const auto& old_policy = previous.evaluation_policy(*old_slot);
        const auto& new_policy = source.evaluation_policy(slots[slot]);
        if (old_policy.original_integrals != new_policy.original_integrals)
            continue;
        if constexpr (Mode == ArithmeticMode::Ordinary)
        {
            const auto& old_selected =
                previous.finite_ordinary_catalog() ? previous.finite_ordinary_catalog() : previous.ordinary_catalog();
            const auto& new_selected =
                source.finite_ordinary_catalog() ? source.finite_ordinary_catalog() : source.ordinary_catalog();
            if (!same_source_owner(PLIC::CellGeometryView<Mode>(old_selected, *old_slot),
                    PLIC::CellGeometryView<Mode>(new_selected, slots[slot])) ||
                old_policy.ordinary_integrals != new_policy.ordinary_integrals)
                continue;
        }
        else if constexpr (Mode == ArithmeticMode::BoundedCertification)
            if (!same_source_owner(PLIC::CellGeometryView<Mode>(previous.bounded_catalog(), *old_slot),
                    PLIC::CellGeometryView<Mode>(source.bounded_catalog(), slots[slot])) ||
                old_policy.bounded_integrals != new_policy.bounded_integrals)
                continue;
        result[slot] = &found->second;
    }
    return result;
}
template<ArithmeticMode Mode>
std::shared_ptr<const PLIC::CellGeometryCatalog<Mode>> evaluation_catalog(
    const std::shared_ptr<const PLIC::CellGeometryCatalog<Mode>>& source, std::span<const std::size_t> slots,
    std::span<const AMR::AdaptiveLeaf* const> leaves, const AMR::AdaptiveGeometrySnapshot& snapshot,
    bool original_policy, wire::Budget& budget, PLIC::GeometryAllocationLedger& owners,
    EvaluationOwners<Mode>& evaluations, std::span<const Cell* const> reused_cells, const GeneralVoF::Data* reuse,
    const CatalogConstructionControls& controls)
{
    using Catalog = PLIC::CellGeometryCatalog<Mode>;
    typename Catalog::SourceTable sources;
    std::vector<typename Catalog::Entry> entries;
    reserve_checked(sources, slots.size(), budget);
    reserve_checked(entries, slots.size(), budget);
    std::map<const PLIC::CartesianCellSource*, std::size_t> source_slots;
    for (std::size_t index = 0; index < slots.size(); ++index)
    {
        const PLIC::CellGeometryView<Mode> view(source, slots[index]);
        if constexpr (Mode == ArithmeticMode::Ordinary)
            if (view.is_procedural())
            {
                const auto ref = PLIC::details::NativeCellGeometrySourceAccess::source_cell(view);
                auto found = source_slots.find(ref.source.get());
                if (found == source_slots.end())
                {
                    charge_items(budget, 1, sizeof(typename decltype(source_slots)::value_type) + 4 * sizeof(void*));
                    found = source_slots.emplace(ref.source.get(), sources.size()).first;
                    sources.push_back(ref.source);
                }
                entries.emplace_back(PLIC::NativeCellGeometrySlot{found->second, ref.cell_slot});
                continue;
            }
        if (!reused_cells.empty() && reused_cells[index] && construction_controls() == controls)
        {
            PLIC::ConvexGeometryPtr<Mode> evaluation;
            if constexpr (Mode == ArithmeticMode::Ordinary)
                evaluation = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                    original_policy ? native_cell_geometry(*reuse, *reused_cells[index])
                                    : cell_geometry<Mode>(*reuse, *reused_cells[index]));
            else
                evaluation = existing_cell_geometry<Mode>(*reuse, *reused_cells[index]);
            owners.retain(evaluation, budget.used, budget.limit);
            entries.emplace_back(std::move(evaluation));
            continue;
        }
        auto geometry = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(view);
        bool share_explicit_original = false;
        if constexpr (Mode == ArithmeticMode::Ordinary)
            share_explicit_original = original_policy && leaves[index]->geometry && geometry == leaves[index]->geometry;
        if (!share_explicit_original)
        {
            const auto& policy = snapshot.evaluation_policy(slots[index]);
            const bool copy_integrals = original_policy                    ? policy.original_integrals
                                        : Mode == ArithmeticMode::Ordinary ? policy.ordinary_integrals
                                                                           : policy.bounded_integrals;
            auto found = evaluations.find(geometry.get());
            if (found == evaluations.end())
            {
                charge_items(budget, 1,
                    sizeof(typename std::remove_cvref_t<decltype(evaluations)>::value_type) + 4 * sizeof(void*));
                found = evaluations.emplace(geometry.get(), std::array<PLIC::ConvexGeometryPtr<Mode>, 2>{}).first;
            }
            auto& evaluation = found->second[copy_integrals ? 1 : 0];
            if (!evaluation)
            {
                evaluation = PLIC::details::clone_cell_evaluation<Mode>(
                    geometry, copy_integrals, geometry_limits(Mode, budget.limit - budget.used));
                owners.retain(evaluation, budget.used, budget.limit);
            }
            geometry = evaluation;
        }
        entries.emplace_back(std::move(geometry));
    }
    auto result =
        Catalog::make(std::move(sources), std::move(entries), geometry_limits(Mode, budget.limit - budget.used));
    owners.retain(result, budget.used, budget.limit);
    return result;
}
} // namespace
template<ArithmeticMode Mode>
CatalogBuild build(const GeneralVoF::Hierarchy& hierarchy, std::span<const AMR::AdaptiveLeaf* const> leaves,
    wire::Budget& budget, const GeneralVoF::Data* reuse, CatalogPurpose purpose,
    PLIC::GeometryAllocationLedger* admitted_originals, const CatalogReuseAuthority* reuse_authority,
    const void* construction_operation, const AMR::AdaptiveNativeDraft* native_draft,
    const AMR::HierarchyObservation* source_observation)
{
    CatalogBuild result;
    PLIC::GeometryAllocationLedger local_ordinary, bounded;
    auto& ordinary = admitted_originals ? *admitted_originals : local_ordinary;
    const auto controls = construction_controls();
    if (purpose != CatalogPurpose::State)
        reuse = nullptr;
    if (reuse)
    {
        if (reuse->hierarchy.get() != &hierarchy || reuse->options.arithmetic_mode != Mode)
            throw std::invalid_argument("Native catalog reuse lacks matching hierarchy/arithmetic authority.");
        // Prior evaluations stay admitted even when controls or operation proof
        // prohibit reuse, preserving old/new overlap and spent prefixes.
        if (reuse->topology->geometry_snapshot)
            reuse->topology->geometry_snapshot->retain(ordinary, bounded, budget.used, budget.limit);
        ordinary.retain(reuse->topology->cell_geometry, budget.used, budget.limit);
        if (reuse->topology->ordinary_cell_geometry)
            ordinary.retain(reuse->topology->ordinary_cell_geometry, budget.used, budget.limit);
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            bounded.retain(reuse->topology->bounded_cell_geometry, budget.used, budget.limit);
    }
    const bool reuse_bindings = reuse && construction_operation && reuse_authority && reuse_authority->stable &&
                                reuse_authority->operation == construction_operation &&
                                reuse_authority->original_catalog == reuse->topology->cell_geometry.get() &&
                                reuse_authority->source_snapshot == reuse->topology->geometry_snapshot.get() &&
                                controls.supported && reuse_authority->controls == controls;
    const auto snapshot_purpose = purpose == CatalogPurpose::State ? AMR::GeometrySnapshotPurpose::State
                                                                   : AMR::GeometrySnapshotPurpose::OriginalSources;
    if (native_draft)
    {
        if (!source_observation)
            throw std::invalid_argument("Native geometry catalog lacks its authenticated source observation.");
        result.snapshot = AMR::adaptive_detail::GeometrySnapshotAccess::native_draft(hierarchy, *native_draft, leaves,
            *source_observation, Mode, snapshot_purpose, budget.used, budget.limit, ordinary, bounded);
    }
    else
        result.snapshot =
            hierarchy.geometry_snapshot(leaves, Mode, snapshot_purpose, budget.used, budget.limit, &ordinary, &bounded);
    const auto& source = *result.snapshot;
    std::vector<std::size_t> slots;
    reserve_checked(slots, leaves.size(), budget);
    if (purpose == CatalogPurpose::State)
        reserve_checked(result.native_metrics, leaves.size(), budget);
    for (const auto* leaf : leaves)
    {
        const auto slot = source.find(*leaf);
        if (!slot)
            throw std::invalid_argument("Mesh geometry snapshot lacks its selected authenticated binding.");
        slots.push_back(*slot);
        if (purpose == CatalogPurpose::State)
            result.native_metrics.push_back(source.native_metrics().at(*slot));
    }
    std::vector<const Cell*> reused_cells;
    if (reuse_bindings && construction_controls() == controls)
        reused_cells = reusable_evaluations<Mode>(hierarchy, leaves, slots, source, *reuse, controls, budget);
    EvaluationOwners<ArithmeticMode::Ordinary> ordinary_evaluations;
    result.ordinary = evaluation_catalog<ArithmeticMode::Ordinary>(source.ordinary_catalog(), slots, leaves, source,
        true, budget, ordinary, ordinary_evaluations, reused_cells, reuse, controls);
    if (source.finite_ordinary_catalog())
        result.finite_ordinary = evaluation_catalog<ArithmeticMode::Ordinary>(source.finite_ordinary_catalog(), slots,
            leaves, source, false, budget, ordinary, ordinary_evaluations, reused_cells, reuse, controls);
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        if (source.bounded_catalog())
        {
            EvaluationOwners<Mode> bounded_evaluations;
            result.bounded = evaluation_catalog<Mode>(source.bounded_catalog(), slots, leaves, source, false, budget,
                bounded, bounded_evaluations, reused_cells, reuse, controls);
        }
    result.reuse_authority = {construction_operation, result.ordinary.get(), result.snapshot.get(), controls,
        controls.supported && source.stable_controls() && construction_controls() == controls};
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
