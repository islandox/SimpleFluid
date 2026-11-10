/** @file CellGeometryDemandCatalog.tcc
 * @brief Compiled typed resolvers and source-only demand catalog state.
 */
#pragma once
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <utility>

namespace SimpleFluid::PLIC
{
template<ArithmeticMode M> struct CellGeometryCatalog<M>::DemandState
{
    details::geometry_demand::GeometryDemandOrder order;
    details::geometry_demand::GeometryOwnerManifest native_sources;
    std::shared_ptr<const ExplicitPage> tail;
    std::vector<details::geometry_demand::OrderStamp> tail_stamps;

    std::size_t exclusive_payload_bytes() const noexcept
    {
        return sizeof(*this) + tail_stamps.capacity() * sizeof(tail_stamps[0]);
    }
};
template<ArithmeticMode M> std::size_t CellGeometryCatalog<M>::source_count() const noexcept
{
    return demand_ ? demand_->native_sources.owner_count() : sources_.size();
}
template<ArithmeticMode M>
details::geometry_demand::SourceSlot CellGeometryCatalog<M>::physical_reference(std::size_t slot) const
{
    using namespace details::geometry_demand;
    if (slot >= size_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid persistent catalog source slot.");
    if (demand_)
        return demand_->order.at(slot).physical;
    const auto selected = select(slot);
    if (selected.is_explicit())
        return {explicit_pages_[selected.index()], selected.slot, SourceKind::Explicit};
    return {sources_[selected.index()], selected.slot, SourceKind::Cartesian};
}
template<ArithmeticMode M> bool CellGeometryCatalog<M>::procedural_slot(std::size_t slot) const
{
    return demand_ ? demand_->order.at(slot).physical.kind == details::geometry_demand::SourceKind::Cartesian
                   : !select(slot).is_explicit();
}
template<ArithmeticMode M> const ConvexGeometryPtr<M>& CellGeometryCatalog<M>::shell_geometry(std::size_t slot) const
{
    const auto selected = physical_reference(slot);
    if (selected.kind != details::geometry_demand::SourceKind::Explicit)
        throw std::bad_variant_access();
    const auto* page = static_cast<const ExplicitPage*>(selected.source.get());
    return page->cells.at(selected.row);
}
template<ArithmeticMode M>
const typename CellGeometryCatalog<M>::SourcePtr& CellGeometryCatalog<M>::source_owner(std::size_t slot) const
{
    if (demand_)
        return demand_->order.native_owner(slot);
    return sources_[select(slot).index()];
}
template<ArithmeticMode M> std::size_t CellGeometryCatalog<M>::physical_row(std::size_t slot) const
{
    return demand_ ? demand_->order.at(slot).physical.row : select(slot).slot;
}
namespace details
{
namespace catalog_demand_detail
{
geometry_demand::Admission admission(CellCatalogAdmission& value)
{
    return {&value,
        [](void* context, std::size_t bytes) { static_cast<CellCatalogAdmission*>(context)->charge(bytes, 1); }};
}
std::size_t native_body_bytes(const std::shared_ptr<const void>& owner)
{
    const auto bytes =
        NativeCellGeometrySourceAccess::exclusive_bytes(*std::static_pointer_cast<const CartesianCellSource>(owner));
    return geometry_demand::checked_sum(geometry_demand::checked_sum(bytes[0], bytes[1]), bytes[2]);
}
} // namespace catalog_demand_detail
template<ArithmeticMode M> bool CellCatalogDemandAccess::persistent(const CellGeometryCatalog<M>& catalog) noexcept
{
    return bool(catalog.demand_);
}
template<ArithmeticMode M>
geometry_demand::OrderStamp CellCatalogDemandAccess::stamp(const CellGeometryCatalog<M>& catalog, std::size_t slot)
{
    if (!catalog.demand_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Catalog has no persistent demand order.");
    return catalog.demand_->order.at(slot).stamp;
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellCatalogDemandAccess::initialize(
    std::shared_ptr<const CellGeometryCatalog<M>> original, GeometryLimits limits)
{
    using Catalog = CellGeometryCatalog<M>;
    namespace G = geometry_demand;
    if (!original)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Missing persistent catalog initialization source.");
    if (original->demand_)
        return original;
    CellCatalogAdmission memory(limits);
    const auto admitted = catalog_demand_detail::admission(memory);
    memory.charge(1, sizeof(Catalog));
    memory.charge(1, sizeof(typename Catalog::DemandState) + G::control_bytes);
    auto state = std::make_shared<typename Catalog::DemandState>();
    std::vector<G::SourceSlot> slots;
    memory.charge(original->size(), sizeof(G::SourceSlot));
    slots.reserve(original->size());
    if (slots.capacity() != original->size())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified catalog demand capture capacity.");
    std::shared_ptr<const typename Catalog::ExplicitPage> tail;
    for (std::size_t slot = 0; slot < original->size(); ++slot)
    {
        auto value = original->physical_reference(slot);
        if (value.kind == G::SourceKind::Cartesian)
            state->native_sources = state->native_sources.retained(
                0, 0, value.source, catalog_demand_detail::native_body_bytes(value.source), admitted);
        else
        {
            auto page = std::static_pointer_cast<const typename Catalog::ExplicitPage>(value.source);
            if (page->cells.size() < G::leaf_capacity)
                tail = std::move(page);
        }
        slots.push_back(std::move(value));
    }
    state->order = G::GeometryDemandOrder{}.prepended(slots, {}, admitted);
    if (tail)
    {
        memory.charge(tail->cells.size(), sizeof(G::OrderStamp));
        state->tail_stamps.resize(tail->cells.size());
        if (state->tail_stamps.capacity() != tail->cells.size())
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified catalog tail stamp capacity.");
        bool unique = true;
        for (std::size_t slot = 0; slot < slots.size(); ++slot)
            if (slots[slot].source.get() == tail.get())
            {
                auto& stamp = state->tail_stamps.at(slots[slot].row);
                unique = unique && stamp.generation == 0;
                stamp = state->order.at(slot).stamp;
            }
        if (unique && std::all_of(state->tail_stamps.begin(), state->tail_stamps.end(),
                          [](const auto& value) { return value.generation != 0; }))
            state->tail = std::move(tail);
        else
            std::vector<G::OrderStamp>().swap(state->tail_stamps);
    }
    auto result = std::shared_ptr<Catalog>(new Catalog);
    result->size_ = state->order.size();
    result->demand_ = std::move(state);
    return result;
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellCatalogDemandAccess::extended(
    std::shared_ptr<const CellGeometryCatalog<M>> accepted, std::shared_ptr<const CellGeometryCatalog<M>> requested,
    std::span<const std::optional<std::size_t>> previous_slots, GeometryLimits limits, CellCatalogDemandWork* measured)
{
    using Catalog = CellGeometryCatalog<M>;
    namespace G = geometry_demand;
    if (!accepted || !accepted->demand_ || !requested || previous_slots.size() != requested->size())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid persistent catalog demand sources.");
    if (!requested->size())
        return accepted;
    const auto& old = *accepted->demand_;
    if (old.order.generation() == std::numeric_limits<std::uint64_t>::max())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Persistent catalog generation overflows.");
    for (const auto& slot : previous_slots)
        if (slot && *slot >= accepted->size())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Persistent catalog demand ordinal is absent.");

    CellCatalogAdmission memory(limits);
    struct Measure
    {
        CellCatalogAdmission& memory;
        CellCatalogDemandWork* output;
        G::PageWork pages;
        ~Measure()
        {
            if (output)
            {
                output->cumulative_metadata_bytes = memory.used_bytes();
                output->pages = pages;
            }
        }
    } measure{memory, measured, {}};
    const auto admitted = catalog_demand_detail::admission(memory);
    const auto reserve = [&]<class T>(std::vector<T>& target, std::size_t count)
    {
        memory.charge(count, sizeof(T));
        target.reserve(count);
        if (target.capacity() != count)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified catalog demand buffer capacity.");
    };
    std::vector<G::SourceSlot> prefix;
    reserve(prefix, requested->size());
    std::vector<G::OrderStamp> retired;
    const auto retired_count =
        std::count_if(previous_slots.begin(), previous_slots.end(), [](const auto& value) { return bool(value); });
    reserve(retired, retired_count);
    memory.charge(1, sizeof(Catalog));
    memory.charge(1, sizeof(typename Catalog::DemandState) + G::control_bytes);
    auto state = std::make_shared<typename Catalog::DemandState>();
    state->native_sources = old.native_sources;

    bool changes_tail = false;
    std::vector<std::uint8_t> pack;
    reserve(pack, requested->size());
    for (std::size_t slot = 0; slot < requested->size(); ++slot)
    {
        auto current = requested->physical_reference(slot);
        bool same = false;
        if (const auto prior = previous_slots[slot])
        {
            const auto previous = accepted->physical_reference(*prior);
            retired.push_back(old.order.at(*prior).stamp);
            changes_tail = changes_tail || (old.tail && previous.kind == G::SourceKind::Explicit &&
                                               previous.source.get() == old.tail.get());
            if (previous.kind == G::SourceKind::Cartesian)
                state->native_sources = state->native_sources.retired(0, 0, previous.source, admitted, &measure.pages);
            if (current.kind == previous.kind)
            {
                if (current.kind == G::SourceKind::Cartesian)
                    same = current.row == previous.row && current.source.get() == previous.source.get() &&
                           !current.source.owner_before(previous.source) &&
                           !previous.source.owner_before(current.source);
                else
                {
                    const auto* first = static_cast<const typename Catalog::ExplicitPage*>(current.source.get());
                    const auto* second = static_cast<const typename Catalog::ExplicitPage*>(previous.source.get());
                    const auto& a = first->cells.at(current.row);
                    const auto& b = second->cells.at(previous.row);
                    same = a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
                }
            }
            if (same)
                current = previous;
        }
        if (current.kind == G::SourceKind::Cartesian)
            state->native_sources = state->native_sources.retained(0, 0, current.source,
                catalog_demand_detail::native_body_bytes(current.source), admitted, &measure.pages);
        const bool partial =
            current.kind == G::SourceKind::Explicit && (!same || (old.tail && current.source.get() == old.tail.get()));
        pack.push_back(std::uint8_t(partial));
        changes_tail = changes_tail || partial;
        prefix.push_back(std::move(current));
    }
    // Validate duplicate old ordinals before constructing replacement shell pages.
    {
        std::vector<G::OrderStamp> unique;
        reserve(unique, retired.size());
        unique.assign(retired.begin(), retired.end());
        const auto before = [](const auto& a, const auto& b)
        { return a.generation != b.generation ? a.generation > b.generation : a.position < b.position; };
        std::sort(unique.begin(), unique.end(), before);
        if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Persistent catalog demands a duplicate old cell.");
    }

    struct ExplicitCell
    {
        ConvexGeometryPtr<M> geometry;
        G::OrderStamp stamp;
        std::size_t prefix = std::numeric_limits<std::size_t>::max();
    };
    std::vector<ExplicitCell> cells;
    std::vector<G::OrderedSourceSlot> rebound;
    if (changes_tail)
    {
        const auto old_count = old.tail ? old.tail->cells.size() : 0;
        const auto added = std::count(pack.begin(), pack.end(), std::uint8_t{1});
        reserve(cells, G::checked_sum(old_count, added));
        reserve(rebound, old_count);
        for (std::size_t row = 0; row < old_count; ++row)
            if (std::find(retired.begin(), retired.end(), old.tail_stamps[row]) == retired.end())
                cells.push_back({old.tail->cells[row], old.tail_stamps[row]});
        const auto generation = old.order.generation() + 1;
        for (std::size_t slot = 0; slot < prefix.size(); ++slot)
            if (pack[slot])
            {
                const auto* page = static_cast<const typename Catalog::ExplicitPage*>(prefix[slot].source.get());
                cells.push_back({page->cells.at(prefix[slot].row), {generation, slot}, slot});
            }
        std::size_t offset = 0;
        while (offset < cells.size())
        {
            const auto count = std::min(G::leaf_capacity, cells.size() - offset);
            memory.charge(1, sizeof(typename Catalog::ExplicitPage));
            auto page = std::make_shared<typename Catalog::ExplicitPage>();
            reserve(page->cells, count);
            for (std::size_t row = 0; row < count; ++row)
                page->cells.push_back(cells[offset + row].geometry);
            for (std::size_t row = 0; row < count; ++row)
            {
                const auto& value = cells[offset + row];
                G::SourceSlot physical{page, row, G::SourceKind::Explicit};
                if (value.prefix == std::numeric_limits<std::size_t>::max())
                    rebound.push_back({value.stamp, std::move(physical)});
                else
                    prefix[value.prefix] = std::move(physical);
            }
            if (count < G::leaf_capacity)
            {
                state->tail = page;
                reserve(state->tail_stamps, count);
                for (std::size_t row = 0; row < count; ++row)
                    state->tail_stamps.push_back(cells[offset + row].stamp);
            }
            offset += count;
        }
    }
    else if (old.tail)
    {
        state->tail = old.tail;
        reserve(state->tail_stamps, old.tail_stamps.size());
        state->tail_stamps.assign(old.tail_stamps.begin(), old.tail_stamps.end());
    }
    state->order = old.order.prepended(prefix, retired, admitted, &measure.pages);
    if (!rebound.empty())
        state->order = state->order.rebound(rebound, admitted, &measure.pages);
    auto result = std::shared_ptr<Catalog>(new Catalog);
    result->size_ = state->order.size();
    result->demand_ = std::move(state);
    return result;
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellCatalogDemandAccess::exported(
    std::shared_ptr<const CellGeometryCatalog<M>> original, std::span<const std::size_t> slots, bool permutation,
    GeometryLimits limits)
{
    using Catalog = CellGeometryCatalog<M>;
    namespace G = geometry_demand;
    if (!original || !original->demand_ || (permutation && slots.size() != original->size()))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid persistent catalog export source.");
    bool identity = slots.size() == original->size();
    for (std::size_t slot = 0; slot < slots.size(); ++slot)
    {
        if (slots[slot] >= original->size())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid persistent catalog export ordinal.");
        identity = identity && slots[slot] == slot;
    }
    if (identity && !permutation)
        return original;
    CellCatalogAdmission memory(limits);
    memory.charge(1, sizeof(Catalog));
    if (permutation)
    {
        std::vector<std::uint8_t> seen;
        memory.charge(slots.size(), sizeof(std::uint8_t));
        seen.resize(slots.size());
        if (seen.capacity() != slots.size())
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified catalog permutation scratch capacity.");
        for (const auto slot : slots)
            if (std::exchange(seen[slot], 1))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Catalog permutation is not one-to-one.");
    }
    using Source = typename Catalog::SourcePtr;
    using OwnerTable = std::map<const void*, std::pair<Source, std::size_t>>;
    OwnerTable owners;
    for (const auto slot : slots)
    {
        const auto value = original->physical_reference(slot);
        if (value.kind != G::SourceKind::Cartesian)
            continue;
        const auto source = std::static_pointer_cast<const CartesianCellSource>(value.source);
        const auto found = owners.find(source.get());
        if (found != owners.end())
        {
            if (found->second.first.owner_before(source) || source.owner_before(found->second.first))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Catalog export source has conflicting controls.");
            continue;
        }
        memory.charge(1, sizeof(typename OwnerTable::value_type) + 4 * sizeof(void*));
        owners.emplace(source.get(), std::pair{source, owners.size()});
    }
    typename Catalog::SourceTable sources;
    memory.charge(owners.size(), sizeof(Source));
    sources.resize(owners.size());
    if (sources.capacity() != owners.size())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified catalog export source capacity.");
    for (const auto& [owner, record] : owners)
        sources[record.second] = record.first;
    std::vector<typename Catalog::Entry> entries;
    memory.charge(slots.size(), sizeof(typename Catalog::Entry));
    entries.reserve(slots.size());
    if (entries.capacity() != slots.size())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified catalog export entry capacity.");
    for (const auto slot : slots)
    {
        const auto value = original->physical_reference(slot);
        if (value.kind == G::SourceKind::Cartesian)
            entries.emplace_back(NativeCellGeometrySlot{owners.at(value.source.get()).second, value.row});
        else
            entries.emplace_back(original->explicit_entry(slot));
    }
    return Catalog::build(std::move(sources), std::move(entries), memory, true);
}
template<ArithmeticMode M>
void CellCatalogDemandAccess::visit_borrowed_dependencies(const CellGeometryCatalog<M>& catalog,
    geometry_demand::BorrowedMetadataSink metadata, geometry_demand::BorrowedSourceSink sources)
{
    namespace G = geometry_demand;
    if (!catalog.demand_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Catalog has no persistent dependency root.");
    const auto& state = catalog.demand_;
    if (!metadata(state.get(), nullptr, state->exclusive_payload_bytes()))
        return;
    state->order.visit_borrowed_dependencies(metadata, sources, state.get());
    state->native_sources.visit_borrowed_dependencies(metadata,
        {&sources, [](void* context, const void* parent, const G::OwnerRecord& row)
            { (*static_cast<G::BorrowedSourceSink*>(context))(parent, G::SourceKind::Cartesian, row.owner.get()); }},
        state.get());
    if (state->tail)
        sources(state.get(), G::SourceKind::Explicit, state->tail.get());
}
template<ArithmeticMode M>
void CellCatalogDemandAccess::visit_dependencies(
    const CellGeometryCatalog<M>& catalog, geometry_demand::MetadataSink metadata, geometry_demand::SourceSink sources)
{
    namespace G = geometry_demand;
    if (!catalog.demand_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Catalog has no persistent dependency root.");
    const auto& state = catalog.demand_;
    const auto payload = state->exclusive_payload_bytes();
    if (!metadata(state, payload, payload + G::control_bytes))
        return;
    state->order.visit_dependencies(metadata, sources);
    state->native_sources.visit_dependencies(
        metadata, {&sources, [](void* context, const G::OwnerRecord& row)
                      { (*static_cast<G::SourceSink*>(context))(G::SourceKind::Cartesian, row.owner); }});
    if (state->tail)
        sources(G::SourceKind::Explicit, state->tail);
}
} // namespace details
} // namespace SimpleFluid::PLIC
