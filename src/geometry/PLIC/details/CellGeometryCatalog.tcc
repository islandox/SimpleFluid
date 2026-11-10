/** @file CellGeometryCatalog.tcc
 * @brief Compiled immutable source-table publication and metadata admission.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include <map>
#include <utility>
#include <vector>
namespace SimpleFluid::PLIC
{
namespace
{
template<class T>
void reserve_catalog(std::vector<T>& buffer, std::size_t count, details::CellCatalogAdmission& admission)
{
    admission.charge(count, sizeof(T));
    buffer.reserve(count);
    if (buffer.capacity() > count)
        admission.charge(buffer.capacity() - count, sizeof(T));
}
} // namespace
template<ArithmeticMode M> std::size_t CellGeometryCatalog<M>::storage_bytes() const noexcept
{
    if (demand_)
        return sizeof(*this);
    auto bytes = sizeof(*this) + sources_.capacity() * sizeof(SourcePtr) +
                 explicit_pages_.capacity() * sizeof(typename decltype(explicit_pages_)::value_type) +
                 selection_.capacity() * sizeof(SelectionPage);
    for (const auto& page : selection_)
        bytes += page.irregular.capacity() * sizeof(Selector);
    return bytes;
}
template<ArithmeticMode M>
typename CellGeometryCatalog<M>::Selector CellGeometryCatalog<M>::select(std::size_t slot) const noexcept
{
    const auto& page = selection_[slot / page_size];
    const auto local = slot % page_size;
    return page.irregular.empty() ? Selector{page.first.owner, page.first.slot + local} : page.irregular[local];
}
template<ArithmeticMode M>
typename CellGeometryCatalog<M>::ExplicitEntry CellGeometryCatalog<M>::explicit_entry(std::size_t slot) const
{
    const auto value = physical_reference(slot);
    if (value.kind != details::geometry_demand::SourceKind::Explicit)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid shell page slot.");
    return {std::static_pointer_cast<const ExplicitPage>(value.source), value.row};
}
template<ArithmeticMode M>
void CellGeometryCatalog<M>::reserve_selection(std::size_t count, details::CellCatalogAdmission& admission)
{
    size_ = count;
    reserve_catalog(selection_, count / page_size + (count % page_size != 0), admission);
}
template<ArithmeticMode M>
void CellGeometryCatalog<M>::append_selection(std::span<const Selector> slots, details::CellCatalogAdmission& admission)
{
    const auto first = slots.front();
    SelectionPage page{first, {}};
    for (std::size_t i = 1; i < slots.size(); ++i)
        if (slots[i].owner != first.owner || slots[i].slot < first.slot || slots[i].slot - first.slot != i)
        {
            reserve_catalog(page.irregular, slots.size(), admission);
            page.irregular.assign(slots.begin(), slots.end());
            break;
        }
    selection_.push_back(std::move(page));
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::build(
    SourceTable sources, std::vector<Entry> entries, details::CellCatalogAdmission& admission, bool distinct_sources)
{
    if constexpr (M != ArithmeticMode::Ordinary)
        if (!sources.empty())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Foreign-mode procedural source catalog.");
    for (const auto& source : sources)
        if (!source)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null Cartesian source table entry.");
    const bool has_new_shells = std::any_of(entries.begin(), entries.end(),
        [](const auto& entry) { return std::holds_alternative<ConvexGeometryPtr<M>>(entry); });
    const auto repack = [&](const ExplicitEntry& entry)
    { return has_new_shells && entry.page_->cells.size() < page_size; };
    std::map<const ExplicitPage*, std::size_t> shared_pages;
    std::size_t explicit_cells = 0;
    for (const auto& entry : entries)
        if (const auto* native = std::get_if<NativeEntry>(&entry))
        {
            if (M != ArithmeticMode::Ordinary || native->source_slot >= sources.size() ||
                native->cell_slot >= sources[native->source_slot]->size())
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid procedural catalog slot.");
        }
        else if (const auto* shared = std::get_if<ExplicitEntry>(&entry))
        {
            if (!shared->page_ || shared->slot_ >= shared->page_->cells.size())
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid shared shell page slot.");
            if (repack(*shared))
                ++explicit_cells;
            else if (!shared_pages.contains(shared->page_.get()))
            {
                admission.charge(1, sizeof(typename decltype(shared_pages)::value_type) + 4 * sizeof(void*));
                shared_pages.emplace(shared->page_.get(), Selector::explicit_page);
            }
        }
        else
        {
            if (!std::get<ConvexGeometryPtr<M>>(entry))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Null materialized catalog geometry.");
            ++explicit_cells;
        }
    if (sources.size() > 1 && !distinct_sources)
    {
        std::vector<std::size_t> remap;
        admission.charge(sources.size(), sizeof(std::size_t));
        remap.reserve(sources.size());
        if (remap.capacity() > sources.size())
            admission.charge(remap.capacity() - sources.size(), sizeof(std::size_t));
        std::size_t distinct = 0;
        for (std::size_t old = 0; old < sources.size(); ++old)
        {
            std::size_t slot = 0;
            while (slot < distinct && sources[slot].get() != sources[old].get())
                ++slot;
            remap.push_back(slot);
            if (slot == distinct)
            {
                if (distinct != old)
                    sources[distinct] = std::move(sources[old]);
                ++distinct;
            }
        }
        for (auto& entry : entries)
            if (auto* native = std::get_if<NativeEntry>(&entry))
                native->source_slot = remap[native->source_slot];
        sources.resize(distinct);
    }
    auto result = std::shared_ptr<CellGeometryCatalog>(new CellGeometryCatalog);
    result->sources_ = std::move(sources);
    result->reserve_selection(entries.size(), admission);
    const auto new_pages = explicit_cells / page_size + (explicit_cells % page_size != 0);
    reserve_catalog(result->explicit_pages_, new_pages + shared_pages.size(), admission);
    std::array<Selector, page_size> selection;
    std::shared_ptr<ExplicitPage> page;
    std::size_t page_index = 0;
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        auto& entry = entries[i];
        const auto slot = i % page_size;
        if (const auto* native = std::get_if<NativeEntry>(&entry))
            selection[slot] = {native->source_slot, native->cell_slot};
        else if (const auto* shared = std::get_if<ExplicitEntry>(&entry); shared && !repack(*shared))
        {
            auto& index = shared_pages.at(shared->page_.get());
            if (index == Selector::explicit_page)
            {
                index = result->explicit_pages_.size();
                result->explicit_pages_.push_back(shared->page_);
            }
            selection[slot] = {Selector::explicit_page | index, shared->slot_};
        }
        else
        {
            if (!page || page->cells.size() == page_size)
            {
                admission.charge(1, sizeof(ExplicitPage));
                page = std::make_shared<ExplicitPage>();
                reserve_catalog(page->cells, std::min(explicit_cells, page_size), admission);
                page_index = result->explicit_pages_.size();
                result->explicit_pages_.push_back(page);
            }
            selection[slot] = {Selector::explicit_page | page_index, page->cells.size()};
            if (const auto* shared = std::get_if<ExplicitEntry>(&entry))
                page->cells.push_back(shared->page_->cells[shared->slot_]);
            else
                page->cells.push_back(std::move(std::get<ConvexGeometryPtr<M>>(entry)));
            --explicit_cells;
        }
        if (slot + 1 == page_size || i + 1 == entries.size())
            result->append_selection(std::span(selection).first(slot + 1), admission);
    }
    return result;
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::make(
    SourceTable sources, std::vector<Entry> entries, GeometryLimits limits)
{
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    admission.charge(sources.capacity(), sizeof(SourcePtr));
    admission.charge(entries.capacity(), sizeof(Entry));
    return build(std::move(sources), std::move(entries), admission);
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::make(
    SourcePtr source, std::vector<Entry> entries, GeometryLimits limits)
{
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    admission.charge(entries.capacity(), sizeof(Entry));
    SourceTable sources;
    if (source)
    {
        admission.charge(1, sizeof(SourcePtr));
        sources.reserve(1);
        if (sources.capacity() > 1)
            admission.charge(sources.capacity() - 1, sizeof(SourcePtr));
        sources.push_back(std::move(source));
    }
    return build(std::move(sources), std::move(entries), admission);
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::cartesian(
    std::shared_ptr<const CartesianCellSource> source, GeometryLimits limits)
    requires(M == ArithmeticMode::Ordinary)
{
    if (!source)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Null Cartesian source catalog.");
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    auto result = std::shared_ptr<CellGeometryCatalog>(new CellGeometryCatalog);
    reserve_catalog(result->sources_, 1, admission);
    const auto count = source->size();
    result->sources_.push_back(std::move(source));
    result->reserve_selection(count, admission);
    for (std::size_t page = 0; page < count / page_size + (count % page_size != 0); ++page)
        result->selection_.push_back({{0, page * page_size}, {}});
    return result;
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::materialized(
    std::vector<ConvexGeometryPtr<M>> cells, GeometryLimits limits)
{
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    admission.charge(cells.capacity(), sizeof(ConvexGeometryPtr<M>));
    std::vector<Entry> entries;
    admission.charge(cells.size(), sizeof(Entry));
    entries.reserve(cells.size());
    if (entries.capacity() > cells.size())
        admission.charge(entries.capacity() - cells.size(), sizeof(Entry));
    for (auto& cell : cells)
        entries.emplace_back(std::move(cell));
    return build({}, std::move(entries), admission);
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::single_existing(
    ConvexGeometryPtr<M> geometry, GeometryLimits limits)
{
    if (!geometry)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Null existing geometry owner.");
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    std::vector<Entry> entries;
    admission.charge(1, sizeof(Entry));
    entries.reserve(1);
    if (entries.capacity() > 1)
        admission.charge(entries.capacity() - 1, sizeof(Entry));
    entries.emplace_back(std::move(geometry));
    return build({}, std::move(entries), admission, true);
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::selected(
    std::shared_ptr<const CellGeometryCatalog> original, std::span<const std::size_t> slots, GeometryLimits limits)
{
    if (original && original->demand_)
        return details::CellCatalogDemandAccess::exported(std::move(original), slots, false, limits);
    if (!original)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Null geometry catalog selection source.");
    bool identity = slots.size() == original->size();
    for (std::size_t i = 0; i < slots.size(); ++i)
    {
        if (slots[i] >= original->size())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid selected geometry catalog slot.");
        identity = identity && slots[i] == i;
    }
    if (identity)
        return original;
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    std::map<std::size_t, std::size_t> owners;
    std::size_t source_count = 0, page_count = 0;
    for (const auto slot : slots)
    {
        const auto entry = original->select(slot);
        if (owners.contains(entry.owner))
            continue;
        admission.charge(1, sizeof(typename decltype(owners)::value_type) + 4 * sizeof(void*));
        const auto index = entry.is_explicit() ? Selector::explicit_page | page_count++ : source_count++;
        owners.emplace(entry.owner, index);
    }
    auto result = std::shared_ptr<CellGeometryCatalog>(new CellGeometryCatalog);
    reserve_catalog(result->sources_, source_count, admission);
    reserve_catalog(result->explicit_pages_, page_count, admission);
    result->sources_.resize(source_count);
    result->explicit_pages_.resize(page_count);
    for (const auto& [old, next] : owners)
    {
        const Selector from{old, 0}, to{next, 0};
        if (from.is_explicit())
            result->explicit_pages_[to.index()] = original->explicit_pages_[from.index()];
        else
            result->sources_[to.index()] = original->sources_[from.index()];
    }
    result->reserve_selection(slots.size(), admission);
    std::array<Selector, page_size> selection;
    for (std::size_t i = 0; i < slots.size(); ++i)
    {
        auto entry = original->select(slots[i]);
        entry.owner = owners.at(entry.owner);
        const auto slot = i % page_size;
        selection[slot] = entry;
        if (slot + 1 == page_size || i + 1 == slots.size())
            result->append_selection(std::span(selection).first(slot + 1), admission);
    }
    return result;
}
template<ArithmeticMode M>
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::reordered(
    std::shared_ptr<const CellGeometryCatalog> original, std::span<const std::size_t> permutation,
    GeometryLimits limits)
{
    if (original && original->demand_)
        return details::CellCatalogDemandAccess::exported(std::move(original), permutation, true, limits);
    if (!original || permutation.size() != original->size())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry catalog permutation size.");
    details::CellCatalogAdmission admission(limits);
    admission.charge(1, sizeof(CellGeometryCatalog));
    std::vector<std::uint8_t> seen;
    admission.charge(permutation.size(), sizeof(std::uint8_t));
    seen.resize(permutation.size());
    if (seen.capacity() > permutation.size())
        admission.charge(seen.capacity() - permutation.size(), sizeof(std::uint8_t));
    for (const auto index : permutation)
    {
        if (index >= original->size() || seen[index])
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry catalog permutation is not one-to-one.");
        seen[index] = 1;
    }
    auto result = std::shared_ptr<CellGeometryCatalog>(new CellGeometryCatalog);
    reserve_catalog(result->sources_, original->sources_.size(), admission);
    result->sources_ = original->sources_;
    reserve_catalog(result->explicit_pages_, original->explicit_pages_.size(), admission);
    result->explicit_pages_ = original->explicit_pages_;
    result->reserve_selection(permutation.size(), admission);
    std::array<Selector, page_size> selection;
    for (std::size_t i = 0; i < permutation.size(); ++i)
    {
        const auto slot = i % page_size;
        selection[slot] = original->select(permutation[i]);
        if (slot + 1 == page_size || i + 1 == permutation.size())
            result->append_selection(std::span(selection).first(slot + 1), admission);
    }
    return result;
}
} // namespace SimpleFluid::PLIC
