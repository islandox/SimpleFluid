/** @file CellGeometryCatalog.tcc
 * @brief Compiled immutable source-table publication and metadata admission.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include <utility>
#include <vector>
namespace SimpleFluid::PLIC
{
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
    for (const auto& entry : entries)
        if (const auto* native = std::get_if<NativeEntry>(&entry))
        {
            if (M != ArithmeticMode::Ordinary || native->source_slot >= sources.size() ||
                native->cell_slot >= sources[native->source_slot]->size())
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid procedural catalog slot.");
        }
        else if (!std::get<ConvexGeometryPtr<M>>(entry))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null materialized catalog geometry.");
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
    result->entries_ = std::move(entries);
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
    SourceTable sources;
    admission.charge(1, sizeof(SourcePtr));
    sources.reserve(1);
    if (sources.capacity() > 1)
        admission.charge(sources.capacity() - 1, sizeof(SourcePtr));
    const auto count = source->size();
    sources.push_back(std::move(source));
    std::vector<Entry> entries;
    admission.charge(count, sizeof(Entry));
    entries.reserve(count);
    if (entries.capacity() > count)
        admission.charge(entries.capacity() - count, sizeof(Entry));
    for (std::size_t slot = 0; slot < count; ++slot)
        entries.emplace_back(NativeEntry{0, slot});
    return build(std::move(sources), std::move(entries), admission);
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
std::shared_ptr<const CellGeometryCatalog<M>> CellGeometryCatalog<M>::reordered(
    std::shared_ptr<const CellGeometryCatalog> original, std::span<const std::size_t> permutation,
    GeometryLimits limits)
{
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
    SourceTable sources;
    admission.charge(original->sources_.size(), sizeof(SourcePtr));
    sources.reserve(original->sources_.size());
    if (sources.capacity() > original->sources_.size())
        admission.charge(sources.capacity() - original->sources_.size(), sizeof(SourcePtr));
    for (const auto& source : original->sources_)
        sources.push_back(source);
    std::vector<Entry> entries;
    admission.charge(permutation.size(), sizeof(Entry));
    entries.reserve(permutation.size());
    if (entries.capacity() > permutation.size())
        admission.charge(entries.capacity() - permutation.size(), sizeof(Entry));
    for (const auto index : permutation)
        entries.push_back(original->entries_[index]);
    return build(std::move(sources), std::move(entries), admission, true);
}
} // namespace SimpleFluid::PLIC
