/** @file FieldStored.ipp
 * @brief Caller-owned descriptor construction, indexed access and generic layout acquisition.
 */
#pragma once

#include "fields/FieldStored.hh"
#include <Teuchos_CommHelpers.hpp>

namespace SimpleFluid
{

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline FieldStored<FieldType, Pack, MeshType>::FieldStored(SP<const mesh_type> mesh, std::string name, bool zero_out)
    : FieldStored(descriptor_type(std::move(name)), std::move(mesh), zero_out)
{
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline FieldStored<FieldType, Pack, MeshType>::FieldStored(
    SP<const mesh_type> mesh, const value_type& initial_value, std::string name)
    : FieldStored(descriptor_type(std::move(name)), std::move(mesh), initial_value)
{
}

template<class Pack, class MeshType, class Location>
template<class MakeIds>
inline auto detail::StoredFieldLayout<Pack, MeshType, Location>::acquire(const MeshType& mesh,
    const Teuchos::RCP<const map_type>& owned, const Teuchos::RCP<const map_type>& overlap, MakeIds make_ids)
    -> std::shared_ptr<const StoredFieldLayout>
{
    const std::lock_guard lock(mutex);
    std::shared_ptr<const StoredFieldLayout> cached;
    for (auto it = entries.begin(); it != entries.end();)
    {
        if (auto entry = it->lock())
        {
            if (entry->mesh == &mesh && entry->owned.getRawPtr() == owned.getRawPtr() &&
                entry->overlap.getRawPtr() == overlap.getRawPtr())
                cached = entry;
            ++it;
        }
        else
            it = entries.erase(it);
    }
    // Import construction is collective. Rank-local field lifetimes may
    // differ, so only reuse when every rank has a live matching layout.
    const int local_hit = cached ? 1 : 0;
    int all_hit = 0;
    Teuchos::reduceAll(*owned->getComm(), Teuchos::REDUCE_MIN, 1, &local_hit, &all_hit);
    if (all_hit)
        return cached;
    std::shared_ptr<StoredFieldLayout> entry;
    int local_valid = 1, all_valid = 0;
    try
    {
        entry = std::make_shared<StoredFieldLayout>();
        entry->mesh = &mesh;
        entry->owned = owned;
        entry->overlap = overlap;
        entry->owned_face_ids = make_ids();
        entries.reserve(entries.size() + 1);
    }
    catch (...)
    {
        local_valid = 0;
    }
    Teuchos::reduceAll(*owned->getComm(), Teuchos::REDUCE_MIN, 1, &local_valid, &all_valid);
    if (!all_valid)
        throw std::runtime_error("FieldStored layout allocation or owned-face validation failed on a rank.");
    entry->importer = Teuchos::rcp(new import_type(owned, overlap));
    entries.emplace_back(entry);
    return entry;
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline auto FieldStored<FieldType, Pack, MeshType>::component_value(local_ordinal_type local_id, size_t component) const
    -> scalar_type
{
    check_component(component);
    if constexpr (value_traits::is_vector)
    {
        return d_owned.getData(component)[owned_row(local_id)];
    }
    else
    {
        if (component != 0)
        {
            throw std::out_of_range("Scalar field has only component zero.");
        }
        return value(local_id);
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline auto FieldStored<FieldType, Pack, MeshType>::local_component_value(
    local_ordinal_type local_id, size_t component) const -> scalar_type
{
    check_component(component);
    if constexpr (value_traits::is_vector)
    {
        return d_overlap.getData(component)[overlap_row(local_id)];
    }
    else
    {
        if (component != 0)
        {
            throw std::out_of_range("Scalar field has only component zero.");
        }
        return local_value(local_id);
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline void FieldStored<FieldType, Pack, MeshType>::set_component_value(
    local_ordinal_type local_id, size_t component, scalar_type value)
{
    check_component(component);
    if constexpr (value_traits::is_vector)
    {
        d_owned.replaceLocalValue(owned_row(local_id), component, value);
        d_overlap.replaceLocalValue(overlap_row(local_id), component, value);
    }
    else
    {
        if (component != 0)
        {
            throw std::out_of_range("Scalar field has only component zero.");
        }
        set_value(local_id, value);
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline void FieldStored<FieldType, Pack, MeshType>::set_owned_component_value(
    local_ordinal_type local_id, size_t component, scalar_type value)
{
    check_component(component);
    if constexpr (value_traits::is_vector)
    {
        d_owned.replaceLocalValue(owned_row(local_id), component, value);
    }
    else
    {
        if (component != 0)
        {
            throw std::out_of_range("Scalar field has only component zero.");
        }
        set_owned_value(local_id, value);
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline auto FieldStored<FieldType, Pack, MeshType>::value_at(const storage_type& storage, local_ordinal_type row)
    -> value_type
{
    if constexpr (value_traits::components == 9)
    {
        value_type value{};
        for (size_t component = 0; component < value_traits::components; ++component)
        {
            value[component / 3].component(component % 3) = storage.getData(component)[row];
        }
        return value;
    }
    else if constexpr (value_traits::is_vector)
    {
        return {storage.getData(0)[row], storage.getData(1)[row], storage.getData(2)[row]};
    }
    else
    {
        return storage.getData()[row];
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
inline void FieldStored<FieldType, Pack, MeshType>::set_at(
    storage_type& storage, local_ordinal_type row, const value_type& value)
{
    if constexpr (value_traits::is_vector)
    {
        for (size_t component = 0; component < value_traits::components; ++component)
        {
            storage.replaceLocalValue(row, component, value_traits::component(value, component));
        }
    }
    else
    {
        storage.replaceLocalValue(row, value);
    }
}

} // namespace SimpleFluid
