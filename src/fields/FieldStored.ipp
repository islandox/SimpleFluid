/** @file FieldStored.ipp
 * @brief Caller-owned descriptor construction, indexed access and generic layout acquisition.
 */
#pragma once

#include "fields/FieldStored.hh"
#include <Teuchos_CommHelpers.hpp>

namespace SimpleFluid
{

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline FieldStored<Descriptor, Pack, MeshT>::FieldStored(SP<const mesh_type> mesh, std::string name, bool zero_out)
    : FieldStored(descriptor_type(std::move(name)), std::move(mesh), zero_out)
{
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline FieldStored<Descriptor, Pack, MeshT>::FieldStored(
    SP<const mesh_type> mesh, const value_type& initial_value, std::string name)
    : FieldStored(descriptor_type(std::move(name)), std::move(mesh), initial_value)
{
}

template<TpetraTypePack Pack, MeshType MeshT, class Location>
template<class MakeIds>
inline auto detail::StoredFieldLayout<Pack, MeshT, Location>::acquire(const MeshT& mesh,
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

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline auto FieldStored<Descriptor, Pack, MeshT>::component_value(local_ordinal_type local_id, size_t component) const
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

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline auto FieldStored<Descriptor, Pack, MeshT>::local_component_value(
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

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline void FieldStored<Descriptor, Pack, MeshT>::set_component_value(
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

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline void FieldStored<Descriptor, Pack, MeshT>::set_owned_component_value(
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

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline auto FieldStored<Descriptor, Pack, MeshT>::row_for_global(
    const storage_type& storage, global_ordinal_type global_id) -> local_ordinal_type
{
    const auto row = storage.getMap()->getLocalElement(global_id);
    if (row == invalid_local_ordinal())
    {
        throw std::out_of_range("Field global ID is not available in the selected map.");
    }
    return row;
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline auto FieldStored<Descriptor, Pack, MeshT>::global_component_value(
    global_ordinal_type global_id, size_t component) const -> scalar_type
{
    check_component(component);
    const auto row = row_for_global(d_owned, global_id);
    if constexpr (value_traits::is_vector)
        return d_owned.getData(component)[row];
    else
        return d_owned.getData()[row];
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline void FieldStored<Descriptor, Pack, MeshT>::set_global_value(global_ordinal_type global_id, const value_type& value)
{
    set_at(d_owned, row_for_global(d_owned, global_id), value);
    set_at(d_overlap, row_for_global(d_overlap, global_id), value);
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline void FieldStored<Descriptor, Pack, MeshT>::set_global_component_value(
    global_ordinal_type global_id, size_t component, scalar_type value)
{
    check_component(component);
    const auto owned = row_for_global(d_owned, global_id);
    const auto overlap = row_for_global(d_overlap, global_id);
    if constexpr (value_traits::is_vector)
    {
        d_owned.replaceLocalValue(owned, component, value);
        d_overlap.replaceLocalValue(overlap, component, value);
    }
    else
    {
        d_owned.replaceLocalValue(owned, value);
        d_overlap.replaceLocalValue(overlap, value);
    }
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline void FieldStored<Descriptor, Pack, MeshT>::sum_into_global_value(global_ordinal_type global_id, scalar_type value)
    requires (!value_traits::is_vector)
{
    d_owned.sumIntoLocalValue(row_for_global(d_owned, global_id), value);
    d_overlap.sumIntoLocalValue(row_for_global(d_overlap, global_id), value);
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline auto FieldStored<Descriptor, Pack, MeshT>::boundary_face_local_id(
    int boundary_id, local_ordinal_type in_batch_id) const -> local_ordinal_type
    requires std::same_as<location_type, BoundaryFaceLocation>
{
    const auto& batch = d_mesh->boundary_face_batch(boundary_id);
    CHECK_BOUNDS(in_batch_id, 0, batch.face_lids.size());
    return batch.face_lids[static_cast<size_t>(in_batch_id)];
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline size_t FieldStored<Descriptor, Pack, MeshT>::tensor_component_index(size_t row, size_t column)
{
    if (row >= 3 || column >= 3)
    {
        throw std::out_of_range("Tensor field row or column is out of range.");
    }
    return row * 3 + column;
}

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline auto FieldStored<Descriptor, Pack, MeshT>::value_at(const storage_type& storage, local_ordinal_type row)
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

template<FieldType Descriptor, TpetraTypePack Pack, MeshType MeshT>
inline void FieldStored<Descriptor, Pack, MeshT>::set_at(
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
