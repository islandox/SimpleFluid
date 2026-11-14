/** @file FieldStored.tcc
 * @brief Compiled field construction, allocation and bulk initialization.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

template<class FieldType, TpetraTypePack Pack, class MeshType>
FieldStored<FieldType, Pack, MeshType>::FieldStored(descriptor_type descriptor, SP<const mesh_type> mesh, bool zero_out)
    : d_descriptor(std::move(descriptor)), d_mesh(require_mesh(std::move(mesh))),
      d_owned(make_storage(map_for<false>(*d_mesh), zero_out)), d_overlap(make_storage(map_for<true>(*d_mesh), false)),
      d_layout(layout_type::acquire(
          *d_mesh, d_owned.getMap(), d_overlap.getMap(), [this] { return make_owned_face_ids(*d_mesh); })),
      d_import(d_layout->importer)
{
    if (zero_out)
    {
        sync_ghosts();
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
FieldStored<FieldType, Pack, MeshType>::FieldStored(
    descriptor_type descriptor, SP<const mesh_type> mesh, const value_type& initial_value)
    : FieldStored(std::move(descriptor), std::move(mesh), false)
{
    put_value(initial_value);
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
void FieldStored<FieldType, Pack, MeshType>::put_value(const value_type& value)
{
    if constexpr (value_traits::is_vector)
    {
        for (size_t component = 0; component < value_traits::components; ++component)
        {
            d_owned.getVectorNonConst(component)->putScalar(value_traits::component(value, component));
            d_overlap.getVectorNonConst(component)->putScalar(value_traits::component(value, component));
        }
    }
    else
    {
        d_owned.putScalar(value);
        d_overlap.putScalar(value);
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
auto FieldStored<FieldType, Pack, MeshType>::make_storage(const Teuchos::RCP<const map_type>& map, bool zero_out)
    -> storage_type
{
    if constexpr (value_traits::is_vector)
    {
        return storage_type(map, value_traits::components, zero_out);
    }
    else
    {
        return storage_type(map, zero_out);
    }
}

template<class FieldType, TpetraTypePack Pack, class MeshType>
auto FieldStored<FieldType, Pack, MeshType>::make_owned_face_ids(const mesh_type& mesh)
    -> std::vector<local_ordinal_type>
{
    std::vector<local_ordinal_type> result;
    if constexpr (std::is_same_v<location_type, FaceLocation>)
    {
        result.reserve(mesh.num_owned_faces());
        for (size_t face = 0; face < mesh.num_faces(); ++face)
        {
            const auto face_lid = static_cast<local_ordinal_type>(face);
            if (mesh.is_owned_face(face_lid))
            {
                result.push_back(face_lid);
            }
        }
        if (result.size() != mesh.num_owned_faces())
        {
            throw std::runtime_error("FieldStored owned-face IDs do not match the owned map.");
        }
        for (size_t row = 0; row < result.size(); ++row)
        {
            if (result[row] != static_cast<local_ordinal_type>(row))
            {
                throw std::runtime_error("FieldStored requires owned faces to form a "
                                         "local-ID prefix.");
            }
        }
    }
    return result;
}

} // namespace SimpleFluid
