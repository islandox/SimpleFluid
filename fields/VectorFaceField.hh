/**
 * @file VectorFaceField.hh
 * @brief Three-component face-centered vector field backed by Tpetra::MultiVector.
 */
#pragma once

#include "geometry/Mesh.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Vector field defined on mesh faces owned by locally owned cells.
 *
 * Stores x/y/z components in a Tpetra MultiVector. A face is owned by this
 * field when its owner cell is locally owned by the mesh.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class VectorFaceField
{
public:
    using mesh_type = Mesh<Pack>;
    using vector_type = typename Pack::multi_vector_type;
    using map_type = typename Pack::map_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using vec_type = typename mesh_type::Vec3;

    static constexpr std::size_t num_components = 3;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    explicit VectorFaceField(SP<const mesh_type> mesh,
                             std::string name = std::string(),
                             bool zero_out = true);

    VectorFaceField(SP<const mesh_type> mesh,
                    const vec_type& initial_value,
                    std::string name = std::string());

    const std::string& name() const noexcept { return d_name; }
    void set_name(std::string name) { d_name = std::move(name); }

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    RCP<const map_type> map() const { return d_data.getMap(); }

    vector_type& data() noexcept { return d_data; }
    const vector_type& data() const noexcept { return d_data; }

    std::size_t num_owned_faces() const noexcept
    {
        return d_owned_face_ids.size();
    }

    const std::vector<local_ordinal_type>& owned_face_ids() const noexcept
    {
        return d_owned_face_ids;
    }

    void put_scalar(const vec_type& value);

    global_ordinal_type face_global_id(local_ordinal_type face_lid) const;

    vec_type value(local_ordinal_type face_lid) const;
    vec_type global_value(global_ordinal_type face_gid) const;

    scalar_type component_value(local_ordinal_type face_lid,
                                std::size_t component) const;
    scalar_type global_component_value(global_ordinal_type face_gid,
                                       std::size_t component) const;

    void set_value(local_ordinal_type face_lid, const vec_type& value);
    void set_global_value(global_ordinal_type face_gid, const vec_type& value);
    void set_component_value(local_ordinal_type face_lid,
                             std::size_t component,
                             const scalar_type& value);
    void set_global_component_value(global_ordinal_type face_gid,
                                    std::size_t component,
                                    const scalar_type& value);

    bool is_owned_face(local_ordinal_type face_lid) const;
    bool is_owned_global_face(global_ordinal_type face_gid) const;

private:
    static RCP<const map_type> make_owned_face_map(
        const SP<const mesh_type>& mesh,
        std::vector<local_ordinal_type>& owned_face_ids,
        std::vector<local_ordinal_type>& face_lid_to_owned_row);

    static local_ordinal_type invalid_owned_row()
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    static void check_component(std::size_t component);

    local_ordinal_type owned_row_for_face(local_ordinal_type face_lid) const;
    local_ordinal_type owned_row_for_global_face(global_ordinal_type face_gid) const;
    void check_face_lid(local_ordinal_type face_lid) const;

    std::string d_name;
    SP<const mesh_type> d_mesh;
    std::vector<local_ordinal_type> d_owned_face_ids;
    std::vector<local_ordinal_type> d_face_lid_to_owned_row;
    vector_type d_data;
};

template<TpetraTypePack Pack>
VectorFaceField<Pack>::VectorFaceField(SP<const mesh_type> mesh,
                                       std::string name,
                                       bool zero_out)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_owned_face_ids(),
      d_face_lid_to_owned_row(),
      d_data(make_owned_face_map(d_mesh, d_owned_face_ids, d_face_lid_to_owned_row),
             num_components,
             zero_out)
{
}

template<TpetraTypePack Pack>
VectorFaceField<Pack>::VectorFaceField(SP<const mesh_type> mesh,
                                       const vec_type& initial_value,
                                       std::string name)
    : VectorFaceField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::make_owned_face_map(
    const SP<const mesh_type>& mesh,
    std::vector<local_ordinal_type>& owned_face_ids,
    std::vector<local_ordinal_type>& face_lid_to_owned_row)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("VectorFaceField requires a non-null mesh.");
    }

    if (mesh->owned_cell_map() == Teuchos::null)
    {
        throw std::runtime_error(
            "VectorFaceField requires an assembled mesh with an owned-cell map.");
    }

    owned_face_ids.clear();
    owned_face_ids.reserve(mesh->num_faces());
    face_lid_to_owned_row.assign(mesh->num_faces(), invalid_owned_row());

    for (std::size_t fid = 0; fid < mesh->num_faces(); ++fid)
    {
        const auto face_lid =
            detail::checked_size_to_ordinal<local_ordinal_type>(fid, "face local id");
        const auto owner = mesh->owner_cell(face_lid);

        if (mesh->is_owned_cell(owner))
        {
            const auto owned_row =
                detail::checked_size_to_ordinal<local_ordinal_type>(
                    owned_face_ids.size(), "owned face row");
            owned_face_ids.push_back(face_lid);
            face_lid_to_owned_row[fid] = owned_row;
        }
    }

    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;
    const auto comm = Tpetra::getDefaultComm();

    return Teuchos::rcp(new map_type(invalid_global_size,
                                     owned_face_ids.size(),
                                     index_base,
                                     comm));
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::check_component(std::size_t component)
{
    if (component >= num_components)
    {
        throw std::out_of_range("VectorFaceField component index is out of bounds.");
    }
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::check_face_lid(local_ordinal_type face_lid) const
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (face_lid < 0)
        {
            throw std::out_of_range("Face local id cannot be negative: "
                                  + std::to_string(face_lid));
        }
    }

    if (static_cast<std::size_t>(face_lid) >= d_face_lid_to_owned_row.size())
    {
        throw std::out_of_range("Face local id is out of bounds: "
                              + std::to_string(face_lid));
    }
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::owned_row_for_face(local_ordinal_type face_lid) const
    -> local_ordinal_type
{
    check_face_lid(face_lid);

    const auto owned_row =
        d_face_lid_to_owned_row[static_cast<std::size_t>(face_lid)];
    if (owned_row == invalid_owned_row())
    {
        throw std::out_of_range("Face local id is not owned by this rank: "
                              + std::to_string(face_lid));
    }

    return owned_row;
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::owned_row_for_global_face(
    global_ordinal_type face_gid) const -> local_ordinal_type
{
    const auto owned_row = d_data.getMap()->getLocalElement(face_gid);
    if (owned_row == invalid_owned_row())
    {
        throw std::out_of_range("Face global id is not owned by this rank: "
                              + std::to_string(face_gid));
    }

    return owned_row;
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::face_global_id(local_ordinal_type face_lid) const
    -> global_ordinal_type
{
    return d_data.getMap()->getGlobalElement(owned_row_for_face(face_lid));
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::value(local_ordinal_type face_lid) const -> vec_type
{
    return {
        component_value(face_lid, 0),
        component_value(face_lid, 1),
        component_value(face_lid, 2)
    };
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::global_value(global_ordinal_type face_gid) const
    -> vec_type
{
    return {
        global_component_value(face_gid, 0),
        global_component_value(face_gid, 1),
        global_component_value(face_gid, 2)
    };
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::component_value(
    local_ordinal_type face_lid,
    std::size_t component) const -> scalar_type
{
    check_component(component);
    return d_data.getData(component)[owned_row_for_face(face_lid)];
}

template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::global_component_value(
    global_ordinal_type face_gid,
    std::size_t component) const -> scalar_type
{
    check_component(component);
    return d_data.getData(component)[owned_row_for_global_face(face_gid)];
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::put_scalar(const vec_type& value)
{
    for (std::size_t component = 0; component < num_components; ++component)
    {
        d_data.getVectorNonConst(component)->putScalar(value.component(component));
    }
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_value(local_ordinal_type face_lid,
                                      const vec_type& value)
{
    for (std::size_t component = 0; component < num_components; ++component)
    {
        set_component_value(face_lid, component, value.component(component));
    }
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_global_value(global_ordinal_type face_gid,
                                             const vec_type& value)
{
    for (std::size_t component = 0; component < num_components; ++component)
    {
        set_global_component_value(face_gid, component, value.component(component));
    }
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_component_value(
    local_ordinal_type face_lid,
    std::size_t component,
    const scalar_type& value)
{
    check_component(component);
    d_data.replaceLocalValue(owned_row_for_face(face_lid), component, value);
}

template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_global_component_value(
    global_ordinal_type face_gid,
    std::size_t component,
    const scalar_type& value)
{
    check_component(component);
    d_data.replaceLocalValue(owned_row_for_global_face(face_gid), component, value);
}

template<TpetraTypePack Pack>
bool VectorFaceField<Pack>::is_owned_face(local_ordinal_type face_lid) const
{
    check_face_lid(face_lid);
    return d_face_lid_to_owned_row[static_cast<std::size_t>(face_lid)]
        != invalid_owned_row();
}

template<TpetraTypePack Pack>
bool VectorFaceField<Pack>::is_owned_global_face(global_ordinal_type face_gid) const
{
    const auto row = d_data.getMap()->getLocalElement(face_gid);
    return row != invalid_owned_row();
}

} // namespace SimpleFluid
