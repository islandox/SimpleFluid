/**
 * @file VectorFaceField.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Three-component face-centered vector field backed by Tpetra::MultiVector.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/FaceFieldBase.hh"

#include <cstddef>
#include <stdexcept>
#include <string>
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
class VectorFaceField : public FaceFieldBase<Pack,
                                              VectorFaceField<Pack>,
                                              typename Pack::multi_vector_type>
{
public:
    using base_type = FaceFieldBase<Pack,
                                     VectorFaceField<Pack>,
                                     typename Pack::multi_vector_type>;
    using mesh_type = typename base_type::mesh_type;
    using vector_type = typename base_type::vector_type;
    using map_type = typename base_type::map_type;
    using scalar_type = typename base_type::scalar_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using global_ordinal_type = typename base_type::global_ordinal_type;
    using vec_type = typename mesh_type::Vec3;

    static constexpr size_t num_components = 3;

    template <class T>
    using RCP = typename base_type::template RCP<T>;

    explicit VectorFaceField(SP<const mesh_type> mesh,
                             std::string name = std::string(),
                             bool zero_out = true);

    VectorFaceField(SP<const mesh_type> mesh,
                    const vec_type& initial_value,
                    std::string name = std::string());

    using base_type::name;
    using base_type::set_name;
    using base_type::mesh;
    using base_type::mesh_ptr;
    using base_type::map;
    using base_type::num_owned_faces;
    using base_type::owned_face_ids;
    using base_type::face_global_id;
    using base_type::is_owned_face;
    using base_type::is_owned_global_face;

    vector_type& data() noexcept { return this->d_data; }
    const vector_type& data() const noexcept { return this->d_data; }

    void put_scalar(const vec_type& value);

    vec_type value(local_ordinal_type face_lid) const;
    vec_type global_value(global_ordinal_type face_gid) const;

    scalar_type component_value(local_ordinal_type face_lid,
                                size_t component) const;
    scalar_type global_component_value(global_ordinal_type face_gid,
                                       size_t component) const;

    void set_value(local_ordinal_type face_lid, const vec_type& value);
    void set_global_value(global_ordinal_type face_gid, const vec_type& value);
    void set_component_value(local_ordinal_type face_lid,
                             size_t component,
                             const scalar_type& value);
    void set_global_component_value(global_ordinal_type face_gid,
                                    size_t component,
                                    const scalar_type& value);

private:
    static void check_component(size_t component);
};

/**
 * @brief Construct a vector face field with three components.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Optional field name for I/O.
 * @param zero_out If true, initialize all entries to zero.
 */
template<TpetraTypePack Pack>
VectorFaceField<Pack>::VectorFaceField(SP<const mesh_type> mesh,
                                       std::string name,
                                       bool zero_out)
    : base_type(std::move(name), mesh)
{
    const auto owned_face_map =
        base_type::make_owned_face_map(mesh, "VectorFaceField",
                                       this->d_owned_face_ids);
    this->d_data = vector_type(owned_face_map, num_components, zero_out);
}

/**
 * @brief Construct a vector face field initialized with a uniform vector value.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param initial_value Vector value to fill all owned face entries.
 * @param name Optional field name for I/O.
 */
template<TpetraTypePack Pack>
VectorFaceField<Pack>::VectorFaceField(SP<const mesh_type> mesh,
                                       const vec_type& initial_value,
                                       std::string name)
    : VectorFaceField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

/**
 * @brief Validate that a component index is in range [0, num_components).
 *
 * @tparam Pack Tpetra type pack.
 * @param component Component index to validate.
 * @throws std::out_of_range if @p component is out of bounds.
 */
template<TpetraTypePack Pack>
void VectorFaceField<Pack>::check_component(size_t component)
{
    if (component >= num_components)
    {
        throw std::out_of_range(
            "VectorFaceField component index is out of bounds.");
    }
}

/**
 * @brief Read the vector value stored at an owned face.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_lid Local ID of the face.
 * @return 3D vector stored at the face.
 * @throws std::out_of_range if @p face_lid is out of bounds or not owned.
 */
template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::value(local_ordinal_type face_lid) const -> vec_type
{
    return {
        component_value(face_lid, 0),
        component_value(face_lid, 1),
        component_value(face_lid, 2)
    };
}

/**
 * @brief Read the vector value stored at a face identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_gid Global ID of the face.
 * @return 3D vector stored at the face.
 * @throws std::out_of_range if @p face_gid is not owned by this rank.
 */
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

/**
 * @brief Read a single component value from an owned face.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_lid Local ID of the face.
 * @param component Component index (0, 1, or 2).
 * @return Scalar component value.
 * @throws std::out_of_range if @p face_lid is out of bounds/not owned, or @p component is invalid.
 */
template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::component_value(
    local_ordinal_type face_lid,
    size_t component) const -> scalar_type
{
    check_component(component);
    return this->d_data.getData(component)[this->owned_row_for_face(face_lid)];
}

/**
 * @brief Read a single component value from a face identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_gid Global ID of the face.
 * @param component Component index (0, 1, or 2).
 * @return Scalar component value.
 * @throws std::out_of_range if @p face_gid is not owned or @p component is invalid.
 */
template<TpetraTypePack Pack>
auto VectorFaceField<Pack>::global_component_value(
    global_ordinal_type face_gid,
    size_t component) const -> scalar_type
{
    check_component(component);
    return this->d_data.getData(component)[
        this->owned_row_for_global_face(face_gid)];
}

/**
 * @brief Set all owned face entries to a uniform vector value.
 *
 * @tparam Pack Tpetra type pack.
 * @param value Vector value to assign to every owned face (per component).
 */
template<TpetraTypePack Pack>
void VectorFaceField<Pack>::put_scalar(const vec_type& value)
{
    for (size_t c = 0; c < num_components; ++c)
    {
        this->d_data.getVectorNonConst(c)->putScalar(value.component(c));
    }
}

/**
 * @brief Write a 3D vector to an owned face.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_lid Local ID of the face.
 * @param value Vector value to store.
 * @throws std::out_of_range if @p face_lid is out of bounds or not owned.
 */
template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_value(local_ordinal_type face_lid,
                                      const vec_type& value)
{
    for (size_t c = 0; c < num_components; ++c)
        set_component_value(face_lid, c, value.component(c));
}

/**
 * @brief Write a 3D vector to a face identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_gid Global ID of the face.
 * @param value Vector value to store.
 * @throws std::out_of_range if @p face_gid is not owned by this rank.
 */
template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_global_value(global_ordinal_type face_gid,
                                             const vec_type& value)
{
    for (size_t c = 0; c < num_components; ++c)
        set_global_component_value(face_gid, c, value.component(c));
}

/**
 * @brief Write a single component value to an owned face.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_lid Local ID of the face.
 * @param component Component index (0, 1, or 2).
 * @param value Scalar value to store for the given component.
 * @throws std::out_of_range if @p face_lid is out of bounds/not owned, or @p component is invalid.
 */
template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_component_value(
    local_ordinal_type face_lid,
    size_t component,
    const scalar_type& value)
{
    check_component(component);
    this->d_data.replaceLocalValue(
        this->owned_row_for_face(face_lid), component, value);
}

/**
 * @brief Write a single component value to a face identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param face_gid Global ID of the face.
 * @param component Component index (0, 1, or 2).
 * @param value Scalar value to store for the given component.
 * @throws std::out_of_range if @p face_gid is not owned or @p component is invalid.
 */
template<TpetraTypePack Pack>
void VectorFaceField<Pack>::set_global_component_value(
    global_ordinal_type face_gid,
    size_t component,
    const scalar_type& value)
{
    check_component(component);
    this->d_data.replaceLocalValue(
        this->owned_row_for_global_face(face_gid), component, value);
}

extern template class VectorFaceField<DefaultTpetraTypes>;

} // namespace SimpleFluid
